// Ampable.app Burn and Send Contract (ETH to Arbitrum for $AMPBL)
pragma solidity ^0.8.0;

/* Minimal ERC20 interface */
interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

/* Updated Arbitrum Inbox interface for L1-to-L2 messaging.
   This interface now exposes createRetryableTicket and calculateRetryableSubmissionFee,
   as defined in Arbitrum’s Nitro contracts.
*/
interface IArbInbox {
    function createRetryableTicket(
        address to,
        uint256 l2CallValue,
        uint256 maxSubmissionCost,
        address excessFeeRefundAddress,
        address callValueRefundAddress,
        uint256 gasLimit,
        uint256 maxFeePerGas,
        bytes calldata data
    ) external payable returns (uint256);

    function calculateRetryableSubmissionFee(uint256 dataLength, uint256 baseFee) external view returns (uint256);
}

contract BridgeBurn {
    address public owner;
    // Burn address: tokens sent here are irretrievable.
    address public constant BURN_ADDRESS = 0x000000000000000000000000000000000000dEaD;

    // Mapping from ERC20 token address to conversion rate (DAO tokens per token unit burned).
    mapping(address => uint256) public tokenRates;
    // Dynamic array to keep track of supported ERC20 token addresses.
    address[] public supportedTokens;

    // Default conversion rate for ETH: 1 ETH = 150 DAO tokens.
    uint256 public constant ETH_RATE = 150;

    // Default gas parameters for retryable ticket creation.
    uint256 public constant DEFAULT_GAS_LIMIT = 1000000;
    uint256 public constant DEFAULT_MAX_FEE_PER_GAS = 2000000000; // 2 Gwei

    // Hardcoded Arbitrum Delayed Inbox address on Ethereum.
    IArbInbox public arbInbox = IArbInbox(0x4Dbd4fc535Ac27206064B68FfCf827b0A60BAB3f);
    // Hardcoded destination L2 contract (BridgeMint) address on Arbitrum.
    address public l2Destination = 0x9c7A755378D108d667568851356428635C24EA61;

    // UNI token is preset to a conversion rate of 1:1.
    address public constant UNI = 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984;

    // Ledger tracking the total DAO tokens minted per user via this bridge.
    mapping(address => uint256) public mintedLedger;

    event TokenBurned(
        address indexed user,
        address indexed token,
        uint256 amountBurned,
        uint256 rate,
        uint256 mintAmount
    );
    event EthBurned(
        address indexed user,
        uint256 ethBurned,
        uint256 rate,
        uint256 mintAmount
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        // Pre-set UNI conversion rate to 1:1 and add UNI to the supported tokens list.
        tokenRates[UNI] = 1;
        supportedTokens.push(UNI);
    }

    ////////// Owner Functions //////////

    /// @notice Add a token to the supported list with its conversion rate.
    function addToken(address token, uint256 rate) external onlyOwner {
        require(token != address(0), "Invalid token");
        require(rate > 0, "Rate must be > 0");
        if (tokenRates[token] == 0) {
            supportedTokens.push(token);
        }
        tokenRates[token] = rate;
    }

    /// @notice Remove a token from the supported list.
    function removeToken(address token) external onlyOwner {
        require(tokenRates[token] > 0, "Token not supported");
        delete tokenRates[token];
        for (uint i = 0; i < supportedTokens.length; i++) {
            if (supportedTokens[i] == token) {
                supportedTokens[i] = supportedTokens[supportedTokens.length - 1];
                supportedTokens.pop();
                break;
            }
        }
    }

    /// @notice Update the conversion rate of a supported token.
    function updateTokenRate(address token, uint256 newRate) external onlyOwner {
        require(tokenRates[token] > 0, "Token not supported");
        require(newRate > 0, "Rate must be > 0");
        tokenRates[token] = newRate;
    }

    /// @notice Update the destination L2 address.
    function updateL2Destination(address newL2Dest) external onlyOwner {
        require(newL2Dest != address(0), "Invalid L2 destination");
        l2Destination = newL2Dest;
    }

    /// @notice Update the Arbitrum Inbox address.
    function updateArbInbox(address newInbox) external onlyOwner {
        require(newInbox != address(0), "Invalid Inbox address");
        arbInbox = IArbInbox(newInbox);
    }

    ////////// View Functions //////////

    function getAllTokenRates() external view returns (address[] memory tokens, uint256[] memory rates) {
        uint256 len = supportedTokens.length;
        tokens = new address[](len + 1);
        rates = new uint256[](len + 1);
        tokens[0] = address(0);
        rates[0] = ETH_RATE;
        for (uint i = 0; i < len; i++) {
            tokens[i + 1] = supportedTokens[i];
            rates[i + 1] = tokenRates[supportedTokens[i]];
        }
    }

    function calculateMintAmount(address token, uint256 amount) external view returns (uint256 mintAmount) {
        if (token == address(0)) {
            mintAmount = amount * ETH_RATE;
        } else {
            uint256 rate = tokenRates[token];
            require(rate > 0, "Token not supported");
            mintAmount = amount * rate;
        }
    }

    function getUserMinted(address user) external view returns (uint256) {
        return mintedLedger[user];
    }

    ////////// Burn and Relay Functions //////////

    /**
     * @notice Burn an ERC20 token and send a cross-chain message to L2.
     * The user must have approved this contract.
     * @param token The ERC20 token address.
     * @param amount The amount to burn (in token's smallest units).
     */
    function burnAndSend(address token, uint256 amount) external payable {
        uint256 rate = tokenRates[token];
        require(rate > 0, "Token not supported");
        require(amount > 0, "Amount must be > 0");

        require(IERC20(token).transferFrom(msg.sender, BURN_ADDRESS, amount), "Token transfer failed");

        uint256 mintAmount = amount * rate;
        mintedLedger[msg.sender] += mintAmount;

        emit TokenBurned(msg.sender, token, amount, rate, mintAmount);

        // Encode the call to mintFromBridge on L2.
        bytes memory l2Data = abi.encodeWithSignature("mintFromBridge(address,uint256)", msg.sender, mintAmount);

        // Calculate submission fee for the retryable ticket.
        uint256 submissionCost = arbInbox.calculateRetryableSubmissionFee(l2Data.length, block.basefee);
        // Call createRetryableTicket with parameters.
        arbInbox.createRetryableTicket{value: msg.value}(
            l2Destination,      // L2 destination contract address
            0,                  // l2CallValue (no additional ETH sent to L2)
            submissionCost,     // maxSubmissionCost
            msg.sender,         // excessFeeRefundAddress (refund any excess fees)
            msg.sender,         // callValueRefundAddress (refund l2CallValue if needed)
            DEFAULT_GAS_LIMIT,  // gasLimit for L2 execution
            DEFAULT_MAX_FEE_PER_GAS, // maxFeePerGas for L2 execution
            l2Data              // Encoded L2 call data
        );
    }

    /**
     * @notice Burn ETH and send a cross-chain message to L2.
     * The caller specifies the ETH amount to "burn" (converted at ETH_RATE).
     * Any extra ETH is used to cover the L2 message fee.
     * @param burnAmount The amount of ETH (in wei) to burn.
     */
    function burnAndSendETH(uint256 burnAmount) external payable {
        require(burnAmount > 0, "Burn amount must be > 0");
        require(msg.value >= burnAmount, "Insufficient ETH sent");
        uint256 fee = msg.value - burnAmount;

        // Burn ETH by sending it to the burn address.
        (bool sent, ) = BURN_ADDRESS.call{value: burnAmount}("");
        require(sent, "ETH burn failed");

        uint256 mintAmount = burnAmount * ETH_RATE;
        mintedLedger[msg.sender] += mintAmount;

        emit EthBurned(msg.sender, burnAmount, ETH_RATE, mintAmount);

        // Encode the call to mintFromBridge on L2.
        bytes memory l2Data = abi.encodeWithSignature("mintFromBridge(address,uint256)", msg.sender, mintAmount);

        uint256 submissionCost = arbInbox.calculateRetryableSubmissionFee(l2Data.length, block.basefee);
        arbInbox.createRetryableTicket{value: fee}(
            l2Destination,      // L2 destination contract address
            0,                  // l2CallValue
            submissionCost,     // maxSubmissionCost
            msg.sender,         // excessFeeRefundAddress
            msg.sender,         // callValueRefundAddress
            DEFAULT_GAS_LIMIT,  // gasLimit
            DEFAULT_MAX_FEE_PER_GAS, // maxFeePerGas
            l2Data              // Encoded call data
        );
    }

    // Allow the contract to receive ETH.
    receive() external payable {}
}