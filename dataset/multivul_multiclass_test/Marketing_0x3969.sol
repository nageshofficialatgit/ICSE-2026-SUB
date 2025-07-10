// SPDX-License-Identifier: MIT
// File: @openzeppelin/contracts/token/ERC20/IERC20.sol


// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC-20 standard as defined in the ERC.
 */
interface IERC20 {
    /**
     * @dev Emitted when `value` tokens are moved from one account (`from`) to
     * another (`to`).
     *
     * Note that `value` may be zero.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Emitted when the allowance of a `spender` for an `owner` is set by
     * a call to {approve}. `value` is the new allowance.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);

    /**
     * @dev Returns the value of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the value of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves a `value` amount of tokens from the caller's account to `to`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address to, uint256 value) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets a `value` amount of tokens as the allowance of `spender` over the
     * caller's tokens.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * IMPORTANT: Beware that changing an allowance with this method brings the risk
     * that someone may use both the old and the new allowance by unfortunate
     * transaction ordering. One possible solution to mitigate this race
     * condition is to first reduce the spender's allowance to 0 and set the
     * desired value afterwards:
     * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
     *
     * Emits an {Approval} event.
     */
    function approve(address spender, uint256 value) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to` using the
     * allowance mechanism. `value` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

// File: @chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol


pragma solidity ^0.8.0;

interface AggregatorV3Interface {
  function decimals() external view returns (uint8);

  function description() external view returns (string memory);

  function version() external view returns (uint256);

  function getRoundData(
    uint80 _roundId
  ) external view returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);

  function latestRoundData()
    external
    view
    returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);
}

// File: contracts/RADWMarketing.sol


pragma solidity ^0.8.0;



contract Marketing {
    address public superAdmin; // Super admin address (deployer)
    address public ownerWallet; // Owner wallet for receiving payments
    IERC20 public radwToken; // RADW token contract
    IERC20 public usdcToken; // USDC token contract
    IERC20 public usdtToken; // USDT token contract
    AggregatorV3Interface internal priceFeed; // Chainlink price feed for ETH/USD
    uint256 public tokenPrice; // Price of RADW token in USD (8 decimals)
    bool public isMarketingEnabled; // Flag to enable or disable buy/sell features

    // Modifier for functions accessible only by the super admin
    modifier onlySuperAdmin() {
        require(msg.sender == superAdmin, "Only super admin can perform this action");
        _;
    }

    // Modifier for functions accessible only by the owner wallet
    modifier onlyOwner() {
        require(msg.sender == ownerWallet, "Only owner can perform this action");
        _;
    }

    // Modifier for functions accessible by either the super admin or owner wallet
    modifier onlyAuthorized() {
        require(
            msg.sender == superAdmin || msg.sender == ownerWallet,
            "Only super admin or owner can perform this action"
        );
        _;
    }

    constructor(
        address _radwTokenAddress,
        address _usdcAddress,
        address _usdtAddress,
        address _aggregatorAddress,
        address _ownerWallet,
        uint256 _initialPrice // Initial price in USD (8 decimals)
    ) {
        superAdmin = msg.sender; // Set the deployer as the super admin
        radwToken = IERC20(_radwTokenAddress);
        usdcToken = IERC20(_usdcAddress);
        usdtToken = IERC20(_usdtAddress);
        priceFeed = AggregatorV3Interface(_aggregatorAddress); // ETH/USD price feed
        ownerWallet = _ownerWallet;
        tokenPrice = _initialPrice; // Example: $5 USD = 500000000 (8 decimals)
        isMarketingEnabled = true; // Marketing is enabled by default
    }

    // Function to update the RADW token address (only super admin)
    function updateRadwTokenAddress(address newRadwTokenAddress) external onlySuperAdmin {
        require(newRadwTokenAddress != address(0), "Invalid RADW token address");
        radwToken = IERC20(newRadwTokenAddress);
    }

    // ------- BUY FUNCTIONALITY -------

    // Function to buy RADW tokens with ETH
    function buyTokensWithETH() external payable {
        require(isMarketingEnabled, "Marketing is disabled");
        require(msg.value > 0, "ETH amount must be greater than 0");

        // Get the latest ETH/USD price
        (, int ethPrice, , , ) = priceFeed.latestRoundData();
        require(ethPrice > 0, "Invalid ETH price");
        uint256 ethUSDPrice = uint256(ethPrice); // ETH/USD price with 8 decimals

        // Calculate the RADW tokens to transfer
        uint256 radwTokens = (msg.value * ethUSDPrice) / tokenPrice;

        // Transfer ETH to the owner wallet
        payable(ownerWallet).transfer(msg.value);

        // Transfer RADW tokens from owner to the buyer
        require(radwToken.transferFrom(ownerWallet, msg.sender, radwTokens), "Token transfer failed");
    }

    // Function to buy RADW tokens with USDC
    function buyTokensWithUSDC(uint256 usdcAmount) external {
        require(isMarketingEnabled, "Marketing is disabled");
        require(usdcAmount > 0, "USDC amount must be greater than 0");

        // Calculate the RADW tokens to transfer
        uint256 radwTokens = (usdcAmount * (10 ** (18 - 6 + 8))) / tokenPrice;

        // Transfer USDC from buyer to owner wallet
        require(usdcToken.transferFrom(msg.sender, ownerWallet, usdcAmount), "USDC transfer failed");

        // Transfer RADW tokens from owner to the buyer
        require(radwToken.transferFrom(ownerWallet, msg.sender, radwTokens), "Token transfer failed");
    }

    // Function to buy RADW tokens with USDT
    function buyTokensWithUSDT(uint256 usdtAmount) external {
        require(isMarketingEnabled, "Marketing is disabled");
        require(usdtAmount > 0, "USDT amount must be greater than 0");

        // Calculate the RADW tokens to transfer
        uint256 radwTokens = (usdtAmount * (10 ** (18 - 6 + 8))) / tokenPrice;

        // Transfer USDT from buyer to owner wallet
        require(usdtToken.transferFrom(msg.sender, ownerWallet, usdtAmount), "USDT transfer failed");

        // Transfer RADW tokens from owner to the buyer
        require(radwToken.transferFrom(ownerWallet, msg.sender, radwTokens), "Token transfer failed");
    }

    // ------- SELL FUNCTIONALITY -------

    // Function to sell RADW tokens for ETH
    function sellTokensForETH(uint256 tokenAmount) external {
        require(isMarketingEnabled, "Marketing is disabled");
        require(tokenAmount > 0, "Token amount must be greater than 0");

        // Get the latest ETH/USD price
        (, int ethPrice, , , ) = priceFeed.latestRoundData();
        require(ethPrice > 0, "Invalid ETH price");
        uint256 ethUSDPrice = uint256(ethPrice); // ETH/USD price with 8 decimals

        // Calculate the ETH equivalent to transfer
        uint256 ethAmount = (tokenAmount * tokenPrice) / ethUSDPrice;

        // Ensure the contract has enough ETH liquidity
        require(address(this).balance >= ethAmount, "Insufficient ETH liquidity");

        // Transfer RADW tokens from the seller to the owner wallet
        require(radwToken.transferFrom(msg.sender, ownerWallet, tokenAmount), "Token transfer failed");

        // Transfer ETH to the seller
        (bool success, ) = payable(msg.sender).call{value: ethAmount}("");
        require(success, "ETH transfer failed");
    }

    // Function to sell RADW tokens for USDC
    function sellTokensForUSDC(uint256 tokenAmount) external {
        require(isMarketingEnabled, "Marketing is disabled");
        require(tokenAmount > 0, "Token amount must be greater than 0");

        // Calculate the USDC equivalent to transfer
        uint256 usdcAmount = (tokenAmount * tokenPrice) / (10 ** (18 - 6 + 8)); // Adjust for USDC decimals (6)

        // Ensure the contract has enough USDC liquidity
        require(usdcToken.balanceOf(address(this)) >= usdcAmount, "Insufficient USDC liquidity");

        // Transfer RADW tokens from the seller to the owner wallet
        require(radwToken.transferFrom(msg.sender, ownerWallet, tokenAmount), "Token transfer failed");

        // Transfer USDC to the seller
        require(usdcToken.transfer(msg.sender, usdcAmount), "USDC transfer failed");
    }

    // Function to sell RADW tokens for USDT
    function sellTokensForUSDT(uint256 tokenAmount) external {
        require(isMarketingEnabled, "Marketing is disabled");
        require(tokenAmount > 0, "Token amount must be greater than 0");

        // Calculate the USDT equivalent to transfer
        uint256 usdtAmount = (tokenAmount * tokenPrice) / (10 ** (18 - 6 + 8)); // Adjust for USDT decimals (6)

        // Ensure the contract has enough USDT liquidity
        require(usdtToken.balanceOf(address(this)) >= usdtAmount, "Insufficient USDT liquidity");

        // Transfer RADW tokens from the seller to the owner wallet
        require(radwToken.transferFrom(msg.sender, ownerWallet, tokenAmount), "Token transfer failed");

        // Transfer USDT to the seller
        require(usdtToken.transfer(msg.sender, usdtAmount), "USDT transfer failed");
    }

    // ------- ADMIN FUNCTIONS -------

    // Function to toggle marketing (only super admin)
    function toggleMarketing(bool enable) external onlySuperAdmin {
        isMarketingEnabled = enable;
    }

    // Function to set the RADW token price (only super admin)
    function setTokenPrice(uint256 newPrice) external onlySuperAdmin {
        require(newPrice > 0, "Price must be greater than 0");
        tokenPrice = newPrice;
    }

    // Function to transfer super admin rights (only super admin)
    function transferSuperAdmin(address newSuperAdmin) external onlySuperAdmin {
        require(newSuperAdmin != address(0), "Invalid address for super admin");
        superAdmin = newSuperAdmin;
    }

    // Function to set the owner wallet (only super admin)
    function setOwnerWallet(address newOwnerWallet) external onlySuperAdmin {
        require(newOwnerWallet != address(0), "Invalid address for owner wallet");
        ownerWallet = newOwnerWallet;
    }

    // ------- LIQUIDITY MANAGEMENT (DEPOSIT/WITHDRAW) -------

    // Deposit ETH into the contract (only authorized accounts)
    receive() external payable {}

    // Deposit USDC (only authorized accounts)
    function depositUSDC(uint256 amount) external onlyAuthorized {
        require(usdcToken.transferFrom(msg.sender, address(this), amount), "USDC transfer failed");
    }

    // Deposit USDT (only authorized accounts)
    function depositUSDT(uint256 amount) external onlyAuthorized {
        require(usdtToken.transferFrom(msg.sender, address(this), amount), "USDT transfer failed");
    }

    // Withdraw ETH (only authorized accounts)
    function withdrawETH(uint256 amount) external onlyAuthorized {
        (bool success, ) = payable(ownerWallet).call{value: amount}("");
        require(success, "ETH withdrawal failed");
    }

    // Withdraw USDC (only authorized accounts)
    function withdrawUSDC(uint256 amount) external onlyAuthorized {
        require(usdcToken.transfer(ownerWallet, amount), "USDC withdrawal failed");
    }

    // Withdraw USDT (only authorized accounts)
    function withdrawUSDT(uint256 amount) external onlyAuthorized {
        require(usdtToken.transfer(ownerWallet, amount), "USDT withdrawal failed");
    }
}