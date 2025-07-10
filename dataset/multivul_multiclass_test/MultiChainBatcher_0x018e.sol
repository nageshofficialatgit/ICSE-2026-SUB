// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

/**
 * @title MultiChainBatcher
 * @notice A multi-chain compatible utility contract for batch transfers, balance queries, and administrative controls across various EVM-compatible chains.
 * @dev Provides functions to distribute native tokens (e.g., ETH, BNB, MATIC) or ERC-20 tokens to multiple addresses, check balances/allowances, and manage fees.
 *      Designed to be deployed on multiple chains such as Ethereum, Binance Smart Chain, Polygon, and other EVM-compatible networks.
 *      Implements gas optimizations and security measures suitable for cross-chain operations.
 */
contract MultiChainBatcher {
    // --- State Variables ---
    address public owner;              /// @notice Owner of the contract (has full privileges)
    address public moderator;          /// @notice Moderator with limited administrative privileges
    uint256 public serviceFeeBP;       /// @notice Service fee in basis points (parts per 10,000) for batch transfers across chains (0 = no fee, 100 = 1%, etc.)
    bool private _notEntered;          // Reentrancy guard state for cross-chain security
    bool public paused;                /// @notice Indicates whether the contract is paused

    // Whitelist and blacklist mappings (using mappings for gas efficiency across chains)
    mapping(address => bool) public whitelisted;        /// @notice Addresses exempted from fees on all chains
    mapping(address => bool) public blacklistedToken;   /// @notice Tokens that are not allowed to be used on any chain
    mapping(address => bool) public blacklistedWallet;  /// @notice Wallets forbidden from using the service on all chains

    // --- Additional storage for iterable lists ---
    address[] private whitelistedAddresses;
    address[] private blacklistedTokenAddresses;
    address[] private blacklistedWalletAddresses;

    // --- Events ---
    event ServiceFeeChanged(uint256 newFeeBP);
    event AddressWhitelisted(address indexed account);
    event AddressUnwhitelisted(address indexed account);
    event TokenBlacklisted(address indexed token);
    event TokenRemovedFromBlacklist(address indexed token);
    event WalletBlacklisted(address indexed wallet);
    event WalletRemovedFromBlacklist(address indexed wallet);
    event ModeratorUpdated(address indexed newModerator);
    event OwnerChanged(address indexed newOwner);
    event FeesWithdrawn(address indexed token, address indexed to, uint256 amount);
    event NativeTokenDistributed(address indexed sender, uint256 totalAmount, uint256 recipientCount, uint256 feeAmount);
    event TokenDistributed(address indexed sender, address indexed token, uint256 totalAmount, uint256 recipientCount, uint256 feeAmount);
    event BalanceDifference(address indexed token, address indexed account, int256 difference, uint256 previousBalance, uint256 newBalance);
    event Received(address indexed sender, uint256 amount);
    event ContractPaused();  /// @notice
    event ContractUnpaused(); /// @notice

    // --- Modifiers ---
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    modifier onlyOwnerOrModerator() {
        require(msg.sender == owner || msg.sender == moderator, "Only owner/moderator");
        _;
    }
    modifier notBlacklistedCaller() {
        require(!blacklistedWallet[msg.sender], "Sender blacklisted");
        _;
    }
    modifier nonReentrant() {
        require(_notEntered, "Reentrant call");
        _notEntered = false;
        _;
        _notEntered = true;
    }
    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }

    constructor() {
        owner = msg.sender;
        serviceFeeBP = 0;
        _notEntered = true;
        paused = false; // Contract starts unpaused
    }

        // --- Internal Fee Calculation Helpers ---

    /**
     * @dev Calculate fee for native token transfers across chains.
     * For native token equal transfers, fee = (msg.value * serviceFeeBP) / (10000 + serviceFeeBP).
     */
    function _calculateFeeNativeToken(uint256 total) internal view returns (uint256) {
        if (whitelisted[msg.sender]) return 0;
        return (total * serviceFeeBP) / (10000 + serviceFeeBP);
    }

    /**
     * @dev Calculate fee for token transfers across chains.
     * For tokens, fee = (totalTokenAmount * serviceFeeBP) / 10000.
     */
    function _calculateFeeToken(uint256 total) internal view returns (uint256) {
        if (whitelisted[msg.sender]) return 0;
        return (total * serviceFeeBP) / 10000;
    }

    // --- Internal utilities for safe token operations ---

    /**
     * @dev Internal function to safely call ERC-20/IERC20 `transferFrom` across chains.
     */
    function _safeTransferFrom(address token, address sender, address recipient, uint256 amount) internal {
        (bool success, bytes memory data) = token.call(
            abi.encodeWithSelector(bytes4(keccak256("transferFrom(address,address,uint256)")), sender, recipient, amount)
        );
        require(success && (data.length == 0 || abi.decode(data, (bool))), "Token transferFrom failed");
    }

    /**
     * @dev Internal function to safely call ERC-20/IERC20 `transfer` across chains.
     */
    function _safeTransfer(address token, address recipient, uint256 amount) internal {
        (bool success, bytes memory data) = token.call(
            abi.encodeWithSelector(bytes4(keccak256("transfer(address,uint256)")), recipient, amount)
        );
        require(success && (data.length == 0 || abi.decode(data, (bool))), "Token transfer failed");
    }

    // --- Pause and Unpause Functions ---
    function pause() external onlyOwner {
        paused = true;
        emit ContractPaused();
    }

    function unpause() external onlyOwner {
        paused = false;
        emit ContractUnpaused();
    }

    // --- Batch Transfer Functions ---

    /**
     * @notice Evenly distribute native tokens (e.g., ETH, BNB, MATIC) among multiple recipients.
     * @param recipients Array of recipient addresses.
     * @dev Requires a minimum of 0.0005 native tokens sent. The fee is deducted and retained in the contract.
     *      This function works across different EVM-compatible chains, adapting to the native token of each chain.
     */
    function batchNativeTokenTransferEqual(address[] calldata recipients) external payable notBlacklistedCaller nonReentrant {
        uint256 count = recipients.length;
        require(count > 0, "No recipients");
        require(msg.value >= 5e14, "Minimum 0.0005 native tokens required");
        uint256 feeAmount = _calculateFeeNativeToken(msg.value);
        uint256 distributionAmount = msg.value - feeAmount;
        require(distributionAmount % count == 0, "Distribution amount not evenly divisible");
        uint256 amountEach = distributionAmount / count;

        for (uint256 i = 0; i < count; ) {
            address recipient = recipients[i];
            require(!blacklistedWallet[recipient], "Recipient blacklisted");
            (bool sent, ) = recipient.call{value: amountEach}("");
            require(sent, "Native token transfer failed");
            unchecked { ++i; }
        }
        emit NativeTokenDistributed(msg.sender, distributionAmount, count, feeAmount);
    }

    /**
     * @notice Distribute custom amounts of native tokens (e.g., ETH, BNB, MATIC) to multiple recipients.
     * @param recipients Array of recipient addresses.
     * @param amounts Array of native token amounts corresponding to each recipient.
     * @dev Requires a minimum of 0.0005 native tokens sent. The caller must send exactly the sum of amounts plus the fee.
     *      This function works across different EVM-compatible chains, adapting to the native token of each chain.
     */
    function batchNativeTokenTransferCustom(address[] calldata recipients, uint256[] calldata amounts) external payable notBlacklistedCaller nonReentrant {
        uint256 count = recipients.length;
        require(count > 0 && count == amounts.length, "Invalid input");
        require(msg.value >= 5e14, "Minimum 0.0005 native tokens required");

        uint256 totalSend = 0;
        for (uint256 i = 0; i < count; ) {
            totalSend += amounts[i];
            unchecked { ++i; }
        }
        uint256 feeAmount = whitelisted[msg.sender] ? 0 : (totalSend * serviceFeeBP) / 10000;
        require(msg.value == totalSend + feeAmount, "Incorrect native token amount sent");

        for (uint256 i = 0; i < count; ) {
            address recipient = recipients[i];
            require(!blacklistedWallet[recipient], "Recipient blacklisted");
            uint256 amount = amounts[i];
            if (amount > 0) {
                (bool sent, ) = recipient.call{value: amount}("");
                require(sent, "Native token transfer failed");
            }
            unchecked { ++i; }
        }
        emit NativeTokenDistributed(msg.sender, totalSend, count, feeAmount);
    }

    /**
     * @notice Evenly distribute a ERC-20/IERC20 token among multiple recipients.
     * @param token Address of the token contract.
     * @param recipients Array of recipient addresses.
     * @param amountEach The amount of tokens each recipient should receive.
     * @dev Requires a minimum per-transfer of 0.0001 tokens. The caller must have approved (totalTransfer + fee).
     */
    function batchTokenTransferEqual(address token, address[] calldata recipients, uint256 amountEach) external notBlacklistedCaller nonReentrant {
        require(token != address(0), "Token address cannot be zero");
        require(!blacklistedToken[token], "Token blacklisted");
        uint256 count = recipients.length;
        require(count > 0, "No recipients");
        require(amountEach >= 1e14, "Each token amount must be at least 0.0001 tokens");
        uint256 totalTransfer = amountEach * count;
        uint256 feeAmount = _calculateFeeToken(totalTransfer);

        _safeTransferFrom(token, msg.sender, address(this), totalTransfer + feeAmount);

        for (uint256 i = 0; i < count; ) {
            address recipient = recipients[i];
            require(!blacklistedWallet[recipient], "Recipient blacklisted");
            _safeTransfer(token, recipient, amountEach);
            unchecked { ++i; }
        }
        emit TokenDistributed(msg.sender, token, totalTransfer, count, feeAmount);
    }

    /**
     * @notice Transfer custom amounts of a ERC-20/IERC20 token to multiple recipients.
     * @param token Address of the token.
     * @param recipients Array of recipient addresses.
     * @param amounts Array of token amounts corresponding to each recipient.
     * @dev Each amount must be at least 0.0001 tokens. Caller must have approved (totalTransfer + fee).
     */
    function batchTokenTransferCustom(address token, address[] calldata recipients, uint256[] calldata amounts) external notBlacklistedCaller nonReentrant {
        require(token != address(0), "Token address cannot be zero");
        require(!blacklistedToken[token], "Token blacklisted");
        uint256 count = recipients.length;
        require(count > 0 && count == amounts.length, "Invalid input");

        uint256 totalTransfer = 0;
        for (uint256 i = 0; i < count; ) {
            require(amounts[i] >= 1e14, "Each token amount must be at least 0.0001 tokens");
            totalTransfer += amounts[i];
            unchecked { ++i; }
        }
        uint256 feeAmount = _calculateFeeToken(totalTransfer);

        _safeTransferFrom(token, msg.sender, address(this), totalTransfer + feeAmount);

        for (uint256 i = 0; i < count; ) {
            address recipient = recipients[i];
            require(!blacklistedWallet[recipient], "Recipient blacklisted");
            _safeTransfer(token, recipient, amounts[i]);
            unchecked { ++i; }
        }
        emit TokenDistributed(msg.sender, token, totalTransfer, count, feeAmount);
    }

    // --- Balance & Allowance Checking Functions ---

    /**
     * @notice Retrieve the allowances of a token for multiple owner addresses to a single spender.
     * @param token Address of the token.
     * @param owners Array of addresses whose allowances will be checked.
     * @param spender The address for which to check allowances.
     * @return allowances An array of allowance values.
     */
    function getAllowances(address token, address[] calldata owners, address spender) external view returns (uint256[] memory allowances) {
        uint256 len = owners.length;
        allowances = new uint256[](len);
        for (uint256 i = 0; i < len; i++) {
            allowances[i] = IERC20(token).allowance(owners[i], spender);
        }
    }

    /**
     * @notice Retrieve a token's symbol and decimals.
     * @param token Address of the token.
     * @return symbol Token symbol string.
     * @return decimals Token decimals.
     * @dev Uses try/catch to handle tokens that might not implement these functions.
     */
    function getTokenMetadata(address token) external view returns (string memory symbol, uint8 decimals) {
        if (token.code.length == 0) {
            return ("", 0);
        }
        try IERC20Metadata(token).symbol() returns (string memory sym) {
            symbol = sym;
        } catch {
            symbol = "";
        }
        try IERC20Metadata(token).decimals() returns (uint8 dec) {
            decimals = dec;
        } catch {
            decimals = 0;
        }
    }

    /**
     * @notice Calculate the percentage of a token's total supply held by a list of addresses.
     * @param token Address of the token.
     * @param holders Array of addresses to check.
     * @return percentageBasisPoints Combined holdings as a percentage of total supply (in basis points).
     */
    function getWalletHoldingsPercentage(address token, address[] calldata holders) external view returns (uint256 percentageBasisPoints) {
        uint256 totalSupply = IERC20(token).totalSupply();
        if (totalSupply == 0) {
            return 0;
        }
        uint256 combinedBalance = 0;
        for (uint256 i = 0; i < holders.length; i++) {
            combinedBalance += IERC20(token).balanceOf(holders[i]);
        }
        percentageBasisPoints = (combinedBalance * 10000) / totalSupply;
    }

    /**
     * @notice Check whether an address is a smart contract.
     * @param addr Address to check.
     * @return True if `addr` contains contract code; false otherwise.
     */
    function isSmartContract(address addr) external view returns (bool) {
        return addr.code.length > 0;
    }

    /**
     * @notice Get balances of multiple tokens for multiple addresses.
     * @param tokens Array of token addresses.
     * @param addresses Array of user addresses.
     * @return balances A 2D array of token balances, indexed as [userIndex][tokenIndex].
     */
    function batchTokenBalances(address[] calldata tokens, address[] calldata addresses)
        external
        view
        returns (uint256[][] memory balances)
    {
        require(tokens.length > 0, "Tokens array is empty");
        require(addresses.length > 0, "Addresses array is empty");

        uint256 addrCount = addresses.length;
        uint256 tokenCount = tokens.length;
        balances = new uint256[][](addrCount);

        for (uint256 i = 0; i < addrCount; ) {
            address addr = addresses[i];
            require(addr != address(0), "Address is zero");
            uint256[] memory addrBalances = new uint256[](tokenCount);
            for (uint256 j = 0; j < tokenCount; ) {
                address tokenAddr = tokens[j];
                require(tokenAddr != address(0), "Token address is zero");
                require(tokenAddr.code.length > 0, "Token address is not a contract");
                (bool success, bytes memory data) =
                    tokenAddr.staticcall(abi.encodeWithSelector(IERC20.balanceOf.selector, addr));
                if (success && data.length >= 32) {
                    addrBalances[j] = abi.decode(data, (uint256));
                } else {
                    addrBalances[j] = 0;
                }
                unchecked { ++j; }
            }
            balances[i] = addrBalances;
            unchecked { ++i; }
        }
    }

    /**
     * @notice Get NativeToken balances for multiple addresses.
     * @param addresses Array of addresses.
     * @return balances An array of NATIVETOKEN balances. batchNativeBalances
     */
    function batchNativeBalances(address[] calldata addresses) external view returns (uint256[] memory balances) {
        require(addresses.length > 0, "Addresses array is empty");

        uint256 len = addresses.length;
        balances = new uint256[](len);
        for (uint256 i = 0; i < len; ) {
            address addr = addresses[i];
            require(addr != address(0), "Address is zero");
            balances[i] = addr.balance;
            unchecked { ++i; }
        }
    }

    // --- Estimate Gas ---

    /**
     * @notice Estimate the gas cost of a batch transfer operation.
     * @param isTokenTransfer True for token transfers, false for NATIVETOKEN transfers.
     * @param numRecipients Number of recipients.
     * @return estimatedGas Approximate gas units expected.
     */
    function estimateBatchGasCost(bool isTokenTransfer, uint256 numRecipients) external pure returns (uint256 estimatedGas) {
        uint256 baseGas = isTokenTransfer ? 50000 : 21000;
        uint256 perRecipientGas = isTokenTransfer ? 55000 : 25000;
        estimatedGas = baseGas + perRecipientGas * numRecipients;
    }

    // --- Security & Admin Functions ---

    /**
     * @notice Set the service fee percentage.
     * @param feeBasisPoints The new fee in basis points (e.g., 100 = 1% fee).
     */
    function setServiceFee(uint256 feeBasisPoints) external onlyOwner {
        require(feeBasisPoints <= 10000, "Fee too high");
        serviceFeeBP = feeBasisPoints;
        emit ServiceFeeChanged(feeBasisPoints);
    }

    /**
     * @notice Add an address to the whitelist (fee exemption).
     * @param account Address to whitelist.
     */
    function addWhitelistedAddress(address account) external onlyOwnerOrModerator {
        require(!whitelisted[account], "Already whitelisted");
        whitelisted[account] = true;
        emit AddressWhitelisted(account);
    }

    /**
     * @notice Remove an address from the whitelist.
     * @param account Address to remove.
     */
    function removeWhitelistedAddress(address account) external onlyOwnerOrModerator {
        require(whitelisted[account], "Not whitelisted");
        whitelisted[account] = false;
        emit AddressUnwhitelisted(account);
    }

    /**
     * @notice Blacklist a token from batch transfers.
     * @param token Address of the token.
     */
    function blacklistToken(address token) external onlyOwnerOrModerator {
        require(!blacklistedToken[token], "Token already blacklisted");
        blacklistedToken[token] = true;
        emit TokenBlacklisted(token);
    }

    /**
     * @notice Remove a token from the blacklist.
     * @param token Address of the token.
     */
    function removeBlacklistedToken(address token) external onlyOwnerOrModerator {
        require(blacklistedToken[token], "Token not blacklisted");
        blacklistedToken[token] = false;
        emit TokenRemovedFromBlacklist(token);
    }

    /**
     * @notice Blacklist a wallet from using batch transfer services.
     * @param wallet Address of the wallet.
     */
    function blacklistWallet(address wallet) external onlyOwnerOrModerator {
        require(!blacklistedWallet[wallet], "Wallet already blacklisted");
        blacklistedWallet[wallet] = true;
        emit WalletBlacklisted(wallet);
    }

    /**
     * @notice Remove a wallet from the blacklist.
     * @param wallet Address of the wallet.
     */
    function removeBlacklistedWallet(address wallet) external onlyOwnerOrModerator {
        require(blacklistedWallet[wallet], "Wallet not blacklisted");
        blacklistedWallet[wallet] = false;
        emit WalletRemovedFromBlacklist(wallet);
    }

    /**
     * @notice Returns the list of all whitelisted addresses.
     */
    function getWhitelistedAddresses() external view returns (address[] memory) {
        return whitelistedAddresses;
    }

    /**
     * @notice Returns the lists of blacklisted token addresses and blacklisted wallet addresses.
     */
    function getBlacklistedAddresses() external view returns (address[] memory blacklistedTokens, address[] memory blacklistedWallets) {
        return (blacklistedTokenAddresses, blacklistedWalletAddresses);
    }

    /**
     * @notice Withdraw collected fees (NATIVETOKEN or token) from the contract.
     * @param token Address of the token (use address(0) for NATIVETOKEN).
     * @param to Recipient address.
     * @param amount Amount to withdraw.
     */
    function withdrawFees(address token, address payable to, uint256 amount) external onlyOwner nonReentrant {
        require(to != address(0), "Invalid recipient");
        if (token == address(0)) {
            require(amount <= address(this).balance, "Insufficient NATIVETOKEN");
            (bool sent, ) = to.call{value: amount}("");
            require(sent, "NATIVETOKEN withdraw failed");
        } else {
            uint256 contractBalance = IERC20(token).balanceOf(address(this));
            require(amount <= contractBalance, "Insufficient token balance");
            _safeTransfer(token, to, amount);
        }
        emit FeesWithdrawn(token, to, amount);
    }

    /**
     * @notice Update the moderator address.
     * @param newModerator Address of the new moderator.
     */
    function updateModerator(address newModerator) external onlyOwner {
        moderator = newModerator;
        emit ModeratorUpdated(newModerator);
    }

    /**
     * @notice Transfer contract ownership.
     * @param newOwner Address of the new owner.
     */
    function changeOwner(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero");
        owner = newOwner;
        emit OwnerChanged(newOwner);
    }

    /**
     * @notice Fallback function to receive NATIVETOKEN.
     */
    receive() external payable {
        emit Received(msg.sender, msg.value);
    }
}

/**
 * @dev Interface for IERC20, as used by this contract.
 */
interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    function totalSupply() external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

/**
 * @dev Extended IERC20 interface for metadata.
 */
interface IERC20Metadata is IERC20 {
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}
