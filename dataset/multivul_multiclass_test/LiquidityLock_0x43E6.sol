// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @title Liquidity Lock Contract
/// @notice This contract allows the owner to lock liquidity tokens for a specified duration (minimum 48 hours).
/// It also provides functions for the owner to claim locked tokens, any ERC20 tokens, and ETH after the lock period.
interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract LiquidityLock {
    // Owner of the contract, set at deployment.
    address public owner;
    // Timestamp when the contract was deployed.
    uint256 public deploymentTime;

    // Structure to store lock details for a given liquidity pool token.
    struct LockInfo {
        uint256 amount;      // Amount of tokens locked.
        uint256 unlockTime;  // Timestamp when tokens can be unlocked.
    }

    // Mapping from token address to its lock information.
    mapping(address => LockInfo) public locks;

    // Emitted when tokens are locked.
    event Locked(address indexed token, uint256 amount, uint256 unlockTime);
    // Emitted when tokens are claimed.
    event Claimed(address indexed token, uint256 amount);

    /// @dev Modifier to restrict functions to only the owner.
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    /// @notice Sets the contract deployer as the owner and records the deployment time.
    constructor() {
        owner = msg.sender;
        deploymentTime = block.timestamp;
    }

    /// @notice Locks liquidity tokens for a specified duration.
    /// @dev The lock duration must be at least 48 hours (172800 seconds).
    /// @param lpToken The address of the liquidity pool token to be locked.
    /// @param amount The amount of tokens to lock.
    /// @param lockDuration Duration in seconds for which the tokens will be locked.
    /// @return unlockTime The timestamp when the tokens will be available to claim.
    function lock(address lpToken, uint256 amount, uint256 lockDuration)
        external
        onlyOwner
        returns (uint256 unlockTime)
    {
        require(lockDuration >= 172800, "Lock duration must be at least 48 hours");
        // Transfer tokens from owner to this contract (owner must approve first)
        require(IERC20(lpToken).transferFrom(msg.sender, address(this), amount), "Token transfer failed");
        // Calculate the unlock time by adding the duration to the current block timestamp.
        unlockTime = block.timestamp + lockDuration;
        locks[lpToken] = LockInfo(amount, unlockTime);
        emit Locked(lpToken, amount, unlockTime);
    }

    /// @notice Claims the locked liquidity tokens after the lock period has expired.
    /// @param lpToken The address of the locked liquidity pool token.
    function claimLockedTokens(address lpToken) external onlyOwner {
        LockInfo storage lockInfo = locks[lpToken];
        require(lockInfo.amount > 0, "No tokens locked");
        require(block.timestamp >= lockInfo.unlockTime, "Lock period not yet expired");
        uint256 amount = lockInfo.amount;
        // Reset the locked amount before transferring to prevent reentrancy.
        lockInfo.amount = 0;
        require(IERC20(lpToken).transfer(msg.sender, amount), "Token transfer failed");
        emit Claimed(lpToken, amount);
    }

    /// @notice Claims any ERC20 token held by the contract after 48 hours from deployment.
    /// @dev This function can be used to rescue tokens that are not part of a specific lock.
    /// @param token The address of the ERC20 token to claim.
    function claimToken(address token) external onlyOwner {
        require(block.timestamp >= deploymentTime + 172800, "Tokens can be claimed only after 48 hours from deployment");
        uint256 tokenBalance = IERC20(token).balanceOf(address(this));
        require(tokenBalance > 0, "No tokens to claim");
        require(IERC20(token).transfer(msg.sender, tokenBalance), "Token transfer failed");
        emit Claimed(token, tokenBalance);
    }

    /// @notice Claims any ETH held by the contract after 48 hours from deployment.
    function claimETH() external onlyOwner {
        require(block.timestamp >= deploymentTime + 172800, "ETH can be claimed only after 48 hours from deployment");
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH to claim");
        payable(msg.sender).transfer(balance);
    }

    /// @notice Allows the contract to receive ETH.
    receive() external payable {}
}