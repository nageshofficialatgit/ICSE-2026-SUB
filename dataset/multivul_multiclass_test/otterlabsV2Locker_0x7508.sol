// SPDX-License-Identifier: MIT
/*
   ___ _____ _____ ___ ___   _      _   ___ ___ 
  / _ \_   _|_   _| __| _ \ | |    /_\ | _ ) __|
 | (_) || |   | | | _||   / | |__ / _ \| _ \__ \
  \___/ |_|   |_| |___|_|_\ |____/_/ \_\___/___/
OTTERLABS UNI V2 Liquidity Locker                 

This contract allows anyone to lock their V2 LP tokens with immutable proof.

*/


pragma solidity ^0.8.0;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

interface IUniswapV2Pair {
    function token0() external view returns (address);
    function token1() external view returns (address);
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
    function totalSupply() external view returns (uint256);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _transferOwnership(_msgSender());
    }

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function _checkOwner() internal view virtual {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

abstract contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
    }

    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }
}

contract otterlabsV2Locker is Ownable, ReentrancyGuard {
    // Lock information structure
    struct LockInfo {
        address owner;
        address lpToken;
        uint256 amount;
        uint256 lockTime;
        uint256 unlockTime;
        bool isLocked;
    }

    // LP token => Lock ID => Lock Info
    mapping(address => mapping(uint256 => LockInfo)) public locks;
    // LP token => total number of locks
    mapping(address => uint256) public totalLocks;
    // LP token => total amount locked
    mapping(address => uint256) public totalLocked;
    // User address => LP token => array of lock IDs
    mapping(address => mapping(address => uint256[])) public userLocks;

    // Fee settings
    uint256 public lockFee = 1; // 0.1% fee (1/1000)
    uint256 public constant FEE_DENOMINATOR = 1000;
    
    // Lock duration limits
    uint256 public constant MIN_LOCK_TIME = 1 days;
    uint256 public constant MAX_LOCK_TIME = 777 days;

    // Events
    event LPLocked(
        address indexed lpToken,
        address indexed owner,
        uint256 amount,
        uint256 lockId,
        uint256 unlockTime
    );
    event LPUnlocked(
        address indexed lpToken,
        address indexed owner,
        uint256 amount,
        uint256 lockId
    );
    event LockExtended(
        address indexed lpToken,
        uint256 lockId,
        uint256 newUnlockTime
    );

    /**
     * @notice Lock LP tokens for a specified duration
     * @param lpToken The LP token address to lock
     * @param amount The amount of LP tokens to lock
     * @param lockDuration Duration in seconds for which to lock the tokens
     * @return lockId The ID of the created lock
     */
    function lockLP(
        address lpToken,
        uint256 amount,
        uint256 lockDuration
    ) external nonReentrant returns (uint256) {
        require(amount > 0, "Amount must be > 0");
        require(
            lockDuration >= MIN_LOCK_TIME && lockDuration <= MAX_LOCK_TIME,
            "Invalid lock duration"
        );

        // Verify it's a valid V2 LP token
        IUniswapV2Pair pair = IUniswapV2Pair(lpToken);
        require(pair.token0() != address(0) && pair.token1() != address(0), "Invalid LP token");

        // Calculate fee
        uint256 fee = (amount * lockFee) / FEE_DENOMINATOR;
        uint256 amountAfterFee = amount - fee;

        // Transfer LP tokens to contract
        bool success = IERC20(lpToken).transferFrom(msg.sender, address(this), amount);
        require(success, "LP token transfer failed");
        
        if (fee > 0) {
            success = IERC20(lpToken).transfer(owner(), fee);
            require(success, "Fee transfer failed");
        }

        // Create lock
        uint256 lockId = totalLocks[lpToken];
        uint256 unlockTime = block.timestamp + lockDuration;

        locks[lpToken][lockId] = LockInfo({
            owner: msg.sender,
            lpToken: lpToken,
            amount: amountAfterFee,
            lockTime: block.timestamp,
            unlockTime: unlockTime,
            isLocked: true
        });

        totalLocks[lpToken]++;
        totalLocked[lpToken] += amountAfterFee;
        userLocks[msg.sender][lpToken].push(lockId);

        emit LPLocked(lpToken, msg.sender, amountAfterFee, lockId, unlockTime);
        return lockId;
    }

    /**
     * @notice Unlock LP tokens after the lock period has expired
     * @param lpToken The LP token address
     * @param lockId The ID of the lock to unlock
     */
    function unlockLP(address lpToken, uint256 lockId) external nonReentrant {
        LockInfo storage lock = locks[lpToken][lockId];
        require(lock.isLocked, "Lock not found or already unlocked");
        require(lock.owner == msg.sender, "Not lock owner");
        require(block.timestamp >= lock.unlockTime, "Lock not expired");

        lock.isLocked = false;
        totalLocked[lpToken] -= lock.amount;

        bool success = IERC20(lpToken).transfer(msg.sender, lock.amount);
        require(success, "LP token transfer failed");

        emit LPUnlocked(lpToken, msg.sender, lock.amount, lockId);
    }

    /**
     * @notice Extend the lock duration of an existing lock
     * @param lpToken The LP token address
     * @param lockId The ID of the lock to extend
     * @param newDuration The new duration to add from current time
     */
    function extendLock(
        address lpToken,
        uint256 lockId,
        uint256 newDuration
    ) external nonReentrant {
        require(
            newDuration >= MIN_LOCK_TIME && newDuration <= MAX_LOCK_TIME,
            "Invalid lock duration"
        );

        LockInfo storage lock = locks[lpToken][lockId];
        require(lock.isLocked, "Lock not found or already unlocked");
        require(lock.owner == msg.sender, "Not lock owner");

        uint256 newUnlockTime = block.timestamp + newDuration;
        require(newUnlockTime > lock.unlockTime, "Can only extend lock");

        lock.unlockTime = newUnlockTime;

        emit LockExtended(lpToken, lockId, newUnlockTime);
    }

    /**
     * @notice Get information about a specific lock
     * @param lpToken The LP token address
     * @param lockId The ID of the lock
     * @return Lock information
     */
    function getLockInfo(address lpToken, uint256 lockId)
        external
        view
        returns (LockInfo memory)
    {
        return locks[lpToken][lockId];
    }

    /**
     * @notice Get the number of locks a user has for a specific LP token
     * @param user The user address
     * @param lpToken The LP token address
     * @return Number of locks
     */
    function getUserLockCount(address user, address lpToken)
        external
        view
        returns (uint256)
    {
        return userLocks[user][lpToken].length;
    }

    /**
     * @notice Get all lock IDs for a user's LP token
     * @param user The user address
     * @param lpToken The LP token address
     * @return Array of lock IDs
     */
    function getUserLocks(address user, address lpToken)
        external
        view
        returns (uint256[] memory)
    {
        return userLocks[user][lpToken];
    }

    /**
     * @notice Get LP token data including total and locked liquidity
     * @param lpToken The LP token address
     * @return token0 Address of token0
     * @return token1 Address of token1
     * @return totalLiquidity Total supply of LP tokens
     * @return lockedLiquidity Amount of LP tokens locked in this contract
     */
    function getLPTokenData(address lpToken)
        external
        view
        returns (
            address token0,
            address token1,
            uint256 totalLiquidity,
            uint256 lockedLiquidity
        )
    {
        IUniswapV2Pair pair = IUniswapV2Pair(lpToken);
        token0 = pair.token0();
        token1 = pair.token1();
        totalLiquidity = IERC20(lpToken).totalSupply();
        lockedLiquidity = totalLocked[lpToken];
    }

    /**
     * @notice Set the lock fee percentage (owner only)
     * @param newFee New fee value (1 = 0.1%)
     */
    function setLockFee(uint256 newFee) external onlyOwner {
        require(newFee <= 50, "Fee too high"); // Max 5%
        lockFee = newFee;
    }

    /**
     * @notice Emergency function to recover stuck tokens (owner only)
     * @param token The token address to recover
     * @param amount The amount to recover
     */
    function withdrawFees(address token, uint256 amount) external onlyOwner {
        bool success = IERC20(token).transfer(msg.sender, amount);
        require(success, "Token transfer failed");
    }
}