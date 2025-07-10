// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @dev Minimal interface for the HyperSolver token (HPS).
///      Includes the methods we need: `transferFrom` and `transfer`.
interface IHyperSolverToken {
    /**
     * @notice Transfers `amount` tokens on behalf of `sender` to `recipient`.
     *         Typical ERC20 'transferFrom'.
     */
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

    /**
     * @notice Transfers `amount` tokens from the caller (contract) to `recipient`.
     *         Typical ERC20 'transfer'.
     */
    function transfer(address recipient, uint256 amount) external returns (bool);
}

/// @dev A minimal ERC20 interface for rescuing tokens other than HPS.
interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
}

/// @dev Interface for the Treasury contract used to claim Frankencoin (ZCHF) rewards.
interface ITreasury {
    /**
     * @notice Called by whitelisted contracts to claim Frankencoin rewards on behalf of a user.
     * @param user The address of the user claiming rewards.
     * @param userTokenVested The amount of tokens the user has locked/vested.
     */
    function claimFrankencoinReward(address user, uint256 userTokenVested) external;

    /**
     * @notice Returns the reward amount in Frankencoin (ZCHF) for a given user
     *         and locked token amount (userTokenVested).
     * @param user The user address for which the reward is being calculated.
     * @param userTokenVested The locked (vested) token amount for the user.
     * @return The reward amount in ZCHF.
     */
    function getUserRewardInFrankencoin(address user, uint256 userTokenVested) external view returns (uint256);
}

/**
 * @title HyperVesting
 *
 * This contract is supported by HypeLoot.com
 * Stay connected with us:
 * - Website: https://hypeloot.com/
 * - X (formerly Twitter): https://x.com/HypeLootCom
 * - Telegram Channel: https://t.me/Hypelootcom
 * - Instagram: https://instagram.com/hypelootcom
 *
 * For platform support, please email: support[at]hypeloot.com
 * We are continuously expanding our activities.
 *
 * @notice This contract enables users to lock their "HyperSolver" tokens (HPS) for a defined
 *         vesting period, during which they cannot withdraw their tokens. Users are allowed
 *         a limited window (e.g., 10 days from `startVesting`) to deposit/lock their tokens,
 *         after which they may also claim rewards in Frankencoin (ZCHF) from the `Treasury`
 *         contract once the deposit window is over.
 *
 *         Key Points:
 *         - Only one lock per user is allowed (they cannot lock multiple times).
 *         - The vesting period (e.g., 190 days) is set at initialization.
 *         - After vesting, users may withdraw their locked HPS tokens.
 *         - A user can claim Frankencoin rewards through `claimRewardsFromTreasury` exactly once.
 *         - They can only start to claim those rewards after the deposit window (e.g., 10 days) has ended.
 *         - The contract can rescue stuck ETH or ERC20 tokens (excluding the locked HPS itself).
 *         - Designed for proxy usage (has an `initialize` function rather than a constructor).
 */
contract HyperVesting {
    //--------------------------------------------------------------------------
    // Storage
    //--------------------------------------------------------------------------

    // The owner of the contract (set in `initialize`).
    address public owner;

    // The HyperSolver token interface (HPS).
    // Used for transferring tokens in and out of this contract.
    address public hyperSolverToken;

    // The Treasury contract interface for claiming Frankencoin rewards.
    address public treasury;

    // The vesting period in seconds (e.g., 190 days).
    uint256 public vestingPeriod;

    // The start time of the vesting deposit window (set by `startVesting`).
    uint256 public vestingStartTime;

    // Whether the owner has already initiated the vesting schedule.
    bool public vestingHasStarted;

    // Once `startVesting` is called, users have `depositOpenWindow` seconds (e.g., 10 days)
    // to deposit/lock their tokens.
    uint256 public depositOpenWindow;

    // Tracks per-user lock information.
    // Each user can lock tokens only once.
    struct UserLock {
        // The amount of HPS tokens locked by the user.
        uint256 amount;
        // The block timestamp when the tokens were locked.
        uint256 depositTime;
        // Whether the user has withdrawn the locked tokens after vesting.
        bool withdrawn;
        // Whether the user has claimed rewards for this lock.
        bool rewardClaimed;
    }

    // Mapping of user address -> UserLock struct.
    mapping(address => UserLock) public userLocks;

    // A flag to ensure this contract is only initialized once for proxy usage.
    bool private _initialized;

    // Reentrancy guard lock.
    bool private _locked;

    //--------------------------------------------------------------------------
    // Events
    //--------------------------------------------------------------------------

    // Emitted when the contract is initialized via `initialize`.
    event VestingInitialized(
        address indexed owner,
        address hyperSolverToken,
        address treasury,
        uint256 vestingPeriod,
        uint256 depositOpenWindow
    );

    // Emitted when the owner sets the vesting start time.
    event VestingStarted(uint256 startTime);

    // Emitted when a user locks their HPS tokens into this contract.
    event TokensLocked(address indexed user, uint256 amount, uint256 depositTime);

    // Emitted when a user withdraws their locked tokens after the vesting period.
    event TokensWithdrawn(address indexed user, uint256 amount, uint256 withdrawTime);

    // Emitted when a user claims Frankencoin (ZCHF) rewards from the Treasury via this contract.
    event RewardsClaimed(address indexed user, uint256 hpsAmount, uint256 claimTime);

    // Emitted when stuck ERC20 tokens (other than locked HPS) are rescued by the owner.
    event StuckERC20Claimed(address indexed operator, address token, uint256 amount);

    // Emitted when stuck ETH is rescued by the owner.
    event StuckETHClaimed(address indexed operator, uint256 amount);

    //--------------------------------------------------------------------------
    // Modifiers
    //--------------------------------------------------------------------------

    /**
     * @dev Ensures the function can only be called by the contract owner.
     */
    modifier onlyOwner() {
        require(msg.sender == owner, "HyperVesting: caller is not the owner");
        _;
    }

    /**
     * @dev Prevents reentrancy for state-changing functions.
     */
    modifier nonReentrant() {
        require(!_locked, "HyperVesting: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }

    //--------------------------------------------------------------------------
    // Initialization
    //--------------------------------------------------------------------------

    /**
     * @notice Initializes the vesting contract (for proxy usage).
     * @dev This function can only be called once.
     * @param _owner The owner of this contract.
     * @param _hyperSolverToken The address of the HyperSolver token (HPS).
     * @param _treasury The address of the Treasury contract for claiming rewards.
     * @param _vestingPeriod The number of seconds tokens must remain locked (e.g., 190 days).
     * @param _depositOpenWindow The number of seconds users have to deposit after `startVesting` (e.g., 10 days).
     */
    function initialize(
        address _owner,
        address _hyperSolverToken,
        address _treasury,
        uint256 _vestingPeriod,
        uint256 _depositOpenWindow
    ) external {
        require(!_initialized, "HyperVesting: already initialized");
        _initialized = true;

        owner = _owner;
        hyperSolverToken = _hyperSolverToken;
        treasury = _treasury;
        vestingPeriod = _vestingPeriod;
        depositOpenWindow = _depositOpenWindow;

        emit VestingInitialized(_owner, _hyperSolverToken, _treasury, _vestingPeriod, _depositOpenWindow);
    }

    //--------------------------------------------------------------------------
    // Core Vesting Logic
    //--------------------------------------------------------------------------

    /**
     * @notice Allows the owner to start the vesting deposit window. This can only be called once.
     * @dev Once called, users have `depositOpenWindow` seconds to lock their tokens.
     */
    function startVesting() external onlyOwner {
        require(!vestingHasStarted, "HyperVesting: vesting has already started");
        vestingHasStarted = true;

        vestingStartTime = block.timestamp;
        emit VestingStarted(vestingStartTime);
    }

    /**
     * @notice Allows a user to lock HPS tokens into this contract within the deposit window.
     *         Each user can only do this once.
     * @param amount The amount of HPS tokens to lock.
     */
    function lockTokens(uint256 amount) external nonReentrant {
        require(vestingStartTime > 0, "HyperVesting: vesting not started");
        require(block.timestamp <= vestingStartTime + depositOpenWindow, "HyperVesting: deposit window ended");
        require(amount > 0, "HyperVesting: amount must be > 0");

        UserLock storage userLock = userLocks[msg.sender];
        require(userLock.amount == 0, "HyperVesting: user already locked tokens");

        // Transfer HPS tokens from user to this contract
        bool success = IHyperSolverToken(hyperSolverToken).transferFrom(msg.sender, address(this), amount);
        require(success, "HyperVesting: token transfer failed");

        // Record the user's lock
        userLock.amount = amount;
        userLock.depositTime = block.timestamp;
        userLock.withdrawn = false;
        userLock.rewardClaimed = false;

        emit TokensLocked(msg.sender, amount, block.timestamp);
    }

    /**
     * @notice Allows the user to withdraw their locked HPS tokens after the vesting period has elapsed.
     */
    function withdrawTokens() external nonReentrant {
        UserLock storage userLock = userLocks[msg.sender];
        require(userLock.amount > 0, "HyperVesting: no tokens locked");
        require(!userLock.withdrawn, "HyperVesting: tokens already withdrawn");
        require(block.timestamp >= userLock.depositTime + vestingPeriod, "HyperVesting: vesting period not over");

        uint256 amount = userLock.amount;
        userLock.withdrawn = true;
        userLock.amount = 0; // Optional cleanup

        bool success = IHyperSolverToken(hyperSolverToken).transfer(msg.sender, amount);
        require(success, "HyperVesting: token transfer failed");

        emit TokensWithdrawn(msg.sender, amount, block.timestamp);
    }

    //--------------------------------------------------------------------------
    // Rewards Claim Logic
    //--------------------------------------------------------------------------

    /**
     * @notice Calls the Treasury contract's `claimFrankencoinReward` function on behalf of the user,
     *         using the locked token amount as the `userTokenVested`. This can only be done once per user lock.
     *         Additionally, the user may only claim after the deposit window (e.g., 10 days) has ended.
     */
    function claimRewardsFromTreasury() external nonReentrant {
        UserLock storage userLock = userLocks[msg.sender];
        require(userLock.amount > 0, "HyperVesting: no tokens locked");
        require(!userLock.rewardClaimed, "HyperVesting: reward already claimed");
        require(vestingStartTime > 0, "HyperVesting: vesting not started");
        // The user can only claim after deposit window is over (10 days, for example).
        require(block.timestamp >= vestingStartTime + depositOpenWindow, "HyperVesting: claim not available yet");

        // Mark reward as claimed
        userLock.rewardClaimed = true;

        // Call the treasury to claim Frankencoin rewards
        ITreasury(treasury).claimFrankencoinReward(msg.sender, userLock.amount);

        emit RewardsClaimed(msg.sender, userLock.amount, block.timestamp);
    }

    /**
     * @notice Returns the amount of Frankencoin (ZCHF) rewards the user would receive
     *         if they claim from the Treasury. This calls the Treasury's
     *         `getUserRewardInFrankencoin` function for `msg.sender`.
     * @return The reward amount in Frankencoin (ZCHF).
     */
    function checkUserReward() external view returns (uint256) {
        UserLock memory userLock = userLocks[msg.sender];
        if (userLock.amount == 0) {
            return 0;
        }
        return ITreasury(treasury).getUserRewardInFrankencoin(msg.sender, userLock.amount);
    }

    /**
     * @notice Returns the amount of Frankencoin (ZCHF) rewards an arbitrary user
     *         would receive if they claim from the Treasury. This calls the Treasury's
     *         `getUserRewardInFrankencoin` function for the specified `user`.
     * @param user The user address to check.
     * @return The reward amount in Frankencoin (ZCHF).
     */
    function checkUserRewardOf(address user) external view returns (uint256) {
        UserLock memory userLock = userLocks[user];
        if (userLock.amount == 0) {
            return 0;
        }
        return ITreasury(treasury).getUserRewardInFrankencoin(user, userLock.amount);
    }

    //--------------------------------------------------------------------------
    // Rescue Stuck Assets
    //--------------------------------------------------------------------------

    /**
     * @notice Allows the owner to rescue any ERC20 tokens mistakenly sent to this contract,
     *         except for the locked HyperSolver tokens (HPS).
     * @param token Address of the ERC20 token to rescue.
     */
    function claimStuckERC20(address token) external onlyOwner {
        // Disallow rescuing the HyperSolver tokens locked in this contract
        require(token != hyperSolverToken, "HyperVesting: cannot rescue locked HPS");
        
        uint256 stuckBalance = IERC20(token).balanceOf(address(this));
        require(stuckBalance > 0, "HyperVesting: no tokens to rescue");

        bool success = IERC20(token).transfer(owner, stuckBalance);
        require(success, "HyperVesting: stuck token transfer failed");

        emit StuckERC20Claimed(msg.sender, token, stuckBalance);
    }

    /**
     * @notice Allows the owner to rescue any stuck ETH in this contract.
     */
    function claimStuckETH() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "HyperVesting: no ETH to rescue");

        (bool success, ) = owner.call{value: balance}("");
        require(success, "HyperVesting: transfer failed");

        emit StuckETHClaimed(msg.sender, balance);
    }
}