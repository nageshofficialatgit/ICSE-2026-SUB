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

// File: @openzeppelin/contracts/utils/Context.sol


// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

pragma solidity ^0.8.20;

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}

// File: @openzeppelin/contracts/access/Ownable.sol


// OpenZeppelin Contracts (last updated v5.0.0) (access/Ownable.sol)

pragma solidity ^0.8.20;


/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * The initial owner is set to the address provided by the deployer. This can
 * later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
    address private _owner;

    /**
     * @dev The caller account is not authorized to perform an operation.
     */
    error OwnableUnauthorizedAccount(address account);

    /**
     * @dev The owner is not a valid owner account. (eg. `address(0)`)
     */
    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Initializes the contract setting the address provided by the deployer as the initial owner.
     */
    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if the sender is not the owner.
     */
    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby disabling any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

// File: @openzeppelin/contracts/security/ReentrancyGuard.sol


// OpenZeppelin Contracts (last updated v4.9.0) (security/ReentrancyGuard.sol)

pragma solidity ^0.8.0;

/**
 * @dev Contract module that helps prevent reentrant calls to a function.
 *
 * Inheriting from `ReentrancyGuard` will make the {nonReentrant} modifier
 * available, which can be applied to functions to make sure there are no nested
 * (reentrant) calls to them.
 *
 * Note that because there is a single `nonReentrant` guard, functions marked as
 * `nonReentrant` may not call one another. This can be worked around by making
 * those functions `private`, and then adding `external` `nonReentrant` entry
 * points to them.
 *
 * TIP: If you would like to learn more about reentrancy and alternative ways
 * to protect against it, check out our blog post
 * https://blog.openzeppelin.com/reentrancy-after-istanbul/[Reentrancy After Istanbul].
 */
abstract contract ReentrancyGuard {
    // Booleans are more expensive than uint256 or any type that takes up a full
    // word because each write operation emits an extra SLOAD to first read the
    // slot's contents, replace the bits taken up by the boolean, and then write
    // back. This is the compiler's defense against contract upgrades and
    // pointer aliasing, and it cannot be disabled.

    // The values being non-zero value makes deployment a bit more expensive,
    // but in exchange the refund on every call to nonReentrant will be lower in
    // amount. Since refunds are capped to a percentage of the total
    // transaction's gas, it is best to keep them low in cases like this one, to
    // increase the likelihood of the full refund coming into effect.
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
    }

    /**
     * @dev Prevents a contract from calling itself, directly or indirectly.
     * Calling a `nonReentrant` function from another `nonReentrant`
     * function is not supported. It is possible to prevent this from happening
     * by making the `nonReentrant` function external, and making it call a
     * `private` function that does the actual work.
     */
    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {
        // On the first call to nonReentrant, _status will be _NOT_ENTERED
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");

        // Any calls to nonReentrant after this point will fail
        _status = _ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = _NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == _ENTERED;
    }
}

// File: @openzeppelin/contracts/security/Pausable.sol


// OpenZeppelin Contracts (last updated v4.7.0) (security/Pausable.sol)

pragma solidity ^0.8.0;


/**
 * @dev Contract module which allows children to implement an emergency stop
 * mechanism that can be triggered by an authorized account.
 *
 * This module is used through inheritance. It will make available the
 * modifiers `whenNotPaused` and `whenPaused`, which can be applied to
 * the functions of your contract. Note that they will not be pausable by
 * simply including this module, only once the modifiers are put in place.
 */
abstract contract Pausable is Context {
    /**
     * @dev Emitted when the pause is triggered by `account`.
     */
    event Paused(address account);

    /**
     * @dev Emitted when the pause is lifted by `account`.
     */
    event Unpaused(address account);

    bool private _paused;

    /**
     * @dev Initializes the contract in unpaused state.
     */
    constructor() {
        _paused = false;
    }

    /**
     * @dev Modifier to make a function callable only when the contract is not paused.
     *
     * Requirements:
     *
     * - The contract must not be paused.
     */
    modifier whenNotPaused() {
        _requireNotPaused();
        _;
    }

    /**
     * @dev Modifier to make a function callable only when the contract is paused.
     *
     * Requirements:
     *
     * - The contract must be paused.
     */
    modifier whenPaused() {
        _requirePaused();
        _;
    }

    /**
     * @dev Returns true if the contract is paused, and false otherwise.
     */
    function paused() public view virtual returns (bool) {
        return _paused;
    }

    /**
     * @dev Throws if the contract is paused.
     */
    function _requireNotPaused() internal view virtual {
        require(!paused(), "Pausable: paused");
    }

    /**
     * @dev Throws if the contract is not paused.
     */
    function _requirePaused() internal view virtual {
        require(paused(), "Pausable: not paused");
    }

    /**
     * @dev Triggers stopped state.
     *
     * Requirements:
     *
     * - The contract must not be paused.
     */
    function _pause() internal virtual whenNotPaused {
        _paused = true;
        emit Paused(_msgSender());
    }

    /**
     * @dev Returns to normal state.
     *
     * Requirements:
     *
     * - The contract must be paused.
     */
    function _unpause() internal virtual whenPaused {
        _paused = false;
        emit Unpaused(_msgSender());
    }
}

// File: contracts/Stake-ECM/ECMStakingv8.sol


pragma solidity ^0.8.0;





/**
 * @title MCPECMStaking
 * @notice A flexible ERC20 token staking contract with multiple staking plans and referral system
 * @dev Implements staking functionality with owner-controlled plans, rewards, referrals, and safety mechanisms
 * @author ECM Development Team
 */
contract MCPECMStaking is Ownable(msg.sender), ReentrancyGuard, Pausable {
    // ================ STATE VARIABLES ================

    /// @notice The ERC20 token used for staking
    IERC20 public immutable ecmCoin;

    /// @notice Minimum amount that can be staked
    uint256 public minimumStake;

    /// @notice Maximum amount that can be staked
    uint256 public maximumStake;

    // @notice Penalty rate for user force unstake (1000 = 10.00%)
    uint256 public penaltyRate = 1000; 
    
    /// @notice Referral commission percentage (500 = 5.00%)
    uint256 public referralCommissionRate = 500;
    
    /// @notice Flag to enable/disable rewards for admin force unstaking
    bool public enableAdminForceUnstakeRewards = true;
    
    /// @notice Total accumulated penalty amount from force unstakes
    uint256 public totalPenaltyCollected;

    /// @notice Tracks the next unique stake ID
    uint256 private nextStakeId = 1;

    /// @notice Annual basis for calculating actual rewards (10000 = 100.00%)
    uint256 public constant YEAR_IN_DAYS = 365;
    
    // ================ STRUCTS ================

    /**
     * @dev Staking plan structure defining duration and reward percentage
     * @param duration Number of days the stake is locked
     * @param rewardPercent Percentage of reward (multiplied by 100 for precision, e.g. 1500 = 15.00%)
     * @param isActive Indicates if the plan is currently available for staking
     * @param maxPoolSize Maximum total amount that can be staked in this plan
     * @param currentPoolSize Current total amount staked in this plan
     */
    struct StakingPlan {
        uint256 duration;
        uint256 rewardPercent;
        bool isActive;
        uint256 maxPoolSize;
        uint256 currentPoolSize;
    }

    /**
     * @dev Detailed information about a specific stake
     * @param stakeId Unique identifier for the stake
     * @param amount Original staked amount
     * @param planIndex Index of the staking plan used
     * @param startTime Timestamp when stake was created
     * @param endTime Timestamp when stake can be unstaked
     * @param claimedAt Timestamp when claimed (0 if not claimed)
     * @param referrer Address of the referrer (address(0) if none)
     */
    struct StakeInfo {
        uint256 stakeId;       // Unique stake identifier
        uint256 amount;        // Staked amount
        uint256 planIndex;     // Index of staking plan used
        uint256 startTime;     // Start timestamp
        uint256 endTime;       // End timestamp
        uint256 claimedAt;     // Timestamp when claimed (0 if not claimed)
        address referrer;      // Referrer address (if any)
    }

    // ================ MAPPINGS ================

    /// @notice Array of available staking plans
    StakingPlan[] public stakingPlans;

    /// @notice Mapping of user addresses to their stakes
    mapping(address => StakeInfo[]) public stakes;

    /// @notice Mapping of stake IDs to their owners
    mapping(uint256 => address) public stakeOwners;

    /// @notice Mapping to quickly find a stake's index for a given user and stake ID
    mapping(address => mapping(uint256 => uint256)) private stakeIdToIndex;

    // ================ EVENTS ================

    // Events for tracking contract activities
    event Staked(address indexed staker, uint256 indexed stakeId, uint256 amount, uint256 planIndex, uint256 endTime);
    event Unstaked(address indexed staker, uint256 indexed stakeId, uint256 amount, uint256 reward, string unstakeType);
    event ECMWithdrawn(address indexed owner, uint256 amount);
    event ReferralCommissionPaid(address indexed referrer, address indexed staker, uint256 indexed stakeId, uint256 amount);
    event PenaltyCollected(address indexed staker, uint256 amount);

    // ================ CONSTRUCTOR ================

    /**
     * @notice Contract constructor
     * @param _ecmCoin Address of the ERC20 token used for staking
     * @param _minimumStake Minimum amount that can be staked
     * @param _maximumStake Maximum amount that can be staked
     * @dev Initializes the contract with default staking plans
     */
    constructor(address _ecmCoin, uint256 _minimumStake, uint256 _maximumStake) {
        require(_ecmCoin != address(0), "Zero address not allowed");
        require(_minimumStake > 0, "Minimum stake must be greater than 0");
        require(_maximumStake > _minimumStake, "Maximum must be greater than minimum");
        
        ecmCoin = IERC20(_ecmCoin);
        minimumStake = _minimumStake;
        maximumStake = _maximumStake;

        // Initialize default staking plans
        _addPlan(7, 500, 10000 * 10**18);   // 7 days, 5.00% yearly, 10K tokens max
        _addPlan(30, 800, 50000 * 10**18); // 30 days, 8.00% yearly, 50K tokens max
        _addPlan(90, 1100, 200000 * 10**18); // 90 days, 11.00% yearly, 200K tokens max
        _addPlan(180, 1500, 1000000 * 10**18); // 180 days, 15.00% yearly, 1M tokens max
        _addPlan(365, 2000, 3000000 * 10**18); // 365 days, 20.00% yearly, 3M tokens max
    }

    // ================ USER STAKING FUNCTIONS ================

    /**
     * @notice Creates a new stake
     * @param _amount Amount of tokens to stake
     * @param _planIndex Index of the staking plan to use
     * @param _referrer Address of the referrer (optional, can be address(0))
     * @return The unique stake ID
     * @dev Requires user to have approved the contract to spend tokens
     */
    function stake(uint256 _amount, uint256 _planIndex, address _referrer) external nonReentrant whenNotPaused returns (uint256) {
        require(_planIndex < stakingPlans.length, "Invalid plan index");
        StakingPlan storage plan = stakingPlans[_planIndex];
        
        require(plan.isActive, "Plan is not active");
        require(_amount >= minimumStake, "Amount below minimum stake");
        require(_amount <= maximumStake, "Amount above maximum stake");
        require(_referrer != msg.sender && _referrer != address(this), "Invalid referrer");
        
        // Check plan pool size limit
        require(plan.currentPoolSize + _amount <= plan.maxPoolSize, "Plan pool size limit reached");

        // Check user balance
        require(ecmCoin.balanceOf(msg.sender) >= _amount, "Insufficient ECM balance");
        require(ecmCoin.allowance(msg.sender, address(this)) >= _amount, "Insufficient allowance");

        uint256 currentStakeId = nextStakeId++;

        // Transfer tokens to contract
        require(ecmCoin.transferFrom(msg.sender, address(this), _amount), "Transfer failed");

        // Update plan pool size
        plan.currentPoolSize += _amount;

        // Create stake with unique ID and plan index
        stakes[msg.sender].push(StakeInfo({
            stakeId: currentStakeId,
            amount: _amount,
            planIndex: _planIndex,
            startTime: block.timestamp,
            endTime: block.timestamp + (plan.duration * 1 days),
            claimedAt: 0,
            referrer: _referrer
        }));

        uint256 newIndex = stakes[msg.sender].length - 1;
        stakeIdToIndex[msg.sender][currentStakeId] = newIndex;

        stakeOwners[currentStakeId] = msg.sender;
        
        emit Staked(msg.sender, currentStakeId, _amount, _planIndex, block.timestamp + (plan.duration * 1 days));
        return currentStakeId;
    }

    /**
     * @notice Unstakes tokens after the lock period
     * @param _stakeId Unique identifier of the stake to unstake
     * @dev Only the stake owner can unstake, and only after the lock period
     * @dev Pays referral commission during unstaking
     */
    function unstake(uint256 _stakeId) external nonReentrant whenNotPaused {
        require(stakeOwners[_stakeId] == msg.sender, "Not stake owner");
        uint256 stakeIndex = findStakeIndex(msg.sender, _stakeId);
        StakeInfo storage userStake = stakes[msg.sender][stakeIndex];
        
        require(userStake.claimedAt == 0, "Stake already claimed");
        require(block.timestamp >= userStake.endTime, "Stake still locked");

        // Calculate current reward based on the current plan settings
        (uint256 rewardAmount, uint256 finalAmount) = calculateCurrentReward(msg.sender, stakeIndex);

        // Update stake status
        userStake.claimedAt = block.timestamp;

        // Update plan pool size
        stakingPlans[userStake.planIndex].currentPoolSize -= userStake.amount;

        // Pay referral commission - always enabled for regular unstaking
        // Commission is based on reward amount, not staked amount
        _processReferralCommission(userStake.referrer, msg.sender, _stakeId, rewardAmount, true);

        require(ecmCoin.transfer(msg.sender, finalAmount), "Transfer failed");
        emit Unstaked(msg.sender, _stakeId, userStake.amount, rewardAmount, "normal");
    }

    /**
     * @notice Force unstake a stake (can be called by owner or stake owner)
     * @param _stakeId Unique identifier of the stake to unstake
     * @dev Behavior depends on caller: admin follows enableAdminWithdrawWithReward flag, users always get penalty
     */
    function forceUnstake(uint256 _stakeId) external nonReentrant whenNotPaused {
        address staker = stakeOwners[_stakeId];
        require(staker != address(0), "Stake does not exist");
        
        bool isAdmin = (msg.sender == owner());
        
        // If not admin, caller must be the stake owner
        if (!isAdmin) {
            require(msg.sender == staker, "Only stake owner or admin can force unstake");
        }
        
        uint256 stakeIndex = findStakeIndex(staker, _stakeId);
        StakeInfo storage userStake = stakes[staker][stakeIndex];
        require(userStake.claimedAt == 0, "Stake already claimed");
        
        uint256 transferAmount;
        uint256 rewardAmount = 0;
        string memory unstakeType = isAdmin ? "admin" : "user";
        
        // Update stake status and plan pool size
        userStake.claimedAt = block.timestamp;
        stakingPlans[userStake.planIndex].currentPoolSize -= userStake.amount;
        
        if (isAdmin && enableAdminForceUnstakeRewards) {
            // Admin unstake with rewards (when enabled)
            (rewardAmount, transferAmount) = calculateCurrentReward(staker, stakeIndex);
            
            // Pay referral commission if applicable (only for admin with rewards enabled)
            if (rewardAmount > 0) {
                _processReferralCommission(userStake.referrer, staker, _stakeId, rewardAmount, true);
            }
        } else if (!isAdmin) {
            // User force unstake with penalty
            // For user force unstake, NO rewards are provided and NO referral commission is paid
            uint256 penalty = (userStake.amount * penaltyRate) / 10000;
            transferAmount = userStake.amount - penalty;
            
            // Track the penalty amount instead of burning
            totalPenaltyCollected += penalty;
            emit PenaltyCollected(staker, penalty);
        } else {
            // Admin unstake without rewards
            transferAmount = userStake.amount;
        }
        
        require(ecmCoin.transfer(staker, transferAmount), "Transfer failed");
        
        emit Unstaked(staker, _stakeId, userStake.amount, rewardAmount, unstakeType);
    }

    // ================ CORE CALCULATION FUNCTIONS ================
    
    /**
     * @notice Calculates reward amounts for a given stake
     * @param _amount Original staked amount
     * @param _rewardPercent Reward percentage (yearly basis, multiplied by 100)
     * @param _duration Duration in days
     * @return Tuple of (rewardAmount, finalAmount)
     * @dev Internal pure function for reward calculation
     */
    function calculateAmounts(
        uint256 _amount, 
        uint256 _rewardPercent, 
        uint256 _duration
    ) internal pure returns (uint256, uint256) {
        // Calculate pro-rated reward based on duration as a fraction of a year
        // rewardPercent is multiplied by 100 (e.g., 1500 = 15.00%)
        uint256 rewardAmount = (_amount * _rewardPercent * _duration) / (YEAR_IN_DAYS * 10000);
        uint256 finalAmount = _amount + rewardAmount;
        return (rewardAmount, finalAmount);
    }
    
    /**
     * @notice Process referral commission payment
     * @param _referrer Address of the referrer
     * @param _staker Address of the staker
     * @param _stakeId Stake identifier
     * @param _rewardAmount Reward amount (not the staked amount)
     * @param _enabled Whether commission payment is enabled
     * @dev Internal function to handle referral commission payments
     */
    function _processReferralCommission(
        address _referrer, 
        address _staker, 
        uint256 _stakeId,
        uint256 _rewardAmount,
        bool _enabled
    ) internal {
        if (_enabled && _referrer != address(0) && _rewardAmount > 0) {
            uint256 commission = (_rewardAmount * referralCommissionRate) / 10000;
            if (commission > 0) {
                require(ecmCoin.transfer(_referrer, commission), "Referral transfer failed");
                emit ReferralCommissionPaid(_referrer, _staker, _stakeId, commission);
            }
        }
    }

    /**
     * @notice Calculates the current reward for a stake
     * @param _staker Address of the stake owner
     * @param _stakeIndex Index of the stake in the user's stakes array
     * @return The calculated reward amount and final amount (principal + rewards)
     */
    function calculateCurrentReward(address _staker, uint256 _stakeIndex) internal view returns (uint256, uint256) {
        StakeInfo storage userStake = stakes[_staker][_stakeIndex];
        StakingPlan storage plan = stakingPlans[userStake.planIndex];
        
        return calculateAmounts(userStake.amount, plan.rewardPercent, plan.duration);
    }

    /**
     * @notice Finds the index of a specific stake for a user
     * @param _staker Address of the stake owner
     * @param _stakeId Unique identifier of the stake
     * @return The index of the stake in the user's stakes array
     * @dev Internal view function to locate a specific stake
     */
    function findStakeIndex(address _staker, uint256 _stakeId) internal view returns (uint256) {
        uint256 index = stakeIdToIndex[_staker][_stakeId];
        require(index < stakes[_staker].length, "Invalid index");
        require(stakes[_staker][index].stakeId == _stakeId, "Stake ID mismatch");
        return index;
    }

    // ================ ADMIN PLAN MANAGEMENT FUNCTIONS ================

    /**
     * @notice Adds a new staking plan
     * @param _duration Number of days the stake will be locked
     * @param _rewardPercent Percentage of reward for the plan (yearly basis, multiplied by 100, e.g. 1500 = 15.00%)
     * @param _maxPoolSize Maximum amount that can be staked in this plan
     * @dev Can only be called by the contract owner
     */
    function addPlan(uint256 _duration, uint256 _rewardPercent, uint256 _maxPoolSize) external onlyOwner {
        require(_duration > 0, "Duration must be greater than 0");
        require(_rewardPercent > 0, "Reward percent must be greater than 0");
        require(_maxPoolSize > 0, "Max pool size must be greater than 0");
        _addPlan(_duration, _rewardPercent, _maxPoolSize);
    }

    /**
     * @notice Internal function to add a new staking plan
     * @param _duration Number of days the stake will be locked
     * @param _rewardPercent Percentage of reward for the plan (yearly basis, multiplied by 100)
     * @param _maxPoolSize Maximum amount that can be staked in this plan
     * @dev Private method called by public addPlan and constructor
     */
    function _addPlan(uint256 _duration, uint256 _rewardPercent, uint256 _maxPoolSize) private {
        stakingPlans.push(StakingPlan({
            duration: _duration,
            rewardPercent: _rewardPercent,
            isActive: true,
            maxPoolSize: _maxPoolSize,
            currentPoolSize: 0
        }));
    }

    /**
     * @notice Updates an existing staking plan
     * @param _planIndex Index of the plan to update
     * @param _duration New duration for the plan
     * @param _rewardPercent New reward percentage (yearly basis, multiplied by 100)
     * @param _isActive Whether the plan is active
     * @param _maxPoolSize Maximum amount that can be staked in this plan
     * @dev Can only be called by the contract owner
     */
    function updatePlan(
        uint256 _planIndex,
        uint256 _duration,
        uint256 _rewardPercent,
        bool _isActive,
        uint256 _maxPoolSize
    ) external onlyOwner {
        require(_planIndex < stakingPlans.length, "Invalid plan index");
        require(_duration > 0, "Duration must be greater than 0");
        require(_rewardPercent > 0, "Reward percent must be greater than 0");
        require(_maxPoolSize >= stakingPlans[_planIndex].currentPoolSize, "Max pool size cannot be less than current pool size");
        
        StakingPlan storage plan = stakingPlans[_planIndex];
        plan.duration = _duration;
        plan.rewardPercent = _rewardPercent;
        plan.isActive = _isActive;
        plan.maxPoolSize = _maxPoolSize;
    }

    // ================ ADMIN CONFIGURATION FUNCTIONS ================

    /**
     * @notice Updates the referral commission rate
     * @param _newRate New commission rate (500 = 5.00%)
     * @dev Can only be called by contract owner
     */
    function updateReferralCommissionRate(uint256 _newRate) external onlyOwner {
        require(_newRate <= 10000, "Rate cannot exceed 100%");
        referralCommissionRate = _newRate;
    }

    /**
     * @notice Updates whether admin force unstaking includes rewards
     * @param _enabled Whether admin force unstakes should include rewards
     * @dev Can only be called by contract owner
     */
    function setAdminForceUnstakeRewards(bool _enabled) external onlyOwner {
        enableAdminForceUnstakeRewards = _enabled;
    }

    /**
     * @notice Updates the minimum stake amount
     * @param _newMinimumStake New minimum stake amount
     * @dev Can only be called by contract owner
     */
    function updateMinimumStake(uint256 _newMinimumStake) external onlyOwner {
        require(_newMinimumStake <= maximumStake, "Min must be less than maximum stake");
        minimumStake = _newMinimumStake;
    }

    /**
     * @notice Updates the maximum stake amount
     * @param _newMaxAmount New maximum stake amount
     * @dev Can only be called by contract owner
     */
    function updateMaximumStake(uint256 _newMaxAmount) external onlyOwner {
        require(_newMaxAmount >= minimumStake, "Max must be greater than minimum stake");
        maximumStake = _newMaxAmount;
    }

    /// @notice Update penalty rate for force unstaking (1000 = 10.00%)
    function updatePenaltyRate(uint256 _newRate) external onlyOwner {
        require(_newRate <= 10000, "Rate cannot exceed 100%");
        penaltyRate = _newRate;
    }

    // ================ ADMIN WITHDRAWAL FUNCTIONS ================
    
    /**
     * @notice Allows owner to withdraw ECM tokens from the contract
     * @param _amount Amount of tokens to withdraw
     * @dev Provides a mechanism for owner to retrieve tokens
     */
    function withdrawECM(uint256 _amount) external onlyOwner {
        require(_amount > 0, "Amount must be greater than 0");
        require(_amount <= ecmCoin.balanceOf(address(this)), "Insufficient balance");
        
        require(ecmCoin.transfer(msg.sender, _amount), "Transfer failed");
        emit ECMWithdrawn(msg.sender, _amount);
    }

    /**
     * @notice Emergency withdrawal of all tokens when contract is paused
     * @dev Provides ultimate fallback for recovering all contract tokens
     */
    function emergencyWithdraw() external onlyOwner whenPaused {
        uint256 balance = ecmCoin.balanceOf(address(this));
        require(balance > 0, "No balance");
        require(ecmCoin.transfer(owner(), balance), "Transfer failed");
        emit ECMWithdrawn(owner(), balance);
    }

    // ================ PAUSE CONTROL FUNCTIONS ================

    /**
     * @notice Pauses all staking activities
     * @dev Can only be called by contract owner
     * Stops stake creation and unstaking while allowing withdrawals
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @notice Unpauses all staking activities
     * @dev Can only be called by contract owner
     * Resumes normal contract functionality
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    // ================ VIEW FUNCTIONS ================

    /**
     * @notice Retrieves a paginated list of user's stakes
     * @param _user Address of the user
     * @param _offset Starting index for retrieval
     * @param _limit Maximum number of stakes to return
     * @return Array of StakeInfo for the user
     */
    function getUserStakes(
        address _user, 
        uint256 _offset, 
        uint256 _limit
    ) external view returns (StakeInfo[] memory) {
        StakeInfo[] storage userStakes = stakes[_user];
        uint256 length = userStakes.length;
        
        if (_offset >= length) {
            return new StakeInfo[](0);
        }
        
        uint256 remaining = length - _offset;
        uint256 count = remaining < _limit ? remaining : _limit;
        
        StakeInfo[] memory result = new StakeInfo[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = userStakes[_offset + i];
        }
        
        return result;
    }

    /**
     * @notice Gets total number of stakes for a user
     * @param _user Address of the user
     * @return Total number of stakes
     */
    function getUserStakeCount(address _user) external view returns (uint256) {
        return stakes[_user].length;
    }

    /**
    * @notice Retrieves comprehensive information about a specific stake
    * @param _stakeId Unique identifier of the stake to retrieve
    * @return owner Address of the stake owner
    * @return amount Original staked token amount
    * @return planIndex Index of the staking plan used
    * @return rewardAmount Calculated reward amount for the stake based on current plan settings
    * @return finalAmount Total amount to be received (principal + rewards)
    * @return startTime Timestamp when the stake was created
    * @return endTime Timestamp when the stake can be unstaked
    * @return rewardPercent Current reward percentage multiplied by 100 (e.g., 1500 = 15.00%)
    * @return claimed Boolean indicating whether the stake has been claimed
    * @return claimedAt Timestamp when the stake was claimed
    * @return referrer Address of the referrer (if any)
    */
    function getStakeInfo(uint256 _stakeId) external view returns (
        address owner,
        uint256 amount,
        uint256 planIndex,
        uint256 rewardAmount,
        uint256 finalAmount,
        uint256 startTime,
        uint256 endTime,
        uint256 rewardPercent,
        bool claimed,
        uint256 claimedAt,
        address referrer
    ) {
        owner = stakeOwners[_stakeId];
        require(owner != address(0), "Stake does not exist");
        
        uint256 stakeIndex = findStakeIndex(owner, _stakeId);
        StakeInfo memory userStake = stakes[owner][stakeIndex];
        StakingPlan memory plan = stakingPlans[userStake.planIndex];
        
        planIndex = userStake.planIndex;
        rewardPercent = plan.rewardPercent;
        (rewardAmount, finalAmount) = calculateAmounts(userStake.amount, plan.rewardPercent, plan.duration);
        claimed = userStake.claimedAt > 0;
        
        return (
            owner,
            userStake.amount,
            planIndex,
            rewardAmount,
            finalAmount,
            userStake.startTime,
            userStake.endTime,
            rewardPercent,
            claimed,
            userStake.claimedAt,
            userStake.referrer
        );
    }

    /**
     * @notice Prevents contract from accepting direct ETH transfers
     * @dev Revert any attempt to send ETH directly to the contract
     */
    receive() external payable {
        revert("Contract does not accept ETH");
    }
}