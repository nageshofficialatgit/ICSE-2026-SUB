// File: https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/token/ERC20/IERC20.sol


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

// File: https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/utils/Context.sol


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

// File: https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/access/Ownable.sol


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

// File: https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/utils/ReentrancyGuard.sol


// OpenZeppelin Contracts (last updated v5.1.0) (utils/ReentrancyGuard.sol)

pragma solidity ^0.8.20;

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
 * TIP: If EIP-1153 (transient storage) is available on the chain you're deploying at,
 * consider using {ReentrancyGuardTransient} instead.
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
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;

    uint256 private _status;

    /**
     * @dev Unauthorized reentrant call.
     */
    error ReentrancyGuardReentrantCall();

    constructor() {
        _status = NOT_ENTERED;
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
        // On the first call to nonReentrant, _status will be NOT_ENTERED
        if (_status == ENTERED) {
            revert ReentrancyGuardReentrantCall();
        }

        // Any calls to nonReentrant after this point will fail
        _status = ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == ENTERED;
    }
}

// File: staking.sol


pragma solidity 0.8.20;





// Contract definition inheriting from Ownable and ReentrancyGuard
contract TLNstaking is Ownable, ReentrancyGuard {
    // Struct to store information about each staking pool
    struct PoolInfo {
        uint256 lockupDuration; // Duration for which funds are locked up
        uint256 returnPer; // Return percentage for the pool APY
    }

    // Struct to store information about each staking order
    struct OrderInfo {
        address beneficiary; // Address of the staker
        uint256 amount; // Amount of tokens staked
        uint256 lockupDuration; // Duration for which the stake is locked up
        uint256 returnPer; // Return percentage for the stake
        uint256 starttime; // Start time of the stake
        uint256 endtime; // End time of the stake
        uint256 claimedReward; // Amount of claimed rewards
        bool claimed; // Flag indicating whether rewards are claimed
    }

    // Constants defining various durations in seconds
    uint256 private constant _days0 = 1 minutes;
    uint256 private constant _days7 = 7 days;
    uint256 private constant _days14 = 14 days;
    uint256 private constant _days30 = 30 days;
    uint256 private constant _days365 = 365 days;

    // Public variables
    IERC20 public token; // Token being staked
    bool public started = true; // Flag indicating whether staking has started
    uint256 public emergencyWithdrawFeesPercentage = 0; // Fees percentage for emergency withdrawals

    // Percentage returns for different lockup durations
    uint256 private _0daysPercentage = 5;
    uint256 private _7daysPercentage = 7;
    uint256 private _14daysPercentage = 9;
    uint256 private _30daysPercentage = 20;

    // Tracking variables
    uint256 private latestOrderId = 0; // Latest staking order ID
    uint public totalStakers; // Total number of unique stakers
    uint public totalStaked; // Total amount of tokens staked
    uint256 public totalStake = 0; // Total amount of tokens currently staked
    uint256 public totalWithdrawal = 0; // Total amount withdrawn by users
    uint256 public totalRewardPending = 0; // Total pending rewards
    uint256 public totalRewardsDistribution = 0; // Total rewards distributed

     // Modifier to allow only EOA callers
    modifier onlyEOA() {
        require(msg.sender == tx.origin, "Staking: Caller must be an EOA");
        _;
    }

    // Mappings to store data
    mapping(uint256 => PoolInfo) public pooldata; // Mapping of lockup duration to pool info
    mapping(address => uint256) public balanceOf; // Balance of tokens for each address
    mapping(address => uint256) public totalRewardEarn; // Total rewards earned by each address
    mapping(uint256 => OrderInfo) public orders; // Mapping of order ID to order info
    mapping(address => uint256[]) private orderIds; // Mapping of address to array of order IDs

    // Additional mappings for tracking staking status
    mapping(address => mapping(uint => bool)) public hasStaked; // Mapping of address and lockup duration to staking status
    mapping(uint => uint) public stakeOnPool; // Total staked amount on each pool
    mapping(uint => uint) public rewardOnPool; // Total rewards on each pool
    mapping(uint => uint) public stakersPlan; // Total number of stakers on each lockup duration

    // Events
    event Deposit(
        address indexed user,
        uint256 indexed lockupDuration,
        uint256 amount,
        uint256 returnPer
    );
    event Withdraw(
        address indexed user,
        uint256 amount,
        uint256 reward,
        uint256 total
    );
    event WithdrawAll(address indexed user, uint256 amount);
    event RewardClaimed(address indexed user, uint256 reward);
    event RefRewardClaimed(address indexed user, uint256 reward);

    // Constructor function
    constructor(address _token, bool _started) Ownable(msg.sender){
        token = IERC20(_token);
        started = _started;

        // Initialize pooldata with lockup durations and return percentages APY 
        pooldata[1].lockupDuration = _days0; // 0 days
        pooldata[1].returnPer = _0daysPercentage;

        pooldata[2].lockupDuration = _days7; // 7 days
        pooldata[2].returnPer = _7daysPercentage;

        pooldata[3].lockupDuration = _days14; // 14 days
        pooldata[3].returnPer = _14daysPercentage;

        pooldata[4].lockupDuration = _days30; // 30 days
        pooldata[4].returnPer = _30daysPercentage;
    }

    // Function to deposit tokens into the staking contract
    function deposit(uint256 _amount, uint256 _lockupDuration) external onlyEOA {
        // Retrieve pool info based on lockup duration
        PoolInfo storage pool = pooldata[_lockupDuration];
        
        // Check validity of staking parameters
        require(pool.lockupDuration > 0, "TokenStakingTLN: asked pool does not exist");
        require(started, "TokenStakingTLN: staking not yet started");
        require(_amount > 0, "TokenStakingTLN: stake amount must be non-zero");

        // Calculate APY (Annual Percentage Yield) and user reward
        uint256 APY = (_amount * pool.returnPer) / 100;
        uint256 userReward = (APY * pool.lockupDuration) / _days365;

        // Transfer tokens from user to staking contract
        require(token.transferFrom(_msgSender(), address(this), _amount), "TokenStakingTLN: token transferFrom via deposit not succeeded");

        // Update staking information
        orders[++latestOrderId] = OrderInfo(
            _msgSender(),
            _amount,
            pool.lockupDuration,
            pool.returnPer,
            block.timestamp,
            block.timestamp + pool.lockupDuration,
            0,
            false
        );

        // Update staking status
        if (!hasStaked[msg.sender][_lockupDuration]) {
            stakersPlan[_lockupDuration] = stakersPlan[_lockupDuration] + 1;
            totalStakers = totalStakers + 1;
        }
        hasStaked[msg.sender][_lockupDuration] = true;
        stakeOnPool[_lockupDuration] = stakeOnPool[_lockupDuration] + _amount;
        totalStaked = totalStaked + _amount;
        totalStake += _amount;
        totalRewardPending += userReward;
        balanceOf[_msgSender()] += _amount;
        orderIds[_msgSender()].push(latestOrderId);

        // Emit deposit event
        emit Deposit(_msgSender(), pool.lockupDuration, _amount, pool.returnPer);
    }

    // Function to withdraw tokens and rewards
    function withdraw(uint256 orderId) external nonReentrant {
        // Retrieve order info based on order ID
        OrderInfo storage orderInfo = orders[orderId];

        // Check validity of order and caller
        require(orderId <= latestOrderId, "TokenStakingTLN: INVALID orderId, orderId greater than latestOrderId");
        require(_msgSender() == orderInfo.beneficiary, "TokenStakingTLN: caller is not the beneficiary");
        require(!orderInfo.claimed, "TokenStakingTLN: order already unstaked");
        require(block.timestamp >= orderInfo.endtime, "TokenStakingTLN: stake locked until lock duration completion");

        // Calculate available rewards for claiming
        uint256 claimAvailable = pendingRewards(orderId);
        uint256 total = orderInfo.amount + claimAvailable;

        // Update reward and withdrawal information
        totalRewardEarn[_msgSender()] += claimAvailable;
        totalRewardsDistribution += claimAvailable;
        orderInfo.claimedReward += claimAvailable;
        totalRewardPending -= claimAvailable;

        // Update balance and staking information
        balanceOf[_msgSender()] -= orderInfo.amount;
        totalWithdrawal += orderInfo.amount;
        orderInfo.claimed = true;
        totalStake -= orderInfo.amount;

        // Transfer tokens to the beneficiary
        require(token.transfer(address(_msgSender()), total), "TokenStakingTLN: token transfer via withdraw not succeeded");
        rewardOnPool[orderInfo.lockupDuration] = rewardOnPool[orderInfo.lockupDuration] + claimAvailable;

        // Emit withdrawal event
        emit Withdraw(_msgSender(), orderInfo.amount, claimAvailable, total);
    }

    // Function to claim rewards
    function claimRewards(uint256 orderId) external nonReentrant {
        // Retrieve order info based on order ID
        OrderInfo storage orderInfo = orders[orderId];

        // Check validity of order and caller
        require(orderId <= latestOrderId, "TokenStakingTLN: INVALID orderId, orderId greater than latestOrderId");
        require(_msgSender() == orderInfo.beneficiary, "TokenStakingTLN: caller is not the beneficiary");
        require(!orderInfo.claimed, "TokenStakingTLN: order already unstaked");

        // Calculate available rewards for claiming
        uint256 claimAvailable = pendingRewards(orderId);

        // Update reward information
        totalRewardEarn[_msgSender()] += claimAvailable;
        totalRewardsDistribution += claimAvailable;
        totalRewardPending -= claimAvailable;
        orderInfo.claimedReward += claimAvailable;

        // Transfer tokens to the beneficiary
        require(token.transfer(address(_msgSender()), claimAvailable), "TokenStakingTLN: token transfer via claim rewards not succeeded");
        rewardOnPool[orderInfo.lockupDuration] = rewardOnPool[orderInfo.lockupDuration] + claimAvailable;

        // Emit reward claimed event
        emit RewardClaimed(address(_msgSender()), claimAvailable);
    }

    // Function to calculate pending rewards for a given order
    function pendingRewards(uint256 orderId) public view returns (uint256) {
        // Retrieve order info based on order ID
        OrderInfo storage orderInfo = orders[orderId];

        // Check if rewards are claimed
        if (!orderInfo.claimed) {
            if (block.timestamp >= orderInfo.endtime) {
                // Calculate rewards if stake duration has ended
                uint256 APY = (orderInfo.amount * orderInfo.returnPer) / 100;
                uint256 reward = (APY * orderInfo.lockupDuration) / _days365;
                uint256 claimAvailable = reward - orderInfo.claimedReward;
                return claimAvailable;
            } else {
                // Calculate rewards based on current time if stake is still active
                uint256 stakeTime = block.timestamp - orderInfo.starttime;
                uint256 APY = (orderInfo.amount * orderInfo.returnPer) / 100;
                uint256 reward = (APY * stakeTime) / _days365;
                uint256 claimAvailableNow = reward - orderInfo.claimedReward;
                return claimAvailableNow;
            }
        } else {
            return 0;
        }
    }

    // Function for emergency withdrawal
    function emergencyWithdraw(uint256 orderId) external nonReentrant {
        // Retrieve order info based on order ID
        OrderInfo storage orderInfo = orders[orderId];

        // Check validity of order and caller
        require(orderId <= latestOrderId, "TokenStakingTLN: INVALID orderId, orderId greater than latestOrderId");
        require(orderInfo.lockupDuration == _days0, "Please run Withdraw function for unstake");
        require(_msgSender() == orderInfo.beneficiary, "TokenStakingTLN: caller is not the beneficiary");
        require(!orderInfo.claimed, "TokenStakingTLN: order already unstaked");

        // Calculate available rewards for claiming
        uint256 claimAvailable = pendingRewards(orderId);
        uint256 fees = (orderInfo.amount * emergencyWithdrawFeesPercentage) / 1000;
        orderInfo.amount -= fees;
        uint256 total = orderInfo.amount + claimAvailable;

        // Update reward information
        totalRewardEarn[_msgSender()] += claimAvailable;
        totalRewardsDistribution += claimAvailable;
        totalRewardPending -= claimAvailable;
        orderInfo.claimedReward += claimAvailable;

        // Calculate total reward for the stake
        uint256 APY = ((orderInfo.amount + fees) * orderInfo.returnPer) / 100;
        uint256 totalReward = (APY * orderInfo.lockupDuration) / _days365;
        totalRewardPending -= (totalReward - orderInfo.claimedReward);

        // Update balance and withdrawal information
        balanceOf[_msgSender()] -= (orderInfo.amount + fees);
        totalWithdrawal += (orderInfo.amount + fees);
        orderInfo.claimed = true;

        // Transfer tokens to the beneficiary and fees to the owner
        require(token.transfer(address(_msgSender()), total), "TokenStakingTLN: token transfer via emergency withdraw not succeeded");
        require(token.transfer(owner(), fees), "TokenStakingTLN: token transfer via emergency withdraw to admin is not succeeded");

        // Emit withdrawal event
        emit WithdrawAll(_msgSender(), total);
    }

    // Function to toggle staking status
    function toggleStaking(bool _start) external onlyOwner returns (bool) {
        started = _start;
        return true;
    }

    // Function to get order IDs of an investor
    function investorOrderIds(address investor) external view returns (uint256[] memory ids) {
        uint256[] memory arr = orderIds[investor];
        return arr;
    }

    // Function to calculate total rewards of an address
    function _totalRewards(address ref) private view returns (uint256) {
        uint256 rewards;
        uint256[] memory arr = orderIds[ref];
        for (uint256 i = 0; i < arr.length; i++) {
            OrderInfo memory order = orders[arr[i]];
            rewards += (order.claimedReward + pendingRewards(arr[i]));
        }
        return rewards;
    }

    // Function to transfer remaining tokens to the owner
    function transferToken() external onlyOwner {
        uint256 amount = IERC20(token).balanceOf(address(this));
        uint256 transferAmount = amount - totalStake;
        IERC20(token).transfer(owner(), transferAmount);
    }
}