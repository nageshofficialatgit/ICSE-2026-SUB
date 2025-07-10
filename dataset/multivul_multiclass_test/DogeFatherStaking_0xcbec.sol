// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

interface IERC165 {
   
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}


library SafeERC20 {
  
    error SafeERC20FailedOperation(address token);

    error SafeERC20FailedDecreaseAllowance(address spender, uint256 currentAllowance, uint256 requestedDecrease);

    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeCall(token.transfer, (to, value)));
    }

    function safeTransferFrom(IERC20 token, address from, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeCall(token.transferFrom, (from, to, value)));
    }

    
    function safeIncreaseAllowance(IERC20 token, address spender, uint256 value) internal {
        uint256 oldAllowance = token.allowance(address(this), spender);
        forceApprove(token, spender, oldAllowance + value);
    }

    
    function safeDecreaseAllowance(IERC20 token, address spender, uint256 requestedDecrease) internal {
        unchecked {
            uint256 currentAllowance = token.allowance(address(this), spender);
            if (currentAllowance < requestedDecrease) {
                revert SafeERC20FailedDecreaseAllowance(spender, currentAllowance, requestedDecrease);
            }
            forceApprove(token, spender, currentAllowance - requestedDecrease);
        }
    }

    function forceApprove(IERC20 token, address spender, uint256 value) internal {
        bytes memory approvalCall = abi.encodeCall(token.approve, (spender, value));

        if (!_callOptionalReturnBool(token, approvalCall)) {
            _callOptionalReturn(token, abi.encodeCall(token.approve, (spender, 0)));
            _callOptionalReturn(token, approvalCall);
        }
    }

    function transferAndCallRelaxed(IERC1363 token, address to, uint256 value, bytes memory data) internal {
        if (to.code.length == 0) {
            safeTransfer(token, to, value);
        } else if (!token.transferAndCall(to, value, data)) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    function transferFromAndCallRelaxed(
        IERC1363 token,
        address from,
        address to,
        uint256 value,
        bytes memory data
    ) internal {
        if (to.code.length == 0) {
            safeTransferFrom(token, from, to, value);
        } else if (!token.transferFromAndCall(from, to, value, data)) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

   
    function approveAndCallRelaxed(IERC1363 token, address to, uint256 value, bytes memory data) internal {
        if (to.code.length == 0) {
            forceApprove(token, to, value);
        } else if (!token.approveAndCall(to, value, data)) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    function _callOptionalReturn(IERC20 token, bytes memory data) private {
        uint256 returnSize;
        uint256 returnValue;
        assembly ("memory-safe") {
            let success := call(gas(), token, 0, add(data, 0x20), mload(data), 0, 0x20)
            // bubble errors
            if iszero(success) {
                let ptr := mload(0x40)
                returndatacopy(ptr, 0, returndatasize())
                revert(ptr, returndatasize())
            }
            returnSize := returndatasize()
            returnValue := mload(0)
        }

        if (returnSize == 0 ? address(token).code.length == 0 : returnValue != 1) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

   
    function _callOptionalReturnBool(IERC20 token, bytes memory data) private returns (bool) {
        bool success;
        uint256 returnSize;
        uint256 returnValue;
        assembly ("memory-safe") {
            success := call(gas(), token, 0, add(data, 0x20), mload(data), 0, 0x20)
            returnSize := returndatasize()
            returnValue := mload(0)
        }
        return success && (returnSize == 0 ? address(token).code.length > 0 : returnValue == 1);
    }
}

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


interface IERC20 {
   
    event Transfer(address indexed from, address indexed to, uint256 value);

   
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function totalSupply() external view returns (uint256);

    function balanceOf(address account) external view returns (uint256);

    function transfer(address to, uint256 value) external returns (bool);

    function allowance(address owner, address spender) external view returns (uint256);

   
     
    function approve(address spender, uint256 value) external returns (bool);

    function transferFrom(address from, address to, uint256 value) external returns (bool);
}


interface IERC1363 is IERC20, IERC165 {
   
  
    function transferAndCall(address to, uint256 value) external returns (bool);

   
    function transferAndCall(address to, uint256 value, bytes calldata data) external returns (bool);

   
    function transferFromAndCall(address from, address to, uint256 value) external returns (bool);

    function transferFromAndCall(address from, address to, uint256 value, bytes calldata data) external returns (bool);

    function approveAndCall(address spender, uint256 value) external returns (bool);

    function approveAndCall(address spender, uint256 value, bytes calldata data) external returns (bool);
}


abstract contract Ownable is Context {
    address private _owner;

   
    error OwnableUnauthorizedAccount(address account);

    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

   
    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

  
    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

  
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

abstract contract ReentrancyGuard {
   
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;

    uint256 private _status;

    error ReentrancyGuardReentrantCall();

    constructor() {
        _status = NOT_ENTERED;
    }

    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {
        if (_status == ENTERED) {
            revert ReentrancyGuardReentrantCall();
        }

        _status = ENTERED;
    }

    function _nonReentrantAfter() private {
     
        _status = NOT_ENTERED;
    }

  
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == ENTERED;
    }
}

contract DogeFatherStaking is ReentrancyGuard, Ownable {
    using SafeERC20 for IERC20;

    // --- Tokens ---
    IERC20 public stakingToken;
    IERC20 public rewardToken;

    // --- Lock-up Periods ---

    uint256 public constant SEVEN_DAYS = 7 days; // 7 days
    uint256 public constant FIVETEEN_DAYS = 15 days; // 15 days
    uint256 public constant THIRTY_DAYS = 30 days; // 30 days

    // --- APRs (in basis points, e.g., 2500 = 25.00%) ---
    uint256 public constant sevenDaysAPR = 2000;
    uint256 public constant fiveteenDaysAPR = 4000;
    uint256 public constant thirtyDaysAPR = 6000;

    // reward tracking
    uint256 public totalDistributedRewards;

    // --- Global Staking State ---
    uint256 public totalStakedAmount;
    uint256 public totalWeightedStake;

    address private penaltyRecipient;
    address private penaltyRecipient2;

    uint256 public constant SWEEP_GRACE_PERIOD = 40 days;

    // --- Injection Parameters ---
    uint256 public monthlyDistributionAmount;
    uint256 public DISTRIBUTION_INTERVAL = THIRTY_DAYS;

    // --- Reward Accumulator ---
    uint256 public constant PRECISION = 1e18;
    uint256 public accRewardPerWeight;
    uint256 public lastRewardUpdate;

    // --- Staking Deadline ---
    uint256 public stakingStartTime;
    uint256 public constant STAKING_DEADLINE = THIRTY_DAYS;

    // --- Stake Data ---
    struct Stake {
        uint256 amount;
        uint256 startTime;
        uint256 lockUpPeriod;
        uint256 apr;
        uint256 weightedStake;
        uint256 rewardDebt;
    }
    mapping(address => Stake[]) public stakes;

    // --- History ---
    struct History {
        uint256 time;
        uint256 totalWeightedStake;
        uint256 accRewardPerWeight;
    }
    History[] public history;

    // --- Events ---
    event Staked(
        address indexed user,
        uint256 amount,
        uint256 lockUpPeriod,
        uint256 apr
    );
    event Unstaked(address indexed user, uint256 amount, uint256 reward);
    event RewardClaimed(address indexed user, uint256 reward);
    event EmergencyUnstaked(
        address indexed user,
        uint256 amount,
        uint256 penalty
    );
    event RewardTokenSweep(address indexed owner, uint256 amount);

    // --- Constructor ---
    constructor(
        address _stakingToken,
        address _rewardToken,
        uint256 _monthlyDistributionAmount,
        address _penaltyRecipient,
        address _penaltyRecipient2
    ) Ownable(msg.sender) {
        require(_stakingToken != address(0), "Zero staking token address");
        require(_rewardToken != address(0), "Zero reward token address");
        require(_penaltyRecipient != address(0), "Zero address is not allowed");
        require(
            _penaltyRecipient2 != address(0),
            "Zero address is not allowed"
        );
        stakingToken = IERC20(_stakingToken);
        rewardToken = IERC20(_rewardToken);
        require(_monthlyDistributionAmount > 0, "Invalid amount");
        penaltyRecipient = _penaltyRecipient;
        penaltyRecipient2 = _penaltyRecipient2;

        // Initialize history time
        history.push(History(block.timestamp, 0, 0));

        monthlyDistributionAmount = _monthlyDistributionAmount;

        stakingStartTime = 0;
        renounceOwnership();
    }

    modifier updateRewards() {
        _updateAccumulator();
        _;
    }

    function _updateAccumulator() internal {
        if (
            block.timestamp <= stakingStartTime + STAKING_DEADLINE &&
            totalWeightedStake > 0
        ) {
            uint256 timeDelta = block.timestamp - lastRewardUpdate;
            uint256 rewardDelta = (monthlyDistributionAmount *
                timeDelta *
                PRECISION) / (totalWeightedStake * DISTRIBUTION_INTERVAL);
            accRewardPerWeight += rewardDelta;
        }
        lastRewardUpdate = block.timestamp;
    }

    function _updateHistory() internal {
        if (
            history.length > 0 &&
            history[history.length - 1].time == block.timestamp
        ) {
            history[history.length - 1].totalWeightedStake = totalWeightedStake;
            history[history.length - 1].accRewardPerWeight = accRewardPerWeight;
        } else {
            history.push(
                History(block.timestamp, totalWeightedStake, accRewardPerWeight)
            );
        }
    }

    function getAccRewardPerWeightAt(
        uint256 targetTime
    ) internal view returns (uint256) {
        if (history.length == 0) return 0;
        // Binary search to find the last entry where time <= targetTime
        uint256 left = 0;
        uint256 right = history.length - 1;
        while (left < right) {
            uint256 mid = left + (right - left + 1) / 2;
            if (history[mid].time <= targetTime) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        History memory h = history[left];
        if (h.time > targetTime) return 0; // Before first update
        if (h.totalWeightedStake == 0) return h.accRewardPerWeight;
        uint256 timeDelta = targetTime - h.time;
        uint256 rewardDelta = (monthlyDistributionAmount *
            timeDelta *
            PRECISION) / (h.totalWeightedStake * DISTRIBUTION_INTERVAL);
        return h.accRewardPerWeight + rewardDelta;
    }

    function stake(
        uint256 amount,
        uint256 lockUpPeriod
    ) external nonReentrant updateRewards {
        require(
            stakingStartTime > 0 && block.timestamp >= stakingStartTime,
            "Staking has not started yet"
        );
        // Check if staking is still allowed (within 30 days of deployment)
        require(
            block.timestamp <= stakingStartTime + STAKING_DEADLINE,
            "Staking period has ended"
        );
        require(amount > 0, "Cannot stake zero tokens");
        uint256 minStake = 1000 * (10 ** 18);
        require(
            amount >= minStake,
            "Amount must be greater than or equal to 1000"
        );

        require(stakes[msg.sender].length < 15, "Maximum stakes reached");

        require(
            lockUpPeriod == SEVEN_DAYS ||
                lockUpPeriod == FIVETEEN_DAYS ||
                lockUpPeriod == THIRTY_DAYS,
            "Invalid lock-up period"
        );

        uint256 apr = getAPRForLockupPeriod(lockUpPeriod);

        uint256 weighted = (amount * apr) / 10000;

        stakes[msg.sender].push(
            Stake({
                amount: amount,
                startTime: block.timestamp,
                lockUpPeriod: lockUpPeriod,
                apr: apr,
                weightedStake: weighted,
                rewardDebt: accRewardPerWeight
            })
        );

        totalStakedAmount += amount;
        totalWeightedStake += weighted;

        _updateHistory();

        stakingToken.safeTransferFrom(msg.sender, address(this), amount);

        emit Staked(msg.sender, amount, lockUpPeriod, apr);
    }

    function unstake(uint256 index) external updateRewards nonReentrant {
        require(index < stakes[msg.sender].length, "Invalid stake index");
        Stake memory userStake = stakes[msg.sender][index];

        uint256 stakeEndTime = userStake.startTime + userStake.lockUpPeriod;
        require(
            block.timestamp >= stakeEndTime,
            "Lock-up period not completed"
        );

        uint256 stakingPeriodEnd = stakingStartTime + STAKING_DEADLINE;

        // Determine the reward end time (earlier of stakeEndTime or stakingPeriodEnd)
        uint256 rewardEndTime = stakeEndTime < stakingPeriodEnd
            ? stakeEndTime
            : stakingPeriodEnd;

        // Since unstaking is only allowed after stakeEndTime, and rewardEndTime <= stakeEndTime,
        // we can directly use the accumulator at rewardEndTime
        uint256 accAtEnd = getAccRewardPerWeightAt(rewardEndTime);

        uint256 reward = (userStake.weightedStake *
            (accAtEnd - userStake.rewardDebt)) / PRECISION;

        require(
            reward <= rewardToken.balanceOf(address(this)),
            "Insufficient rewards pool"
        );

        totalStakedAmount -= userStake.amount;
        if (totalWeightedStake > 0) {
            totalWeightedStake -= userStake.weightedStake;
        }

        if (index != stakes[msg.sender].length - 1) {
            stakes[msg.sender][index] = stakes[msg.sender][
                stakes[msg.sender].length - 1
            ];
        }
        stakes[msg.sender].pop();

        _updateHistory();
        _updateRewardDistributeTracker(reward);

        // Transfer staked tokens and rewards to user
        stakingToken.safeTransfer(msg.sender, userStake.amount);
        rewardToken.safeTransfer(msg.sender, reward);

        emit Unstaked(msg.sender, userStake.amount, reward);
    }

    function claimReward(uint256 index) external nonReentrant updateRewards {
        require(index < stakes[msg.sender].length, "Invalid stake index");
        Stake storage userStake = stakes[msg.sender][index];

        uint256 stakeEndTime = userStake.startTime + userStake.lockUpPeriod;
        uint256 stakingPeriodEnd = stakingStartTime + STAKING_DEADLINE;

        // Determine the reward end time (earlier of stakeEndTime or stakingPeriodEnd)
        uint256 rewardEndTime = stakeEndTime < stakingPeriodEnd
            ? stakeEndTime
            : stakingPeriodEnd;

        uint256 effectiveAcc;

        if (block.timestamp >= rewardEndTime) {
            effectiveAcc = getAccRewardPerWeightAt(rewardEndTime);
        } else {
            effectiveAcc = accRewardPerWeight;
        }
        // Calculate pending reward
        uint256 reward = (userStake.weightedStake *
            (effectiveAcc - userStake.rewardDebt)) / PRECISION;
        require(reward > 0, "No rewards to claim");
        require(
            reward <= rewardToken.balanceOf(address(this)),
            "Insufficient rewards pool"
        );

        _updateRewardDistributeTracker(reward);

        // Update rewardDebt to prevent double-claiming
        userStake.rewardDebt = effectiveAcc;

        // Transfer reward to user
        rewardToken.safeTransfer(msg.sender, reward);

        emit RewardClaimed(msg.sender, reward);
    }

    // For frontend
    function calculateReward(
        Stake memory userStake
    ) public view returns (uint256) {
        uint256 stakeEndTime = userStake.startTime + userStake.lockUpPeriod;
        uint256 stakingPeriodEnd = stakingStartTime + STAKING_DEADLINE;

        // Determine the reward end time (earlier of stakeEndTime or stakingPeriodEnd)
        uint256 rewardEndTime = stakeEndTime < stakingPeriodEnd
            ? stakeEndTime
            : stakingPeriodEnd;

        uint256 effectiveAcc;

        if (block.timestamp >= rewardEndTime) {
            // If current time is past the reward end time, use the accumulator at rewardEndTime
            effectiveAcc = getAccRewardPerWeightAt(rewardEndTime);
        } else {
            // Otherwise, use the current accumulator and add any pending rewards up to the current time
            effectiveAcc = accRewardPerWeight;
            if (
                block.timestamp > lastRewardUpdate &&
                totalWeightedStake > 0 &&
                block.timestamp <= stakingPeriodEnd
            ) {
                uint256 timeDelta = block.timestamp - lastRewardUpdate;
                uint256 pending = (monthlyDistributionAmount *
                    timeDelta *
                    PRECISION) / (totalWeightedStake * DISTRIBUTION_INTERVAL);
                effectiveAcc += pending;
            }
        }

        return
            (userStake.weightedStake * (effectiveAcc - userStake.rewardDebt)) /
            PRECISION;
    }

    function emergencyUnstake(
        uint256 stakeIndex
    ) external nonReentrant updateRewards {
        require(stakeIndex < stakes[msg.sender].length, "Invalid stake index");
        Stake memory userStake = stakes[msg.sender][stakeIndex];

        uint256 penaltyPercent = 2000; // 20%

        uint256 totalPenalty = (userStake.amount * penaltyPercent) / 10000;
        uint256 penaltyRecipientPortion = (totalPenalty * 75) / 100;
        uint256 remainingPenalty = totalPenalty - penaltyRecipientPortion;

        stakingToken.safeTransfer(msg.sender, userStake.amount - totalPenalty);
        stakingToken.safeTransfer(penaltyRecipient, penaltyRecipientPortion);
        stakingToken.safeTransfer(penaltyRecipient2, remainingPenalty);

        // Update global staking totals
        totalStakedAmount -= userStake.amount;
        if (totalWeightedStake > 0) {
            totalWeightedStake -= userStake.weightedStake;
        }

        // Remove the stake
        uint256 lastIndex = stakes[msg.sender].length - 1;
        if (stakeIndex != lastIndex) {
            stakes[msg.sender][stakeIndex] = stakes[msg.sender][lastIndex];
        }
        stakes[msg.sender].pop();

        _updateHistory();

        emit EmergencyUnstaked(
            msg.sender,
            userStake.amount - totalPenalty,
            totalPenalty
        );
    }

    function startStaking() external {
        require(msg.sender == penaltyRecipient, "Unknow caller");
        require(stakingStartTime == 0, "Staking already started");
        stakingStartTime = block.timestamp;
        lastRewardUpdate = block.timestamp;
    }

    function _updateRewardDistributeTracker(uint256 rewardAmount) internal {
        totalDistributedRewards += rewardAmount;
    }

    function getStakeCount(address user) external view returns (uint256) {
        return stakes[user].length;
    }

    function getAPRForLockupPeriod(
        uint256 lockupPeriod
    ) public pure returns (uint256) {
        if (lockupPeriod == SEVEN_DAYS) return sevenDaysAPR;
        if (lockupPeriod == FIVETEEN_DAYS) return fiveteenDaysAPR;
        if (lockupPeriod == THIRTY_DAYS) return thirtyDaysAPR;
        return 0;
    }

    function recoverERC20(address token, uint256 amount) external {
        require(msg.sender == penaltyRecipient, "Unknow caller");
        require(
            token != address(stakingToken) && token != address(rewardToken),
            "Cannot recover staking/reward tokens"
        );
        IERC20(token).safeTransfer(msg.sender, amount);
    }

    function getRewardPoolBalance() public view returns (uint256) {
        return rewardToken.balanceOf(address(this));
    }

    function sweepRemainingRewardTokens() external {
        require(msg.sender == penaltyRecipient, "Unknow caller");
        uint256 sweepUnlockTime = stakingStartTime +
            STAKING_DEADLINE +
            SWEEP_GRACE_PERIOD;

        require(
            block.timestamp > sweepUnlockTime,
            "Sweep grace period not yet ended"
        );

        uint256 remainingBalance = rewardToken.balanceOf(address(this));
        require(remainingBalance > 0, "No reward tokens to sweep");

        rewardToken.safeTransfer(msg.sender, remainingBalance);

        emit RewardTokenSweep(msg.sender, remainingBalance);
    }
}