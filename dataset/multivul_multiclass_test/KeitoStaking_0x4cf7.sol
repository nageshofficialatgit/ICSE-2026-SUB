// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract KeitoStaking {
    IERC20 public stakingToken;
    address public owner;

    struct Stake {
        uint256 amount;
        uint256 timestamp;  
        uint256 lockTimestamp; 
        bool unstakeForced;
        uint256 claimedReward;
    }

    mapping(address => Stake[]) public stakes;
    uint256 public apy1 = 50;
    uint256 public apy2 = 100;
    uint256 public apy3 = 200;
    uint256 public constant FORCE_UNSTAKE_FEE = 30;

    event Staked(address indexed user, uint256 amount, uint256 lockTimestamp);
    event Unstaked(address indexed user, uint256 amount, uint256 reward);
    event ForcedUnstake(address indexed user, uint256 amount, uint256 fee);
    event RewardClaimed(address indexed user, uint256 reward);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can execute");
        _;
    }

    constructor(address _stakingToken) {
        owner = msg.sender;
        stakingToken = IERC20(_stakingToken);
    }

    function stake(uint256 amount, uint256 lockDuration) external {
        require(amount > 0);
        require(lockDuration == 7 days || lockDuration == 30 days || lockDuration == 90 days);
        require(stakingToken.transferFrom(msg.sender, address(this), amount));

        uint256 lockTimestamp = block.timestamp + lockDuration; 

        stakes[msg.sender].push(Stake({
            amount: amount,
            timestamp: block.timestamp,
            lockTimestamp: lockTimestamp,
            unstakeForced: false,
            claimedReward: 0
        }));

        emit Staked(msg.sender, amount, lockTimestamp);
    }

    function calculateReward(address user, uint256 stakeIndex) public view returns (uint256) {
        Stake storage userStake = stakes[user][stakeIndex];
        if (userStake.amount == 0) {
            return 0;
        }

        uint256 stakedDuration = block.timestamp - userStake.timestamp;
        uint256 apy = getAPYForDuration(userStake.lockTimestamp - userStake.timestamp);

        uint256 reward = (userStake.amount * apy * stakedDuration) / (365 days * 100);
        return reward - userStake.claimedReward; 
    }

    function claimReward(uint256 stakeIndex) external {
        Stake storage userStake = stakes[msg.sender][stakeIndex];
        uint256 reward = calculateReward(msg.sender, stakeIndex);

        require(reward > 0, "No reward to claim");

        userStake.claimedReward += reward;
        require(stakingToken.transfer(msg.sender, reward));

        emit RewardClaimed(msg.sender, reward);
    }

    function unstake(uint256 stakeIndex) external {
        Stake storage userStake = stakes[msg.sender][stakeIndex];
        require(userStake.amount > 0);
        require(block.timestamp >= userStake.lockTimestamp);

        uint256 reward = calculateReward(msg.sender, stakeIndex);
        uint256 totalAmount = userStake.amount + reward;

        userStake.amount = 0;

        require(stakingToken.transfer(msg.sender, totalAmount));

        emit Unstaked(msg.sender, userStake.amount, reward);
    }

    function forcedUnstake(uint256 stakeIndex) external {
        Stake storage userStake = stakes[msg.sender][stakeIndex];
        require(userStake.amount > 0);
        require(!userStake.unstakeForced);

        uint256 reward = calculateReward(msg.sender, stakeIndex);
        uint256 totalAmount = userStake.amount + reward;
        uint256 fee = (totalAmount * FORCE_UNSTAKE_FEE) / 100;
        uint256 amountAfterFee = totalAmount - fee;

        userStake.unstakeForced = true;

        userStake.amount = 0;

        require(stakingToken.transfer(msg.sender, amountAfterFee));

        emit ForcedUnstake(msg.sender, totalAmount, fee);
    }

    function infoUserStake(address user) external view returns (uint256[] memory amounts, uint256[] memory timestamps, uint256[] memory lockTimestamps, uint256[] memory claimedRewards, bool[] memory unstakeForced) {
        uint256 length = stakes[user].length;

        uint256[] memory _amounts = new uint256[](length);
        uint256[] memory _timestamps = new uint256[](length);
        uint256[] memory _lockTimestamps = new uint256[](length);
        uint256[] memory _claimedRewards = new uint256[](length);
        bool[] memory _unstakeForced = new bool[](length);

        for (uint256 i = 0; i < length; i++) {
            Stake storage userStake = stakes[user][i];
            _amounts[i] = userStake.amount;
            _timestamps[i] = userStake.timestamp;
            _lockTimestamps[i] = userStake.lockTimestamp;
            _claimedRewards[i] = userStake.claimedReward;
            _unstakeForced[i] = userStake.unstakeForced;
        }

        return (_amounts, _timestamps, _lockTimestamps, _claimedRewards, _unstakeForced);
    }

    function getAPYForDuration(uint256 lockDuration) public view returns (uint256) {
        if (lockDuration == 7 days) {
            return apy1;
        } else if (lockDuration == 30 days) {
            return apy2;
        } else if (lockDuration == 90 days) {
            return apy3;
        }
        return 0;
    }

    function setAPY(uint256 _apy1, uint256 _apy2, uint256 _apy3) external onlyOwner {
        apy1 = _apy1;
        apy2 = _apy2;
        apy3 = _apy3;
    }

    function withdraw(uint256 _amount) external onlyOwner {
        uint256 contractBalance = stakingToken.balanceOf(address(this));
        require(_amount <= contractBalance, "Insufficient balance");
        require(stakingToken.transfer(owner, _amount), "Transfer failed");
    }

    function contractBalance() external view returns (uint256) {
        return stakingToken.balanceOf(address(this));
    }
}