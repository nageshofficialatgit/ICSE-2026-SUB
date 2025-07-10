// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;


library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");

        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return sub(a, b, "SafeMath: subtraction overflow");
    }

    function sub(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        require(b <= a, errorMessage);
        uint256 c = a - b;

        return c;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        // Gas optimization: this is cheaper than requiring 'a' not being zero, but the
        // benefit is lost if 'b' is also tested.
        // See: https://github.com/OpenZeppelin/openzeppelin-contracts/pull/522
        if (a == 0) {
            return 0;
        }

        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");

        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return div(a, b, "SafeMath: division by zero");
    }

    function div(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        require(b > 0, errorMessage);
        uint256 c = a / b;
        // assert(a == b * c + a % b); // There is no case in which this doesn't hold

        return c;
    }

    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return mod(a, b, "SafeMath: modulo by zero");
    }

    function mod(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        require(b != 0, errorMessage);
        return a % b;
    }
}

interface IERC20 {
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );

    event Transfer(address indexed from, address indexed to, uint256 value);

    function name() external view returns (string memory);

    function symbol() external view returns (string memory);

    function decimals() external view returns (uint8);

    function totalSupply() external view returns (uint256);

    function balanceOf(address owner) external view returns (uint256);

    function allowance(address owner, address spender)
        external
        view
        returns (uint256);

    function approve(address spender, uint256 value) external;

    function transfer(address to, uint256 value) external;

    function transferFrom(
        address from,
        address to,
        uint256 value
    ) external;
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address payable) {
        return payable(msg.sender);
    }

    function _msgData() internal view virtual returns (bytes memory) {
        this; // silence state mutability warning without generating bytecode - see https://github.com/ethereum/solidity/issues/2691
        return msg.data;
    }
}

contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    constructor() {
        _owner = _msgSender();
        emit OwnershipTransferred(address(0), _owner);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(
            newOwner != address(0),
            "Ownable: new owner is the zero address"
        );
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

contract Stake_de_RenQ is Ownable{
    using SafeMath for uint256;

    IERC20 public stakeToken;
    address public distributor;

    uint256 public totalStaked;
    uint256 public totalUnStaked;
    uint256 public totalClaimedReward;
    uint256 public totalStakers;
    uint256 public unstakePercent;
    uint256 public percentDivider;

    uint256[3] public Duration = [90 days, 180 days, 365 days];
    uint256[3] public Bonus = [7_50, 12_50, 25_00];
    uint256[3] public totalStakedPerPlan;
    uint256[3] public totalStakersPerPlan;

    struct Stake {
        uint256 plan;
        uint256 withdrawtime;
        uint256 staketime;
        uint256 amount;
        uint256 reward;
        uint256 persecondreward;
        bool withdrawan;
        bool unstaked;
    }

    struct User {
        uint256 totalStaked;
        uint256 totalUnStaked;
        uint256 totalClaimedReward;
        uint256 stakeCount;
        bool alreadyExists;
    }

    mapping(address => User) public Stakers;
    mapping(uint256 => address) public StakersID;
    mapping(address => mapping(uint256 => Stake)) public stakersRecord;
    mapping(address => mapping(uint256 => uint256)) public userStakedPerPlan;

    event STAKE(address Staker, uint256 amount);
    event UNSTAKE(address Staker, uint256 amount);
    event WITHDRAW(address Staker, uint256 amount);

    constructor(address _distributor, address _token) {
        distributor = payable(_distributor);
        stakeToken = IERC20(_token);
        unstakePercent = 2000;
        percentDivider = 100_00;
    }

    function stake(uint256 amount, uint256 planIndex) public {
        require(planIndex >= 0 && planIndex <= 2, "Invalid Time Period");
        require(amount >= 0, "stake more than 0");

        if (!Stakers[msg.sender].alreadyExists) {
            Stakers[msg.sender].alreadyExists = true;
            StakersID[totalStakers] = msg.sender;
            totalStakers++;
        }

        stakeToken.transferFrom(msg.sender, address(this), amount);

        uint256 index = Stakers[msg.sender].stakeCount;
        Stakers[msg.sender].totalStaked = Stakers[msg.sender]
            .totalStaked
            .add(amount);
        totalStaked = totalStaked.add(amount);
        stakersRecord[msg.sender][index].withdrawtime = block.timestamp.add(
            Duration[planIndex]
        );
        stakersRecord[msg.sender][index].staketime = block.timestamp;
        stakersRecord[msg.sender][index].amount = amount;
        stakersRecord[msg.sender][index].reward = amount
            .mul(Bonus[planIndex])
            .div(percentDivider);
        stakersRecord[msg.sender][index].persecondreward = stakersRecord[
            msg.sender
        ][index].reward.div(Duration[planIndex]);
        stakersRecord[msg.sender][index].plan = planIndex;
        Stakers[msg.sender].stakeCount++;
        userStakedPerPlan[msg.sender][planIndex] = userStakedPerPlan[
            msg.sender
        ][planIndex].add(amount);
        totalStakedPerPlan[planIndex] = totalStakedPerPlan[planIndex].add(
            amount
        );
        totalStakersPerPlan[planIndex]++;

        emit STAKE(msg.sender, amount);
    }

    function unstake(uint256 index) public {
        require(
            !stakersRecord[msg.sender][index].withdrawan,
            "already withdrawan"
        );
        require(!stakersRecord[msg.sender][index].unstaked, "already unstaked");
        require(index < Stakers[msg.sender].stakeCount, "Invalid index");

        stakersRecord[msg.sender][index].unstaked = true;
        uint256 penalty = stakersRecord[msg.sender][index]
            .amount
            .mul(unstakePercent)
            .div(percentDivider);
        stakeToken.transfer(distributor, penalty);
        stakeToken.transfer(
            msg.sender,
            (stakersRecord[msg.sender][index].amount).sub(penalty)
        );
        totalUnStaked = totalUnStaked.add(
            stakersRecord[msg.sender][index].amount.sub(penalty)
        );
        Stakers[msg.sender].totalUnStaked = Stakers[msg.sender]
            .totalUnStaked
            .add(stakersRecord[msg.sender][index].amount.sub(penalty));
        uint256 planIndex = stakersRecord[msg.sender][index].plan;
        userStakedPerPlan[msg.sender][planIndex] = userStakedPerPlan[
            msg.sender
        ][planIndex].sub(stakersRecord[msg.sender][index].amount, "user stake");
        totalStakedPerPlan[planIndex] = totalStakedPerPlan[planIndex].sub(
            stakersRecord[msg.sender][index].amount,
            "total stake"
        );
        totalStakersPerPlan[planIndex]--;

        emit UNSTAKE(msg.sender, stakersRecord[msg.sender][index].amount);
    }

    function withdraw(uint256 index) public {
        require(
            !stakersRecord[msg.sender][index].withdrawan,
            "already withdrawan"
        );
        require(!stakersRecord[msg.sender][index].unstaked, "already unstaked");
        require(
            stakersRecord[msg.sender][index].withdrawtime < block.timestamp,
            "cannot withdraw before stake duration"
        );
        require(index < Stakers[msg.sender].stakeCount, "Invalid index");

        stakersRecord[msg.sender][index].withdrawan = true;
        stakeToken.transfer(
            msg.sender,
            stakersRecord[msg.sender][index].amount
        );
        stakeToken.transferFrom(
            distributor,
            msg.sender,
            stakersRecord[msg.sender][index].reward
        );
        totalUnStaked = totalUnStaked.add(
            stakersRecord[msg.sender][index].amount
        );
        totalClaimedReward = totalClaimedReward.add(
            stakersRecord[msg.sender][index].reward
        );
        Stakers[msg.sender].totalUnStaked = Stakers[msg.sender]
            .totalUnStaked
            .add(stakersRecord[msg.sender][index].amount);
        Stakers[msg.sender].totalClaimedReward = Stakers[msg.sender]
            .totalClaimedReward
            .add(stakersRecord[msg.sender][index].reward);
        uint256 planIndex = stakersRecord[msg.sender][index].plan;
        userStakedPerPlan[msg.sender][planIndex] = userStakedPerPlan[
            msg.sender
        ][planIndex].sub(stakersRecord[msg.sender][index].amount, "user stake");
        totalStakedPerPlan[planIndex] = totalStakedPerPlan[planIndex].sub(
            stakersRecord[msg.sender][index].amount,
            "total stake"
        );
        totalStakersPerPlan[planIndex]--;

        emit WITHDRAW(
            msg.sender,
            stakersRecord[msg.sender][index].reward.add(
                stakersRecord[msg.sender][index].amount
            )
        );
    }

    function SetStakeDuration(
        uint256 first,
        uint256 second,
        uint256 third
    ) external onlyOwner {
        Duration[0] = first;
        Duration[1] = second;
        Duration[2] = third;
    }

    function SetStakeBonus(
        uint256 first,
        uint256 second,
        uint256 third
    ) external onlyOwner {
        Bonus[0] = first;
        Bonus[1] = second;
        Bonus[2] = third;
    }

    function SetDivider(uint256 percent) external onlyOwner {
        percentDivider = percent;
    }

    function SetPenalty(uint256 percent) external onlyOwner {
        unstakePercent = percent;
    }

    function SetDistributor(address _wallet) external onlyOwner {
        distributor = _wallet;
    }

    function realtimeReward(address user) public view returns (uint256) {
        uint256 ret;
        for (uint256 i; i < Stakers[user].stakeCount; i++) {
            if (
                !stakersRecord[user][i].withdrawan &&
                !stakersRecord[user][i].unstaked
            ) {
                uint256 val;
                val = block.timestamp - stakersRecord[user][i].staketime;
                val = val.mul(stakersRecord[user][i].persecondreward);
                if (val < stakersRecord[user][i].reward) {
                    ret += val;
                } else {
                    ret += stakersRecord[user][i].reward;
                }
            }
        }
        return ret;
    }
}