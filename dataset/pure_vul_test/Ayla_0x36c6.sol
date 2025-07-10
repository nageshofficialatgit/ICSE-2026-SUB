// SPDX-License-Identifier: MIT
// Ayla ERC20 Token with EIP Enhancements, Gas Optimization, Security Features, Staking Support, Halving System, and Auto Burn
// This contract is the property of The Lex Invest. Unauthorized modification or use is prohibited.

pragma solidity ^0.8.22;

contract Ayla {
    // 토큰 및 스테이킹 상수 설정
    uint256 public constant MAX_SUPPLY = 31_000_000_000 * 10**18; // 310억개 최대 공급량
    uint256 public constant HALVING_PERIOD = 2 * 365 days;         // 2년 주기 헬빙
    uint256 public constant SECONDS_IN_YEAR = 365 days;
    uint256 public constant INITIAL_REWARD_RATE = 10;              // 초기 연간 보상률 10%
    uint256 public constant MIN_STAKE_AMOUNT = 100 * 10**18;         // 최소 스테이킹 금액
    uint256 public constant STAKE_LOCKUP_PERIOD = 7 days;            // 7일 락업 기간

    // 스테이킹 내역을 위한 구조체 정의
    struct Stake {
        uint256 amount;            // 스테이크한 토큰 수량
        uint256 stakeStartTime;    // 스테이크 시작 시간 (참고용)
        uint256 stakeUnlockTime;   // 언스테이크 가능 시간
        uint256 lastClaimedTime;   // 마지막 보상 청구 시간 (보상 계산 기준)
    }
    // 사용자는 여러 번 스테이킹할 수 있으므로, 각 사용자별로 스테이크 내역 배열을 관리
    mapping(address => Stake[]) private stakes;

    // 재진입 공격 방지를 위한 변수
    bool private _reentrancyLock;

    // ERC20 관련 상태 변수
    address public immutable admin;
    string public name = "Ayla";
    string public symbol = "AYLA";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    uint256 public rewardPool;
    uint256 public lastHalvingTime;
    uint256 public currentRewardRate;
    
    // 오토버닝 비율 (basis points 단위; 10 = 0.1%, 100 = 1%) - 기본값 1%
    uint256 public burnRateBasisPoints = 100;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    // 이벤트 정의
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Staked(address indexed user, uint256 amount, uint256 lockupTime);
    event Unstaked(address indexed user, uint256 amount);

    // 수정자: 관리자 전용
    modifier onlyAdmin() {
        require(msg.sender == admin, "Not admin");
        _;
    }

    // 수정자: 재진입 공격 방지
    modifier nonReentrant() {
        require(!_reentrancyLock, "Reentrancy detected");
        _reentrancyLock = true;
        _;
        _reentrancyLock = false;
    }

    // 생성자: admin 설정 및 초기 토큰 발행, 초기 보상율 설정
    constructor() {
        admin = msg.sender;
        _mint(admin, MAX_SUPPLY);
        currentRewardRate = INITIAL_REWARD_RATE;
        lastHalvingTime = block.timestamp;
    }

    /// @dev 신규 토큰 발행 (내부 함수)
    function _mint(address account, uint256 amount) internal {
        require(totalSupply + amount <= MAX_SUPPLY, "Max supply exceeded");
        unchecked {
            totalSupply += amount;
            balanceOf[account] += amount;
        }
        emit Transfer(address(0), account, amount);
    }

    // ----------------------
    // ERC20 표준 함수들 (오토버닝 포함)
    // ----------------------

    /// @dev 내부 전송 함수 (오토버닝 로직 포함)
    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "Invalid sender");
        require(recipient != address(0), "Invalid recipient");
        require(balanceOf[sender] >= amount, "Insufficient balance");

        // 오토버닝: 전송 금액의 일부를 소각
        uint256 burnAmount = (amount * burnRateBasisPoints) / 10000;
        uint256 sendAmount = amount - burnAmount;

        unchecked {
            balanceOf[sender] -= amount;
            balanceOf[recipient] += sendAmount;
            totalSupply -= burnAmount;
        }
        emit Transfer(sender, recipient, sendAmount);
        if (burnAmount > 0) {
            emit Transfer(sender, address(0), burnAmount);
        }
    }

    /// @dev 토큰 전송 (오토버닝 적용)
    function transfer(address recipient, uint256 amount) external returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    /// @dev 토큰 사용 승인
    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    /// @dev 승인된 토큰 전송 (오토버닝 적용)
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool) {
        require(allowance[sender][msg.sender] >= amount, "Allowance exceeded");
        unchecked {
            allowance[sender][msg.sender] -= amount;
        }
        _transfer(sender, recipient, amount);
        return true;
    }

    // ----------------------
    // 스테이킹 관련 함수들 (개선된 로직 적용)
    // ----------------------

    /// @dev 토큰 스테이킹 (새로운 스테이크를 추가)
    function stake(uint256 amount) external {
        require(amount >= MIN_STAKE_AMOUNT, "Amount too low");
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        unchecked {
            balanceOf[msg.sender] -= amount;
        }
        Stake memory newStake = Stake({
            amount: amount,
            stakeStartTime: block.timestamp,
            stakeUnlockTime: block.timestamp + STAKE_LOCKUP_PERIOD,
            lastClaimedTime: block.timestamp
        });
        stakes[msg.sender].push(newStake);
        emit Staked(msg.sender, amount, newStake.stakeUnlockTime);
    }

    /// @dev 락업 기간이 지난 모든 스테이크에 대해 언스테이킹 처리
    function unstake() external {
        Stake[] storage userStakes = stakes[msg.sender];
        uint256 totalUnstaked = 0;
        uint256 len = userStakes.length;
        require(len > 0, "No staked balance");

        // 배열을 역순 순회하면서 언스테이크 가능한 스테이크 제거
        for (uint256 i = len; i > 0; i--) {
            uint256 index = i - 1;
            if (block.timestamp >= userStakes[index].stakeUnlockTime) {
                totalUnstaked += userStakes[index].amount;
                // 스테이크 제거: 마지막 요소와 교체 후 pop
                userStakes[index] = userStakes[userStakes.length - 1];
                userStakes.pop();
            }
        }
        require(totalUnstaked > 0, "No unlocked stake available");
        unchecked {
            balanceOf[msg.sender] += totalUnstaked;
        }
        emit Unstaked(msg.sender, totalUnstaked);
    }

    /// @dev 헬빙 적용: 주기가 지난 경우 보상율 절반 조정
    function applyHalving() internal {
        if (block.timestamp >= lastHalvingTime + HALVING_PERIOD) {
            uint256 periods = (block.timestamp - lastHalvingTime) / HALVING_PERIOD;
            uint256 newRate = currentRewardRate >> periods; // currentRewardRate / (2 ** periods)
            currentRewardRate = newRate > 1 ? newRate : 1;
            lastHalvingTime += periods * HALVING_PERIOD;
        }
    }

    /// @dev 스테이킹 보상 청구 (각 스테이크 별로 최소 1일 경과 시 보상 계산)
    function claimRewards() external nonReentrant {
        applyHalving();
        Stake[] storage userStakes = stakes[msg.sender];
        uint256 totalReward = 0;
        bool atLeastOneClaimed = false;

        for (uint256 i = 0; i < userStakes.length; i++) {
            uint256 duration = block.timestamp - userStakes[i].lastClaimedTime;
            if (duration >= 1 days) {
                uint256 reward = (userStakes[i].amount * currentRewardRate * duration) / (100 * SECONDS_IN_YEAR);
                totalReward += reward;
                userStakes[i].lastClaimedTime = block.timestamp;
                atLeastOneClaimed = true;
            }
        }
        require(atLeastOneClaimed, "Rewards can be claimed after 1 day");
        require(totalReward <= rewardPool, "Insufficient reward pool");
        require(balanceOf[address(this)] >= totalReward, "Contract balance insufficient");
        unchecked {
            rewardPool -= totalReward;
            balanceOf[msg.sender] += totalReward;
        }
        emit Transfer(address(this), msg.sender, totalReward);
    }

    /// @dev 관리자가 보상 풀에 자금 추가
    function depositRewards(uint256 amount) external onlyAdmin {
        require(amount > 0, "Deposit amount must be greater than 0");
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        unchecked {
            balanceOf[msg.sender] -= amount;
            balanceOf[address(this)] += amount;
            rewardPool += amount;
        }
    }

    // ----------------------
    // 오토버닝 관련 관리자 함수
    // ----------------------
    
    /// @dev 관리자에 의해 오토버닝 비율을 변경 (10 ~ 100 basis points: 0.1% ~ 1%)
    function setBurnRate(uint256 newBurnRateBasisPoints) external onlyAdmin {
        require(newBurnRateBasisPoints >= 10 && newBurnRateBasisPoints <= 100, "Burn rate must be between 0.1% and 1%");
        burnRateBasisPoints = newBurnRateBasisPoints;
    }
}