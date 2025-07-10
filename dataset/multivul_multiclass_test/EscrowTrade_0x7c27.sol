// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// ERC20 토큰 인터페이스
interface IERC20 {
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);

    function transfer(address recipient, uint256 amount)
        external
        returns (bool);

    function balanceOf(address account) external view returns (uint256);

    function allowance(address owner, address spender)
        external
        view
        returns (uint256);
}

/**
 * @title EscrowTrade
 * @dev 개별 거래를 위한 에스크로 컨트랙트
 */
contract EscrowTrade {
    address public owner;
    address public feeOwner;
    uint256 public feePercentage;
    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant MAX_FEE = 200; // 최대 2% 수수료

    // 거래 상태
    enum Status {
        AWAITING_DEPOSIT, // 예치금 대기
        APPROVED, // 거래 승인됨(진행중)
        DISPUTED, // 분쟁 발생
        COMPLETED, // 거래 완료
        REFUNDED // 환불 완료
    }

    struct Trade {
        address buyer;
        address seller;
        address arbiter;
        uint256 fee;
        address feeOwner;
        address tokenAddress;
        uint256 amount;
        Status status;
    }

    // 거래 ID로 거래 정보를 저장
    mapping(bytes32 => Trade) public trades;

    // 전체 거래 ID 목록
    bytes32[] public allTradeIds;

    event TradeCreated(bytes32 indexed tradeId, address buyer, address seller);
    event Deposited(bytes32 indexed tradeId, address buyer, uint256 amount);
    event FundsReleased(
        bytes32 indexed tradeId,
        address seller,
        uint256 amount,
        uint256 fee
    );
    event FundsRefunded(
        bytes32 indexed tradeId,
        address buyer,
        uint256 amount,
        uint256 fee
    );
    event FeePaid(bytes32 indexed tradeId, address arbiter, uint256 fee);
    event FeeUpdated(uint256 oldFee, uint256 newFee);
    event Disputed(bytes32 indexed tradeId, address sender);
    event OwnershipTransferred(
        address indexed oldOwner,
        address indexed newOwner
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }

    modifier onlyTradeBuyerOrSeller(bytes32 tradeId) {
        require(
            msg.sender == trades[tradeId].buyer ||
                msg.sender == trades[tradeId].seller,
            "Only trade buyer or seller can call this"
        );
        _;
    }

    constructor(uint256 _initialFeePercentage) {
        require(_initialFeePercentage <= MAX_FEE, "Fee too high");
        owner = msg.sender;
        feeOwner = msg.sender;
        feePercentage = _initialFeePercentage;
    }

    // 새로운 거래 생성
    function createTrade(
        bytes32 _tradeId,
        address _buyer,
        address _seller,
        address _tokenAddress,
        uint256 _amount
    ) public {
        require(_tokenAddress != address(0), "Invalid token address");
        require(
            _seller != address(0) && _buyer != address(0),
            "Invalid addresses"
        );

      

        Trade storage newTrade = trades[_tradeId];
        newTrade.buyer = _buyer;
        newTrade.seller = _seller;
        newTrade.arbiter = owner;
        newTrade.fee = feePercentage;
        newTrade.feeOwner = feeOwner;
        newTrade.amount = _amount;
        newTrade.tokenAddress = _tokenAddress;
        newTrade.status = Status.AWAITING_DEPOSIT;

        allTradeIds.push(_tradeId);

        emit TradeCreated(_tradeId, _buyer, _seller);

    }

    // 구매자가 토큰 예치
    function deposit(bytes32 tradeId) public {
        Trade storage trade = trades[tradeId];

        // 1. 체크(Checks)
        require(msg.sender == trade.buyer, "Only buyer can deposit");
        require(
            trade.status == Status.AWAITING_DEPOSIT,
            "Invalid status for deposit"
        );

        IERC20 token = IERC20(trade.tokenAddress);
        require(
            token.allowance(msg.sender, address(this)) >= trade.amount,
            "Insufficient allowance"
        );

        // 2. 이펙트(Effects) + 3. 상호작용(Interactions)
        // 거래 상태를 일시적인 중간 상태로 설정하여 재진입을 방지
        trade.status = Status.APPROVED;

        // 외부 호출
        bool success = token.transferFrom(
            msg.sender,
            address(this),
            trade.amount
        );

        // 전송이 실패하면 상태를 원래대로 되돌림
        require(success, "Transfer failed");

        emit Deposited(tradeId, msg.sender, trade.amount);
    }

    // 분쟁 발생
    function disputeArises(bytes32 tradeId)
        public
        onlyTradeBuyerOrSeller(tradeId)
    {
        Trade storage trade = trades[tradeId];
        require(trade.status == Status.APPROVED, "Invalid status for dispute");

        trade.status = Status.DISPUTED;

        emit Disputed(tradeId, msg.sender);
    }

    // 수수료 계산
    function calculateFee(uint256 _amount, uint256 _fee)
        private
        pure
        returns (uint256)
    {
        return (_amount * _fee) / BASIS_POINTS;
    }

    // 판매자에게 자금 지급
    function _releaseFunds(bytes32 tradeId) private {
        Trade storage trade = trades[tradeId];
        uint256 fee = calculateFee(trade.amount, trade.fee);
        uint256 remainingAmount = trade.amount - fee;

        trade.status = Status.COMPLETED;

        IERC20 token = IERC20(trade.tokenAddress);
        require(token.transfer(trade.feeOwner, fee), "Fee transfer failed");
        require(
            token.transfer(trade.seller, remainingAmount),
            "Transfer failed"
        );

        emit FeePaid(tradeId, trade.feeOwner, fee);
        emit FundsReleased(tradeId, trade.seller, remainingAmount, fee);
    }

    // 구매자에게 환불
    function _refund(bytes32 tradeId) private {
        Trade storage trade = trades[tradeId];
        uint256 fee = calculateFee(trade.amount, trade.fee);
        uint256 remainingAmount = trade.amount - fee;

        trade.status = Status.REFUNDED;

        IERC20 token = IERC20(trade.tokenAddress);
        require(token.transfer(trade.feeOwner, fee), "Fee transfer failed");
        require(
            token.transfer(trade.buyer, remainingAmount),
            "Transfer failed"
        );

        emit FeePaid(tradeId, trade.feeOwner, fee);
        emit FundsRefunded(tradeId, trade.buyer, remainingAmount, fee);
    }

    // 정상 거래 완료 - 판매자에게 자금 지급
    function releaseFunds(bytes32 tradeId) public {
        Trade storage trade = trades[tradeId];
        require(msg.sender == trade.buyer, "Not authorized");
        require(trade.status == Status.APPROVED, "Invalid status for release");

        _releaseFunds(tradeId);
    }

    // 분쟁거래 - 판매자에게 자금 지급
    function disputedTransactionToSeller(bytes32 tradeId) public {
        Trade storage trade = trades[tradeId];
        require(msg.sender == owner, "Not authorized");
        require(trade.status == Status.DISPUTED, "Invalid status for release");

        _releaseFunds(tradeId);
    }

    // 분쟁거래 - 구매자에게 자금 지급
    function disputedTransactionToBuyer(bytes32 tradeId) public {
        Trade storage trade = trades[tradeId];
        require(msg.sender == owner, "Not authorized");
        require(trade.status == Status.DISPUTED, "Invalid status for release");

        _refund(tradeId);
    }

    // 수수료율 변경 (소유자만 가능)
    function setFeePercentage(uint256 _newFeePercentage) public onlyOwner {
        require(_newFeePercentage <= MAX_FEE, "Fee too high");
        uint256 oldFee = feePercentage;
        feePercentage = _newFeePercentage;
        emit FeeUpdated(oldFee, _newFeePercentage);
    }

    // 수수료율 지급주소 변경
    function setFeeOwner(address newfeeOwner) public onlyOwner {
        feeOwner = newfeeOwner;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner is zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}