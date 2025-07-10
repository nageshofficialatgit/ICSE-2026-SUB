// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/* =================================
 * =========== SafeMath ============
 * ================================= */
library SafeMath {
    function tryAdd(
        uint256 a,
        uint256 b
    ) internal pure returns (bool, uint256) {
        unchecked {
            uint256 c = a + b;
            if (c < a) return (false, 0);
            return (true, c);
        }
    }
    function trySub(
        uint256 a,
        uint256 b
    ) internal pure returns (bool, uint256) {
        unchecked {
            if (b > a) return (false, 0);
            return (true, a - b);
        }
    }
    function tryMul(
        uint256 a,
        uint256 b
    ) internal pure returns (bool, uint256) {
        unchecked {
            if (a == 0) return (true, 0);
            uint256 c = a * b;
            if (c / a != b) return (false, 0);
            return (true, c);
        }
    }
    function tryDiv(
        uint256 a,
        uint256 b
    ) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a / b);
        }
    }
    function tryMod(
        uint256 a,
        uint256 b
    ) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a % b);
        }
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return a - b;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        return a * b;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return a / b;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return a % b;
    }
    function sub(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        unchecked {
            require(b <= a, errorMessage);
            return a - b;
        }
    }
    function div(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a / b;
        }
    }
    function mod(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a % b;
        }
    }
}

/* =================================
 * ============ IERC20 ============
 * ================================= */
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(
        address recipient,
        uint256 amount
    ) external returns (bool);
    function allowance(
        address owner,
        address spender
    ) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
}

/* =================================
 * ============ Context ============
 * ================================= */
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

/* =================================
 * ============ Ownable ============
 * ================================= */
abstract contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );
    constructor() {
        _transferOwnership(_msgSender());
    }
    function owner() public view virtual returns (address) {
        return _owner;
    }
    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }
    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(
            newOwner != address(0),
            "Ownable: new owner is the zero address"
        );
        _transferOwnership(newOwner);
    }
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

/* =================================
 * ========== Address Library ========
 * ================================= */
library Address {
    function isContract(address account) internal view returns (bool) {
        uint256 size;
        assembly {
            size := extcodesize(account)
        }
        return size > 0;
    }
    function sendValue(address payable recipient, uint256 amount) internal {
        require(
            address(this).balance >= amount,
            "Address: insufficient balance"
        );
        (bool success, ) = recipient.call{value: amount}("");
        require(
            success,
            "Address: unable to send value, recipient may have reverted"
        );
    }
    function functionCall(
        address target,
        bytes memory data
    ) internal returns (bytes memory) {
        return functionCall(
            target,
            data,
            "Address: low-level call failed"
        );
    }
    function functionCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal returns (bytes memory) {
        return functionCallWithValue(
            target,
            data,
            0,
            errorMessage
        );
    }
    function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value
    ) internal returns (bytes memory) {
        return functionCallWithValue(
            target, 
            data, 
            value, 
            "Address: low-level call with value failed"
        );
    }
    function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value,
        string memory errorMessage
    ) internal returns (bytes memory) {
        require(
            address(this).balance >= value,
            "Address: insufficient balance for call"
        );
        require(isContract(target), "Address: call to non-contract");
        (bool success, bytes memory returndata) = target.call{value: value}(
            data
        );
        return verifyCallResult(success, returndata, errorMessage);
    }
    function functionStaticCall(
        address target,
        bytes memory data
    ) internal view returns (bytes memory) {
        return
            functionStaticCall(
                target,
                data,
                "Address: low-level static call failed"
            );
    }
    function functionStaticCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal view returns (bytes memory) {
        require(isContract(target), "Address: static call to non-contract");
        (bool success, bytes memory returndata) = target.staticcall(data);
        return verifyCallResult(success, returndata, errorMessage);
    }
    function functionDelegateCall(
        address target,
        bytes memory data
    ) internal returns (bytes memory) {
        return
            functionDelegateCall(
                target,
                data,
                "Address: low-level delegate call failed"
            );
    }
    function functionDelegateCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal returns (bytes memory) {
        require(isContract(target), "Address: delegate call to non-contract");
        (bool success, bytes memory returndata) = target.delegatecall(data);
        return verifyCallResult(success, returndata, errorMessage);
    }
    function verifyCallResult(
        bool success, 
        bytes memory returndata,
        string memory errorMessage
    ) internal pure returns (bytes memory) {
        if (success) {
            return returndata;
        } else {
            if (returndata.length > 0) {
                assembly {
                    let returndata_size := mload(returndata)
                    revert(add(32, returndata), returndata_size)
                }
            } else {
                revert(errorMessage);
            }
        }
    }
}

/* =================================
 * ========== SafeERC20 ============
 * ================================= */
library SafeERC20 {
    using Address for address;
    function safeTransfer(
        IERC20 token, 
        address to, 
        uint256 value
    ) internal {
        _callOptionalReturn(
            token, 
            abi.encodeWithSelector(token.transfer.selector, to, value)
        );
    }
    function safeTransferFrom(
        IERC20 token,
        address from,
        address to,
        uint256 value
    ) internal {
        _callOptionalReturn(
            token, 
            abi.encodeWithSelector(token.transferFrom.selector, from, to, value)
        );
    }
    function _callOptionalReturn(
        IERC20 token, 
        bytes memory data
    ) private {
        bytes memory returndata = address(token).functionCall(
            data, 
            "SafeERC20: low-level call failed"
        );
        if (returndata.length > 0) {
            require(
                abi.decode(returndata, (bool)), 
                "SafeERC20: ERC20 operation did not succeed"
            );
        }
    }
}

/* =================================
 * ======== IERC20Metadata ==========
 * ================================= */
interface IERC20Metadata is IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}

/* =================================
 * ========== OwnerWithdrawable =====
 * ================================= */
contract OwnerWithdrawable is Ownable {
    using SafeMath for uint256;
    using SafeERC20 for IERC20;
    receive() external payable {}
    fallback() external payable {}
    function withdraw(address token, uint256 amt) public onlyOwner {
        IERC20(token).safeTransfer(msg.sender, amt);
    }
    function withdrawAll(address token) public onlyOwner {
        uint256 amt = IERC20(token).balanceOf(address(this));
        withdraw(token, amt);
    }
    function withdrawCurrency(uint256 amt) public onlyOwner {
        payable(msg.sender).transfer(amt);
    }
}

/* =================================
 * ============ ReentrancyGuard =====
 * ================================= */
abstract contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status;
    constructor () {
        _status = _NOT_ENTERED;
    }
    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }
}

/* =================================
 * [NEW] LayerZero 인터페이스
 * ================================= */
interface ILayerZeroEndpoint {
    function send(
        uint16 _dstChainId,
        bytes calldata _destination,
        bytes calldata _payload,
        address payable _refundAddress,
        address _zroPaymentAddress,
        bytes calldata _adapterParams
    ) external payable;
}

/* =================================
 * ============ PresaleV2 ==========
 * ================================= */
contract PresaleV2 is OwnerWithdrawable, ReentrancyGuard {
    using SafeMath for uint256;
    using SafeERC20 for IERC20;
    using SafeERC20 for IERC20Metadata;

    /*  
     * [1] 기존 상태 변수
     */
    mapping(address => bool) public payableTokens;
    mapping(address => uint256) public tokenPrices;
    uint256 public rate;
    bool public saleStatus;
    uint256 public totalTokensforSale;
    uint256 public totalTokensSold;
    address public saleToken;           
    uint public saleTokenDec;
    uint256 public stageLevel = 0;
    mapping(uint256 => uint256) public stageSoldAmount;
    mapping(uint256 => uint256) public stageTargetAmount;
    mapping(uint256 => uint256) public stageStartTime;
    mapping(uint256 => uint256) public stageDuration;
    address[] public buyers;
    
    struct BuyerDetails {
        uint amount;
        bool exists;
        bool hasClaimed;
    }
    mapping(address => BuyerDetails) public buyersDetails;
    uint256 public totalBuyers;
    bool public presaleEnded;
    event PresaleEnded(uint256 timestamp);
    event TokensPurchased(address indexed buyer, uint256 amount);
    event TokensClaimed(address indexed claimer, uint256 amount);

    /*  
     * [3] 스테이킹 관련 상태 변수
     */
    struct StakerDetails {
        uint256 amount;
        uint256 timestamp;
        uint256 stakedInterestRate;
        uint256 stakingPeriod;
        bool hasClaimed;
    }
    mapping(address => StakerDetails[]) public stakers;
    uint256 public interestRate = 300;
    event TokensStaked(address indexed staker, uint256 amount, uint256 timestamp, uint256 stakingPeriod);
    event StakedTokensClaimed(address indexed staker, uint256 amount, uint256 interest);

    /*  
     * [5] 레퍼럴 관련 상태 변수
     */
    struct ReferralInfo {
        uint256 referralCount;
        uint256 totalReferralAmount;
    }
    mapping(address => ReferralInfo) public referralInfos;
    address[] public referrers;
    address public topReferrerByCount;
    uint256 public topReferrerCount;
    address public topReferrerByAmount;
    uint256 public topReferrerAmount;
    uint256 public ourTokenPrice = 9_500_000_000_000_000;
    uint256 public nativeCoinPrice;

    /* 
     * [NEW] LayerZero 관련 상태 변수
     */
    ILayerZeroEndpoint public lzEndpoint = ILayerZeroEndpoint(0x1a44076050125825900e736c501f859c50fE728c);
    uint16 public ethereumChainIdLZ = 30101;
    bytes public ethereumDestination;
    bool public isEthereumChain = false;

    constructor() {
        saleStatus = false;
        presaleEnded = false;
    }

    /*  
     * [6] Sale Token 설정 함수
     */
    function setSaleToken(
        uint256 _decimals,
        uint256 _totalTokensforSale,
        uint256 _rate,
        uint256 _nativeCoinPrice,
        uint256 _ourTokenPrice,
        bool _saleStatus
    ) external onlyOwner {
        require(_rate != 0, "Rate cannot be zero");
        nativeCoinPrice = _nativeCoinPrice;
        ourTokenPrice = _ourTokenPrice;
        rate = _rate;
        saleStatus = _saleStatus;
        saleTokenDec = _decimals;
        totalTokensforSale = _totalTokensforSale;
    }

    /**
     * @dev Sale Token 주소 수정 함수
     */
    function updateSaleToken(address _newSaleToken, uint256 _newDecimals) external onlyOwner {
        require(!presaleEnded, "Cannot update sale token after presale ended");
        require(_newSaleToken != address(0), "Sale token address cannot be zero");
        saleToken = _newSaleToken;
        saleTokenDec = _newDecimals;
    }

    /*  
     * [7] 결제 가능 토큰 추가 함수
     */
    function addPayableTokens(
        address[] memory _tokens,
        uint256[] memory _prices
    ) external onlyOwner {
        require(
            _tokens.length == _prices.length,
            "tokens & prices length mismatch"
        );
        for (uint256 i = 0; i < _tokens.length; i++) {
            require(_prices[i] != 0, "Price cannot be zero");
            payableTokens[_tokens[i]] = true;
            tokenPrices[_tokens[i]] = _prices[i];
        }
    }

    /*  
     * [8] 특정 토큰의 결제 허용/비허용 스위치
     */
    function payableTokenStatus(
        address _token,
        bool _status
    ) external onlyOwner {
        require(payableTokens[_token] != _status, "Status is already set");
        payableTokens[_token] = _status;
    }

    /*  
     * [9] 결제 토큰의 가격 업데이트 함수
     */
    function updateTokenRate(
        address[] memory _tokens,
        uint256[] memory _prices,
        uint256 _rate
    ) external onlyOwner {
        require(
            _tokens.length == _prices.length,
            "tokens & prices length mismatch"
        );
        if (_rate != 0) {
            rate = _rate;
        }
        for (uint256 i = 0; i < _tokens.length; i++) {
            require(payableTokens[_tokens[i]] == true, "Not allowed token");
            require(_prices[i] != 0, "Price cannot be zero");
            tokenPrices[_tokens[i]] = _prices[i];
        }
    }

    /*  
     * [10] Presale 일시중지 및 재개 함수
     */
    function stopSale() external onlyOwner {
        require(!presaleEnded, "Presale: already ended");
        saleStatus = false;
    }
    function resumeSale() external onlyOwner {
        require(!presaleEnded, "Presale: already ended");
        saleStatus = true;
    }

    /*  
     * [11] 최종 Presale 종료 함수
     */
    function endPresale() external onlyOwner {
        require(!presaleEnded, "Presale: already ended");
        saleStatus = false;
        presaleEnded = true;
        emit PresaleEnded(block.timestamp);
    }

    /*  
     * [12] 토큰 구매 시 saleTokenAmt 계산 함수
     */
    function getTokenAmount(
        address token,
        uint256 amount
    ) public view returns (uint256) {
        uint256 amtOut;
        if (token != address(0)) {
            require(payableTokens[token], "Token not allowed");
            uint256 price = tokenPrices[token];
            uint256 tokenDecimals = IERC20Metadata(token).decimals();
            require(saleTokenDec >= tokenDecimals, "Invalid decimals");
            amtOut = amount.mul(price).mul(10 ** saleTokenDec).div(10 ** tokenDecimals).div(ourTokenPrice);
        } else {
            amtOut = amount.mul(nativeCoinPrice).mul(10 ** saleTokenDec).div(10 ** 18).div(ourTokenPrice);
        }
        return amtOut;
    }

    /*  
     * [13] ETH 전송 함수
     */
    function transferETH() private {
        payable(owner()).transfer(msg.value);
    }

    /*  
     * [14] ERC20 토큰 전송 함수
     */
    function transferToken(address _token, uint256 _amount) private {
        IERC20(_token).safeTransferFrom(
            msg.sender,
            owner(),
            _amount
        );
    }

    /*  
     * [15] 토큰 구매 함수
     */
    function buyToken(
        address _token,
        uint256 _amount,
        address _referral
    ) external payable nonReentrant {
        require(!presaleEnded, "Presale: already ended");
        require(saleStatus, "Presale: not active");
        require(isCurrentStageActive(), "Current stage is not active");
        uint256 saleTokenAmt = (_token != address(0))
            ? getTokenAmount(_token, _amount)
            : getTokenAmount(address(0), msg.value);
        require(saleTokenAmt != 0, "Amount is 0");
        uint256 referralBonus = 0;
        if (_referral != address(0) && _referral != msg.sender) {
            referralBonus = saleTokenAmt.mul(5).div(100);
        }
        require(
            totalTokensSold.add(saleTokenAmt).add(referralBonus) 
                <= totalTokensforSale,
            "Not enough tokens for sale"
        );
        if (_token != address(0)) {
            IERC20(_token).safeTransferFrom(
                msg.sender,
                address(this),
                _amount
            );
        } else {
            transferETH();
        }
        totalTokensSold = totalTokensSold.add(saleTokenAmt).add(referralBonus);
        stageSoldAmount[stageLevel] = stageSoldAmount[stageLevel].add(saleTokenAmt).add(referralBonus);
        if (!buyersDetails[msg.sender].exists) {
            buyersDetails[msg.sender].exists = true;
            buyers.push(msg.sender);
            totalBuyers = totalBuyers.add(1);
        }
        buyersDetails[msg.sender].amount = buyersDetails[msg.sender].amount.add(saleTokenAmt);
        if (referralBonus > 0) {
            if (!buyersDetails[_referral].exists) {
                buyersDetails[_referral].exists = true;
                buyers.push(_referral);
                totalBuyers = totalBuyers.add(1);
            }
            buyersDetails[_referral].amount = buyersDetails[_referral].amount.add(referralBonus);
            _updateReferral(_referral, referralBonus);
        }
        emit TokensPurchased(msg.sender, saleTokenAmt);
    }

    /*  
     * [16] 토큰 구매 후 스테이킹을 동시에 하는 함수
     */
    function buyAndStakeToken(
        address _token,
        uint256 _amount,
        address _referral,
        uint256 _stakingPeriod
    ) external payable nonReentrant {
        require(!presaleEnded, "Presale: already ended");
        require(saleStatus, "Presale: not active");
        require(isCurrentStageActive(), "Current stage is not active");
        require(_stakingPeriod > 0, "Staking period must be greater than 0");
        uint256 saleTokenAmt = (_token != address(0))
            ? getTokenAmount(_token, _amount)
            : getTokenAmount(address(0), msg.value);
        require(saleTokenAmt != 0, "Amount is 0");
        uint256 referralBonus = 0;
        if (_referral != address(0) && _referral != msg.sender) {
            referralBonus = saleTokenAmt.mul(5).div(100);
        }
        require(
            totalTokensSold.add(saleTokenAmt).add(referralBonus) 
                <= totalTokensforSale,
            "Not enough tokens for sale"
        );
        if (_token != address(0)) {
            IERC20(_token).safeTransferFrom(
                msg.sender,
                address(this),
                _amount
            );
        } else {
            transferETH();
        }
        totalTokensSold = totalTokensSold.add(saleTokenAmt).add(referralBonus);
        stageSoldAmount[stageLevel] = stageSoldAmount[stageLevel].add(saleTokenAmt).add(referralBonus);
        if (!buyersDetails[msg.sender].exists) {
            buyersDetails[msg.sender].exists = true;
            buyers.push(msg.sender);
            totalBuyers = totalBuyers.add(1);
        }
        buyersDetails[msg.sender].amount = buyersDetails[msg.sender].amount.add(saleTokenAmt);
        if (referralBonus > 0) {
            if (!buyersDetails[_referral].exists) {
                buyersDetails[_referral].exists = true;
                buyers.push(_referral);
                totalBuyers = totalBuyers.add(1);
            }
            buyersDetails[_referral].amount = buyersDetails[_referral].amount.add(referralBonus);
            _updateReferral(_referral, referralBonus);
        }
        _stakeTokens(msg.sender, saleTokenAmt, _stakingPeriod);
        emit TokensPurchased(msg.sender, saleTokenAmt);
    }

    /*  
     * [17] 내부 스테이킹 처리 함수
     */
    function _stakeTokens(address staker, uint256 amount, uint256 stakingPeriod) internal {
        StakerDetails memory newStake = StakerDetails({
            amount: amount,
            timestamp: block.timestamp,
            stakedInterestRate: interestRate,
            stakingPeriod: stakingPeriod,
            hasClaimed: false
        });
        stakers[staker].push(newStake);
        emit TokensStaked(staker, amount, block.timestamp, stakingPeriod);
    }

    /*  
     * [18] Staking Claim을 위한 이자율 설정 함수
     */
    function setInterestRate(uint256 _interestRate) external onlyOwner {
        require(_interestRate <= 1000, "Interest rate too high");
        interestRate = _interestRate;
    }

    /*  
     * [19] 직접 클레임 함수 (LayerZero 통합)
     *      - 이더리움 체인에서는 saleToken을 직접 전송
     *      - 그 외 체인에서는 LayerZero를 통해 이더리움 체인 PresaleV2로 클레임 요청 전달
     */
    function claim() external nonReentrant payable {
        require(presaleEnded, "Presale: not ended yet");
        require(saleToken != address(0), "Sale token not set");
        BuyerDetails storage buyer = buyersDetails[msg.sender];
        uint256 amount = buyer.amount;
        require(amount > 0, "No tokens to claim");
        require(!buyer.hasClaimed, "Tokens already claimed");
        // 클레임 중복 방지를 위해 미리 처리
        buyer.hasClaimed = true;
        if (isEthereumChain) {
            // 이더리움 체인: 직접 토큰 전송
            require(IERC20(saleToken).transfer(msg.sender, amount), "Token transfer failed");
            emit TokensClaimed(msg.sender, amount);
        } else {
            // 그 외 체인: LayerZero 메시지 전송 (payload: 클레임 대상 주소와 amount)
            bytes memory payload = abi.encode(msg.sender, amount);
            bytes memory adapterParams = "";
            require(msg.value > 0, "Insufficient msg.value for LayerZero fee");
            lzEndpoint.send{value: msg.value}(
                ethereumChainIdLZ,
                ethereumDestination,
                payload,
                payable(msg.sender),
                address(0),
                adapterParams
            );
            emit TokensClaimed(msg.sender, amount);
        }
    }

    /*  
     * [20] Presale 종료 후 남은 토큰 회수 함수
     */
    function withdrawUnsoldTokens() external onlyOwner {
        require(presaleEnded, "Presale: not ended yet");
        require(saleToken != address(0), "Sale token not set");
        uint256 unsoldTokens = IERC20(saleToken).balanceOf(address(this)).sub(totalTokensSold);
        require(unsoldTokens > 0, "No unsold tokens to withdraw");
        IERC20(saleToken).transfer(owner(), unsoldTokens);
    }

    /*  
     * [21] Presale 전에 SaleToken 입금 함수 (오너 전용)
     */
    function depositSaleTokens(uint256 amount) external onlyOwner {
        require(saleToken != address(0), "Sale token not set");
        IERC20(saleToken).transferFrom(_msgSender(), address(this), amount);
    }

    /*  
     * [22] 구매자 정보 조회 함수 (페이지네이션)
     */
    struct BuyerAmount {
        uint amount;
        address buyer;
    }
    function buyersAmountList(
        uint _from,
        uint _to
    ) external view returns (BuyerAmount[] memory) {
        require(_from < _to, "_from should be less than _to");
        uint to = _to > totalBuyers ? totalBuyers : _to;
        uint from = _from > totalBuyers ? totalBuyers : _from;
        BuyerAmount[] memory buyersAmt = new BuyerAmount[](to - from);
        for (uint i = from; i < to; i++) {
            buyersAmt[i - from].amount = buyersDetails[ buyers[i] ].amount;
            buyersAmt[i - from].buyer = buyers[i];
        }
        return buyersAmt;
    }

    /*  
     * [23] 레퍼럴 정보를 업데이트하는 내부 함수
     */
    function _updateReferral(address _referral, uint256 _referralBonus) internal {
        ReferralInfo storage refInfo = referralInfos[_referral];
        refInfo.referralCount = refInfo.referralCount.add(1);
        refInfo.totalReferralAmount = refInfo.totalReferralAmount.add(_referralBonus);
        if (refInfo.referralCount == 1 && refInfo.totalReferralAmount == _referralBonus) {
            referrers.push(_referral);
        }
        if (refInfo.referralCount > topReferrerCount) {
            topReferrerCount = refInfo.referralCount;
            topReferrerByCount = _referral;
        }
        if (refInfo.totalReferralAmount > topReferrerAmount) {
            topReferrerAmount = refInfo.totalReferralAmount;
            topReferrerByAmount = _referral;
        }
    }

    /*  
     * [24] 레퍼럴 리스트 조회 함수 (페이지네이션)
     */
    struct ReferrerInfo {
        address referrer;
        uint256 referralCount;
        uint256 totalReferralAmount;
    }
    function getReferrerInfoList(
        uint _from,
        uint _to
    ) external view returns (ReferrerInfo[] memory referrerInfos_) {
        require(_from < _to, "_from should be less than _to");
        uint to = _to > referrers.length ? referrers.length : _to;
        uint from = _from > referrers.length ? referrers.length : _from;
        ReferrerInfo[] memory infos = new ReferrerInfo[](to - from);
        for (uint i = from; i < to; i++) {
            address ref = referrers[i];
            ReferralInfo storage info = referralInfos[ref];
            infos[i - from] = ReferrerInfo({
                referrer: ref,
                referralCount: info.referralCount,
                totalReferralAmount: info.totalReferralAmount
            });
        }
        return infos;
    }

    /*  
     * [25] 상위 레퍼럴 조회 함수
     */
    function getTopReferrerByCount() external view returns (address) {
        return topReferrerByCount;
    }
    function getTopReferrerCount() external view returns (uint256) {
        return topReferrerCount;
    }
    function getTopReferrerByAmount() external view returns (address) {
        return topReferrerByAmount;
    }
    function getTopReferrerAmount() external view returns (uint256) {
        return topReferrerAmount;
    }

    /*  
     * [NEW] 우리 토큰 가격 업데이트 함수
     */
    function updateTokenPrice(uint256 _tokenPrice) external onlyOwner {
        require(_tokenPrice > 0, "Token price must be greater than 0");
        ourTokenPrice = _tokenPrice;
        emit TokenPriceUpdated(_tokenPrice);
    }
    /*  
     * [NEW] 네이티브 코인 가격 업데이트 함수
     */
    function updateNativeCoinPrice(uint256 _nativeCoinPrice) external onlyOwner {
        require(_nativeCoinPrice > 0, "Native coin price must be greater than 0");
        nativeCoinPrice = _nativeCoinPrice;
        emit NativeCoinPriceUpdated(_nativeCoinPrice);
    }
    event TokenPriceUpdated(uint256 tokenPrice);
    event NativeCoinPriceUpdated(uint256 nativeCoinPrice);

    /*  
     * [NEW] 스테이지별 목표 판매량 설정 함수
     */
    function setStageTargetAmount(uint256 _stageLevel, uint256 _targetAmount) external onlyOwner {
        require(_targetAmount > 0, "Target amount must be greater than 0");
        stageTargetAmount[_stageLevel] = _targetAmount;
    }
    /*  
     * [NEW] 현재 스테이지 목표 판매량 조회 함수
     */
    function getCurrentStageTargetAmount() external view returns (uint256) {
        return stageTargetAmount[stageLevel];
    }
    /*  
     * [NEW] 스테이지 기간 설정 함수
     */
    function setStageDuration(
        uint256 _stageLevel,
        uint256 _duration
    ) external onlyOwner {
        require(_duration > 0, "Duration must be greater than 0");
        uint256 startTime = block.timestamp;
        uint256 endTime = startTime + _duration;
        stageStartTime[_stageLevel] = startTime;
        stageDuration[_stageLevel] = _duration;
        emit StageTimeUpdated(_stageLevel, startTime, endTime);
    }
    /*  
     * [NEW] 현재 스테이지 남은 시간 조회 함수 (초 단위)
     */
    function getCurrentStageTimeLeft() external view returns (uint256) {
        uint256 endTime = stageStartTime[stageLevel] + stageDuration[stageLevel];
        if (endTime == 0 || block.timestamp >= endTime) {
            return 0;
        }
        return endTime - block.timestamp;
    }
    /*  
     * [NEW] 현재 스테이지 활성화 여부 확인 함수
     */
    function isCurrentStageActive() public view returns (bool) {
        uint256 startTime = stageStartTime[stageLevel];
        uint256 endTime = startTime + stageDuration[stageLevel];
        if (startTime == 0 || endTime == 0) {
            return true;
        }
        return block.timestamp >= startTime && block.timestamp < endTime;
    }
    event StageTimeUpdated(uint256 stageLevel, uint256 startTime, uint256 endTime);

    /*  
     * [NEW] 스테이지 레벨, 토큰 가격, 이자율, 최대 판매 수량을 업데이트하는 함수
     */
    function updatePresaleParameters(
        uint256 _stageLevel,
        uint256 _ourTokenPrice,
        uint256 _interestRate,
        uint256 _maxSaleAmount
    ) external onlyOwner {
        require(!presaleEnded, "Presale: already ended");
        require(_ourTokenPrice > 0, "Token price must be greater than 0");
        require(_maxSaleAmount > totalTokensSold, "Max sale amount must be greater than total tokens sold");
        stageLevel = _stageLevel;
        ourTokenPrice = _ourTokenPrice;
        interestRate = _interestRate;
        totalTokensforSale = _maxSaleAmount;
    }

    /*  
     * [NEW] LayerZero 구성 함수  
     * 이더리움 체인에 배포된 PresaleV2 컨트랙트의 LayerZero 정보를 설정합니다.
     * @param _endpoint LayerZero 엔드포인트 주소
     * @param _ethChainIdLZ LayerZero 상의 이더리움 체인 id
     * @param _ethDestination 이더리움 체인에 배포된 PresaleV2 컨트랙트 주소 (바이트 형태, 예: abi.encodePacked(ethPresaleAddress))
     * @param _isEthereumChain 현재 컨트랙트가 이더리움 체인에 배포되었는지 여부
     */
    function setLayerZeroConfig(
        address _endpoint,
        uint16 _ethChainIdLZ,
        bytes calldata _ethDestination,
        bool _isEthereumChain
    ) external onlyOwner {
        lzEndpoint = ILayerZeroEndpoint(_endpoint);
        ethereumChainIdLZ = _ethChainIdLZ;
        ethereumDestination = _ethDestination;
        isEthereumChain = _isEthereumChain;
    }
}