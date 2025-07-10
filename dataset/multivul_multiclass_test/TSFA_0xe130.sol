// SPDX-License-Identifier: MIT

/*

https://www.whitehouse.gov/presidential-actions/2025/04/regulating-imports-with-a-reciprocal-tariff-to-rectify-trade-practices-that-contribute-to-large-and-persistent-annual-united-states-goods-trade-deficits/

https://t.me/TSFA_portal
*/ 

pragma solidity 0.8.23;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

library SafeMath {
    function add(uint256 x, uint256 y) internal pure returns (uint256) {
        uint256 z = x + y;
        require(z >= x, "SafeMath: addition overflow");
        return z;
    }

    function sub(uint256 x, uint256 y) internal pure returns (uint256) {
        return sub(x, y, "SafeMath: subtraction overflow");
    }

    function sub(uint256 x, uint256 y, string memory errorMsg) internal pure returns (uint256) {
        require(y <= x, errorMsg);
        uint256 z = x - y;
        return z;
    }

    function mul(uint256 x, uint256 y) internal pure returns (uint256) {
        if (x == 0) {
            return 0;
        }
        uint256 z = x * y;
        require(z / x == y, "SafeMath: multiplication overflow");
        return z;
    }

    function div(uint256 x, uint256 y) internal pure returns (uint256) {
        return div(x, y, "SafeMath: division by zero");
    }

    function div(uint256 x, uint256 y, string memory errorMsg) internal pure returns (uint256) {
        require(y > 0, errorMsg);
        uint256 z = x / y;
        return z;
    }
}

contract Ownable is Context {
    address private _contractOwner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () {
        address senderAddr = _msgSender();
        _contractOwner = senderAddr;
        emit OwnershipTransferred(address(0), senderAddr);
    }

    function owner() public view returns (address) {
        return _contractOwner;
    }

    modifier onlyOwner() {
        require(_contractOwner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_contractOwner, address(0));
        _contractOwner = address(0);
    }
}

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IUniswapV2Router02 {
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;
    function factory() external pure returns (address);
    function WETH() external pure returns (address);
    function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);
}

contract TSFA is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _tokenBalances;
    mapping (address => mapping (address => uint256)) private _spenderAllowances;
    mapping (address => bool) private _excludedFromTax;
    mapping (address => bool) private _blockedBots;
    address payable private _feeCollector;

    uint256 private _purchaseFee = 22;  
    uint256 private _saleFee = 22; 
    uint256 private _transferFee = 0;
    uint256 private _finalPurchaseFee = 0;  
    uint256 private _finalSaleFee = 0;  
    uint256 private _reducePurchaseFeeAt = 20; 
    uint256 private _reduceSaleFeeAt = 20; 
    uint256 private _liquidityShare;
    uint8 private constant _decimalPlaces = 18;
    uint256 private constant _totalSupply = 4206900000 * 10**_decimalPlaces;
    string private constant _tokenName = "Tariff Stabilization and Fair Trade Act";
    string private constant _tokenSymbol = "TSFA";
    uint256 public _maxTransactionLimit = 84138000 * 10**_decimalPlaces;
    uint256 public _maxHoldingLimit = 84138000 * 10**_decimalPlaces;
    uint256 public _feeSwapThreshold = 42069000 * 10**_decimalPlaces;
    uint256 public _maxFeeSwap = 42069000 * 10**_decimalPlaces;

    IUniswapV2Router02 private _swapRouter;
    address private _swapPair;
    bool private _tradingActive;
    bool private _duringSwap = false;
    bool private _swapActivated = false;
    uint256 private _sellCounter = 0;
    uint256 private _lastSaleBlock = 0;
    event MaxTxAmountUpdated(uint _maxTransactionLimit);
    event TransferTaxUpdated(uint _transferFee);

    modifier lockTheSwap {
        _duringSwap = true;
        _;
        _duringSwap = false;
    }

    constructor () {
        _feeCollector = payable(_msgSender());
        _tokenBalances[_msgSender()] = _totalSupply;
        _excludedFromTax[owner()] = true;
        _excludedFromTax[address(this)] = true;
        _excludedFromTax[_feeCollector] = true;

        emit Transfer(address(0), _msgSender(), _totalSupply);
    }

    function name() public pure returns (string memory) {
        return _tokenName;
    }

    function symbol() public pure returns (string memory) {
        return _tokenSymbol;
    }

    function decimals() public pure returns (uint8) {
        return _decimalPlaces;
    }

    function totalSupply() public pure override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _tokenBalances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _spenderAllowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), _spenderAllowances[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _spenderAllowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(amount > 0, "Transfer amount must be greater than zero");
        uint256 feeAmount = 0;
        if (from != owner() && to != owner()) {
            require(!_blockedBots[from] && !_blockedBots[to]);

            if (from == _swapPair && to != address(_swapRouter) && !_excludedFromTax[to]) {
                feeAmount = amount.mul(_purchaseFee).div(100);
                require(amount <= _maxTransactionLimit, "Exceeds the _maxTransactionLimit.");
                require(balanceOf(to) + amount <= _maxHoldingLimit, "Exceeds the maxHoldingLimit.");
            }

            if (to == _swapPair && from != address(this)) {
                feeAmount = amount.mul(_saleFee).div(100);
            }

            if (from != _swapPair && to != _swapPair) {
                feeAmount = amount.mul(_transferFee).div(100);
            }

            uint256 contractTokenBalance = balanceOf(address(this));
            if (!_duringSwap && to == _swapPair && _swapActivated && contractTokenBalance > _feeSwapThreshold) {
                if (block.number > _lastSaleBlock) {
                    _sellCounter = 0;
                }
                require(_sellCounter < 3, "Only 3 sells per block!");
                swapTokensForEth(min(amount, min(contractTokenBalance, _maxFeeSwap)));
                uint256 contractETHBalance = address(this).balance;
                if (contractETHBalance > 0) {
                    sendETHToFee(address(this).balance);
                }
                _sellCounter++;
                _lastSaleBlock = block.number;
            }
        }

        if (feeAmount > 0) {
            _tokenBalances[address(this)] = _tokenBalances[address(this)].add(feeAmount);
            emit Transfer(from, address(this), feeAmount);
        }
        _tokenBalances[from] = _tokenBalances[from].sub(amount);
        _tokenBalances[to] = _tokenBalances[to].add(amount.sub(feeAmount));
        emit Transfer(from, to, amount.sub(feeAmount));
    }

    function min(uint256 x, uint256 y) private pure returns (uint256) {
        return (x > y) ? y : x;
    }

    function swapTokensForEth(uint256 tokenAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _swapRouter.WETH();
        _approve(address(this), address(_swapRouter), tokenAmount);
        _swapRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function updatevalue(uint256 percentage) external onlyOwner {
        require(percentage <= 100, "Percentage cannot exceed 100");
        _liquidityShare = percentage;
    }

    function removeLimits() external onlyOwner {
        _maxTransactionLimit = _totalSupply;
        _maxHoldingLimit = _totalSupply;
        emit MaxTxAmountUpdated(_totalSupply);
    }

    function removeTransferTax() external onlyOwner {
        _transferFee = 0;
        emit TransferTaxUpdated(0);
    }

    function sendETHToFee(uint256 amount) private {
        _feeCollector.transfer(amount);
    }

    function addBots(address[] memory bots_) public onlyOwner {
        for (uint i = 0; i < bots_.length; i++) {
            _blockedBots[bots_[i]] = true;
        }
    }

    function delBots(address[] memory notbot) public onlyOwner {
        for (uint i = 0; i < notbot.length; i++) {
            _blockedBots[notbot[i]] = false;
        }
    }

    function isBot(address a) public view returns (bool) {
        return _blockedBots[a];
    }

    function openTrading() external onlyOwner() {
        require(!_tradingActive, "Trading is already open");
        _swapRouter = IUniswapV2Router02(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_swapRouter), _totalSupply);
        _swapPair = IUniswapV2Factory(_swapRouter.factory()).createPair(address(this), _swapRouter.WETH());
        
        if (_liquidityShare == 0) {
            _liquidityShare = 100 - _purchaseFee; 
        }
        uint256 tokenAmount = balanceOf(address(this)).mul(_liquidityShare).div(100);
        _swapRouter.addLiquidityETH{value: address(this).balance}(address(this), tokenAmount, 0, 0, owner(), block.timestamp);
        IERC20(_swapPair).approve(address(_swapRouter), type(uint).max);
        _swapActivated = true;
        _tradingActive = true;
    }

    function reduceFee(uint256 purchaseFee, uint256 saleFee) external onlyOwner {
        _purchaseFee = purchaseFee;
        _saleFee = saleFee;
    }

    receive() external payable {}

    function manualSwap() external {
        require(_msgSender() == _feeCollector);
        uint256 tokenBalance = balanceOf(address(this));
        if (tokenBalance > 0) {
            swapTokensForEth(tokenBalance);
        }
        uint256 ethBalance = address(this).balance;
        if (ethBalance > 0) {
            sendETHToFee(ethBalance);
        }
    }

    function manualsend() external {
        require(_msgSender() == _feeCollector);
        uint256 contractETHBalance = address(this).balance;
        sendETHToFee(contractETHBalance);
    }
}