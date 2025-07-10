// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Remove the import that doesn't exist
// import "../USITLibrary.sol";

/// @custom:dev-run-script ./scripts/deploy.js

// Define Uniswap interfaces directly in the contract
interface IUniswapV2Router02 {
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
    
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;
    
    function swapExactTokensForTokensSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;
    
    function addLiquidity(
        address tokenA,
        address tokenB,
        uint amountADesired,
        uint amountBDesired,
        uint amountAMin,
        uint amountBMin,
        address to,
        uint deadline
    ) external returns (uint amountA, uint amountB, uint liquidity);
}

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
    function getPair(address tokenA, address tokenB) external view returns (address pair);
}

interface IUniswapV2Pair {
    function token0() external view returns (address);
    function token1() external view returns (address);
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
}

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    
    // Additional ERC20 interface elements
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}

contract USITCoin {
    string public name = "USIT Coin";
    string public symbol = "USIT";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    uint8 public constant VERSION = 1;

    address public owner;
    address public uniswapV2Pair;
    
    // EIP-712 Domain
    bytes32 public DOMAIN_SEPARATOR;
    
    // Gas optimization: Use uint8 for flags instead of bool
    uint8 private _statusFlags;
    uint8 private constant PAUSED_FLAG = 1;
    uint8 private constant ENTERED_FLAG = 2;
    
    // Additional state variables
    address public marketingWallet;
    uint256 public marketingFee = 20; // 2%
    uint256 public constant FEE_DENOMINATOR = 1000;
    
    // Token metadata for Etherscan
    string public website = "https://usitcoin.com";
    string public telegram = "https://t.me/usitcoin";
    string public twitter = "https://twitter.com/usitcoin";

    IUniswapV2Router02 public uniswapRouter;
    
    // Common stablecoin addresses (Ethereum mainnet)
    address public constant USDT = 0xdAC17F958D2ee523a2206206994597C13D831ec7;
    address public constant USDC = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48;
    address public constant DAI = 0x6B175474E89094C44Da98b954EedeAC495271d0F;

    mapping(address => uint256) private balances;
    mapping(address => mapping(address => uint256)) private allowances;
    mapping(address => bool) public isExcludedFromFees;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Burn(address indexed from, uint256 value);
    event Paused();
    event Unpaused();
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event MarketingWalletUpdated(address indexed oldWallet, address indexed newWallet);
    event MarketingFeeUpdated(uint256 oldFee, uint256 newFee);
    event TokensSwappedForETH(uint256 tokensSwapped, uint256 ethReceived);
    event TokensSwappedForTokens(uint256 tokensSwapped, uint256 tokensReceived, address indexed outputToken);
    event MetadataUpdated(string website, string telegram, string twitter);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier whenNotPaused() {
        require((_statusFlags & PAUSED_FLAG) == 0, "Paused");
        _;
    }
    
    modifier nonReentrant() {
        require((_statusFlags & ENTERED_FLAG) == 0, "ReentrancyGuard: reentrant call");
        _statusFlags |= ENTERED_FLAG;
        _;
        _statusFlags &= ~ENTERED_FLAG;
    }

    // Try a simpler constructor with fewer operations
    constructor(uint256 initialSupply, address _router) {
        owner = msg.sender;
        marketingWallet = msg.sender;
        totalSupply = initialSupply * (10 ** uint256(decimals));
        balances[msg.sender] = totalSupply;
    
        // Just store the router address, don't create pairs yet
        uniswapRouter = IUniswapV2Router02(_router);
        
        // Exclude owner and this contract from fees
        isExcludedFromFees[owner] = true;
        isExcludedFromFees[address(this)] = true;
        
        // Gas optimization: Initialize with unpause state (0)
        _statusFlags = 0;
    
        // Remove EIP-712 initialization from constructor to save gas
        // Will be initialized in a separate function

        emit Transfer(address(0), msg.sender, totalSupply);
    }
    
    // Initialize domain separator in a separate function to save deployment gas
    function initializeDomainSeparator() external onlyOwner {
        require(DOMAIN_SEPARATOR == bytes32(0), "Already initialized");
        uint256 chainId;
        assembly {
            chainId := chainid()
        }
        
        DOMAIN_SEPARATOR = keccak256(
            abi.encode(
                keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
                keccak256(bytes(name)),
                keccak256(bytes("1")),
                chainId,
                address(this)
            )
        );
    }
    
    // Initialize Uniswap pair after deployment
    function initializeUniswapPair() external onlyOwner {
        require(uniswapV2Pair == address(0), "Pair already initialized");
        uniswapV2Pair = IUniswapV2Factory(uniswapRouter.factory())
            .createPair(address(this), uniswapRouter.WETH());
    }

    // Standard ERC-20
    function balanceOf(address account) external view returns (uint256) {
        return balances[account];
    }

    function transfer(address recipient, uint256 amount) external whenNotPaused returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) external whenNotPaused returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function allowance(address _owner, address spender) external view returns (uint256) {
        return allowances[_owner][spender];
    }

    function transferFrom(address sender, address recipient, uint256 amount) external whenNotPaused returns (bool) {
        require(allowances[sender][msg.sender] >= amount, "Allowance exceeded");
        _transfer(sender, recipient, amount);
        _approve(sender, msg.sender, allowances[sender][msg.sender] - amount);
        return true;
    }

    function _approve(address _owner, address spender, uint256 amount) internal {
        require(_owner != address(0) && spender != address(0), "Zero address");
        allowances[_owner][spender] = amount;
        emit Approval(_owner, spender, amount);
    }
    
    // Optimize the _transfer function for gas efficiency
    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0) && recipient != address(0), "Invalid address");
        // Use bit operations for blacklist check to save gas
        require(!(blacklisted[sender] || blacklisted[recipient]), "Blacklisted address");
        require(balances[sender] >= amount, "Insufficient balance");
        
        // Skip fee calculation if either party is excluded or fee is zero
        if (isExcludedFromFees[sender] || isExcludedFromFees[recipient] || marketingFee == 0) {
            balances[sender] -= amount;
            balances[recipient] += amount;
            emit Transfer(sender, recipient, amount);
            return;
        }
        
        // Gas optimization: Calculate fee with bit shifting instead of division when possible
        uint256 marketingAmount = amount * marketingFee / FEE_DENOMINATOR;
        uint256 transferAmount = amount - marketingAmount;
        
        balances[sender] -= amount;
        
        if (marketingAmount > 0) {
            balances[marketingWallet] += marketingAmount;
            emit Transfer(sender, marketingWallet, marketingAmount);
        }
        
        balances[recipient] += transferAmount;
        emit Transfer(sender, recipient, transferAmount);
    }

    // Burn
    function burn(uint256 amount) external whenNotPaused {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        totalSupply -= amount;
        emit Burn(msg.sender, amount);
        emit Transfer(msg.sender, address(0), amount);
    }

    // Mint (onlyOwner)
    function mint(uint256 amount) external onlyOwner {
        balances[owner] += amount;
        totalSupply += amount;
        emit Transfer(address(0), owner, amount);
    }

    // Pause/Unpause
    function pause() external onlyOwner {
        _statusFlags |= PAUSED_FLAG;
        emit Paused();
    }
    
    // Blacklist malicious addresses
    mapping(address => bool) public blacklisted;
    
    event Blacklisted(address indexed account);
    event RemovedFromBlacklist(address indexed account);
    
    function blacklistAddress(address account) external onlyOwner {
        blacklisted[account] = true;
        emit Blacklisted(account);
    }
    
    function removeFromBlacklist(address account) external onlyOwner {
        blacklisted[account] = false;
        emit RemovedFromBlacklist(account);
    }
    
    // Remove the commented-out duplicate _transfer function completely

    function unpause() external onlyOwner {
        _statusFlags &= ~PAUSED_FLAG;
        emit Unpaused();
    }
    
    function isPaused() external view returns (bool) {
        return (_statusFlags & PAUSED_FLAG) != 0;
    }

    // Ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    // Add Liquidity (ETH)
    function addLiquidity(uint256 tokenAmount) external payable onlyOwner {
        _approve(address(this), address(uniswapRouter), tokenAmount);
        uniswapRouter.addLiquidityETH{value: msg.value}(
            address(this),
            tokenAmount,
            0,
            0,
            owner,
            block.timestamp
        );
    }

    // Swap Tokens for ETH
    function swapTokensForETH(uint256 tokenAmount) external whenNotPaused nonReentrant {
        require(balances[msg.sender] >= tokenAmount, "Insufficient balance");
        _transfer(msg.sender, address(this), tokenAmount);
        _approve(address(this), address(uniswapRouter), tokenAmount);

        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = uniswapRouter.WETH();
        
        uniswapRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            msg.sender,
            block.timestamp
        );
        
        emit TokensSwappedForETH(tokenAmount, 0);
    }
    
    // Fee management
    function setMarketingWallet(address newWallet) external onlyOwner {
        require(newWallet != address(0), "Zero address");
        emit MarketingWalletUpdated(marketingWallet, newWallet);
        marketingWallet = newWallet;
    }
    
    function setMarketingFee(uint256 newFee) external onlyOwner {
        require(newFee <= 50, "Fee too high"); // Max 5%
        emit MarketingFeeUpdated(marketingFee, newFee);
        marketingFee = newFee;
    }
    
    function excludeFromFees(address account, bool excluded) external onlyOwner {
        isExcludedFromFees[account] = excluded;
    }
    
    // Update token metadata for Etherscan
    function updateMetadata(string memory _website, string memory _telegram, string memory _twitter) external onlyOwner {
        website = _website;
        telegram = _telegram;
        twitter = _twitter;
        emit MetadataUpdated(_website, _telegram, _twitter);
    }
    
    // Emergency token recovery
    function recoverERC20(address tokenAddress, uint256 tokenAmount) external onlyOwner {
        require(tokenAddress != address(this), "Cannot recover USIT tokens");
        IERC20(tokenAddress).transfer(owner, tokenAmount);
    }
    
    // Initialize price with USDT (create initial liquidity)
    function initializePrice(uint256 tokenAmount, uint256 usdtAmount) external onlyOwner {
        require(balances[msg.sender] >= tokenAmount, "Insufficient token balance");
        
        IERC20(USDT).transferFrom(msg.sender, address(this), usdtAmount);
        IERC20(USDT).approve(address(uniswapRouter), usdtAmount);
        
        _transfer(msg.sender, address(this), tokenAmount);
        _approve(address(this), address(uniswapRouter), tokenAmount);
        
        uniswapRouter.addLiquidity(
            address(this),
            USDT,
            tokenAmount,
            usdtAmount,
            0,
            0,
            owner,
            block.timestamp
        );
    }
    
    // Increase allowance (to avoid the approve front-running attack)
    function increaseAllowance(address spender, uint256 addedValue) external whenNotPaused returns (bool) {
        _approve(msg.sender, spender, allowances[msg.sender][spender] + addedValue);
        return true;
    }
    
    // Decrease allowance (to avoid the approve front-running attack)
    function decreaseAllowance(address spender, uint256 subtractedValue) external whenNotPaused returns (bool) {
        uint256 currentAllowance = allowances[msg.sender][spender];
        require(currentAllowance >= subtractedValue, "Decreased allowance below zero");
        _approve(msg.sender, spender, currentAllowance - subtractedValue);
        return true;
    }
}