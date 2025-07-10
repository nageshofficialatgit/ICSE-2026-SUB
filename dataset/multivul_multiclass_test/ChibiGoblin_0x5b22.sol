// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * ChibiGoblin (GOB)
 *
 * Telegram: https://t.me/ChibiGoblinn
 * Website:  https://chibigoblin.com/
 *
 
 */

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

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
    function getPair(address tokenA, address tokenB) external view returns (address pair);
}

interface IUniswapV2Router02 {
    function factory() external pure returns (address);
    function WETH() external pure returns (address);

    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external;
    
    function addLiquidityETH(
        address token,
        uint256 amountTokenDesired,
        uint256 amountTokenMin,
        uint256 amountETHMin,
        address to,
        uint256 deadline
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);
}

abstract contract Ownable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }

    modifier onlyOwner() {
        require(owner() == msg.sender, "Ownable: caller is not the owner");
        _;
    }

    function owner() public view returns (address) {
        return _owner;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Zero address not allowed");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
    
    function renounceOwnership() public onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }
}

contract ChibiGoblin is IERC20, Ownable {
    string private constant _name = "ChibiGoblin";
    string private constant _symbol = "GOB";
    uint8 private constant _decimals = 9;

    // 1,000,000,000 * 10^9 = 1e18 total tokens
    uint256 private constant _totalSupply = 1_000_000_000 * (10 ** _decimals);

    // Balances & allowances
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    // Whitelist addresses that don't pay tax
    mapping(address => bool) private _isTaxExempt;

    // Uniswap references
    address public uniswapV2Pair;
    IUniswapV2Router02 public uniswapV2Router;

    // Tax rates (buy/sell) in %, e.g. 5% = 5
    uint256 public buyTax = 5;
    uint256 public sellTax = 5;

    // Tax wallet to receive ETH
    address public taxWallet = 0x0021Aa0d5C7102dbcD389a551B91016C07A3A472;

    // Swapping & trading flags
    bool private inSwap = false;
    bool public swapEnabled = false;
    bool public tradingOpen = false;

    // Minimum tokens in contract before auto-swap -> ETH
    uint256 public taxSwapThreshold = 100_000 * (10 ** _decimals); // example: 100k tokens

    // Events
    event SwapEnabledUpdated(bool enabled);
    event TaxWalletUpdated(address indexed oldWallet, address indexed newWallet);
    event TradingEnabled(address indexed uniswapPair);

    modifier lockTheSwap() {
        inSwap = true;
        _;
        inSwap = false;
    }

    constructor() {
        // Assign entire supply to deployer
        _balances[msg.sender] = _totalSupply;

        // Tax-exempt addresses
        _isTaxExempt[msg.sender] = true;
        _isTaxExempt[address(this)] = true;
        _isTaxExempt[taxWallet] = true;

        emit Transfer(address(0), msg.sender, _totalSupply);
    }

    // Basic ERC20
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
        return _totalSupply;
    }
    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }
    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }
    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }
    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }
    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, msg.sender, _allowances[sender][msg.sender] - amount);
        return true;
    }

    // Private approval helper
    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "Approve from zero");
        require(spender != address(0), "Approve to zero");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    // Whitelist or remove whitelist
    function setTaxExempt(address account, bool exempt) external onlyOwner {
        _isTaxExempt[account] = exempt;
    }

    function isTaxExempt(address account) external view returns (bool) {
        return _isTaxExempt[account];
    }

    // Set tax wallet
    function setTaxWallet(address newWallet) external onlyOwner {
        require(newWallet != address(0), "Zero address not allowed");
        address old = taxWallet;
        taxWallet = newWallet;
        emit TaxWalletUpdated(old, newWallet);
    }

    // Optionally adjust buy/sell taxes
    function setBuyTax(uint256 newBuyTax) external onlyOwner {
        require(newBuyTax <= 10, "Tax too high");
        buyTax = newBuyTax;
    }

    function setSellTax(uint256 newSellTax) external onlyOwner {
        require(newSellTax <= 10, "Tax too high");
        sellTax = newSellTax;
    }

    // set threshold for auto-swap
    function setTaxSwapThreshold(uint256 threshold) external onlyOwner {
        taxSwapThreshold = threshold;
    }

    // Turn on/off auto-swap
    function setSwapEnabled(bool onOff) external onlyOwner {
        swapEnabled = onOff;
        emit SwapEnabledUpdated(onOff);
    }

    // IMPROVED: Enable trading + set up Uniswap pair with clear error handling
    function openTrading(address router) external onlyOwner {
        require(!tradingOpen, "Trading already open");
        require(router != address(0), "Router address cannot be zero");
        
        IUniswapV2Router02 _uniswapRouter = IUniswapV2Router02(router);
        uniswapV2Router = _uniswapRouter;
        
        // Check if pair already exists
        address factory = _uniswapRouter.factory();
        address weth = _uniswapRouter.WETH();
        
        // Try to get existing pair first
        address existingPair = IUniswapV2Factory(factory).getPair(address(this), weth);
        
        if (existingPair == address(0)) {
            // Pair doesn't exist, create it
            uniswapV2Pair = IUniswapV2Factory(factory).createPair(address(this), weth);
        } else {
            // Pair already exists, use it
            uniswapV2Pair = existingPair;
        }
        
        // Enable trading
        tradingOpen = true;
        swapEnabled = true;
        
        emit TradingEnabled(uniswapV2Pair);
    }
    
    // NEW: Function to just enable trading if a pair already exists
    function enableTradingOnly() external onlyOwner {
        require(!tradingOpen, "Trading already open");
        require(uniswapV2Pair != address(0), "No Uniswap pair set");
        
        tradingOpen = true;
        swapEnabled = true;
        
        emit TradingEnabled(uniswapV2Pair);
    }
    
    // NEW: Function to manually set the Uniswap pair if needed
    function setUniswapPair(address pair) external onlyOwner {
        require(pair != address(0), "Pair cannot be zero address");
        uniswapV2Pair = pair;
    }
    
    // NEW: Function to set the Uniswap router if needed
    function setUniswapRouter(address router) external onlyOwner {
        require(router != address(0), "Router cannot be zero address");
        uniswapV2Router = IUniswapV2Router02(router);
    }

    // Internal transfer, applying tax if not whitelisted
    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0), "Transfer from zero");
        require(to != address(0), "Transfer to zero");
        require(amount > 0, "Zero amount");

        // If trading not open, only taxExempt can transfer
        if(!tradingOpen){
            require(_isTaxExempt[from] || _isTaxExempt[to], "Trading not open");
        }

        // If selling (to == uniswapV2Pair), do we have enough tokens in contract to swap for ETH?
        // Avoid reentrancy with 'lockTheSwap'
        if(!inSwap && swapEnabled && to == uniswapV2Pair && uniswapV2Pair != address(0)){
            uint256 contractTokenBalance = _balances[address(this)];
            if(contractTokenBalance >= taxSwapThreshold) {
                _swapTokensForEth(contractTokenBalance);
            }
        }

        _balances[from] -= amount;

        // Calculate tax if not exempt
        uint256 taxAmount = 0;
        if(!_isTaxExempt[from] && !_isTaxExempt[to]){
            // detect buy or sell
            if(from == uniswapV2Pair) {
                // Buy
                taxAmount = (amount * buyTax) / 100;
            } else if(to == uniswapV2Pair) {
                // Sell
                taxAmount = (amount * sellTax) / 100;
            }
        }

        if(taxAmount > 0){
            _balances[address(this)] += taxAmount;
            emit Transfer(from, address(this), taxAmount);
        }

        // Send the remainder to recipient
        uint256 transferAmount = amount - taxAmount;
        _balances[to] += transferAmount;
        emit Transfer(from, to, transferAmount);
    }

    // Swap the entire contract token balance for ETH, then send ETH to tax wallet
    function _swapTokensForEth(uint256 tokenAmount) private lockTheSwap {
        require(tokenAmount > 0, "Zero tokenAmount");
        require(address(uniswapV2Router) != address(0), "Router not set");

        _approve(address(this), address(uniswapV2Router), tokenAmount);

        // Declare and set the path array
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = uniswapV2Router.WETH();

        // Swap tokens -> ETH
        uniswapV2Router.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0, // no slippage protection
            path,
            address(this),
            block.timestamp
        );

        // Transfer the resulting ETH to tax wallet
        uint256 ethBal = address(this).balance;
        if(ethBal > 0){
            (bool success, ) = payable(taxWallet).call{value: ethBal}("");
            require(success, "ETH Transfer failed");
        }
    }

    // Allow the contract to receive ETH from uniswap
    receive() external payable {}
}