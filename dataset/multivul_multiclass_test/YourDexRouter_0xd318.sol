// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function decimals() external view returns (uint8);
}

interface IUniswapV2Router02 {
    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);
    
    function swapExactETHForTokens(
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable returns (uint[] memory amounts);
    
    function swapExactTokensForETH(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    
    function WETH() external pure returns (address);
}

/**
 * @title YourDexRouter
 * @dev A wrapper contract for Uniswap that adds a fee structure
 * @notice This contract allows users to swap tokens via Uniswap while paying a fee to the protocol
 */
contract YourDexRouter {
    // Constants
    address private constant UNISWAP_ROUTER = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D; // Uniswap V2 Router
    address private constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2; // WETH on Ethereum
    
    // State variables
    address public feeCollector;
    uint256 public feeRate;
    uint256 private constant FEE_DENOMINATOR = 10000;
    string public constant PROTOCOL_NAME = "YourDex Protocol";
    string public constant VERSION = "1.0.0";
    bool private locked;
    
    // Events
    event SwapExecuted(
        address indexed user,
        address indexed tokenIn,
        address indexed tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 feeAmount
    );
    
    event FeeCollectorUpdated(
        address indexed oldCollector,
        address indexed newCollector
    );
    
    event FeeRateUpdated(
        uint256 oldFeeRate,
        uint256 newFeeRate
    );
    
    // Modifiers
    modifier nonReentrant() {
        require(!locked, "YourDexRouter: REENTRANCY_GUARD");
        locked = true;
        _;
        locked = false;
    }
    
    modifier onlyFeeCollector() {
        require(msg.sender == feeCollector, "YourDexRouter: CALLER_NOT_FEE_COLLECTOR");
        _;
    }
    
    /**
     * @dev Constructor sets the initial fee collector and fee rate
     * @param _feeCollector Address that will receive the fees
     * @param _feeRate Fee rate in basis points (e.g., 60 = 0.6%)
     */
    constructor(address _feeCollector, uint256 _feeRate) {
        require(_feeCollector != address(0), "YourDexRouter: ZERO_ADDRESS");
        require(_feeRate <= 300, "YourDexRouter: FEE_TOO_HIGH"); // Max 3% fee
        
        feeCollector = _feeCollector;
        feeRate = _feeRate;
    }
    
    /**
     * @dev Allows the contract to receive ETH
     */
    receive() external payable {}
    
    /**
     * @dev Updates the fee collector address
     * @param _newFeeCollector New address to collect fees
     */
    function updateFeeCollector(address _newFeeCollector) external onlyFeeCollector {
        require(_newFeeCollector != address(0), "YourDexRouter: ZERO_ADDRESS");
        
        address oldFeeCollector = feeCollector;
        feeCollector = _newFeeCollector;
        
        emit FeeCollectorUpdated(oldFeeCollector, _newFeeCollector);
    }
    
    /**
     * @dev Updates the fee rate
     * @param _newFeeRate New fee rate in basis points
     */
    function updateFeeRate(uint256 _newFeeRate) external onlyFeeCollector {
        require(_newFeeRate <= 300, "YourDexRouter: FEE_TOO_HIGH"); // Max 3% fee
        
        uint256 oldFeeRate = feeRate;
        feeRate = _newFeeRate;
        
        emit FeeRateUpdated(oldFeeRate, _newFeeRate);
    }
    
    /**
     * @dev Swaps ETH for tokens
     * @param tokenOut Address of the token to receive
     * @param minAmountOut Minimum amount of tokens to receive
     * @return amountOut Amount of tokens received
     */
    function swapETHForTokens(
        address tokenOut,
        uint256 minAmountOut
    ) external payable nonReentrant returns (uint256 amountOut) {
        require(msg.value > 0, "YourDexRouter: ZERO_ETH_SENT");
        
        // Calculate fee
        uint256 fee = (msg.value * feeRate) / FEE_DENOMINATOR;
        uint256 swapAmount = msg.value - fee;
        
        // Send fee to collector
        (bool success, ) = feeCollector.call{value: fee}("");
        require(success, "YourDexRouter: FEE_TRANSFER_FAILED");
        
        // Set up the swap path
        address[] memory path = new address[](2);
        path[0] = WETH;
        path[1] = tokenOut;
        
        // Get initial balance
        uint256 initialBalance = IERC20(tokenOut).balanceOf(msg.sender);
        
        // Execute swap
        IUniswapV2Router02(UNISWAP_ROUTER).swapExactETHForTokens{value: swapAmount}(
            minAmountOut,
            path,
            msg.sender,
            block.timestamp + 300 // 5 minute deadline
        );
        
        // Calculate amount received
        uint256 finalBalance = IERC20(tokenOut).balanceOf(msg.sender);
        amountOut = finalBalance - initialBalance;
        
        emit SwapExecuted(
            msg.sender,
            WETH,
            tokenOut,
            msg.value,
            amountOut,
            fee
        );
        
        return amountOut;
    }
    
    /**
     * @dev Swaps tokens for ETH
     * @param tokenIn Address of the token to swap
     * @param amountIn Amount of tokens to swap
     * @param minAmountOut Minimum amount of ETH to receive
     * @return amountOut Amount of ETH received
     */
    function swapTokensForETH(
        address tokenIn,
        uint256 amountIn,
        uint256 minAmountOut
    ) external nonReentrant returns (uint256 amountOut) {
        require(amountIn > 0, "YourDexRouter: ZERO_AMOUNT_IN");
        
        // Calculate fee
        uint256 fee = (amountIn * feeRate) / FEE_DENOMINATOR;
        uint256 swapAmount = amountIn - fee;
        
        // Transfer tokens to this contract
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn);
        
        // Send fee to collector
        IERC20(tokenIn).transfer(feeCollector, fee);
        
        // Approve router
        IERC20(tokenIn).approve(UNISWAP_ROUTER, swapAmount);
        
        // Set up the swap path
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = WETH;
        
        // Get initial balance
        uint256 initialBalance = address(msg.sender).balance;
        
        // Execute swap
        IUniswapV2Router02(UNISWAP_ROUTER).swapExactTokensForETH(
            swapAmount,
            minAmountOut,
            path,
            msg.sender,
            block.timestamp + 300 // 5 minute deadline
        );
        
        // Calculate amount received
        uint256 finalBalance = address(msg.sender).balance;
        amountOut = finalBalance - initialBalance;
        
        emit SwapExecuted(
            msg.sender,
            tokenIn,
            WETH,
            amountIn,
            amountOut,
            fee
        );
        
        return amountOut;
    }
    
    /**
     * @dev Swaps tokens for tokens
     * @param tokenIn Address of the token to swap
     * @param tokenOut Address of the token to receive
     * @param amountIn Amount of tokens to swap
     * @param minAmountOut Minimum amount of tokens to receive
     * @return amountOut Amount of tokens received
     */
    function swapTokensForTokens(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 minAmountOut
    ) external nonReentrant returns (uint256 amountOut) {
        require(amountIn > 0, "YourDexRouter: ZERO_AMOUNT_IN");
        
        // Calculate fee
        uint256 fee = (amountIn * feeRate) / FEE_DENOMINATOR;
        uint256 swapAmount = amountIn - fee;
        
        // Transfer tokens to this contract
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn);
        
        // Send fee to collector
        IERC20(tokenIn).transfer(feeCollector, fee);
        
        // Approve router
        IERC20(tokenIn).approve(UNISWAP_ROUTER, swapAmount);
        
        // Set up the swap path
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        // Get initial balance
        uint256 initialBalance = IERC20(tokenOut).balanceOf(msg.sender);
        
        // Execute swap
        IUniswapV2Router02(UNISWAP_ROUTER).swapExactTokensForTokens(
            swapAmount,
            minAmountOut,
            path,
            msg.sender,
            block.timestamp + 300 // 5 minute deadline
        );
        
        // Calculate amount received
        uint256 finalBalance = IERC20(tokenOut).balanceOf(msg.sender);
        amountOut = finalBalance - initialBalance;
        
        emit SwapExecuted(
            msg.sender,
            tokenIn,
            tokenOut,
            amountIn,
            amountOut,
            fee
        );
        
        return amountOut;
    }
    
    /**
     * @dev Gets the expected amount out for a swap
     * @param tokenIn Address of the token to swap
     * @param tokenOut Address of the token to receive
     * @param amountIn Amount of tokens to swap
     * @return amountOut Expected amount of tokens to receive
     */
    function getAmountOut(
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) external view returns (uint256 amountOut) {
        require(amountIn > 0, "YourDexRouter: ZERO_AMOUNT_IN");
        
        // Calculate fee
        uint256 fee = (amountIn * feeRate) / FEE_DENOMINATOR;
        uint256 swapAmount = amountIn - fee;
        
        // Set up the swap path
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        // Get expected amount from Uniswap
        uint[] memory amounts = IUniswapV2Router02(UNISWAP_ROUTER).getAmountsOut(swapAmount, path);
        
        return amounts[1];
    }
    
    /**
     * @dev Gets protocol information
     * @return name Protocol name
     * @return version Protocol version
     * @return currentFeeRate Current fee rate in basis points
     */
    function getProtocolInfo() external view returns (string memory name, string memory version, uint256 currentFeeRate) {
        return (PROTOCOL_NAME, VERSION, feeRate);
    }
    
    /**
     * @dev Rescues tokens accidentally sent to the contract
     * @param token Address of the token to rescue
     * @param amount Amount of tokens to rescue
     */
    function rescueTokens(address token, uint256 amount) external onlyFeeCollector {
        IERC20(token).transfer(feeCollector, amount);
    }
    
    /**
     * @dev Rescues ETH accidentally sent to the contract
     * @param amount Amount of ETH to rescue
     */
    function rescueETH(uint256 amount) external onlyFeeCollector {
        (bool success, ) = feeCollector.call{value: amount}("");
        require(success, "YourDexRouter: ETH_TRANSFER_FAILED");
    }
}