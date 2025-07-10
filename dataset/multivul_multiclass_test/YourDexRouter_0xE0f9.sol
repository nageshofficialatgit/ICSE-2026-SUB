// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

interface IUniswapRouter {
    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);
    
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    
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
}

contract YourDexRouter {
    address private constant UNISWAP_ROUTER = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D; // Uniswap V2 Router
    address public constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2; // WETH on Ethereum
    address public feeCollector;
    uint256 public feeRate;
    uint256 private constant FEE_DENOMINATOR = 10000;
    string public constant PROTOCOL_NAME = "YourDex Protocol";
    string public constant VERSION = "1.0.0";
    
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

    constructor(address _feeCollector, uint256 _feeRate) {
        require(_feeCollector != address(0), "YourDexRouter: ZERO_ADDRESS");
        require(_feeRate <= 300, "YourDexRouter: FEE_TOO_HIGH"); // Max 3% fee
        
        feeCollector = _feeCollector;
        feeRate = _feeRate;
    }

    receive() external payable {}

    // Function to swap ETH for tokens
    function swapETHForTokens(address tokenOut, uint256 minAmountOut) external payable returns (uint256 amountOut) {
        require(msg.value > 0, "YourDexRouter: ZERO_ETH_SENT");
        
        // Calculate fee
        uint256 fee = (msg.value * feeRate) / FEE_DENOMINATOR;
        uint256 swapAmount = msg.value - fee;
        
        // Send fee to collector
        payable(feeCollector).transfer(fee);
        
        // Set up the swap path
        address[] memory path = new address[](2);
        path[0] = WETH;
        path[1] = tokenOut;
        
        // Get expected amount out
        uint[] memory amounts = IUniswapRouter(UNISWAP_ROUTER).getAmountsOut(swapAmount, path);
        uint256 expectedAmount = amounts[1];
        
        // Ensure minimum amount out is met
        require(expectedAmount >= minAmountOut, "YourDexRouter: INSUFFICIENT_OUTPUT_AMOUNT");
        
        // Execute the swap
        uint[] memory receivedAmounts = IUniswapRouter(UNISWAP_ROUTER).swapExactETHForTokens{value: swapAmount}(
            minAmountOut,
            path,
            msg.sender,
            block.timestamp + 300 // 5 minute deadline
        );
        
        // Get the actual amount received
        amountOut = receivedAmounts[1];
        
        emit SwapExecuted(msg.sender, WETH, tokenOut, msg.value, amountOut, fee);
        return amountOut;
    }

    // Function to swap tokens for ETH
    function swapTokensForETH(address tokenIn, uint256 amountIn, uint256 minAmountOut) external returns (uint256 amountOut) {
        require(amountIn > 0, "YourDexRouter: ZERO_AMOUNT_IN");
        
        // Calculate fee
        uint256 fee = (amountIn * feeRate) / FEE_DENOMINATOR;
        uint256 swapAmount = amountIn - fee;
        
        // Transfer tokens to this contract
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn);
        
        // Send fee to collector
        IERC20(tokenIn).transfer(feeCollector, fee);
        
        // Approve Uniswap router to spend tokens
        IERC20(tokenIn).approve(UNISWAP_ROUTER, swapAmount);
        
        // Set up the swap path
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = WETH;
        
        // Get expected amount out
        uint[] memory amounts = IUniswapRouter(UNISWAP_ROUTER).getAmountsOut(swapAmount, path);
        uint256 expectedAmount = amounts[1];
        
        // Ensure minimum amount out is met
        require(expectedAmount >= minAmountOut, "YourDexRouter: INSUFFICIENT_OUTPUT_AMOUNT");
        
        // Execute the swap
        uint[] memory receivedAmounts = IUniswapRouter(UNISWAP_ROUTER).swapExactTokensForETH(
            swapAmount,
            minAmountOut,
            path,
            msg.sender,
            block.timestamp + 300 // 5 minute deadline
        );
        
        // Get the actual amount received
        amountOut = receivedAmounts[1];
        
        emit SwapExecuted(msg.sender, tokenIn, WETH, amountIn, amountOut, fee);
        return amountOut;
    }

    // Function to swap tokens for tokens
    function swapTokensForTokens(
        address tokenIn, 
        address tokenOut, 
        uint256 amountIn, 
        uint256 minAmountOut
    ) external returns (uint256 amountOut) {
        require(amountIn > 0, "YourDexRouter: ZERO_AMOUNT_IN");
        
        // Calculate fee
        uint256 fee = (amountIn * feeRate) / FEE_DENOMINATOR;
        uint256 swapAmount = amountIn - fee;
        
        // Transfer tokens to this contract
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn);
        
        // Send fee to collector
        IERC20(tokenIn).transfer(feeCollector, fee);
        
        // Approve Uniswap router to spend tokens
        IERC20(tokenIn).approve(UNISWAP_ROUTER, swapAmount);
        
        // Set up the swap path
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        // Get expected amount out
        uint[] memory amounts = IUniswapRouter(UNISWAP_ROUTER).getAmountsOut(swapAmount, path);
        uint256 expectedAmount = amounts[1];
        
        // Ensure minimum amount out is met
        require(expectedAmount >= minAmountOut, "YourDexRouter: INSUFFICIENT_OUTPUT_AMOUNT");
        
        // Execute the swap
        uint[] memory receivedAmounts = IUniswapRouter(UNISWAP_ROUTER).swapExactTokensForTokens(
            swapAmount,
            minAmountOut,
            path,
            msg.sender,
            block.timestamp + 300 // 5 minute deadline
        );
        
        // Get the actual amount received
        amountOut = receivedAmounts[1];
        
        emit SwapExecuted(msg.sender, tokenIn, tokenOut, amountIn, amountOut, fee);
        return amountOut;
    }

    // Function to get the expected amount out for a swap
    function getAmountOut(address tokenIn, address tokenOut, uint256 amountIn) external view returns (uint256) {
        if (amountIn == 0) return 0;
        
        // Calculate amount after fee
        uint256 amountAfterFee = amountIn - ((amountIn * feeRate) / FEE_DENOMINATOR);
        
        // Set up the path
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        // Get amounts out from Uniswap
        uint[] memory amounts = IUniswapRouter(UNISWAP_ROUTER).getAmountsOut(amountAfterFee, path);
        
        return amounts[1];
    }

    // Function to get protocol information
    function getProtocolInfo() external view returns (string memory name, string memory version, uint256 currentFeeRate) {
        return (PROTOCOL_NAME, VERSION, feeRate);
    }

    // Function to update fee collector (only current fee collector can update)
    function updateFeeCollector(address _newFeeCollector) external {
        require(msg.sender == feeCollector, "YourDexRouter: NOT_AUTHORIZED");
        require(_newFeeCollector != address(0), "YourDexRouter: ZERO_ADDRESS");
        
        address oldCollector = feeCollector;
        feeCollector = _newFeeCollector;
        
        emit FeeCollectorUpdated(oldCollector, _newFeeCollector);
    }

    // Function to update fee rate (only current fee collector can update)
    function updateFeeRate(uint256 _newFeeRate) external {
        require(msg.sender == feeCollector, "YourDexRouter: NOT_AUTHORIZED");
        require(_newFeeRate <= 300, "YourDexRouter: FEE_TOO_HIGH"); // Max 3% fee
        
        uint256 oldFeeRate = feeRate;
        feeRate = _newFeeRate;
        
        emit FeeRateUpdated(oldFeeRate, _newFeeRate);
    }

    // Function to rescue ETH accidentally sent to the contract
    function rescueETH(uint256 amount) external {
        require(msg.sender == feeCollector, "YourDexRouter: NOT_AUTHORIZED");
        payable(feeCollector).transfer(amount);
    }

    // Function to rescue tokens accidentally sent to the contract
    function rescueTokens(address token, uint256 amount) external {
        require(msg.sender == feeCollector, "YourDexRouter: NOT_AUTHORIZED");
        IERC20(token).transfer(feeCollector, amount);
    }
}