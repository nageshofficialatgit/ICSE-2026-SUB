// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

//
// Minimal interfaces for external calls
//

interface IERC20 {
    function approve(address spender, uint amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
}

interface IUniswapV2Router02 {
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

interface IUniswapV2Factory {
    function getPair(address tokenA, address tokenB) external view returns (address);
}

interface IUniswapV2Pair {
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
    function token0() external view returns (address);
    function token1() external view returns (address);
}

//
// Main Contract
//
contract UniswapTradingBot {
    // ------------------------------------------------------------------------
    // State Variables
    // ------------------------------------------------------------------------
    address public owner;
    bool   public isActive;

    // Hardcoded addresses for mainnet
    address public constant WETH           = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address public constant UNISWAP_ROUTER = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
    address public constant UNISWAP_FACTORY= 0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f;

    // Configurable parameters
    uint public slippageTolerance = 300;   // 3% = 300 basis points
    uint public gasLimit          = 300_000;

    // ------------------------------------------------------------------------
    // Events
    // ------------------------------------------------------------------------
    event BotStarted();
    event BotStopped();
    event TradeExecuted(
        string tradeType,
        uint amountIn,
        uint amountOut,
        address token
    );
    event ProfitabilityDetected(
        address token,
        uint profit,
        uint amountIn,
        uint gasCost,
        uint slippage
    );
    event FundsWithdrawn(address indexed to, uint amount);
    event SlippageUpdated(uint newSlippage);
    event GasLimitUpdated(uint newGasLimit);

    // ------------------------------------------------------------------------
    // Modifiers
    // ------------------------------------------------------------------------
    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    // ------------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------------
    constructor() {
        owner = msg.sender;
    }

    // ------------------------------------------------------------------------
    // Fallback to receive ETH
    // ------------------------------------------------------------------------
    receive() external payable {}

    // ------------------------------------------------------------------------
    // Start the Bot
    // ------------------------------------------------------------------------
    function start() external onlyOwner {
        require(!isActive, "Bot is already running");
        require(
            address(this).balance >= 0.01 ether,
            "Insufficient contract balance"
        );
        isActive = true;
        emit BotStarted();
    }

    // ------------------------------------------------------------------------
    // Stop the Bot
    // ------------------------------------------------------------------------
    function stop() external onlyOwner {
        require(isActive, "Bot is not running");
        isActive = false;
        emit BotStopped();
    }

    // ------------------------------------------------------------------------
    // Execute Trade
    // ------------------------------------------------------------------------
    function executeTrade(address token, uint amountIn) external onlyOwner {
        require(isActive, "Bot is not active");

        uint profit = checkProfitability(token, amountIn);
        require(profit > 0, "No profitable trade found");

        // 1) Buy tokens using ETH
        uint amountOut = buyTokens(token, amountIn);

        // 2) Sell tokens back to ETH
        sellTokens(token, amountOut);

        emit ProfitabilityDetected(
            token,
            profit,
            amountIn,
            gasLimit * tx.gasprice,
            slippageTolerance
        );
    }

    // ------------------------------------------------------------------------
    // Buy tokens from Uniswap
    // ------------------------------------------------------------------------
    function buyTokens(address token, uint amountIn)
        private
        returns (uint)
    {
        address[] memory path = getPath(WETH, token);

        uint[] memory amounts = IUniswapV2Router02(UNISWAP_ROUTER)
            .swapExactETHForTokens{ value: amountIn }(
                1,      // minAmountOut, set to 1 to allow high slippage
                path,
                address(this),
                block.timestamp + 300
            );

        emit TradeExecuted("Buy", amountIn, amounts[1], token);
        return amounts[1];
    }

    // ------------------------------------------------------------------------
    // Sell tokens to get ETH
    // ------------------------------------------------------------------------
    function sellTokens(address token, uint amountIn) private {
        IERC20(token).approve(UNISWAP_ROUTER, amountIn);

        address[] memory path = getPath(token, WETH);

        uint[] memory amounts = IUniswapV2Router02(UNISWAP_ROUTER)
            .swapExactTokensForETH(
                amountIn,
                1, // minAmountOut
                path,
                address(this),
                block.timestamp + 300
            );

        emit TradeExecuted("Sell", amountIn, amounts[1], token);
    }

    // ------------------------------------------------------------------------
    // Check Profitability
    // ------------------------------------------------------------------------
    function checkProfitability(address token, uint amountIn)
        private
        view
        returns (uint)
    {
        address pair = IUniswapV2Factory(UNISWAP_FACTORY)
            .getPair(token, WETH);

        require(pair != address(0), "Pair not found");

        (uint112 reserve0, uint112 reserve1, ) =
            IUniswapV2Pair(pair).getReserves();

        (uint tokenReserve, uint wethReserve) =
            (IUniswapV2Pair(pair).token0() == WETH)
                ? (reserve1, reserve0)
                : (reserve0, reserve1);

        // Price in terms of WETH
        uint tokenPrice = (wethReserve * 1e18) / tokenReserve;

        // Subtract slippage from the perceived sell price
        uint sellPrice = tokenPrice - (tokenPrice * slippageTolerance) / 10_000;

        // Calculate the approximate profit in wei
        uint profit = (sellPrice * amountIn) / 1e18;

        // Calculate approximate gas cost
        uint gasCost = gasLimit * tx.gasprice;

        // Return net profit if above cost
        return profit > (amountIn + gasCost)
            ? profit - (amountIn + gasCost)
            : 0;
    }

    // ------------------------------------------------------------------------
    // Build Uniswap Path
    // ------------------------------------------------------------------------
    function getPath(address fromToken, address toToken)
        private
        pure
        returns (address[] memory path)
    {
        path = new address[](2);
        path[0] = fromToken;
        path[1] = toToken;
        return path;
    }

    // ------------------------------------------------------------------------
    // Withdraw All ETH from Contract
    // ------------------------------------------------------------------------
    function withdraw() external onlyOwner {
        uint contractBalance = address(this).balance;
        require(contractBalance > 0, "No funds to withdraw");

        (bool success, ) = payable(owner).call{value: contractBalance}("");
        require(success, "Withdrawal failed");

        emit FundsWithdrawn(owner, contractBalance);
    }

    // ------------------------------------------------------------------------
    // Update Slippage Tolerance
    // ------------------------------------------------------------------------
    function setSlippageTolerance(uint _slippage) external onlyOwner {
        require(_slippage <= 1000, "Slippage too high");
        slippageTolerance = _slippage;
        emit SlippageUpdated(_slippage);
    }

    // ------------------------------------------------------------------------
    // Update Gas Limit
    // ------------------------------------------------------------------------
    function setGasLimit(uint _gasLimit) external onlyOwner {
        require(_gasLimit >= 100_000, "Gas limit too low");
        gasLimit = _gasLimit;
        emit GasLimitUpdated(_gasLimit);
    }

    // ------------------------------------------------------------------------
    // Get Contract Balance
    // ------------------------------------------------------------------------
    function getBalance() external view returns (uint) {
        return address(this).balance;
    }

    // ------------------------------------------------------------------------
    // Check Token Allowance
    // ------------------------------------------------------------------------
    function checkAllowance(address token, address spender)
        external
        view
        returns (uint)
    {
        return IERC20(token).allowance(address(this), spender);
    }
}