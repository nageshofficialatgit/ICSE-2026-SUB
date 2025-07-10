// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Full Uniswap V3 Interfaces (Flattened)
interface ISwapRouter {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }
    function exactInputSingle(ExactInputSingleParams calldata params) external payable returns (uint256 amountOut);
}

interface IQuoter {
    function quoteExactInputSingle(
        address tokenIn,
        address tokenOut,
        uint24 fee,
        uint256 amountIn,
        uint160 sqrtPriceLimitX96
    ) external returns (uint256 amountOut);
}

// Full Uniswap V2 Interface (Flattened)
interface IUniswapV2Router02 {
    function getAmountsOut(uint amountIn, address[] memory path) external view returns (uint[] memory amounts);
    function swapExactETHForTokens(
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable returns (uint[] memory amounts);
}

contract MEVUniswapBot {
    address public owner;
    address private immutable uniswapV2Router;
    address private immutable uniswapV3Router;
    address private immutable uniswapV3Quoter;
    bool public isRunning = false;

    event BotStarted();
    event BotPaused();
    event SwapExecuted(address indexed dex, address tokenIn, address tokenOut, uint amountIn, uint amountOut);
    event ProfitCompounded(uint256 newCapital);
    event Withdrawn(address recipient, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    constructor(
        address _v2Router,
        address _v3Router,
        address _v3Quoter
    ) {
        owner = msg.sender;
        uniswapV2Router = _v2Router;
        uniswapV3Router = _v3Router;
        uniswapV3Quoter = _v3Quoter;
    }

    function startBot() external onlyOwner {
        require(!isRunning, "Bot is already running");
        isRunning = true;
        emit BotStarted();
    }

    function pauseBot() external onlyOwner {
        require(isRunning, "Bot is already paused");
        isRunning = false;
        emit BotPaused();
    }

    function isBotLive() external view returns (bool) {
        return isRunning;
    }

    function executeTrade(address tokenIn, address tokenOut, uint amountIn) internal {
        require(isRunning, "Bot is paused");
        (string memory bestDEX, uint amountOut) = getBestTrade(tokenIn, tokenOut, amountIn);

        if (keccak256(abi.encodePacked(bestDEX)) == keccak256(abi.encodePacked("UniswapV2"))) {
            executeUniswapV2Swap(tokenIn, tokenOut, amountIn, amountOut);
        } else {
            executeUniswapV3Swap(tokenIn, tokenOut, amountIn, amountOut);
        }
    }

    function getBestTrade(address tokenIn, address tokenOut, uint amountIn) public returns (string memory bestDEX, uint amountOut) {
        uint amountOutV2 = getUniswapV2Price(tokenIn, tokenOut, amountIn);
        uint amountOutV3 = getUniswapV3Price(tokenIn, tokenOut, amountIn);

        if (amountOutV3 >= amountOutV2) {
            return ("UniswapV3", amountOutV3);
        } else {
            return ("UniswapV2", amountOutV2);
        }
    }

    function getUniswapV2Price(address tokenIn, address tokenOut, uint amountIn) internal view returns (uint) {
        IUniswapV2Router02 router = IUniswapV2Router02(uniswapV2Router);
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        uint[] memory amounts = router.getAmountsOut(amountIn, path);
        return amounts[1];
    }

    function getUniswapV3Price(address tokenIn, address tokenOut, uint amountIn) internal returns (uint) {
        IQuoter quoter = IQuoter(uniswapV3Quoter);
        return quoter.quoteExactInputSingle(tokenIn, tokenOut, 3000, amountIn, 0);
    }

    function executeUniswapV2Swap(address tokenIn, address tokenOut, uint amountIn, uint minAmountOut) internal {
        IUniswapV2Router02 router = IUniswapV2Router02(uniswapV2Router);
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        router.swapExactETHForTokens{ value: amountIn }(
            minAmountOut, path, address(this), block.timestamp + 15
        );
        emit SwapExecuted(uniswapV2Router, tokenIn, tokenOut, amountIn, minAmountOut);
    }

    function executeUniswapV3Swap(address tokenIn, address tokenOut, uint amountIn, uint minAmountOut) internal {
        ISwapRouter.ExactInputSingleParams memory params = ISwapRouter.ExactInputSingleParams({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            fee: 3000,
            recipient: address(this),
            deadline: block.timestamp + 15,
            amountIn: amountIn,
            amountOutMinimum: minAmountOut,
            sqrtPriceLimitX96: 0
        });

        ISwapRouter router = ISwapRouter(uniswapV3Router);
        router.exactInputSingle{ value: amountIn }(params);
        emit SwapExecuted(uniswapV3Router, tokenIn, tokenOut, amountIn, minAmountOut);
    }

    function withdrawETH() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH balance to withdraw");
        payable(owner).transfer(balance);
        emit Withdrawn(owner, balance);
    }

    receive() external payable {
        if (isRunning) {
            executeTrade(address(0), 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2, msg.value);
        }
    }
}