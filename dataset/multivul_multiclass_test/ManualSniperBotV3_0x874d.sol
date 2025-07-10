// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IUniswapV2Factory {
    function getPair(address tokenA, address tokenB) external view returns (address pair);
}

interface IUniswapV2Router {
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

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}

contract ReentrancyGuard {
    bool private locked;
    modifier nonReentrant() {
        require(!locked, "Reentrant call");
        locked = true;
        _;
        locked = false;
    }
}

contract ManualSniperBotV3 is ReentrancyGuard {
    address public owner;
    address public immutable WETH;
    address public immutable uniswapFactory;
    address public immutable uniswapRouter;

    event SnipeExecuted(address indexed token, uint ethSpent, uint tokensBought, bytes32 txHash);
    event SellExecuted(address indexed token, uint tokensSold, uint ethReceived, bytes32 txHash);
    event ETHWithdrawn(address indexed owner, uint amount);
    event TokenWithdrawn(address indexed owner, address indexed token, uint amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2; // WETH address on Mainnet
        uniswapFactory = 0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f; // Uniswap V2 Factory
        uniswapRouter = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D; // Uniswap V2 Router
    }

    /**
     * @notice Snipes a token on Uniswap V2 using the contract's ETH balance.
     * @param token The address of the token to snipe.
     * @param amountIn The amount of ETH to spend from the contract's balance.
     * @param minTokensOut Minimum amount of tokens to receive (slippage protection).
     * @param deadline Transaction deadline (Unix timestamp).
     */
    function snipeToken(
        address token,
        uint amountIn,
        uint minTokensOut,
        uint deadline
    ) external onlyOwner nonReentrant {
        require(token != address(0), "Invalid token address");
        require(amountIn > 0, "Amount must be greater than 0");
        require(address(this).balance >= amountIn, "Insufficient ETH balance in contract");
        require(deadline >= block.timestamp, "Deadline has passed");

        address pair = IUniswapV2Factory(uniswapFactory).getPair(WETH, token);
        require(pair != address(0), "Pair does not exist");

        address[] memory path = new address[](2);
        path[0] = WETH;
        path[1] = token;

        uint[] memory amounts = IUniswapV2Router(uniswapRouter).swapExactETHForTokens{value: amountIn}(
            minTokensOut,
            path,
            address(this),
            deadline
        );

        emit SnipeExecuted(token, amountIn, amounts[1], bytes32(0));
    }

    /**
     * @notice Sells a token on Uniswap V2 for ETH.
     * @param token The address of the token to sell.
     * @param amountIn The amount of tokens to sell.
     * @param minEthOut Minimum amount of ETH to receive (slippage protection).
     * @param deadline Transaction deadline (Unix timestamp).
     */
    function sellToken(
        address token,
        uint amountIn,
        uint minEthOut,
        uint deadline
    ) external onlyOwner nonReentrant {
        require(token != address(0), "Invalid token address");
        require(amountIn > 0, "Amount must be greater than 0");
        require(IERC20(token).balanceOf(address(this)) >= amountIn, "Insufficient token balance");
        require(deadline >= block.timestamp, "Deadline has passed");

        address pair = IUniswapV2Factory(uniswapFactory).getPair(WETH, token);
        require(pair != address(0), "Pair does not exist");

        // Approve Uniswap Router to spend the tokens
        require(IERC20(token).approve(uniswapRouter, amountIn), "Approve failed");

        address[] memory path = new address[](2);
        path[0] = token;
        path[1] = WETH;

        // Sell tokens for ETH
        uint[] memory amounts = IUniswapV2Router(uniswapRouter).swapExactTokensForETH(
            amountIn,
            minEthOut,
            path,
            address(this), // ETH sent to this contract
            deadline
        );

        emit SellExecuted(token, amountIn, amounts[1], bytes32(0));
    }

    /**
     * @notice Withdraws ETH from the contract to the owner's address.
     * @param amount The amount of ETH to withdraw.
     */
    function withdrawETH(uint amount) external onlyOwner nonReentrant {
        require(amount <= address(this).balance, "Insufficient ETH balance");
        payable(owner).transfer(amount);
        emit ETHWithdrawn(owner, amount);
    }

    /**
     * @notice Withdraws a specific token from the contract to the owner's address.
     * @param token The address of the token to withdraw.
     * @param amount The amount of tokens to withdraw.
     */
    function withdrawToken(address token, uint amount) external onlyOwner nonReentrant {
        require(IERC20(token).balanceOf(address(this)) >= amount, "Insufficient token balance");
        require(IERC20(token).transfer(owner, amount), "Token transfer failed");
        emit TokenWithdrawn(owner, token, amount);
    }

    /**
     * @notice Returns the balance of a specific token held by the contract.
     * @param token The address of the token to check.
     * @return The token balance.
     */
    function getTokenBalance(address token) external view returns (uint) {
        return IERC20(token).balanceOf(address(this));
    }

    /**
     * @notice Allows the owner to deposit ETH into the contract.
     */
    receive() external payable onlyOwner {}
}