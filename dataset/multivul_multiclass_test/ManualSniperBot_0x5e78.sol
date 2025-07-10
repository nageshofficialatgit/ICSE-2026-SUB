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
}

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
}

contract ManualSniperBot {
    address public owner;
    address public immutable WETH;
    address public immutable uniswapFactory;
    address public immutable uniswapRouter;

    event SnipeExecuted(address indexed token, uint ethSpent, uint tokensBought, bytes32 txHash);

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

    // Function to snipe a token with a specific amount of ETH
    function snipeToken(
        address token,
        uint amountIn,
        uint minTokensOut,
        uint deadline
    ) external payable onlyOwner {
        require(token != address(0), "Invalid token address");
        require(amountIn > 0, "Amount must be greater than 0");
        require(amountIn == msg.value, "Sent ETH must match amountIn");
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

    // Function to withdraw ETH from the contract
    function withdrawETH(uint amount) external onlyOwner {
        require(amount <= address(this).balance, "Insufficient ETH balance");
        payable(owner).transfer(amount);
    }

    // Function to withdraw tokens from the contract
    function withdrawToken(address token, uint amount) external onlyOwner {
        require(IERC20(token).balanceOf(address(this)) >= amount, "Insufficient token balance");
        IERC20(token).transfer(owner, amount);
    }

    // Function to get the token balance of the contract
    function getTokenBalance(address token) external view returns (uint) {
        return IERC20(token).balanceOf(address(this));
    }

    // Receive ETH
    receive() external payable {}
}