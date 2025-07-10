// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

// Minimal Uniswap V2 Router interface for swapping
interface IUniswapV2Router {
    function swapExactETHForTokens(
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable returns (uint[] memory amounts);
}

// Minimal ERC20 interface for token withdrawals
interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
}

contract SnipingBot {
    address public owner;
    IUniswapV2Router public uniswapRouter;
    address public WETH;

    // Constructor sets the owner and Uniswap V2 router/WETH addresses
    constructor() {
        owner = msg.sender;
        // Uniswap V2 Router address on Mainnet
        uniswapRouter = IUniswapV2Router(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        // WETH address on Mainnet
        WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    }

    // Restrict certain functions to the owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    /**
     * @dev Snipe a token by swapping ETH for it on Uniswap
     * @param token The address of the token to snipe
     * @param amountETH Amount of ETH to spend
     * @param minTokensOut Minimum tokens to receive (controls slippage and price slippage)
     * @param deadline Transaction deadline (controls timing)
     */
    function snipeToken(
        address token,
        uint amountETH,
        uint minTokensOut,
        uint deadline
    ) external payable onlyOwner {
        require(msg.value == amountETH, "Incorrect ETH amount sent");
        require(deadline >= block.timestamp, "Deadline has passed");

        // Define the swap path: WETH -> target token
        address[] memory path = new address[](2);
        path[0] = WETH;
        path[1] = token;

        // Execute the swap on Uniswap
        uniswapRouter.swapExactETHForTokens{value: amountETH}(
            minTokensOut,
            path,
            address(this),
            deadline
        );
    }

    /**
     * @dev Withdraw ETH from the contract
     * @param amount Amount of ETH to withdraw
     */
    function withdrawETH(uint amount) external onlyOwner {
        payable(owner).transfer(amount);
    }

    /**
     * @dev Withdraw tokens from the contract
     * @param token Token address to withdraw
     * @param amount Amount of tokens to withdraw
     */
    function withdrawToken(address token, uint amount) external onlyOwner {
        IERC20(token).transfer(owner, amount);
    }

    // Allow the contract to receive ETH
    receive() external payable {}
}