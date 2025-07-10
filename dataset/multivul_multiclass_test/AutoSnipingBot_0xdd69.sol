// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

interface IUniswapV2Factory {
    event PairCreated(address indexed token0, address indexed token1, address pair, uint);
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
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract AutoSnipingBot {
    address public owner;
    IUniswapV2Router public uniswapRouter;
    IUniswapV2Factory public uniswapFactory;
    address public WETH;

    event SnipeExecuted(address indexed token, uint ethSpent, uint tokensBought, bytes32 txHash);
    event TokensSold(address indexed token, uint tokensSold, uint ethReceived);

    constructor() {
        owner = msg.sender;
        uniswapRouter = IUniswapV2Router(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        uniswapFactory = IUniswapV2Factory(0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f);
        WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    function autoSnipe(address token0, address token1) external onlyOwner {
        address pair = uniswapFactory.getPair(token0, token1);
        require(pair != address(0), "Pair does not exist");

        address tokenToSnipe;
        if (token0 == WETH) {
            tokenToSnipe = token1;
        } else if (token1 == WETH) {
            tokenToSnipe = token0;
        } else {
            return;
        }

        uint amountETH = 0.05 ether;
        uint minTokensOut = 0;
        uint deadline = block.timestamp + 60;

        address[] memory path = new address[](2);
        path[0] = WETH;
        path[1] = tokenToSnipe;

        uint[] memory amounts = uniswapRouter.swapExactETHForTokens{value: amountETH}(
            minTokensOut,
            path,
            address(this),
            deadline
        );
        emit SnipeExecuted(tokenToSnipe, amountETH, amounts[1], keccak256(abi.encodePacked(blockhash(block.number - 1))));
    }

    function sellToken(address token, uint amount, uint minEthOut, uint deadline) external onlyOwner {
        require(deadline >= block.timestamp, "Deadline passed");
        uint balance = IERC20(token).balanceOf(address(this));
        require(balance >= amount, "Insufficient token balance");

        IERC20(token).approve(address(uniswapRouter), amount);

        address[] memory path = new address[](2);
        path[0] = token;
        path[1] = WETH;

        uint[] memory amounts = uniswapRouter.swapExactTokensForETH(
            amount,
            minEthOut,
            path,
            address(this),
            deadline
        );
        emit TokensSold(token, amount, amounts[1]);
    }

    function withdrawETH(uint amount) external onlyOwner {
        payable(owner).transfer(amount);
    }

    function withdrawToken(address token, uint amount) external onlyOwner {
        IERC20(token).transfer(owner, amount);
    }

    function getTokenBalance(address token) external view returns (uint) {
        return IERC20(token).balanceOf(address(this));
    }

    receive() external payable {}
}