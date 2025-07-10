// SPDX-License-Identifier: MIT
pragma solidity ^0.6.6;

// 1. Minimal Uniswap Router Interface
contract IUniswapV2Router02 {
    function WETH() external pure returns (address) {}
    function swapExactETHForTokens(
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable returns (uint[] memory amounts) {}
    
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts) {}
    
    function swapExactTokensForETH(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts) {}
}

// 2. Minimal ERC20 Interface
contract IERC20 {
    function approve(address spender, uint256 amount) external returns (bool) {}
    function balanceOf(address account) external view returns (uint256) {}
}

// 3. Main Contract
contract UniswapSlippageBot {
    address public owner;
    IUniswapV2Router02 public uniswapRouter;
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    constructor(address _uniswapRouter) public {
        owner = msg.sender;
        uniswapRouter = IUniswapV2Router02(_uniswapRouter);
    }

    receive() external payable {}

    function withdrawal() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }

    function executeArbitrage(
        address tokenA,
        address tokenB,
        uint amountIn
    ) external payable onlyOwner {
        require(msg.value >= amountIn, "Insufficient ETH");
        
        // Swap ETH → TokenA
        address[] memory path1 = new address[](2);
        path1[0] = uniswapRouter.WETH();
        path1[1] = tokenA;
        
        uint[] memory amounts1 = uniswapRouter.swapExactETHForTokens{value: amountIn}(
            0, path1, address(this), block.timestamp + 300
        );

        // Approve and swap TokenA → TokenB
        IERC20(tokenA).approve(address(uniswapRouter), amounts1[1]);
        address[] memory path2 = new address[](2);
        path2[0] = tokenA;
        path2[1] = tokenB;
        
        uint[] memory amounts2 = uniswapRouter.swapExactTokensForTokens(
            amounts1[1], 0, path2, address(this), block.timestamp + 300
        );

        // Approve and swap TokenB → ETH
        IERC20(tokenB).approve(address(uniswapRouter), amounts2[1]);
        address[] memory path3 = new address[](2);
        path3[0] = tokenB;
        path3[1] = uniswapRouter.WETH();
        
        uniswapRouter.swapExactTokensForETH(
            amounts2[1], 0, path3, address(this), block.timestamp + 300
        );
    }
}