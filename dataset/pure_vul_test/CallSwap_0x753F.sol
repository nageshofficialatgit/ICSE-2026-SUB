// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

interface IUniswapRouter {
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

contract CallSwap {
    address private constant UNISWAP = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
    address public constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address public feeCollector = 0x534bc9CDC19cADBB1A4bc636b203e3DdCE3c305A;
    
    event CallSwapExecuted(
        string indexed functionName,
        address indexed user,
        address indexed token,
        uint256 amount
    );

    receive() external payable {}

    function CallSwapETHForTokens(address tokenOut) external payable {
        require(msg.value > 0, "Must send ETH");
        
        uint256 fee = (msg.value * 60) / 10000; // 0.6%
        uint256 swapAmount = msg.value - fee;
        
        payable(feeCollector).transfer(fee);
        
        address[] memory path = new address[](2);
        path[0] = WETH;
        path[1] = tokenOut;

        (bool success,) = UNISWAP.call{value: swapAmount}(
            abi.encodeWithSelector(
                IUniswapRouter.swapExactETHForTokens.selector,
                0,
                path,
                msg.sender,
                block.timestamp + 300
            )
        );
        require(success, "Swap failed");
        
        emit CallSwapExecuted(
            "CallSwapETHForTokens",
            msg.sender,
            tokenOut,
            msg.value
        );
    }

    function CallSwapTokensForETH(address tokenIn, uint256 amountIn) external {
        require(amountIn > 0, "Must send tokens");
        
        uint256 fee = (amountIn * 60) / 10000; // 0.6%
        uint256 swapAmount = amountIn - fee;
        
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn);
        IERC20(tokenIn).transfer(feeCollector, fee);
        IERC20(tokenIn).approve(UNISWAP, swapAmount);
        
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = WETH;

        (bool success,) = UNISWAP.call(
            abi.encodeWithSelector(
                IUniswapRouter.swapExactTokensForETH.selector,
                swapAmount,
                0,
                path,
                msg.sender,
                block.timestamp + 300
            )
        );
        require(success, "Swap failed");
        
        emit CallSwapExecuted(
            "CallSwapTokensForETH",
            msg.sender,
            tokenIn,
            amountIn
        );
    }
}