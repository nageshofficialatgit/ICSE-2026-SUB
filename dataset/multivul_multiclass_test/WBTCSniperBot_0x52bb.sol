// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function approve(address spender, uint256 amount) external returns (bool);
}

interface IUniswapV2Router {
    function WETH() external pure returns (address);
    function swapExactETHForTokensSupportingFeeOnTransferTokens(
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external payable;
}

contract WBTCSniperBot {
    address private owner;
    IUniswapV2Router private uniswapRouter;

    constructor(address _uniswapRouter) {
        owner = msg.sender;
        uniswapRouter = IUniswapV2Router(_uniswapRouter);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    function buyWBTC(uint256 minOut) external payable onlyOwner {
        address WBTC = 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599; // WBTC Contract Address

        address[] memory path;
        path[0] = uniswapRouter.WETH(); 
        path[1] = WBTC;

        uniswapRouter.swapExactETHForTokensSupportingFeeOnTransferTokens{value: msg.value}(
            minOut,
            path,
            msg.sender,
            block.timestamp + 300
        );
    }

    function withdraw() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }

    receive() external payable {}
}