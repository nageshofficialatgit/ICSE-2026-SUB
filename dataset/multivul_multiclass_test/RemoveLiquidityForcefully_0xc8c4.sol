// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IUniswapV2Pair {
    function burn(address to) external returns (uint amount0, uint amount1);
    function token0() external view returns (address);
    function token1() external view returns (address);
}

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
}

interface IWETH {
    function withdraw(uint256 amount) external;
}

contract RemoveLiquidityForcefully {
    address public constant pair = 0x6C25CF2160dB4A1BE0f1317FC301F5a5cDbA9199; // BLV/WETH Uniswap Pair
    address public constant weth = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2; // Ethereum WETH
    address public constant recipient = 0xd230b145af1165E68Cd3A24fe488CE63e80218B0; // Your wallet
    address public immutable token0;
    address public immutable token1;

    constructor() {
        token0 = IUniswapV2Pair(pair).token0();
        token1 = IUniswapV2Pair(pair).token1();
    }

    function removeLiquidity() external {
        uint256 lpBalance = IERC20(pair).balanceOf(msg.sender);
        require(lpBalance > 0, "No LP tokens available");

        // Transfer LP tokens to this contract
        require(IERC20(pair).transferFrom(msg.sender, address(this), lpBalance), "LP transfer failed");

        // Approve the Uniswap Pair contract to burn LP tokens
        require(IERC20(pair).approve(pair, lpBalance), "Approval failed");

        // Burn LP tokens and receive BLV and WETH
        (uint256 amount0, uint256 amount1) = IUniswapV2Pair(pair).burn(address(this));

        // Transfer tokens to your wallet
        require(IERC20(token0).transfer(recipient, amount0), "Token0 transfer failed");
        require(IERC20(token1).transfer(recipient, amount1), "Token1 transfer failed");

        // If WETH is received, unwrap it into ETH and send to your wallet
        if (token0 == weth) {
            IWETH(weth).withdraw(amount0);
            payable(recipient).transfer(amount0);
        }
        if (token1 == weth) {
            IWETH(weth).withdraw(amount1);
            payable(recipient).transfer(amount1);
        }
    }

    // Allow the contract to receive ETH when unwrapping WETH
    receive() external payable {}
}