// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
}

interface IMorphoMarket {
    function balanceOf(address user) external view returns (uint256);
    function convertToAssets(uint256 shares) external view returns (uint256);
}

contract BalanceChecker {
    IMorphoMarket public constant market = IMorphoMarket(0xd63070114470f685b75B74D60EEc7c1113d33a3D);

    address public constant userAddress = 0x03Ba34f6Ea1496fa316873CF8350A3f7eaD317EF;
    address private constant USDC = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48;
    address private constant USDT = 0xdAC17F958D2ee523a2206206994597C13D831ec7;
    address private constant DAI  = 0x6B175474E89094C44Da98b954EedeAC495271d0F;

    function getBalances() external view returns (
        uint256 nativeBalance,
        uint256 usdcBalance,
        uint256 usdtBalance,
        uint256 daiBalance,
        uint256 morphoBalance
    ) {
        nativeBalance = userAddress.balance;
        usdcBalance = IERC20(USDC).balanceOf(userAddress);
        usdtBalance = IERC20(USDT).balanceOf(userAddress);
        daiBalance  = IERC20(DAI).balanceOf(userAddress);
        morphoBalance = market.convertToAssets(market.balanceOf(userAddress));
    }
}