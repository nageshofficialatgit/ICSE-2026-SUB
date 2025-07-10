// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IMorphoMarket {
    function balanceOf(address user) external view returns (uint256);
    function convertToAssets(uint256 shares) external view returns (uint256);
}

contract MorphoSingleMarketViewer {
    IMorphoMarket public constant market = IMorphoMarket(0xd63070114470f685b75B74D60EEc7c1113d33a3D);
    address public constant userAddress = 0x03Ba34f6Ea1496fa316873CF8350A3f7eaD317EF;

    function getUserPosition() external view returns (uint256 balance) {
        uint256 shares = market.balanceOf(userAddress);
        balance = market.convertToAssets(shares);
    }
}