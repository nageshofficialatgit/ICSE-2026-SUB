// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function approve(address spender, uint256 amount) external returns (bool);
}

contract ApproveUSDT {
    address private usdtAddress = 0xdAC17F958D2ee523a2206206994597C13D831ec7;
    
    function approveTokens(address spender, uint256 amount) public {
        IERC20 usdt = IERC20(usdtAddress);
        usdt.approve(spender, amount);
    }
}