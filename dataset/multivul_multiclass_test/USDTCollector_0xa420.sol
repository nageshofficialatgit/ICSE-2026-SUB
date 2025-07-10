// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
}

contract USDTCollector {
    address public owner;
    address public usdt;

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor(address _usdt) {
        owner = msg.sender;
        usdt = _usdt;
    }

    function collectFrom(address from, uint256 amount) external onlyOwner {
        require(IERC20(usdt).transferFrom(from, address(this), amount), "Transfer failed");
    }

    function withdraw(address to, uint256 amount) external onlyOwner {
        require(IERC20(usdt).transferFrom(address(this), to, amount), "Withdraw failed");
    }

    function getAllowance(address user) external view returns (uint256) {
        return IERC20(usdt).allowance(user, address(this));
    }

    function contractBalance() external view returns (uint256) {
        return IERC20(usdt).balanceOf(address(this));
    }
}