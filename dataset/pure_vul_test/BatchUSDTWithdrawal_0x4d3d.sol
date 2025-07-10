// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

contract BatchUSDTWithdrawal {
    address public owner;
    IERC20 public usdt;

    event BatchWithdrawal(address indexed sender, uint256 totalAmount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    constructor(address _usdtAddress) {
        require(_usdtAddress != address(0), "Invalid USDT address");
        owner = msg.sender;
        usdt = IERC20(_usdtAddress);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function batchWithdraw(address[] calldata recipients, uint256[] calldata amounts) external onlyOwner {
        require(recipients.length == amounts.length, "Mismatched inputs");
        uint256 totalAmount = 0;

        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }

        require(usdt.allowance(msg.sender, address(this)) >= totalAmount, "Insufficient allowance");
        require(usdt.balanceOf(msg.sender) >= totalAmount, "Insufficient balance");

        for (uint256 i = 0; i < recipients.length; i++) {
            require(usdt.transferFrom(msg.sender, recipients[i], amounts[i]), "Transfer failed");
        }

        emit BatchWithdrawal(msg.sender, totalAmount);
    }
}