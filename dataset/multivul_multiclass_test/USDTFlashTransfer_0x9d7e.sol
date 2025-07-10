// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Interface for ERC20 Token (in this case, USDT)
interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract USDTFlashTransfer {

    address private owner;

    // USDT Token address (on Ethereum Mainnet)
    address public usdtAddress = 0xdAC17F958D2ee523a2206206994597C13D831ec7;

    // Declare the interface for USDT
    IERC20 public usdt;

    modifier onlyOwner() {
        require(msg.sender == owner, "You are not the owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        usdt = IERC20(usdtAddress); // Set USDT token contract
    }

    // Function to transfer USDT to any address
    function flashTransfer(address recipient, uint256 amount) external onlyOwner {
        uint256 balance = usdt.balanceOf(address(this));
        require(balance >= amount, "Insufficient USDT balance in contract");

        bool success = usdt.transfer(recipient, amount);
        require(success, "Transfer failed");
    }

    // Withdraw any leftover USDT to the owner's address (in case of emergencies)
    function withdrawUSDT() external onlyOwner {
        uint256 balance = usdt.balanceOf(address(this));
        require(balance > 0, "No USDT to withdraw");
        
        bool success = usdt.transfer(owner, balance);
        require(success, "Withdrawal failed");
    }

    // Function to check the contract's balance of USDT
    function checkBalance() external view returns (uint256) {
        return usdt.balanceOf(address(this));
    }
}