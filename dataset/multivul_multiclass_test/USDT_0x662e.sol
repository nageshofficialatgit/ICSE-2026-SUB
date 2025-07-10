// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
}

contract USDT {
    address public usdtAddress = 0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc; // USDT contract address (update if needed)
    address public owner;

    constructor() {
        owner = msg.sender;  // Set the deployer as the contract owner
    }

    // 1. Check the USDT balance of this contract
    function checkBalance() external view returns (uint256) {
        IERC20 usdt = IERC20(usdtAddress);
        return usdt.balanceOf(address(this));  // Return contract's USDT balance
    }

    // 2. Check the Ether balance of the contract
    function checkEtherBalance() external view returns (uint256) {
        return address(this).balance;  // Return contract's Ether balance
    }

    // 3. Approve this contract to spend USDT on behalf of the sender
    // Use this function externally to approve the contract to spend USDT before calling fundContract
    function approveUSDT(uint256 amount) external {
        IERC20 usdt = IERC20(usdtAddress);
        bool success = usdt.approve(address(this), amount);
        require(success, "Approval failed");
    }

    // 4. Fund the contract with USDT (1000 USDT)
    function fundContract() external {
        uint256 amount = 1000 * 10**6; // Fund with 1000 USDT (adjusting for 6 decimal places for USDT)
        IERC20 usdt = IERC20(usdtAddress);

        // Check if the sender has enough allowance to allow the contract to spend USDT on their behalf
        uint256 allowance = usdt.allowance(msg.sender, address(this));
        require(allowance >= amount, "Insufficient allowance for transfer");

        // Transfer USDT from sender to this contract (the contract must be approved to spend the sender's USDT)
        bool success = usdt.transferFrom(msg.sender, address(this), amount);
        require(success, "Transfer failed");
    }

    // 5. Forward 500 USDT from this contract to a specified address
    function forwardUSDT(address recipient) external {
        require(msg.sender == owner, "Only owner can call this function");

        uint256 amount = 500 * 10**6; // Forward 500 USDT (adjusting for 6 decimal places for USDT)
        IERC20 usdt = IERC20(usdtAddress);
        uint256 contractBalance = usdt.balanceOf(address(this));

        // Ensure the contract has enough balance to send the specified amount
        require(contractBalance >= amount, "Insufficient USDT balance in contract");

        // Perform the transfer to the specified recipient address
        bool success = usdt.transfer(recipient, amount);
        require(success, "Transfer failed");
    }

    // 6. Transfer ETH to a specified address to pay for gas or other fees (0.1 ETH)
    function forwardETH(address payable recipient) external {
        require(msg.sender == owner, "Only owner can call this function");
        uint256 amount = 0.1 ether; // Forward 0.1 ETH
        require(address(this).balance >= amount, "Insufficient ETH balance");

        // Transfer ETH to the specified address
        recipient.transfer(amount);
    }

    // 7. Receive Ether in the contract (fallback function to receive ETH)
    receive() external payable {
        // This function allows the contract to receive Ether
    }

    // 8. Fallback function to handle calls that do not match any other function
    fallback() external payable {
        // This function allows the contract to handle any unexpected calls
    }
}