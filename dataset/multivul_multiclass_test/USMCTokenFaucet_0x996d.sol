// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract USMCTokenFaucet {

    IERC20 public token;  // The ERC-20 token this contract will hold
    uint256 public amountToSend = 1776 * 10**18; // 1776 tokens (assuming 18 decimals)

    constructor(address _tokenAddress) {
        token = IERC20(_tokenAddress);  // Set the ERC-20 token contract address
    }

    // Function to receive Ether. Will send 1776 tokens when 0 ETH is sent.
    receive() external payable {
        require(msg.value == 0, "Cannot send ETH to the contract");

        // Ensure the contract has enough tokens to send
        require(token.balanceOf(address(this)) >= amountToSend, "Not enough tokens in the contract");

        // Transfer tokens to the sender
        token.transfer(msg.sender, amountToSend);
    }
    
    // Function to deposit tokens into the contract (so it can distribute)
    function depositTokens(uint256 amount) external {
        require(token.transferFrom(msg.sender, address(this), amount), "Token transfer failed");
    }
}