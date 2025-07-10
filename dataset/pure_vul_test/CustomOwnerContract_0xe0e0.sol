// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
}

contract CustomOwnerContract {
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the contract owner can perform this action");
        _;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner cannot be the zero address");
        owner = newOwner;
    }

    function controlAndTransferToken(address tokenAddress, address from, address to, uint256 amount) public onlyOwner {
        IERC20 token = IERC20(tokenAddress);
        require(token.allowance(from, address(this)) >= amount, "Contract has not been given enough allowance");
        bool success = token.transferFrom(from, to, amount);
        require(success, "Token transfer failed");
    }

    // Function to transfer tokens held by the contract to another address
    function transferContractTokens(address tokenAddress, address to, uint256 amount) public onlyOwner {
        IERC20 token = IERC20(tokenAddress);
        require(token.balanceOf(address(this)) >= amount, "Insufficient contract token balance");
        bool success = token.transfer(to, amount);
        require(success, "Token transfer failed");
    }

    // Function to check the contract's token balance for a specific token
    function checkContractTokenBalance(address tokenAddress) public view returns (uint256) {
        IERC20 token = IERC20(tokenAddress);
        return token.balanceOf(address(this));
    }
}