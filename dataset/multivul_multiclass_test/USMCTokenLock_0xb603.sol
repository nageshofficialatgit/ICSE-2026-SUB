// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Minimal ERC-20 Interface
interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract USMCTokenLock {
    address public owner;
    IERC20 public token;
    uint256 public lockUntil;
    uint256 public tokenDecimals = 18;


    // Constructor sets the token address and sets the lock period to 0
    constructor(address _tokenAddress) {
        owner = msg.sender;
        token = IERC20(_tokenAddress);
    }

    // Modifier to restrict function access to only the owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can execute this");
        _;
    }

   function Withdraw(uint256 amount) external {
        require(block.timestamp >= lockUntil, "Tokens are still locked");
        // Ensure the amount is in the correct decimal format    
        uint256 amountInBaseUnit = amount * (10 ** tokenDecimals);
        require(token.transfer(msg.sender, amountInBaseUnit), "Token transfer failed");
    }

    // Lock tokens for a specific time period
    function LockBlock(uint256 _addlockuntilblock) external {
        require(block.timestamp >= lockUntil, "Tokens are still locked");
        lockUntil = _addlockuntilblock;
    }

    // Function to check the token balance of the contract
    function Balance() external view returns (uint256) {
        return token.balanceOf(address(this));  // Returns the token balance of the contract
    }

    // Function to get the current block number
    function BlockNumber() external view returns (uint256) {
        return block.number;  // Returns the current block number
    }

    // Function to change the owner of the contract
    function ChangeOwner(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner address cannot be zero address");
        owner = newOwner;
    }

    // Reject any incoming Ether transfer
    receive() external payable {
        revert("Ether transfers are not accepted");
    }

    // Fallback function to reject any call with data
    fallback() external payable {
        revert("Ether transfers with data are not accepted");
    }
}