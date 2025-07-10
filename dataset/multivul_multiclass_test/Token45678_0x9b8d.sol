// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

contract Token45678 {
// Basic ERC20 token details
string public name = "Token45678";
string public symbol = "$sai";
uint8 public decimals = 18;
uint256 public totalSupply; 

// Mapping to store token balances for each address
mapping(address => uint256) public balanceOf; 

// Modifier to check if the caller is the contract owner
modifier onlyOwner() {
require(msg.sender == owner, "Only owner can call this function");
_;
}

// Contract owner address
address payable public owner;

// Constructor to set the owner
constructor() {
owner = payable(msg.sender);
}

// Mint function to create new tokens, only callable by the owner
function mint(address to, uint256 amount) public onlyOwner {
require(amount > 0, "Cannot mint zero tokens");
totalSupply += amount;
balanceOf[to] += amount;
emit Transfer(address(0), to, amount); // Emit transfer event
}

function burn(address to, uint256 amount) public onlyOwner {
require(amount > 0, "Cannot mint zero tokens");
require(amount > totalSupply , "can't burn then totalsupply");
totalSupply -= amount;
balanceOf[to] -= amount;
emit Transfer(address(0), to, amount); // Emit transfer event
}

// Standard ERC20 transfer function 
function transfer(address recipient, uint256 amount) public {
require(balanceOf[msg.sender] >= amount, "Insufficient balance");
balanceOf[msg.sender] -= amount;
balanceOf[recipient] += amount;
emit Transfer(msg.sender, recipient, amount);
}

// Event to log token transfers
event Transfer(address indexed from, address indexed to, uint256 value);
}