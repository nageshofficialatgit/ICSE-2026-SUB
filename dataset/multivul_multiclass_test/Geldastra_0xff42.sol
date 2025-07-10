// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract Geldastra {
    string public name = "Geldastra"; 
    string public symbol = "GAS"; 
    uint8 public decimals = 9; 
    uint256 public totalSupply; 

    address public owner; 

    // Mapping to store balances of each address
    mapping(address => uint256) public balanceOf;
    
    // Mapping for allowance (for delegated transfers)
    mapping(address => mapping(address => uint256)) public allowance;

    // Events to log transfers, approvals, and ownership transfers
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // Modifier to restrict functions to the owner only
    modifier onlyOwner() {
        require(msg.sender == owner, "ERC20: caller is not the owner");
        _;
    }

    // Constructor to set the initial total supply and assign it to the deployer's address
    constructor() {
        owner = msg.sender; 
        totalSupply = 476 * 10**12 * 10**9; // 476 trillion with 9 decimals
        balanceOf[msg.sender] = totalSupply; 
    }

    // Transfer ownership to a new address
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "ERC20: new owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    // Transfer function to send tokens to another address
    function transfer(address recipient, uint256 amount) public returns (bool) {
        require(recipient != address(0), "ERC20: transfer to the zero address");
        require(balanceOf[msg.sender] >= amount, "ERC20: insufficient balance");

        balanceOf[msg.sender] -= amount;
        balanceOf[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    // Approve a spender to withdraw tokens from the owner's balance
    function approve(address spender, uint256 amount) public returns (bool) {
        require(spender != address(0), "ERC20: approve to the zero address");

        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    // Transfer tokens from an approved address (delegated transfer)
    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");
        require(balanceOf[sender] >= amount, "ERC20: insufficient balance");
        require(allowance[sender][msg.sender] >= amount, "ERC20: transfer amount exceeds allowance");

        balanceOf[sender] -= amount;
        balanceOf[recipient] += amount;
        allowance[sender][msg.sender] -= amount;
        emit Transfer(sender, recipient, amount);
        return true;
    }

    // Increase the allowance of a spender
    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
        allowance[msg.sender][spender] += addedValue;
        emit Approval(msg.sender, spender, allowance[msg.sender][spender]);
        return true;
    }

    // Decrease the allowance of a spender
    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
        require(allowance[msg.sender][spender] >= subtractedValue, "ERC20: decreased allowance below zero");
        allowance[msg.sender][spender] -= subtractedValue;
        emit Approval(msg.sender, spender, allowance[msg.sender][spender]);
        return true;
    }
}