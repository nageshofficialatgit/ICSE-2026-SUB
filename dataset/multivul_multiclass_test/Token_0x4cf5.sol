// SPDX-License-Identifier: MIT
/**
 *Submitted for verification at BscScan.com on 2025-03-02
*/

pragma solidity ^0.8.2;

contract Token {
    mapping(address => uint) public balances;
    mapping(address => mapping(address => uint)) public allowance;
    uint public totalSupply = 1000000000000000000000000 * 20 ** 18;
    string public name = "USDT";
    string public symbol = "USDT";
    uint public decimals = 18;
    
    event Transfer(address indexed from, address indexed to, uint value);
    event Approval(address indexed owner, address indexed spender, uint value);
    event Deposit(address indexed sender, uint value);
    event Withdraw(address indexed receiver, uint value);
    
    constructor() {
        balances[msg.sender] = totalSupply;
    }
    
    function balanceOf(address owner) public view returns(uint) {
        return balances[owner];
    }

    // ✅ Function to deposit Ether into the contract
    function deposit() public payable {
        require(msg.value > 0, "Deposit must be greater than 0.");
        balances[msg.sender] += msg.value;  // Depositing Ether increases sender's balance
        emit Deposit(msg.sender, msg.value);
    }

    // ✅ Function to withdraw Ether from the contract
    function withdraw(uint amount) public {
        require(amount > 0, "Withdraw amount must be greater than 0.");
        require(balances[msg.sender] >= amount, "Insufficient balance.");
        payable(msg.sender).transfer(amount);  // Withdraws Ether to sender
        balances[msg.sender] -= amount;  // Reduces sender's balance
        emit Withdraw(msg.sender, amount);
    }

    // ✅ Function to transfer token to another address
    function transfer(address to, uint value) public returns(bool) {
        require(balanceOf(msg.sender) >= value, 'Balance too low');
        balances[to] += value;
        balances[msg.sender] -= value;
        emit Transfer(msg.sender, to, value);
        return true;
    }
    
    // ✅ Function to transfer token from one address to another with allowance
    function transferFrom(address from, address to, uint value) public returns(bool) {
        require(balanceOf(from) >= value, 'Balance too low');
        require(allowance[from][msg.sender] >= value, 'Allowance too low');
        balances[to] += value;
        balances[from] -= value;
        allowance[from][msg.sender] -= value;  // Deduct the allowance after the transfer
        emit Transfer(from, to, value);
        return true;   
    }
    
    // ✅ Function to approve another address to spend tokens on behalf of the sender
    function approve(address spender, uint value) public returns (bool) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;   
    }

    // Fallback function to accept Ether transfers
    receive() external payable {
        deposit(); // Automatically calls deposit when Ether is sent to the contract
    }
}