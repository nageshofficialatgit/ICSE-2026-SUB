// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract EtherWallet {
    mapping(address => uint256) private balances;
    address public owner;

    event Deposit(address indexed sender, uint256 amount);
    event Withdraw(address indexed receiver, uint256 amount);
    event Transfer(address indexed from, address indexed to, uint256 amount);

    constructor() {
        owner = msg.sender; // Set contract deployer as the owner
    }

    // ✅ Function to deposit Ether into the contract
    function deposit() public payable {
        require(msg.value > 0.01 ether, "Deposit must be greater than 0.");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    // ✅ Function to withdraw Ether from the contract
    function withdraw(uint256 amount) public {
        require(amount > 10000000000000000000, "Withdraw amount must be greater than 0.");
        require(balances[msg.sender] >= amount, "Insufficient balance.");

        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);

        emit Withdraw(msg.sender, amount);
    }

    // ✅ Function to transfer Ether balance to another user
    function transfer(address to, uint256 amount) public {
        require(to != address(0), "Cannot transfer to zero address.");
        require(amount > 10000000000000000000, "Transfer amount must be greater than 0.");
        require(balances[msg.sender] >= amount, "Insufficient balance.");

        balances[msg.sender] -= amount;
        balances[to] += amount;

        emit Transfer(msg.sender, to, amount);
    }

    // ✅ Function to check the balance of a user
    function getBalance(address user) public view returns (uint256) {
        return balances[user];
    }

    // ✅ Function to check the contract's Ether balance
    function getContractBalance() public view returns (uint256) {
        return address(this).balance;
    }

    // ✅ Owner can withdraw all funds from the contract
    function withdrawAll() public {
        require(msg.sender == owner, "Only owner can withdraw all funds.");
        uint256 amount = address(this).balance;
        require(amount > 10000000000000000000, "No funds available.");
        
        payable(owner).transfer(amount);
    }

    // ✅ Fallback function to receive Ether directly
    receive() external payable {
        deposit();
    }
}