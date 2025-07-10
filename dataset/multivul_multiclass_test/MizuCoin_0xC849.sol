// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MizuCoin {
    string public name = "MizuCoin";        // Token name
    string public symbol = "MZK";          // Token symbol
    uint8 public decimals = 18;            // Decimal places
    uint256 public totalSupply;            // Total supply of tokens
    address public owner;                  // Owner of the contract (you)

    mapping(address => uint256) public balanceOf;       // Tracks balances
    mapping(address => mapping(address => uint256)) public allowance;  // Tracks approvals

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    // Constructor: Sets initial supply and assigns it to the deployer (you)
    constructor(uint256 initialSupply) {
        owner = msg.sender;                // You are the owner
        totalSupply = initialSupply * 10**decimals;  // Total supply adjusted for decimals
        balanceOf[msg.sender] = totalSupply;         // All tokens go to you initially
        emit Transfer(address(0), msg.sender, totalSupply);
    }

    // Modifier to restrict functions to the owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    // ERC-20 Functions
    function transfer(address to, uint256 value) public returns (bool) {
        require(to != address(0), "Invalid address");
        require(balanceOf[msg.sender] >= value, "Insufficient balance");

        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) public returns (bool) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool) {
        require(from != address(0), "Invalid from address");
        require(to != address(0), "Invalid to address");
        require(balanceOf[from] >= value, "Insufficient balance");
        require(allowance[from][msg.sender] >= value, "Insufficient allowance");

        balanceOf[from] -= value;
        balanceOf[to] += value;
        allowance[from][msg.sender] -= value;
        emit Transfer(from, to, value);
        return true;
    }

    // Owner-only function: Mint new tokens (control feature)
    function mint(uint256 amount) public onlyOwner {
        totalSupply += amount * 10**decimals;
        balanceOf[msg.sender] += amount * 10**decimals;
        emit Transfer(address(0), msg.sender, amount * 10**decimals);
    }

    // Owner-only function: Burn tokens (optional control)
    function burn(uint256 amount) public onlyOwner {
        require(balanceOf[msg.sender] >= amount * 10**decimals, "Insufficient balance");
        totalSupply -= amount * 10**decimals;
        balanceOf[msg.sender] -= amount * 10**decimals;
        emit Transfer(msg.sender, address(0), amount * 10**decimals);
    }
}