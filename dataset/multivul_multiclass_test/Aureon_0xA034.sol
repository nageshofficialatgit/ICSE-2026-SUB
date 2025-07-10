// SPDX-License-Identifier: MIT
pragma solidity ^0.8.21;
contract Aureon { 
    string public name = "Aureon"; 
    string public symbol = "AUR"; 
    uint8 public decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    address public owner; // The deployer is the owner
    uint256 public sellFee = 5; // 5% sell tax

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;
    }

    constructor(uint256 initialSupply) {
        owner = msg.sender; // Deployer is the owner
        totalSupply = initialSupply * (10 ** uint256(decimals));
        balanceOf[owner] = totalSupply; // Assign total supply to the deployer
        emit Transfer(address(0), owner, totalSupply);
    }

    function transfer(address to, uint256 value) public returns (bool success) {
        _transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) public returns (bool success) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool success) {
        require(allowance[from][msg.sender] >= value, "Allowance exceeded");
        allowance[from][msg.sender] -= value;
        _transfer(from, to, value);
        return true;
    }

    function _transfer(address from, address to, uint256 value) internal {
        require(balanceOf[from] >= value, "Insufficient balance");

        // Apply 5% fee on every transfer
        uint256 fee = (value * sellFee) / 100; // 5% fee
        uint256 newAmount = value - fee;

        balanceOf[from] -= value;
        balanceOf[to] += newAmount;
        balanceOf[owner] += fee; // Fee is sent to the deployer (owner)

        emit Transfer(from, to, newAmount);
        emit Transfer(from, owner, fee); // Fee sent to the owner (deployer)
    }

    // Optionally, you can change the fee with this function
    function setSellFee(uint256 fee) external onlyOwner {
        require(fee <= 10, "Fee too high");
        sellFee = fee;
    }
}