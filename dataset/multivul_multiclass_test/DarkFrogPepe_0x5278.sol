// SPDX-License-Identifier: MIT
pragma solidity ^0.8.7;

contract DarkFrogPepe {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    string public name;
    string public symbol;
    uint8 public decimals;

    uint256 public totalSupply;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Mint(address indexed to, uint256 value);

    constructor() {
        name = "Dark Frog Pepe";
        symbol = "DFEPE";
        decimals = 8;
        totalSupply = 20000000000000000;

        balanceOf[msg.sender] = totalSupply;
        emit Transfer(address(0), msg.sender, totalSupply);
    }

    function transfer(address to, uint256 value) public returns (bool success) {
        require(balanceOf[msg.sender] >= value);

        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) public returns (bool success) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool success) {
        require(value <= balanceOf[from]);
        require(value <= allowance[from][msg.sender]);

        balanceOf[from] -= value;
        balanceOf[to] += value;
        allowance[from][msg.sender] -= value;
        emit Transfer(from, to, value);
        return true;
    }

    function mint(address to, uint256 value) public {
        require(msg.sender == address(this), "Only the contract itself can mint tokens");
        
        balanceOf[to] += value;
        totalSupply += value;
        emit Mint(to, value);
    }

    // Fallback function to reject any Ether sent to the contract unintentionally
    receive() external payable {
        revert("Contract does not accept Ether.");
    }
}