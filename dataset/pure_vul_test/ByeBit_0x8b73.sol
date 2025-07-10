// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ByeBit {
    string public name = "ByeBit";
    string public symbol = "BYE";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    address public owner;

    event Transfer(address indexed from, address indexed to, uint256 value);

    constructor(address friendAddress) {
        owner = msg.sender;
        totalSupply = 100_000_000 * (10 ** uint256(decimals));
        uint256 twoPercent = 2_000_000 * (10 ** uint256(decimals));
        balanceOf[msg.sender] = twoPercent;
        balanceOf[friendAddress] = twoPercent;
        balanceOf[address(this)] = totalSupply - (2 * twoPercent);
        emit Transfer(address(0), msg.sender, twoPercent);
        emit Transfer(address(0), friendAddress, twoPercent);
        emit Transfer(address(0), address(this), totalSupply - (2 * twoPercent));
    }

    function mint(address to, uint256 amount) public {
        require(msg.sender == owner, "Only the owner can mint tokens");
        uint256 amountWithDecimals = amount * (10 ** uint256(decimals));
        totalSupply += amountWithDecimals;
        balanceOf[to] += amountWithDecimals;
        emit Transfer(address(0), to, amountWithDecimals);
    }

    function burn(uint256 amount) public {
        require(msg.sender == owner, "Only the owner can burn tokens");
        uint256 amountWithDecimals = amount * (10 ** uint256(decimals));
        require(balanceOf[msg.sender] >= amountWithDecimals, "Insufficient balance to burn");
        balanceOf[msg.sender] -= amountWithDecimals;
        totalSupply -= amountWithDecimals;
        emit Transfer(msg.sender, address(0), amountWithDecimals);
    }

    function transfer(address to, uint256 value) public returns (bool) {
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }
}