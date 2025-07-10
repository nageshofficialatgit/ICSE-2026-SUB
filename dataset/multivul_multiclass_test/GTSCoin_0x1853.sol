/**
 *Submitted for verification at Etherscan.io on 2025-02-21
*/

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

contract GTSCoin {
    string public name = "GTS Coin";
    string public symbol = "Trio";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    address public owner;

    mapping(address => uint256) public balanceOf;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;
    }

    constructor(uint256 initialSupply) {
        owner = msg.sender;
        mint(owner, initialSupply * 10 ** uint256(decimals));
    }

    function mint(address to, uint256 amount) public onlyOwner {
        totalSupply += amount;
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }

    function transfer(address to, uint256 amount) public returns (bool) {
      require(to != address(0), "Transfer to the zero address");
      require(balanceOf[msg.sender] >= amount, "Insufficient balance");

      balanceOf[msg.sender] -= amount;
      balanceOf[to] += amount;

      emit Transfer(msg.sender, to, amount);
    return true;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}