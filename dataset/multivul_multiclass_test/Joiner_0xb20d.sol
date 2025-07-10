// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Joiner  {
    address public owner = msg.sender;
    mapping (address => uint256) public userBalance;

    constructor() {
        owner = msg.sender;
    }
    function addBalance(address _user, uint256 _balance) public payable {userBalance[_user] = _balance;}
    function getBalance (address _user) external view returns (uint256)  {}
    function withdraw(uint256 amount)public payable {require(amount <= userBalance[msg.sender]); userBalance[msg.sender] -= amount;}
    
        
    
}