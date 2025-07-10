// SPDX-License-Identifier: MIT

pragma solidity ^0.8.26;


contract TokenStorage {
    mapping(address => uint) public tokenBalance;

    event TokenTransfer(address indexed  _from, address indexed  _to, uint _amount);

    constructor() {
        tokenBalance[msg.sender] = 1000;
    }

    function sendToken(address _to, uint _amount) public {
        require(tokenBalance[msg.sender] >= _amount, "Not enough balance");
        tokenBalance[msg.sender] -= _amount;
        tokenBalance[_to] += _amount;

        emit TokenTransfer(msg.sender, _to, _amount);
    }
}