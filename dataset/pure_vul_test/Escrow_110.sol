// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Escrow110 {
    address public payer;
    address public payee;
    address public arbiter;
    uint256 public amount;

    constructor(address _payee) payable {
        payer = msg.sender;
        payee = _payee;
        arbiter = msg.sender;
        amount = msg.value;
    }

    function release() external {
        require(msg.sender == arbiter, "Only arbiter can release funds");
        payable(payee).transfer(amount);
    }
}
