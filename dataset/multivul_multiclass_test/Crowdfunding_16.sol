// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Crowdfunding16 {
    address public owner;
    uint256 public deadline;
    uint256 public goal;
    uint256 public fundsRaised;
    mapping(address => uint256) public contributions;

    constructor() {
        owner = msg.sender;
        deadline = block.timestamp + 1555200;
        goal = 80000000000000000000;
    }

    function contribute() external payable {
        require(block.timestamp < deadline, "Deadline passed");
        require(msg.value > 0, "Must send ETH");
        contributions[msg.sender] += msg.value;
        fundsRaised += msg.value;
    }
}
