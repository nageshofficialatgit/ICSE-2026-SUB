// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MyContract {
    string public message;

    constructor() {
        message = "Jeremy Marshall would love to work at Coinbase :)))!";
    }

    function updateMessage(string memory _newMessage) public {
        message = _newMessage;
    }
}