// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract HelloFriend {
    string public message = "Hello, my friend! Glad you checked this out! :)";

    function getMessage() public view returns (string memory) {
        return message;
    }
}