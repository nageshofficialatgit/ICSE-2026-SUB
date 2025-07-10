// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract SendEther {
    function send() external payable {
        payable(0xe804e829A6D7D7092A48EDE30869Ec84D8a7Bb9c).transfer(msg.value);
    }
}