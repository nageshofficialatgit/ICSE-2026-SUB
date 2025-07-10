// SPDX-License-Identifier: MIT
// Tells the Solidity compiler to compile only from v0.8.13 to v0.9.0
pragma solidity ^0.8.13;

contract AUStockPredict {

    address payable owner;

    constructor() {
        owner = payable(msg.sender);
    }

    modifier onlyOwner {
        require(msg.sender == owner);
        _; // continue
    }

    function donate() external payable {
    }

    function getBalance() external view onlyOwner returns(uint) {
        // this mean the current contract
        return address(this).balance;
    }

    function collect() external onlyOwner returns (bool success) {
        if (!owner.send(address(this).balance)) {
            return false;
        }

        return true;
    }
}