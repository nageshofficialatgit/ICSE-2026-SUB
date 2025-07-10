// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleStorage {
    uint256 private storedNumber;

    // Event to emit when a number is stored
    event NumberStored(uint256 number);

    // Store a number
    function store(uint256 _number) public {
        storedNumber = _number;
        emit NumberStored(_number);
    }

    // Retrieve the stored number
    function retrieve() public view returns (uint256) {
        return storedNumber;
    }
}