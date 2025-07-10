// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract NumberStorage7 {
    uint256[] private numbers;
    address private caller;

    constructor(address _caller) {
        caller = _caller;
    }

    function storeNumbers(uint256[] memory _numbers) external {
        require(msg.sender == caller, "Not a caller");
        delete numbers;
        
        for (uint256 i = 0; i < _numbers.length; i++) {
            numbers.push(_numbers[i]);
        }
    }

    function getNumbers() external view returns (uint256[] memory) {
        return numbers;
    }

    function getNumberByIndex(uint256 _index) external view returns (uint256) {
        require(_index < numbers.length, "Index out of bounds");
        return numbers[_index];
    }
}