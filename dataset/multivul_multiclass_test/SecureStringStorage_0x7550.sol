// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SecureStringStorage {
    address private caller;
    string private storedString;

    event StringUpdated(address indexed by, string newValue);

    constructor(address _caller) {
        caller = _caller;
    }

    function updateString(string memory _newString) external {
        require(msg.sender == caller, "Not a caller");
        storedString = _newString;
        emit StringUpdated(msg.sender, _newString);
    }

    function getString() external view returns (string memory) {
        return storedString;
    }
}