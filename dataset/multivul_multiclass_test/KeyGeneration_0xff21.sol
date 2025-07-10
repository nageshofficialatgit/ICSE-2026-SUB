// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.28;

contract KeyGeneration {

    // Event for emitting key generation
    event KeyCodeGenerated(bytes32 key);

    // Function to generate a key based on an account address
    function getKeyCode(address account) public  returns (bytes32 key) {
        key = keccak256(abi.encode(account));
        emit KeyCodeGenerated(key);  // Emit the event
        return key;
    }

    // Function to generate a hash from identity, claimTopic, and data
    function generateDataHash(address _identity, uint256 claimTopic, bytes memory data) external pure returns (bytes32) {
        return keccak256(abi.encode(_identity, claimTopic, data));  // Hash the inputs
    }

    // Function to generate a prefixed hash with the Ethereum signed message prefix
    function generatePrefixedHash(bytes32 dataHash) public pure returns (bytes32) {
        // Prefix the hash with the Ethereum signed message format and hash it
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", dataHash));
    }
}