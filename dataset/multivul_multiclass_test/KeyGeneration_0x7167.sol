// SPDX-License-Identifier: GPL-3.0
pragma solidity 0.8.28;
 
contract KeyGeneration{
    
    function getKeyCode(address account) public view returns (bytes32 key) {
        key = keccak256(abi.encode(account));
     //  emit KeyCodeGenerated(key);
        return key;
    }
 
    event KeyCodeGenerated(bytes32 key);
}