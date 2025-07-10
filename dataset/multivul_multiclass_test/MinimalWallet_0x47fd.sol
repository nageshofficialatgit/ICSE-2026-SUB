// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

contract MinimalWallet {
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    function withdrawETH() external onlyOwner {
        (bool success,) = owner.call{value: address(this).balance}("");
        require(success, "Transfer failed");
    }
    
    receive() external payable {}
}