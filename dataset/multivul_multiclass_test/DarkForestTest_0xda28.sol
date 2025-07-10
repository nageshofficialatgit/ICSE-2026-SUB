// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

contract DarkForestTest {
    uint256 public lastDepositTime;

    // Function to receive Ether
    receive() external payable {
        lastDepositTime = block.timestamp;
    }

    // Fallback function to receive Ether
    fallback() external payable {
        lastDepositTime = block.timestamp;
    }
    
    // Function to withdraw all ETH, but only after 24 seconds have passed since last deposit
    function freeMoney() external {
        require(block.timestamp >= lastDepositTime + 24, "Too early, need to wait 24 seconds after last deposit");
        
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH in contract");
        
        // Send all ETH to the caller
        (bool success, ) = payable(msg.sender).call{value: balance}("");
        require(success, "Transfer failed");
    }
    
    // Function to check how much time remains until freeMoney can be called
    function timeRemaining() external view returns (uint256) {
        if (block.timestamp >= lastDepositTime + 24) {
            return 0;
        }
        return lastDepositTime + 24 - block.timestamp;
    }
}