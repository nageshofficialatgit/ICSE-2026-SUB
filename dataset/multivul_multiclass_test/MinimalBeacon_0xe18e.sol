// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IOpenOracle {
    function settle(uint256 reportId) external returns (uint256 price, uint256 settlementTimestamp);
}

contract MinimalBeacon {
    IOpenOracle public immutable oracle;
    bool private _locked; // Simple reentrancy lock
    
    constructor(address oracleAddress) {
        oracle = IOpenOracle(oracleAddress);
    }
    
    // Call this to get free money
    function freeMoney(uint256 reportId) external {
        // Prevent reentrancy
        require(!_locked, "Reentrant call");
        _locked = true;
        
        // Get starting balance to calculate the reward received
        uint256 startBalance = address(this).balance;
        
        // Call settle() - the oracle sends reward to this contract
        oracle.settle(reportId);
        
        // Calculate received amount
        uint256 received = address(this).balance - startBalance;
        
        // Forward the reward to the original caller
        if (received > 0) {
            (bool success, ) = payable(msg.sender).call{value: received}("");
            require(success, "ETH transfer failed");
        }
        
        // Release the lock
        _locked = false;
    }
    
    // Allow the contract to receive ETH (needed for the oracle's reward)
    receive() external payable {}
}