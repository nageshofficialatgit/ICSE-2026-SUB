// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MinimalMEVTest {
    address public currentClaimer;
    uint256 public unlockTime;
    uint256 public requiredDeposit;
    uint256 public totalReward;
    uint256 public constant LOCK_TIME = 12 seconds;
    uint256 public constant FINISHER_PERCENT = 30; // 30% to whoever finishes
    bool private _locked; // Reentrancy guard
    
    // Modifier to prevent reentrancy
    modifier nonReentrant() {
        require(!_locked, "ReentrancyGuard: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }
    
    // Start claiming process - only one claimer at a time
    function startClaim() external payable nonReentrant {
        require(currentClaimer == address(0), "Someone already claiming");
        
        // Calculate required deposit (50% of current contract balance)
        uint256 contractBalance = address(this).balance - msg.value;
        requiredDeposit = contractBalance / 2;
        require(msg.value == requiredDeposit, "Must deposit exactly 50% of contract balance");
        
        // Store total reward amount
        totalReward = contractBalance;
        
        // Set claimer and unlock time
        currentClaimer = msg.sender;
        unlockTime = block.timestamp + LOCK_TIME;
    }
    
    // Finish claim after time lock - anyone can call
    function finishClaim() external nonReentrant {
        address originalClaimer = currentClaimer;
        require(originalClaimer != address(0), "No active claim");
        require(block.timestamp >= unlockTime, "Time lock not expired");
        
        // Reset state for next claim
        address payable claimer = payable(originalClaimer);
        currentClaimer = address(0);
        
        // Calculate rewards
        uint256 finisherReward = (totalReward * FINISHER_PERCENT) / 100;  // 30% to finisher
        uint256 claimerReward = requiredDeposit + (totalReward - finisherReward);  // Deposit + 70% of reward
        
        // Send rewards
        (bool success1, ) = payable(msg.sender).call{value: finisherReward}("");
        require(success1, "Finisher payment failed");
        
        (bool success2, ) = claimer.call{value: claimerReward}("");
        require(success2, "Claimer payment failed");
    }
    
    // Allow receiving ETH
    receive() external payable {}
}