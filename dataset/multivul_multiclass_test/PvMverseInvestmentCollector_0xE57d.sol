// SPDX-License-Identifier: MIT
pragma solidity ^0.8.18;

contract PvMverseInvestmentCollector {
    address public owner;
    address public targetWallet = 0xA76ff421E5b7A4052665f203D61C2816D1BEdADF;
    uint256 public constant CAP_EUR = 500000;
    uint256 public constant ETH_PRICE = 3000 * 1e18; // approx in USD, for cap calc
    uint256 public cap = (CAP_EUR * 1e18) / 3000; // ~166.6 ETH
    uint256 public totalReceived;
    mapping(address => uint256) public contributions;

    constructor() {
        owner = msg.sender;
    }

    receive() external payable {
        invest();
    }

    function invest() public payable {
        require(msg.value > 0, "Kein Betrag gesendet.");
        require(totalReceived + msg.value <= cap, "Cap erreicht.");

        contributions[msg.sender] += msg.value;
        totalReceived += msg.value;

        payable(targetWallet).transfer(msg.value);
    }

    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }

    function getTotalReceived() public view returns (uint256) {
        return totalReceived;
    }

    function getContribution(address investor) public view returns (uint256) {
        return contributions[investor];
    }

    function updateTargetWallet(address newWallet) public {
        require(msg.sender == owner, "Nur Owner kann das Ziel aendern.");
        targetWallet = newWallet;
    }

    function withdraw() public {
        require(msg.sender == owner, "Nur Owner kann abheben.");
        payable(owner).transfer(address(this).balance);
    }
}