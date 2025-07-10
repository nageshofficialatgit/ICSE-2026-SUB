// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract USDTDistribution {
    address public owner;
    uint256 public endTime;
    uint256 public dailyDistribution;
    uint256 public recipientsLimit;

    mapping(address => uint256) public distributions;
    mapping(address => uint256) public claimed;

    constructor(uint256 _dailyDistribution, uint256 _recipientsLimit, uint256 _durationInDays) {
        owner = msg.sender;
        dailyDistribution = _dailyDistribution;
        recipientsLimit = _recipientsLimit;
        endTime = block.timestamp + (_durationInDays * 1 days);
    }

    function distribute(address[] memory recipients, uint256[] memory amounts) external {
        require(msg.sender == owner, "Not authorized");
        require(block.timestamp < endTime, "Distribution period ended");
        require(recipients.length == amounts.length, "Mismatched arrays");
        require(recipients.length <= recipientsLimit, "Exceeds recipient limit");

        for (uint256 i = 0; i < recipients.length; i++) {
            distributions[recipients[i]] += amounts[i];
        }
    }

    function claim() external {
        uint256 amount = distributions[msg.sender] - claimed[msg.sender];
        require(amount > 0, "Nothing to claim");
        claimed[msg.sender] += amount;

        // Add ERC20 transfer logic here (e.g., USDT transfer)
        // Example: IERC20(token).transfer(msg.sender, amount);
    }
}