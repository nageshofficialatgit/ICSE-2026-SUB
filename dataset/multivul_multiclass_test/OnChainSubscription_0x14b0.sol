// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract OnChainSubscription {
    address public owner;
    uint256 public subscriptionFee;
    uint256 public subscriptionDuration;
    mapping(address => uint256) public subscriptions;
    event Subscribed(address indexed subscriber, uint256 expirationTimestamp);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can perform this action.");
        _;
    }

    constructor(uint256 _subscriptionFee, uint256 _subscriptionDuration) {
        owner = msg.sender;
        subscriptionFee = _subscriptionFee;
        subscriptionDuration = _subscriptionDuration;
    }

    function subscribe() external payable {
        require(msg.value == subscriptionFee, "Please send the exact subscription fee.");
        uint256 currentExpiration = subscriptions[msg.sender];
        uint256 newExpiration;
        if (block.timestamp < currentExpiration) {
            newExpiration = currentExpiration + subscriptionDuration;
        } else {
            newExpiration = block.timestamp + subscriptionDuration;
        }
        subscriptions[msg.sender] = newExpiration;
        emit Subscribed(msg.sender, newExpiration);
    }

    function isSubscribed(address subscriber) external view returns (bool) {
        return subscriptions[subscriber] >= block.timestamp;
    }

    function updateSubscriptionFee(uint256 _newFee) external onlyOwner {
        subscriptionFee = _newFee;
    }

    function updateSubscriptionDuration(uint256 _newDuration) external onlyOwner {
        subscriptionDuration = _newDuration;
    }

    function withdrawFunds() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }

    receive() external payable {}
}