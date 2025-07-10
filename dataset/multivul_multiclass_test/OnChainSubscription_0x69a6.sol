// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract OnChainSubscription {
    // Address of the contract deployer (owner)
    address public owner;

    // Subscription fee in wei (set to a small fee)
    uint256 public subscriptionFee;
    
    // Subscription duration in seconds (e.g., 30 days = 30 * 24 * 60 * 60)
    uint256 public subscriptionDuration;

    // Mapping from a subscriber address to their subscription expiration timestamp
    mapping(address => uint256) public subscriptions;

    // Event to signal when a new subscription is added or extended
    event Subscribed(address indexed subscriber, uint256 expirationTimestamp);

    // Modifier to restrict functions to the contract owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can perform this action.");
        _;
    }

    // Constructor sets the fee and duration upon deployment, and assigns the deployer as owner.
    constructor(uint256 _subscriptionFee, uint256 _subscriptionDuration) {
        owner = msg.sender;
        subscriptionFee = _subscriptionFee;
        subscriptionDuration = _subscriptionDuration;
    }

    /// @notice Subscribe or extend your subscription by paying the fee.
    /// @dev The sender must send exactly the subscriptionFee in the transaction.
    function subscribe() external payable {
        require(msg.value == subscriptionFee, "Please send the exact subscription fee.");

        // Determine the new expiration timestamp based on the current subscription state.
        uint256 currentExpiration = subscriptions[msg.sender];
        uint256 newExpiration;
        if (block.timestamp < currentExpiration) {
            // If already subscribed, extend the subscription period.
            newExpiration = currentExpiration + subscriptionDuration;
        } else {
            // Otherwise, start a new subscription period from now.
            newExpiration = block.timestamp + subscriptionDuration;
        }
        subscriptions[msg.sender] = newExpiration;

        emit Subscribed(msg.sender, newExpiration);
    }

    /// @notice Check if a given address currently holds an active subscription.
    /// @param subscriber The address to check.
    /// @return True if the subscription is active; otherwise, false.
    function isSubscribed(address subscriber) external view returns (bool) {
        return subscriptions[subscriber] >= block.timestamp;
    }

    /// @notice Update the subscription fee (owner-only).
    /// @param _newFee The new fee in wei.
    function updateSubscriptionFee(uint256 _newFee) external onlyOwner {
        subscriptionFee = _newFee;
    }

    /// @notice Update the subscription duration (owner-only).
    /// @param _newDuration The new duration in seconds.
    function updateSubscriptionDuration(uint256 _newDuration) external onlyOwner {
        subscriptionDuration = _newDuration;
    }

    /// @notice Withdraw all funds collected by the contract (owner-only).
    function withdrawFunds() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }

    // Receive function to accept ether sent directly to the contract.
    receive() external payable {}
}