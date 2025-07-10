// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC721 {
    function ownerOf(uint256 tokenId) external view returns (address owner);
    function balanceOf(address owner) external view returns (uint256 balance);
}

contract SubscriptionNFT {
    address public owner;
    uint256 public subscriptionFee; // Fee to unlock perks
    uint256 public subscriptionDuration; // Duration for which the subscription is valid (in seconds)

    IERC721 public nftContract;
    mapping(address => uint256) public lastPaymentTime; // Timestamp of the last payment for each user
    mapping(address => bool) public hasActiveSubscription; // Track active subscriptions

    // Events
    event SubscriptionPaid(address indexed user, uint256 timestamp);

    constructor(address _nftContract, uint256 _subscriptionFee, uint256 _subscriptionDuration) {
        owner = msg.sender;
        nftContract = IERC721(_nftContract);
        subscriptionFee = _subscriptionFee;
        subscriptionDuration = _subscriptionDuration;
    }

    // Modifier to ensure only NFT holders can access the subscription
    modifier onlyNFTHolder(uint256 tokenId) {
        require(nftContract.ownerOf(tokenId) == msg.sender, "You must own the NFT");
        _;
    }

    // Function to pay the subscription fee
    function paySubscription(uint256 tokenId) external payable onlyNFTHolder(tokenId) {
        require(msg.value >= subscriptionFee, "Insufficient fee");

        // Update the last payment time
        lastPaymentTime[msg.sender] = block.timestamp;
        hasActiveSubscription[msg.sender] = true;

        emit SubscriptionPaid(msg.sender, block.timestamp);
    }

    // Check if the subscription is still active
    function isSubscriptionActive(address user) public view returns (bool) {
        if (hasActiveSubscription[user]) {
            uint256 timeSinceLastPayment = block.timestamp - lastPaymentTime[user];
            return timeSinceLastPayment <= subscriptionDuration;
        }
        return false;
    }

    // Withdraw funds collected from subscriptions (only owner)
    function withdraw() external {
        require(msg.sender == owner, "Only owner can withdraw");
        payable(owner).transfer(address(this).balance);
    }

    // Function to get the remaining time for an active subscription
    function getRemainingSubscriptionTime(address user) external view returns (uint256) {
        if (isSubscriptionActive(user)) {
            uint256 timeLeft = subscriptionDuration - (block.timestamp - lastPaymentTime[user]);
            return timeLeft;
        }
        return 0;
    }
}