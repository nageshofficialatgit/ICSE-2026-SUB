// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ProductReview
 * @dev Stores and retrieves product reviews on the blockchain
 */
contract ProductReview {
    address public owner;
    
    // Events
    event ReviewSubmitted(
        address indexed reviewer,
        bytes32 indexed productId,
        bytes32 indexed originalTxHash,
        bytes reviewData,
        uint256 timestamp
    );
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev Submit a product review with transaction hash reference
     * @param productId The ID of the product being reviewed 
     * @param originalTxHash The transaction hash of the original purchase
     * @param reviewData The encrypted review data (includes rating, text, etc.)
     */
    function submitReview(
        bytes32 productId,
        bytes32 originalTxHash, 
        bytes calldata reviewData
    ) external {
        // Basic validation
        require(reviewData.length > 0, "Review data cannot be empty");
        
        // Emit the review event
        emit ReviewSubmitted(
            msg.sender,
            productId,
            originalTxHash,
            reviewData,
            block.timestamp
        );
    }
    
    /**
     * @dev Allow the owner to update the contract ownership
     * @param newOwner The address of the new owner
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid new owner address");
        owner = newOwner;
    }
}