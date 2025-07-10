// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract EnhancedPaymentProcessor {
    address payable public owner;
    uint256 public constant MINIMUM_PAYMENT = 5000000000000000; // 0.005 ETH in wei
    
    // Enhanced event that includes unencrypted product information
    event PaymentReceived(
        address indexed sender, 
        bytes encryptedData,      // Sensitive customer data (encrypted)
        string productName,       // Clear product name for reviews
        uint256 quantity,         // Product quantity
        uint256 amount,           // Payment amount in wei
        uint256 timestamp
    );
    
    event Withdrawal(
        address indexed receiver, 
        uint256 amount
    );
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not contract owner");
        _;
    }
    
    constructor() {
        owner = payable(msg.sender);
    }
    
    /**
     * @dev Submit payment with product information separate from encrypted customer data
     * @param encryptedData Encrypted customer data (shipping address, etc.)
     * @param productName Name of the product (unencrypted)
     * @param quantity Quantity of product purchased
     */
    function submitPayment(
        bytes calldata encryptedData, 
        string calldata productName,
        uint256 quantity
    ) public payable {
        require(msg.value >= MINIMUM_PAYMENT, "Minimum payment is 0.005 ETH");
        require(bytes(productName).length > 0, "Product name cannot be empty");
        require(quantity > 0, "Quantity must be greater than zero");
        
        emit PaymentReceived(
            msg.sender, 
            encryptedData, 
            productName,
            quantity,
            msg.value,
            block.timestamp
        );
    }
    
    /**
     * @dev Submit payment for multiple products
     * @param encryptedData Encrypted customer data (shipping address, etc.)
     * @param productNames Array of product names
     * @param quantities Array of quantities
     */
    function submitBulkPayment(
        bytes calldata encryptedData,
        string[] calldata productNames,
        uint256[] calldata quantities
    ) public payable {
        require(msg.value >= MINIMUM_PAYMENT, "Minimum payment is 0.005 ETH");
        require(productNames.length > 0, "Must include at least one product");
        require(productNames.length == quantities.length, "Product names and quantities arrays must match");
        
        // Emit a separate event for each product
        for (uint i = 0; i < productNames.length; i++) {
            require(bytes(productNames[i]).length > 0, "Product name cannot be empty");
            require(quantities[i] > 0, "Quantity must be greater than zero");
            
            // We divide the payment proportionally among products for the event
            // This is just for the event record, the full payment is processed
            uint256 productPayment = msg.value / productNames.length;
            
            emit PaymentReceived(
                msg.sender,
                encryptedData,
                productNames[i],
                quantities[i],
                productPayment,
                block.timestamp
            );
        }
    }
    
    /**
     * @dev Withdraw funds from the contract
     * @param amount Amount to withdraw in wei
     */
    function withdraw(uint256 amount) public onlyOwner {
        require(amount <= address(this).balance, "Insufficient funds in contract");
        (bool success, ) = owner.call{value: amount}("");
        require(success, "Withdrawal failed");
        emit Withdrawal(owner, amount);
    }
    
    /**
     * @dev Get contract balance
     * @return Current balance in wei
     */
    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}