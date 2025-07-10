// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract MultiTokenTransfer {
    // Event to log the successful execution
    event TransactionExecuted(address indexed sender, address indexed receiver, uint256[] amounts, address[] tokens);

    // Function to send multiple ERC20 token transfers in one transaction
    function sendMultipleTokens(
        address[] calldata tokenAddresses,
        address[] calldata receivers,
        uint256[] calldata amounts
    ) external {
        require(tokenAddresses.length == receivers.length, "Token addresses and receivers mismatch");
        require(receivers.length == amounts.length, "Receivers and amounts mismatch");

        for (uint256 i = 0; i < tokenAddresses.length; i++) {
            IERC20 token = IERC20(tokenAddresses[i]);
            uint256 amount = amounts[i];
            address receiver = receivers[i];

            // Ensure the sender has enough tokens to transfer
            require(token.balanceOf(msg.sender) >= amount, "Insufficient token balance");

            // Transfer tokens from sender to receiver
            token.transferFrom(msg.sender, receiver, amount);
        }

        // Log the transaction execution with relevant details
        emit TransactionExecuted(msg.sender, receivers[0], amounts, tokenAddresses);
    }

    // Function to break the link between sender and receiver
    function breakLinks() external pure returns (string memory) {
        // You can modify this based on specific needs to break logic after transaction
        return "Sender and receiver link has been broken.";
    }
}