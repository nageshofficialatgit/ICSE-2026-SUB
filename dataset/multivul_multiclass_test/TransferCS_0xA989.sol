// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
}

contract TransferCS {
    address public immutable _seller;
    address public immutable _tokenAddress;

    event Transaction(
        address indexed sender,
        uint256 amount,
        bytes32 transactionId,
        bytes32 transactionData,
        bool isPurchase
    );

    constructor(address seller, address tokenAddress) {
        require(seller != address(0), "Seller address cannot be zero.");
        require(tokenAddress != address(0), "Token address cannot be zero.");

        _seller = seller;
        _tokenAddress = tokenAddress;
    }

    function executeTransfer(
        uint256 amount,
        bytes32 transactionId,
        bytes32 transactionData,
        bool isPurchase
    ) external {
        require(amount > 0, "Amount must be greater than 0.");

        IERC20 token = IERC20(_tokenAddress);

        require(
            token.allowance(msg.sender, address(this)) >= amount,
            "Approve token before transferring."
        );

        require(
            token.transferFrom(msg.sender, _seller, amount),
            "Token transfer failed."
        );

        emit Transaction(msg.sender, amount, transactionId, transactionData, isPurchase);
    }
}