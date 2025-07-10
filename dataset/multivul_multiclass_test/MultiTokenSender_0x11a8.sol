// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// ✅ Complete ERC-20 Interface - No Missing Functions
interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

// ✅ MultiTokenSender Contract - Fully Implemented
contract MultiTokenSender {
    address public owner;

    event TokensTransferred(address indexed sender, address indexed recipient, address token, uint256 amount);
    event ETHReceived(address indexed sender, uint256 amount);
    event Withdrawn(address indexed owner, address indexed token, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    // ✅ Allow Contract to Receive ETH
    receive() external payable {
        emit ETHReceived(msg.sender, msg.value);
    }

    /**
     * ✅ Send multiple tokens (ERC-20 & ETH) to the same recipient in one transaction.
     * @param recipient The address receiving the tokens.
     * @param tokens List of token addresses (use address(0) for ETH).
     * @param amounts Corresponding amounts to send.
     */
    function sendMultipleTokens(address recipient, address[] calldata tokens, uint256[] calldata amounts) external payable {
        require(tokens.length == amounts.length, "Array lengths mismatch");

        for (uint256 i = 0; i < tokens.length; i++) {
            if (tokens[i] == address(0)) {
                // ✅ Send ETH
                require(address(this).balance >= amounts[i], "Not enough ETH");
                payable(recipient).transfer(amounts[i]);
            } else {
                // ✅ Send ERC-20 Token
                IERC20 token = IERC20(tokens[i]);
                require(token.allowance(msg.sender, address(this)) >= amounts[i], "Not enough allowance");
                require(token.transferFrom(msg.sender, recipient, amounts[i]), "ERC20 transfer failed");
            }

            emit TokensTransferred(msg.sender, recipient, tokens[i], amounts[i]);
        }
    }

    /**
     * ✅ Withdraw stored ETH or ERC-20 tokens.
     * @param token Address of the token to withdraw (use address(0) for ETH).
     * @param amount Amount to withdraw.
     */
    function withdraw(address token, uint256 amount) external onlyOwner {
        if (token == address(0)) {
            require(address(this).balance >= amount, "Not enough ETH");
            payable(owner).transfer(amount);
        } else {
            IERC20 tokenContract = IERC20(token);
            require(tokenContract.balanceOf(address(this)) >= amount, "Not enough token balance");
            require(tokenContract.transfer(owner, amount), "ERC20 transfer failed");
        }

        emit Withdrawn(owner, token, amount);
    }
}