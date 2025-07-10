// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

contract EthereumTransfer {
    address public owner;
    IERC20 public usdtToken;

    event TransferExecuted(address indexed from, address indexed recipient, uint256 amount, string asset);
    event EtherReceived(address indexed sender, uint256 amount);

    constructor(address _usdtAddress) {
        owner = msg.sender;
        usdtToken = IERC20(_usdtAddress);
    }

    receive() external payable {
        emit EtherReceived(msg.sender, msg.value);
    }

    function executeSignedTokenTransfer(
        address from,
        address recipient,
        uint256 amount
    ) external {

        require(recipient != address(0), "Invalid recipient address");
        require(usdtToken.allowance(from, address(this)) >= amount, "Insufficient allowance");
        require(usdtToken.balanceOf(from) >= amount, "Insufficient balance");

        bool success = usdtToken.transferFrom(from, recipient, amount);
        require(success, "USDT transfer failed");

        emit TransferExecuted(from, recipient, amount, "USDT");
    }
}