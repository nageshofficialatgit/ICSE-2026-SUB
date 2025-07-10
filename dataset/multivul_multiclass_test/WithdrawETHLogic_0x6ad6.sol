// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract WithdrawETHLogic {
    event Withdrawn(address indexed recipient, uint256 amount);

    function withdrawETH(address payable recipient, uint256 amount) external {
        require(recipient != address(0), "Invalid recipient");
        require(amount > 0 && amount <= address(this).balance, "Invalid amount");

        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Transfer failed");

        emit Withdrawn(recipient, amount);
    }

    receive() external payable {} // Allow contract to receive ETH
}