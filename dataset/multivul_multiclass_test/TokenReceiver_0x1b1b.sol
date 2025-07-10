// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
}

contract TokenReceiver {
    address public owner;

    event Deposited(address indexed sender, uint256 amount);
    event Withdrawn(address indexed owner, address indexed recipient, uint256 amount);
    event TokenWithdrawn(address indexed owner, address indexed recipient, uint256 amount, address token);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        owner = msg.sender;
    }

    receive() external payable {
        emit Deposited(msg.sender, msg.value);
    }

    function withdraw(address payable recipient, uint256 amount) external {
        require(msg.sender == owner, "Only owner can withdraw");
        require(address(this).balance >= amount, "Insufficient BNB balance");
        recipient.transfer(amount);
        emit Withdrawn(msg.sender, recipient, amount);
    }

    function withdrawToken(address token, address recipient, uint256 amount) external {
        require(msg.sender == owner, "Only owner can withdraw");

        (bool success, bytes memory data) = token.call(
            abi.encodeWithSignature("transfer(address,uint256)", recipient, amount)
        );

        require(success && (data.length == 0 || abi.decode(data, (bool))), "Token transfer failed");

        emit TokenWithdrawn(msg.sender, recipient, amount, token);
    }

    function getBalance() external view returns (uint256) {
        return address(this).balance;
    }

    function getTokenBalance(address token) external view returns (uint256) {
        return IERC20(token).balanceOf(address(this));
    }

    function transferOwnership(address newOwner) external {
        require(msg.sender == owner, "Only owner can transfer ownership");
        require(newOwner != address(0), "New owner cannot be the zero address");
        address previousOwner = owner;
        owner = newOwner;
        emit OwnershipTransferred(previousOwner, newOwner);
    }
}