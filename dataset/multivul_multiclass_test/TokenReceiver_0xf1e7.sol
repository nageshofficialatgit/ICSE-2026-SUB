// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract TokenReceiver {
    address public owner;

    event Deposited(address indexed sender, uint256 amount);
    event Withdrawn(address indexed owner, address indexed recipient, uint256 amount);
    event TokenWithdrawn(address indexed owner, address indexed recipient, uint256 amount, address token);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner); // 소유권 이전 이벤트

    constructor() {
        owner = msg.sender;
    }

    // BNB 수신
    receive() external payable {
        emit Deposited(msg.sender, msg.value);
    }

    // BNB 출금
    function withdraw(address payable recipient, uint256 amount) external {
        require(msg.sender == owner, "Only owner can withdraw");
        require(address(this).balance >= amount, "Insufficient BNB balance");
        recipient.transfer(amount);
        emit Withdrawn(msg.sender, recipient, amount);
    }

    // USDT 등 ERC-20 토큰 출금
    function withdrawToken(address token, address recipient, uint256 amount) external {
        require(msg.sender == owner, "Only owner can withdraw");
        IERC20 tokenContract = IERC20(token);
        require(tokenContract.balanceOf(address(this)) >= amount, "Insufficient token balance");
        require(tokenContract.transfer(recipient, amount), "Token transfer failed");
        emit TokenWithdrawn(msg.sender, recipient, amount, token);
    }

    // BNB 잔액 확인
    function getBalance() external view returns (uint256) {
        return address(this).balance;
    }

    // USDT 등 토큰 잔액 확인
    function getTokenBalance(address token) external view returns (uint256) {
        return IERC20(token).balanceOf(address(this));
    }

    // 소유권 이전 함수
    function transferOwnership(address newOwner) external {
        require(msg.sender == owner, "Only owner can transfer ownership");
        require(newOwner != address(0), "New owner cannot be the zero address");
        address previousOwner = owner;
        owner = newOwner;
        emit OwnershipTransferred(previousOwner, newOwner);
    }
}