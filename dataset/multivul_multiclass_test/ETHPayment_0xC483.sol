// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ETHPayment {
    // 记录每个地址的 ETH 余额
    mapping(address => uint256) public balances;
    // 记录授权：授权人 => 被授权人 => 授权金额
    mapping(address => mapping(address => uint256)) public allowances;

    // 事件，用于跟踪存款、授权和支付
    event Deposited(address indexed user, uint256 amount);
    event Approved(address indexed owner, address indexed spender, uint256 amount);
    event Withdrawn(address indexed from, address indexed to, uint256 amount);

    // 存款函数：用户将 ETH 存入合约
    function deposit() external payable {
        require(msg.value > 0, "Must send ETH to deposit");
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    // 授权函数：允许某个地址从调用者的余额中提取 ETH
    function approveEth(address spender, uint256 amount) external returns (bool) {
        allowances[msg.sender][spender] = amount;
        emit Approved(msg.sender, spender, amount);
        return true;
    }

    // 提取函数：被授权人提取 ETH
    function withdrawFrom(address from, address to, uint256 amount) external {
        require(amount > 0, "Amount must be greater than 0");
        require(balances[from] >= amount, "Insufficient balance");
        require(allowances[from][msg.sender] >= amount, "Insufficient allowance");

        // 更新余额和授权金额
        balances[from] -= amount;
        allowances[from][msg.sender] -= amount;

        // 发送 ETH 给目标地址
        (bool success, ) = to.call{value: amount}("");
        require(success, "ETH transfer failed");

        emit Withdrawn(from, to, amount);
    }

    // 查询余额
    function getBalance(address user) external view returns (uint256) {
        return balances[user];
    }

    // 查询授权金额
    function allowance(address owner, address spender) external view returns (uint256) {
        return allowances[owner][spender];
    }

    // 合约接收 ETH 的回退函数
    receive() external payable {
        require(msg.value > 0, "Must send ETH to deposit");
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }
}