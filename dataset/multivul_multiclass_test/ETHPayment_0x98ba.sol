// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

contract ETHPayment {

    // 存储每个地址的授权额度（允许支付ETH的额度）
    mapping(address => uint256) private _allowances;

    address private _owner;

    event Approval(address indexed spender, uint256 amount);
    event Payment(address indexed sender, uint256 amount);

    // 构造函数，初始化合约所有者
    constructor() {
        _owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == _owner, "ETHPayment: caller is not the owner");
        _;
    }

    modifier onlyAuthorized() {
        require(_allowances[msg.sender] > 0, "ETHPayment: caller is not authorized");
        _;
    }

    // 允许合约所有者授权某个地址支付ETH
    function approve(address spender, uint256 amount) external onlyOwner {
        _allowances[spender] = amount;
        emit Approval(spender, amount);
    }

    // 查看某个地址的授权额度
    function allowance(address spender) external view returns (uint256) {
        return _allowances[spender];
    }

    // 被授权的地址通过该方法向合约支付ETH
    function transferFrom(address, uint256 amount) external onlyAuthorized payable {
        require(msg.value == amount, "ETHPayment: ETH value mismatch");
        require(amount <= _allowances[msg.sender], "ETHPayment: amount exceeds allowance");

        // 更新授权余额
        _allowances[msg.sender] -= amount;

        // 合约接收ETH
        emit Payment(msg.sender, amount);
    }

    // 提取合约中的ETH到所有者
    function withdraw(uint256 amount) external onlyOwner {
        require(address(this).balance >= amount, "ETHPayment: insufficient balance");
        payable(_owner).transfer(amount);
    }

    // 合约接收ETH
    receive() external payable {
        emit Payment(msg.sender, msg.value);
    }
}