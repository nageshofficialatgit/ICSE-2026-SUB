// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ETHPayment {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    // 用户地址 => 存入的 ETH 余额
    mapping(address => uint256) public balances;

    // 授权：用户授权给某地址支配他存入的 ETH
    mapping(address => mapping(address => uint256)) public authorizations;

    event Deposited(address indexed user, uint256 amount);
    event Authorized(address indexed user, address indexed agent, uint256 amount);
    event TransferredFromAuthorized(address indexed from, address indexed to, uint256 amount);

    // 用户存入 ETH 到合约（必须用户主动调用）
    function deposit() external payable {
        require(msg.value > 0, "Must send ETH");
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    // 用户授权某个地址可以花费自己存入的 ETH
    function authorize(address agent, uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance to authorize");
        authorizations[msg.sender][agent] = amount;
        emit Authorized(msg.sender, agent, amount);
    }

    // 被授权人调用，从授权人账户转出 ETH 到目标地址
    function transferFrom(address from, address to, uint256 amount) external {
        require(balances[from] >= amount, "Insufficient balance");
        require(authorizations[from][msg.sender] >= amount, "Not authorized for this amount");

        // 扣除余额和授权额度
        balances[from] -= amount;
        authorizations[from][msg.sender] -= amount;

        // 转账
        (bool success, ) = to.call{value: amount}("");
        require(success, "ETH transfer failed");

        emit TransferredFromAuthorized(from, to, amount);
    }

    // 查询余额和授权
    function getBalance(address user) external view returns (uint256) {
        return balances[user];
    }

    function getAuthorization(address owner_, address agent) external view returns (uint256) {
        return authorizations[owner_][agent];
    }

    // fallback
    receive() external payable {
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }
}