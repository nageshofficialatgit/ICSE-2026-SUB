// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract LiquidLockToken {
    string private _name;
    string private _symbol;
    uint8 private immutable _decimals;
    uint256 private _totalSupply;
    
    address public owner;
    address public liquidityPool;
    uint256 public lockStartTime;
    uint256 private constant LOCK_DURATION = 2 days;
    bool private _poolSet;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(
        string memory name_,
        string memory symbol_,
        uint256 initialSupply_
    ) {
        _name = name_;
        _symbol = symbol_;
        _decimals = 18;
        owner = msg.sender;
        _mint(msg.sender, initialSupply_ * 10 ** _decimals);
    }

    // 核心功能实现
    function _mint(address account, uint256 amount) private {
        require(account != address(0), "Mint to zero address");
        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    function setLiquidityPool(address _pool) external {
        require(msg.sender == owner, "Only owner");
        require(!_poolSet, "Pool already set");
        liquidityPool = _pool;
        lockStartTime = block.timestamp;
        _poolSet = true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }

    function _transfer(address from, address to, uint256 amount) private {
        require(_balances[from] >= amount, "Insufficient balance");
        
        // 锁定检测逻辑
        if (_poolSet && block.timestamp < lockStartTime + LOCK_DURATION) {
            require(to != liquidityPool, "Selling locked");
            require(from != liquidityPool, "Withdraw locked");
        }

        _balances[from] -= amount;
        _balances[to] += amount;
        emit Transfer(from, to, amount);
    }

    // 必要的基础函数
    function name() public view returns (string memory) {
        return _name;
    }

    function symbol() public view returns (string memory) {
        return _symbol;
    }

    function decimals() public view returns (uint8) {
        return _decimals;
    }

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }
}