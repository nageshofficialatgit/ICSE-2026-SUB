// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract SecureToken {
    string private _name;
    string private _symbol;
    uint256 private _totalSupply;
    address public owner;
    
    address public liquidityProvider;
    uint256 public lockEndTime;
    bool public liquidityLocked;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _blacklist;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event BlacklistUpdated(address indexed account, bool status);
    event LiquidityLocked(address indexed provider, uint256 unlockTime);

    modifier onlyOwner() {
        require(msg.sender == owner, "Owner only");
        _;
    }

    constructor(string memory name_, string memory symbol_, uint256 totalSupply_) {
        _name = name_;
        _symbol = symbol_;
        _totalSupply = totalSupply_ * 1e18; // 自动处理18位小数
        _balances[msg.sender] = _totalSupply;
        owner = msg.sender;
        
        emit Transfer(address(0), msg.sender, _totalSupply);
    }

    // 黑名单管理功能
    function addToBlacklist(address account) external onlyOwner {
        require(!_blacklist[account], "Already blacklisted");
        _blacklist[account] = true;
        emit BlacklistUpdated(account, true);
    }

    function removeFromBlacklist(address account) external onlyOwner {
        require(_blacklist[account], "Not blacklisted");
        _blacklist[account] = false;
        emit BlacklistUpdated(account, false);
    }

    function isBlacklisted(address account) public view returns(bool) {
        return _blacklist[account];
    }

    // 流动性锁定功能
    function lockLiquidity() external onlyOwner {
        require(!liquidityLocked, "Already locked");
        liquidityProvider = msg.sender;
        lockEndTime = block.timestamp + 2 days;
        liquidityLocked = true;
        emit LiquidityLocked(msg.sender, lockEndTime);
    }

    // 转账功能（集成双重检查）
    function transfer(address to, uint256 amount) external returns(bool) {
        _validateTransfer(msg.sender, to);
        _transfer(msg.sender, to, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns(bool) {
        _validateTransfer(from, to);
        
        uint256 currentAllowance = _allowances[from][msg.sender];
        require(currentAllowance >= amount, "ERC20: insufficient allowance");
        unchecked {
            _approve(from, msg.sender, currentAllowance - amount);
        }
        
        _transfer(from, to, amount);
        return true;
    }

    // 验证逻辑（黑名单+流动性锁定）
    function _validateTransfer(address from, address to) internal view {
        // 黑名单检查
        require(!_blacklist[from], "Sender blacklisted");
        require(!_blacklist[to], "Recipient blacklisted");
        
        // 流动性锁定检查
        if(from == liquidityProvider) {
            require(block.timestamp >= lockEndTime, "Liquidity locked");
        }
    }

    // 核心转账逻辑
    function _transfer(address from, address to, uint256 amount) internal {
        require(from != address(0), "ERC20: transfer from zero");
        require(to != address(0), "ERC20: transfer to zero");

        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "ERC20: insufficient balance");
        
        unchecked {
            _balances[from] = fromBalance - amount;
        }
        _balances[to] += amount;

        emit Transfer(from, to, amount);
    }

    // ERC20标准函数
    function name() public view returns(string memory) {
        return _name;
    }

    function symbol() public view returns(string memory) {
        return _symbol;
    }

    function decimals() public pure returns(uint8) {
        return 18;
    }

    function totalSupply() public view returns(uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns(uint256) {
        return _balances[account];
    }

    function approve(address spender, uint256 amount) public returns(bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function allowance(address _owner, address spender) public view returns(uint256) {
        return _allowances[_owner][spender];
    }

    function _approve(address _owner, address spender, uint256 amount) internal {
        require(_owner != address(0), "ERC20: approve from zero");
        require(spender != address(0), "ERC20: approve to zero");

        _allowances[_owner][spender] = amount;
        emit Approval(_owner, spender, amount);
    }
}