// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MyToken {
    string private _name;
    string private _symbol;
    uint256 private _totalSupply;
    address public owner;
    
    // 添加流动性锁定参数
    address public liquidityPair;
    uint256 public lockEndTime;
    bool public liquidityLocked;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _blacklist;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event BlacklistUpdated(address indexed account, bool isBlacklisted);
    event LiquidityLocked(address indexed pairAddress, uint256 unlockTime);

    modifier onlyOwner() {
        require(msg.sender == owner, "Owner only");
        _;
    }

    constructor(string memory name_, string memory symbol_, uint256 totalSupply_) {
        _name = name_;
        _symbol = symbol_;
        _totalSupply = totalSupply_ * 10**18; // 自动添加18位小数
        _balances[msg.sender] = _totalSupply;
        owner = msg.sender;
    }

    // 设置流动性对并锁定（只能调用一次）
    function setLiquidityPair(address pairAddress) external onlyOwner {
        require(!liquidityLocked, "Already locked");
        liquidityPair = pairAddress;
        lockEndTime = block.timestamp + 2 days;
        liquidityLocked = true;
        emit LiquidityLocked(pairAddress, lockEndTime);
    }

    // 自动生效的流动性锁定检查
    modifier liquidityLockCheck(address from) {
        if(from == liquidityPair) {
            require(block.timestamp >= lockEndTime, "Liquidity locked");
        }
        _;
    }

    // 黑名单管理功能
    function addToBlacklist(address account) public onlyOwner {
        _blacklist[account] = true;
        emit BlacklistUpdated(account, true);
    }

    function removeFromBlacklist(address account) public onlyOwner {
        _blacklist[account] = false;
        emit BlacklistUpdated(account, false);
    }

    // 代币转账功能
    function transfer(address to, uint256 amount) 
        public
        liquidityLockCheck(msg.sender)
        returns (bool)
    {
        require(!_blacklist[msg.sender], "Sender blacklisted");
        require(!_blacklist[to], "Recipient blacklisted");
        _transfer(msg.sender, to, amount);
        return true;
    }

    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) public liquidityLockCheck(from) returns (bool) {
        require(!_blacklist[from], "Sender blacklisted");
        require(!_blacklist[to], "Recipient blacklisted");
        
        _spendAllowance(from, msg.sender, amount);
        _transfer(from, to, amount);
        return true;
    }

    // 标准ERC20函数
    function name() public view returns (string memory) {
        return _name;
    }

    function symbol() public view returns (string memory) {
        return _symbol;
    }

    function decimals() public pure returns (uint8) {
        return 18;
    }

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function allowance(address _owner, address spender) public view returns (uint256) {
        return _allowances[_owner][spender];
    }

    // 内部函数
    function _transfer(
        address from,
        address to,
        uint256 amount
    ) internal {
        require(from != address(0), "From zero address");
        require(to != address(0), "To zero address");

        _balances[from] -= amount;
        _balances[to] += amount;
        emit Transfer(from, to, amount);
    }

    function _approve(
        address _owner,
        address spender,
        uint256 amount
    ) internal {
        require(_owner != address(0), "Approve from zero");
        require(spender != address(0), "Approve to zero");

        _allowances[_owner][spender] = amount;
        emit Approval(_owner, spender, amount);
    }

    function _spendAllowance(
        address _owner,
        address spender,
        uint256 amount
    ) internal {
        uint256 currentAllowance = allowance(_owner, spender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "Insufficient allowance");
            _approve(_owner, spender, currentAllowance - amount);
        }
    }
}