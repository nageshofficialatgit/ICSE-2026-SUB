// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Define the IERC20 interface
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

// Implementing the ERC20 contract
contract MyUSDTToken is IERC20 {
    string public constant name = "usd";
    string public constant symbol = "USDT";
    uint8 public constant decimals = 18;  // USDT uses 18 decimals

    uint256 private _totalSupply;
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    constructor(uint256 initialSupply) {
        _totalSupply = initialSupply * 10**decimals;
        _balances[msg.sender] = _totalSupply;
    }

    // Returns the total supply of tokens
    function totalSupply() external view override returns (uint256) {
        return _totalSupply;
    }

    // Returns the balance of a specific address
    function balanceOf(address account) external view override returns (uint256) {
        return _balances[account];
    }

    // Transfers tokens to a recipient address
    function transfer(address recipient, uint256 amount) external override returns (bool) {
        require(recipient != address(0), "ERC20: transfer to the zero address");
        require(_balances[msg.sender] >= amount, "ERC20: transfer amount exceeds balance");

        _balances[msg.sender] -= amount;
        _balances[recipient] += amount;

        emit Transfer(msg.sender, recipient, amount);  // Emit the Transfer event
        return true;
    }

    // Approve a spender to spend tokens on behalf of the sender
    function approve(address spender, uint256 amount) external override returns (bool) {
        require(spender != address(0), "Cannot authorize an empty account");

        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);  // Emit the Approval event
        return true;
    }

    // Returns the amount of tokens that a spender is allowed to spend on behalf of an owner
    function allowance(address owner, address spender) external view override returns (uint256) {
        return _allowances[owner][spender];
    }

    // Transfers tokens from one address to another on behalf of the sender
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external override returns (bool) {
        require(sender != address(0), "Cannot send from empty account");
        require(recipient != address(0), "Cannot send to empty account");
        require(_balances[sender] >= amount, "ERC20: transfer amount exceeds balance");
        require(_allowances[sender][msg.sender] >= amount, "ERC20: transfer amount exceeds allowance");

        _balances[sender] -= amount;
        _balances[recipient] += amount;

        _allowances[sender][msg.sender] -= amount;

        emit Transfer(sender, recipient, amount);  // Emit the Transfer event
        return true;
    }

    // Events to log transfers and approvals
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}