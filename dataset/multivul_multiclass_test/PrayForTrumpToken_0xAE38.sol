// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract PrayForTrumpToken {
    // Token details
    string public constant name = "PRAY FOR TRUMP";
    string public constant symbol = "PFT";
    uint8 public constant decimals = 18;
    uint256 public constant totalSupply = 1_000_000_000_000 * 10**uint256(decimals); // 1 trillion tokens

    // Balances and allowances
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    // Reentrancy guard
    bool private _locked;

    // Events
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    // Modifier to prevent reentrancy
    modifier noReentrancy() {
        require(!_locked, "Reentrant call");
        _locked = true;
        _;
        _locked = false;
    }

    // Constructor to mint the total supply to the deployer
    constructor() {
        _balances[msg.sender] = totalSupply;
        emit Transfer(address(0), msg.sender, totalSupply);
    }

    // Returns the balance of an account
    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    // Transfers tokens from the sender to another account
    function transfer(address to, uint256 amount) public noReentrancy returns (bool) {
        require(to != address(0), "ERC20: transfer to the zero address");
        require(_balances[msg.sender] >= amount, "ERC20: transfer amount exceeds balance");

        _balances[msg.sender] -= amount;
        _balances[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    // Approves another account to spend tokens on behalf of the sender
    function approve(address spender, uint256 amount) public noReentrancy returns (bool) {
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    // Returns the remaining number of tokens that `spender` is allowed to spend on behalf of `owner`
    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowances[owner][spender];
    }

    // Transfers tokens from one account to another using an allowance
    function transferFrom(address from, address to, uint256 amount) public noReentrancy returns (bool) {
        require(to != address(0), "ERC20: transfer to the zero address");
        require(_balances[from] >= amount, "ERC20: transfer amount exceeds balance");
        require(_allowances[from][msg.sender] >= amount, "ERC20: transfer amount exceeds allowance");

        // Update the allowance before transferring tokens
        _allowances[from][msg.sender] -= amount;

        // Update balances
        _balances[from] -= amount;
        _balances[to] += amount;

        emit Transfer(from, to, amount);
        return true;
    }
}