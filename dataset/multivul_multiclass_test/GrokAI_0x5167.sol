// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title GrokAI Token
 * @dev ERC-20 Token Implementation
 * Token Name: GrokAI
 * Symbol: XAI
 * Total Supply: 100 million tokens
 */

/**
 * @dev ERC-20 Interface
 */
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

/**
 * @dev GrokAI Token Contract
 */
contract GrokAI is IERC20 {
    string public constant name = "GrokAI";
    string public constant symbol = "XAI";
    uint8 public constant decimals = 18;
    uint256 public constant TOTAL_SUPPLY = 100_000_000 * 10**18; // 100 million tokens with 18 decimals

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    /**
     * @dev Constructor: Mints total supply to deployer
     */
    constructor() {
        _balances[msg.sender] = TOTAL_SUPPLY;
        emit Transfer(address(0), msg.sender, TOTAL_SUPPLY);
    }

    /**
     * @dev Returns the total supply of tokens
     */
    function totalSupply() external pure override returns (uint256) {
        return TOTAL_SUPPLY;
    }

    /**
     * @dev Returns the balance of an account
     */
    function balanceOf(address account) external view override returns (uint256) {
        return _balances[account];
    }

    /**
     * @dev Transfers tokens to a specified address
     * @param to Recipient address
     * @param amount Amount to transfer
     */
    function transfer(address to, uint256 amount) external override returns (bool) {
        require(to != address(0), "GrokAI: transfer to zero address");
        require(_balances[msg.sender] >= amount, "GrokAI: insufficient balance");

        _balances[msg.sender] -= amount;
        _balances[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    /**
     * @dev Approves a spender to spend tokens on behalf of the caller
     * @param spender Address allowed to spend tokens
     * @param amount Amount approved for spending
     */
    function approve(address spender, uint256 amount) external override returns (bool) {
        require(spender != address(0), "GrokAI: approve to zero address");

        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    /**
     * @dev Returns the remaining allowance for a spender
     * @param owner Token owner's address
     * @param spender Spender's address
     */
    function allowance(address owner, address spender) external view override returns (uint256) {
        return _allowances[owner][spender];
    }

    /**
     * @dev Transfers tokens from one address to another using an allowance
     * @param from Sender's address
     * @param to Recipient's address
     * @param amount Amount to transfer
     */
    function transferFrom(address from, address to, uint256 amount) external override returns (bool) {
        require(from != address(0), "GrokAI: transfer from zero address");
        require(to != address(0), "GrokAI: transfer to zero address");
        require(_balances[from] >= amount, "GrokAI: insufficient balance");
        require(_allowances[from][msg.sender] >= amount, "GrokAI: insufficient allowance");

        _balances[from] -= amount;
        _balances[to] += amount;
        _allowances[from][msg.sender] -= amount;
        emit Transfer(from, to, amount);
        return true;
    }
}