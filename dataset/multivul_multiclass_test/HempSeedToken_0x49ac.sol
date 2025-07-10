// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title HempSeed Token
 * @dev Implementation of the ERC-20 Token Standard
 * Based on the EBikeToken but improved to allow contract interactions
 */
contract HempSeedToken {
    // Token details
    string public constant name = "HempSeed";
    string public constant symbol = "HEMP";
    uint8 public constant decimals = 18;
    
    // Total supply is 1,000,000 tokens with 18 decimals (1,000,000 * 10^18)
    uint256 private constant _totalSupply = 1000000 * 10**18;
    
    // Balances and allowances storage
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    
    // Events required by the ERC-20 standard
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    /**
     * @dev Constructor that gives the deployer the entire initial supply
     */
    constructor() {
        _balances[msg.sender] = _totalSupply;
        emit Transfer(address(0), msg.sender, _totalSupply);
    }
    
    /**
     * @dev Returns the total supply of tokens
     */
    function totalSupply() external view returns (uint256) {
        return _totalSupply;
    }
    
    /**
     * @dev Returns the balance of the specified address
     * @param account The address to query the balance of
     * @return The amount owned by the passed address
     */
    function balanceOf(address account) external view returns (uint256) {
        return _balances[account];
    }
    
    /**
     * @dev Moves `amount` tokens from the caller's account to `recipient`
     * @param recipient The address to transfer to
     * @param amount The amount to transfer
     * @return A boolean indicating whether the operation succeeded
     */
    function transfer(address recipient, uint256 amount) external returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }
    
    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through transferFrom
     * @param owner The address which owns the funds
     * @param spender The address which will spend the funds
     * @return The amount of tokens still available for the spender
     */
    function allowance(address owner, address spender) external view returns (uint256) {
        return _allowances[owner][spender];
    }
    
    /**
     * @dev Sets `amount` as the allowance of `spender` over the caller's tokens
     * @param spender The address which will spend the funds
     * @param amount The amount of tokens to allow the spender to use
     * @return A boolean indicating whether the operation succeeded
     */
    function approve(address spender, uint256 amount) external returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }
    
    /**
     * @dev Moves `amount` tokens from `sender` to `recipient` using the
     * allowance mechanism. `amount` is then deducted from the caller's allowance.
     * @param sender The address to transfer tokens from
     * @param recipient The address to transfer to
     * @param amount The amount to transfer
     * @return A boolean indicating whether the operation succeeded
     */
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool) {
        // Check allowance
        require(_allowances[sender][msg.sender] >= amount, "HempSeed: transfer amount exceeds allowance");
        
        _transfer(sender, recipient, amount);
        _approve(sender, msg.sender, _allowances[sender][msg.sender] - amount);
        return true;
    }
    
    /**
     * @dev Atomically increases the allowance granted to `spender` by the caller
     * @param spender The address which will spend the funds
     * @param addedValue The amount of tokens to increase the allowance by
     * @return A boolean indicating whether the operation succeeded
     */
    function increaseAllowance(address spender, uint256 addedValue) external returns (bool) {
        _approve(msg.sender, spender, _allowances[msg.sender][spender] + addedValue);
        return true;
    }
    
    /**
     * @dev Atomically decreases the allowance granted to `spender` by the caller
     * @param spender The address which will spend the funds
     * @param subtractedValue The amount of tokens to decrease the allowance by
     * @return A boolean indicating whether the operation succeeded
     */
    function decreaseAllowance(address spender, uint256 subtractedValue) external returns (bool) {
        uint256 currentAllowance = _allowances[msg.sender][spender];
        require(currentAllowance >= subtractedValue, "HempSeed: decreased allowance below zero");
        _approve(msg.sender, spender, currentAllowance - subtractedValue);
        return true;
    }
    
    /**
     * @dev Optional function to check if an address is a contract
     * This is for informational purposes only and NOT used to restrict transfers
     * @param addr The address to check
     * @return bool True if the address is a contract, false otherwise
     */
    function isContract(address addr) external view returns (bool) {
        uint size;
        assembly {
            size := extcodesize(addr)
        }
        return size > 0;
    }
    
    /**
     * @dev Internal transfer function
     * @param sender The address to transfer from
     * @param recipient The address to transfer to
     * @param amount The amount to transfer
     */
    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "HempSeed: transfer from the zero address");
        require(recipient != address(0), "HempSeed: transfer to the zero address");
        require(_balances[sender] >= amount, "HempSeed: transfer amount exceeds balance");
        
        _balances[sender] = _balances[sender] - amount;
        _balances[recipient] = _balances[recipient] + amount;
        emit Transfer(sender, recipient, amount);
    }
    
    /**
     * @dev Internal approve function
     * @param owner The address that approves the spending
     * @param spender The address that can spend the funds
     * @param amount The amount of tokens to allow
     */
    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0), "HempSeed: approve from the zero address");
        require(spender != address(0), "HempSeed: approve to the zero address");
        
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
}