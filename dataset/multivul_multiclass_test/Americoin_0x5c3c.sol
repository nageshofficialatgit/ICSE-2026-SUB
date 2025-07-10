// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract Americoin {
    string public name = "Americoin";
    string public symbol = "AMCO";
    uint8 public decimals = 18;
    uint256 public totalSupply = 10_000_000_000 * 10**uint256(decimals);

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    address public owner;
    bool public paused;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Burn(address indexed burner, uint256 value);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Paused();
    event Unpaused();

    constructor() {
        owner = msg.sender;
        balanceOf[msg.sender] = totalSupply; // Assign total supply to contract creator
        paused = false; // Contract is not paused by default
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can perform this action");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }

    modifier validAddress(address _address) {
        require(_address != address(0), "Invalid address");
        _;
    }

    // Transfer tokens
    function transfer(address _to, uint256 _value) public validAddress(_to) whenNotPaused returns (bool success) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");

        unchecked {
            balanceOf[msg.sender] -= _value;
            balanceOf[_to] += _value;
        }

        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    // Approve tokens for another address
    function approve(address _spender, uint256 _value) public validAddress(_spender) whenNotPaused returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    // Transfer tokens from one address to another using allowance
    function transferFrom(address _from, address _to, uint256 _value) public validAddress(_from) validAddress(_to) whenNotPaused returns (bool success) {
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Allowance exceeded");

        unchecked {
            balanceOf[_from] -= _value;
            balanceOf[_to] += _value;
            allowance[_from][msg.sender] -= _value;
        }

        emit Transfer(_from, _to, _value);
        return true;
    }

    // Increase allowance for a spender
    function increaseAllowance(address _spender, uint256 _addedValue) public validAddress(_spender) whenNotPaused returns (bool success) {
        allowance[msg.sender][_spender] += _addedValue;
        emit Approval(msg.sender, _spender, allowance[msg.sender][_spender]);
        return true;
    }

    // Decrease allowance for a spender
    function decreaseAllowance(address _spender, uint256 _subtractedValue) public validAddress(_spender) whenNotPaused returns (bool success) {
        uint256 currentAllowance = allowance[msg.sender][_spender];
        require(currentAllowance >= _subtractedValue, "ERC20: decreased allowance below zero");

        unchecked {
            allowance[msg.sender][_spender] = currentAllowance - _subtractedValue;
        }

        emit Approval(msg.sender, _spender, allowance[msg.sender][_spender]);
        return true;
    }

    // Burn tokens from sender's account
    function burn(uint256 _value) public whenNotPaused returns (bool success) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance to burn");

        unchecked {
            balanceOf[msg.sender] -= _value;
            totalSupply -= _value;
        }

        emit Burn(msg.sender, _value);
        return true;
    }

    // Mint new tokens (only the owner can mint)
    function mint(address _to, uint256 _value) public onlyOwner validAddress(_to) returns (bool success) {
        totalSupply += _value;
        balanceOf[_to] += _value;

        emit Transfer(address(0), _to, _value);
        return true;
    }

    // Pause the contract (only the owner can pause)
    function pause() public onlyOwner returns (bool success) {
        paused = true;
        emit Paused();
        return true;
    }

    // Unpause the contract (only the owner can unpause)
    function unpause() public onlyOwner returns (bool success) {
        paused = false;
        emit Unpaused();
        return true;
    }

    // Transfer ownership
    function transferOwnership(address newOwner) public onlyOwner validAddress(newOwner) returns (bool success) {
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
        return true;
    }
}