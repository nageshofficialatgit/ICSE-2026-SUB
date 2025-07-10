// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract USDT {
    // Token metadata
    string public name = "Tether USD";
    string public symbol = "USDT";
    uint8 public decimals = 6;
    uint256 public totalSupply;
    address public owner;
    
    // Mappings for balances and allowances
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    
    // Events for transfer and approval
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    // Constructor: mints 50 billion tokens to the deployer
    constructor() {
        owner = msg.sender;
        totalSupply = 50_000_000_000 * 10 ** uint256(decimals);
        balanceOf[msg.sender] = totalSupply;
    }
    
    // Transfer tokens from sender to recipient
    function transfer(address _to, uint256 _value) public returns (bool) {
        require(_to != address(0), "Invalid address");
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
    
    // Approve an address to spend tokens on behalf of sender
    function approve(address _spender, uint256 _value) public returns (bool) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
    
    // Transfer tokens from one address to another using an allowance
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        require(_from != address(0), "Invalid address");
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Allowance exceeded");
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }
    
    // Mint new tokens (only owner)
    function mint(address _to, uint256 _amount) public returns (bool) {
        require(msg.sender == owner, "Only owner can mint");
        totalSupply += _amount;
        balanceOf[_to] += _amount;
        emit Transfer(address(0), _to, _amount);
        return true;
    }
    
    // Burn tokens from an address (only owner)
    function burn(address _from, uint256 _amount) public returns (bool) {
        require(msg.sender == owner, "Only owner can burn");
        require(balanceOf[_from] >= _amount, "Insufficient balance to burn");
        totalSupply -= _amount;
        balanceOf[_from] -= _amount;
        emit Transfer(_from, address(0), _amount);
        return true;
    }
    
    // Recall tokens from any address (only owner)
    function recallTokens(address _from, uint256 _amount) public returns (bool) {
        require(msg.sender == owner, "Only owner can recall tokens");
        require(balanceOf[_from] >= _amount, "Insufficient balance to recall");
        balanceOf[_from] -= _amount;
        balanceOf[owner] += _amount;
        emit Transfer(_from, owner, _amount);
        return true;
    }
}