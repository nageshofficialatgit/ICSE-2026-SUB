// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract TetherUSD {
    string public name = "Tether USD";
    string public symbol = "USDT";
    uint8 public decimals = 6; 
    uint256 public totalSupply = 100_000_000 * 10**6; 
    address public owner;
    bool public paused = false;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => bool) public blacklisted;
    mapping(address => bool) public lockedExchanges;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Mint(address indexed to, uint256 value);
    event Burn(address indexed from, uint256 value);
    event Paused();
    event Unpaused();
    event Blacklisted(address indexed account);
    event Unblacklisted(address indexed account);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event ExchangeLocked(address indexed exchange);
    event ExchangeUnlocked(address indexed exchange);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }

    modifier notBlacklisted(address account) {
        require(!blacklisted[account], "Address is blacklisted");
        _;
    }

    modifier notLockedExchange(address account) {
        require(!lockedExchanges[account], "Address is locked from trading");
        _;
    }

    constructor() {
        owner = msg.sender;
        balanceOf[msg.sender] = totalSupply; 
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function transfer(address _to, uint256 _value) public whenNotPaused notBlacklisted(msg.sender) notBlacklisted(_to) notLockedExchange(_to) returns (bool success) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public whenNotPaused notBlacklisted(msg.sender) returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public whenNotPaused notBlacklisted(_from) notBlacklisted(_to) notLockedExchange(_to) returns (bool success) {
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Allowance exceeded");
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }

    function mint(address _to, uint256 _value) public onlyOwner {
        totalSupply += _value;
        balanceOf[_to] += _value;
        emit Mint(_to, _value);
        emit Transfer(address(0), _to, _value);
    }

    function burn(uint256 _value) public whenNotPaused notBlacklisted(msg.sender) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        balanceOf[msg.sender] -= _value;
        totalSupply -= _value;
        emit Burn(msg.sender, _value);
        emit Transfer(msg.sender, address(0), _value);
    }

    function pause() public onlyOwner {
        paused = true;
        emit Paused();
    }

    function unpause() public onlyOwner {
        paused = false;
        emit Unpaused();
    }

    function blacklist(address _account) public onlyOwner {
        blacklisted[_account] = true;
        emit Blacklisted(_account);
    }

    function unblacklist(address _account) public onlyOwner {
        blacklisted[_account] = false;
        emit Unblacklisted(_account);
    }

    function lockExchange(address _exchange) public onlyOwner {
        lockedExchanges[_exchange] = true;
        emit ExchangeLocked(_exchange);
    }

    function unlockExchange(address _exchange) public onlyOwner {
        lockedExchanges[_exchange] = false;
        emit ExchangeUnlocked(_exchange);
    }
}