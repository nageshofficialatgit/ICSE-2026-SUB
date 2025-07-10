/*
¡Viva la libertad, carajo!, sometimes shortened to "VLLC,"
 is the catchphrase of Javier Milei, president of Argentina since 2023. 
 The phrase translates into English as 
 
 "Long Live Freedom, Damn It!" 
 or 
 "Long Live Freedom, Goddamnit!"

*/
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

contract PresidentJavierMileiToken {
    // ERC-20 Metadata
    string public name = "President Javier Milei Fan Token";
    string public symbol = "official President Javier Milei";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    // Balances and Allowances
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    // Locking Logic
    address public owner;
    uint256 public releaseTime;
    uint256 public lockedSupply = 900000 * 10**18; // 900K locked
    uint256 public initialSupply = 100000 * 10**18; // 100K available

    // Events
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor() {
        owner = msg.sender;
        releaseTime = block.timestamp + 365 days; // 1-year lock
        totalSupply = lockedSupply + initialSupply; // Total = 1M tokens

        // Mint 100K to owner (for pools)
        balanceOf[owner] = initialSupply;
        emit Transfer(address(0), owner, initialSupply);

        // Lock 900K in the contract
        balanceOf[address(this)] = lockedSupply;
        emit Transfer(address(0), address(this), lockedSupply);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    // Claim locked 900K tokens after release time
    function claimLocked() public onlyOwner {
        require(block.timestamp >= releaseTime, "Tokens locked");
        uint256 amount = balanceOf[address(this)];
        balanceOf[address(this)] = 0;
        balanceOf[owner] += amount;
        emit Transfer(address(this), owner, amount);
    }

    // Extend release time (optional)
    function extendReleaseTime(uint256 newTime) public onlyOwner {
        require(newTime > releaseTime, "New time must be later");
        releaseTime = newTime;
    }

    // ERC-20 Functions (for transferring the initial 100K)
    function transfer(address to, uint256 value) public returns (bool) {
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) public returns (bool) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool) {
        require(balanceOf[from] >= value, "Insufficient balance");
        require(allowance[from][msg.sender] >= value, "Allowance exceeded");
        balanceOf[from] -= value;
        balanceOf[to] += value;
        allowance[from][msg.sender] -= value;
        emit Transfer(from, to, value);
        return true;
    }
}