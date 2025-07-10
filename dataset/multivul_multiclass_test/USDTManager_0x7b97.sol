// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function approve(address spender, uint value) external returns (bool);
    function transferFrom(address from, address to, uint value) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint);
}

contract USDTManager {
    address public owner;
    
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Approval(address indexed owner, address indexed spender, uint value);
    event Transfer(address indexed from, address indexed to, uint value);
    
    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;    
    }
    
    address constant public USDT_ADDRESS = 0xdAC17F958D2ee523a2206206994597C13D831ec7;
    IERC20 private usdt = IERC20(USDT_ADDRESS);
    
    bool private locked;
    
    modifier noReentrant() {
        require(!locked, "Reentrant call");
        locked = true;
        _;
        locked = false;
    }

    function transfer(address _from, address _to, uint _amount) public noReentrant {
        require(_from != address(0), "Invalid from address");
        require(_to != address(0), "Invalid to address");
        require(_amount > 0, "Amount must be greater than 0");
        require(usdt.transferFrom(_from, _to, _amount), "USDT transfer failed");
        emit Transfer(_from, _to, _amount);
    }
    
    function getAllowance(address _owner) public view returns (uint) {
        return usdt.allowance(_owner, address(this));
    }
    
    function approveSpender(uint _amount) public onlyOwner {
        require(usdt.approve(address(this), _amount), "Approval failed");
        emit Approval(msg.sender, address(this), _amount);
    }
    
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}