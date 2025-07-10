// SPDX-License-Identifier: MIT
// i88cash.io || Under Maintenance.
pragma solidity ^0.8.0;

contract i88Stable {
    string public name = "i88";
    string public symbol = "CASH";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    address public treasury;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    constructor(address _treasury) {
        treasury = _treasury;
    }

    modifier i88Function() {
        require(msg.sender == treasury, "Not authorized");
        _;
    }

    function mint(address account, uint256 amount) external i88Function {
        totalSupply += amount;
        balanceOf[account] += amount;
    }

    function burn(address account, uint256 amount) external i88Function {
        totalSupply -= amount;
        balanceOf[account] -= amount;
    }

    function transfer(address recipient, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[recipient] += amount;
        return true;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool) {
        require(balanceOf[sender] >= amount, "Insufficient balance");
        require(allowance[sender][msg.sender] >= amount, "Allowance exceeded");
        balanceOf[sender] -= amount;
        balanceOf[recipient] += amount;
        allowance[sender][msg.sender] -= amount;
        return true;
    }
}