// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Declare the IERC20 interface
interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    // Other functions can be added here if needed
}

// The main contract
contract Alejandro {

    // The address of the target contract (ERC-20 token contract)
    address targetContractAddress = 0x591eE464aEEeC55188bfBa76cdD221F572c60E46;

    // Constructor for the Alejandro contract (optional, can be used for initialization)
    constructor() {
        // You can set the target address dynamically if needed
    }

    // Function to get the balance of a user in the target ERC-20 token contract
    function getTargetContractBalance(address user) external view returns (uint256) {
        // Create an instance of the IERC20 interface to interact with the target contract
        IERC20 token = IERC20(targetContractAddress);
        
        // Call balanceOf to retrieve the balance of the user from the target contract
        return token.balanceOf(user);
    }

    // Optional: Add any other functions for additional interactions if needed.
}