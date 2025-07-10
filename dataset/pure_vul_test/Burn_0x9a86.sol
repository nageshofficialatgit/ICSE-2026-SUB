// SPDX-License-Identifier: Apache 2.0
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);    
}

contract Burn {
    function burn(address tokenAddress, uint256 amount) external {
        require(amount > 0, "Amount must be greater than zero");

        // Pull tokens from the sender to this contract
        bool success = IERC20(tokenAddress).transferFrom(msg.sender, address(this), amount);
        require(success, "Transfer to contract failed");

        // Burn tokwns by sending to the zero address
        success = IERC20(tokenAddress).transfer(address(0), amount);
        require(success, "Burn failed");
    }
}