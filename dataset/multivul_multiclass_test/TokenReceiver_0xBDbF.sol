// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function balanceOf(address account) external view returns (uint);
    function transfer(address recipient, uint amount) external returns (bool);
}

contract TokenReceiver {
    event ReceivedETH(address indexed sender, uint amount);
    event ReceivedToken(address indexed token, address indexed from, uint amount);

    receive() external payable {
        emit ReceivedETH(msg.sender, msg.value);
    }

    fallback() external payable {
        emit ReceivedETH(msg.sender, msg.value);
    }

    function notifyTokenReceived(address token, address from, uint amount) public {
        emit ReceivedToken(token, from, amount);
    }

    function getTokenBalance(address tokenAddress) public view returns (uint) {
        return IERC20(tokenAddress).balanceOf(address(this));
    }
}