// SPDX-License-Identifier: MIT
pragma solidity ^0.8.18;

interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

library TransferHelper {
    function safeTransferETH(address to, uint256 value) internal {
        (bool success, ) = to.call{value: value}("");
        require(success, "ETH transfer failed");
    }

    function safeTransferWithoutRequire(address token, address to, uint256 value) internal {
        (bool success, ) = token.call(abi.encodeWithSelector(IERC20.transfer.selector, to, value));
        require(success, "Token transfer failed");
    }

    function isETH(address token) internal pure returns (bool) {
        return (token == address(0));
    }
}

contract Disperse {
    address public executor; 

    event Withdraw(address indexed token, address indexed from, address indexed to, uint256 amount);

    constructor() {
        executor = msg.sender;
    }

    modifier onlyExecutor() {
        require(msg.sender == executor, "Only executor can call this function");
        _;
    }

    function disperseEther(address payable[] memory recipients, uint256[] memory values) external payable {
        for (uint256 i = 0; i < recipients.length; i++)
            recipients[i].transfer(values[i]);
        uint256 balance = address(this).balance;
        if (balance > 0)
            payable(msg.sender).transfer(balance);
    }

    function disperseToken(IERC20 token, address[] memory recipients, uint256[] memory values) external {
        uint256 total = 0;
        for (uint256 i = 0; i < recipients.length; i++)
            total += values[i];
        require(token.transferFrom(msg.sender, address(this), total));
        for (uint256 i = 0; i < recipients.length; i++)
            require(token.transfer(recipients[i], values[i]));
    }

    function disperseTokenSimple(IERC20 token, address[] memory recipients, uint256[] memory values) external {
        for (uint256 i = 0; i < recipients.length; i++)
            require(token.transferFrom(msg.sender, recipients[i], values[i]));
    }

    function withdrawTokens(address[] memory tokens, address recipient) external onlyExecutor {
        for (uint256 index = 0; index < tokens.length; index++) {  
            uint256 amount;
            if (TransferHelper.isETH(tokens[index])) {
                amount = address(this).balance;
                TransferHelper.safeTransferETH(recipient, amount);
            } else {
                amount = IERC20(tokens[index]).balanceOf(address(this));
                TransferHelper.safeTransferWithoutRequire(tokens[index], recipient, amount);
            }
            emit Withdraw(tokens[index], msg.sender, recipient, amount);
        }
    }
}