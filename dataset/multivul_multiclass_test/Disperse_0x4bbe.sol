// SPDX-License-Identifier: MIT

pragma solidity ^0.8.13;

interface IERC20 {
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

contract Disperse {
    bool private locked;
    
    modifier nonReentrant() {
        require(!locked, "Reentrant call");
        locked = true;
        _;
        locked = false;
    }

    function disperseEther(address[] calldata recipients, uint256[] calldata values) external payable nonReentrant {
        for (uint256 i = 0; i < recipients.length; i++)
            recipients[i].call{value: values[i], gas: 87700}("");
        uint256 balance = address(this).balance;
        if (balance > 0)
            payable(msg.sender).transfer(balance);
    }

    function disperseToken(IERC20 token, address[] calldata recipients, uint256[] calldata values, uint256 gasLimit) external nonReentrant {
        for (uint256 i = 0; i < recipients.length; i++) {
            try token.transferFrom{gas: gasLimit}(msg.sender, recipients[i], values[i]) {
            } catch {
            }
        }
    }
}