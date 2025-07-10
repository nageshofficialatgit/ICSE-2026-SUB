// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function approve(address spender, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
}

contract MultiRouterPermitHelper {
    function approveToken(address token, address router, uint256 amount) external {
        unchecked {
            require(amount > 0, "Amount must be greater than zero");
            require(router != address(0), "Invalid router address");
        }
        IERC20(token).approve(router, amount);
    }

    function revokeApproval(address token, address router) external {
        require(router != address(0), "Invalid router address");
        IERC20(token).approve(router, 0);
    }

    function checkApproval(address token, address router, address owner) external view returns (uint256) {
        return IERC20(token).allowance(owner, router);
    }

    function checkMyApproval(address token, address router) external view returns (uint256) {
        return IERC20(token).allowance(msg.sender, router);
    }
}