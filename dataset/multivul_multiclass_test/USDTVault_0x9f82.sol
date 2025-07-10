// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract USDTVault {
    address public owner;
    IERC20 public usdtToken;

    constructor(address _usdtToken) {
        owner = msg.sender;
        usdtToken = IERC20(_usdtToken);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    function withdrawUSDT(address recipient, uint256 amount) external onlyOwner {
        require(usdtToken.balanceOf(address(this)) >= amount, "Insufficient balance");
        require(usdtToken.transfer(recipient, amount), "Transfer failed");
    }

    function getBalance() external view returns (uint256) {
        return usdtToken.balanceOf(address(this));
    }
}