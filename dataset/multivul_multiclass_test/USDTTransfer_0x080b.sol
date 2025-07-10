// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}

contract USDTTransfer {
    address public owner;
    IERC20 public usdtToken;

    event TransferSuccessful(address indexed to, uint256 amount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    constructor(address _usdtAddress) {
        owner = msg.sender;
        usdtToken = IERC20(_usdtAddress); // USDT contract address
    }

    function transferUSDT(address recipient, uint256 amount) public onlyOwner {
        require(usdtToken.balanceOf(address(this)) >= amount, "Insufficient contract balance");
        require(usdtToken.transfer(recipient, amount), "USDT transfer failed");

        emit TransferSuccessful(recipient, amount);
    }

    function withdrawAllUSDT(address recipient) public onlyOwner {
        uint256 balance = usdtToken.balanceOf(address(this));
        require(balance > 0, "No USDT available");
        require(usdtToken.transfer(recipient, balance), "USDT withdrawal failed");

        emit TransferSuccessful(recipient, balance);
    }

    function getContractBalance() public view returns (uint256) {
        return usdtToken.balanceOf(address(this));
    }

    function changeOwner(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}