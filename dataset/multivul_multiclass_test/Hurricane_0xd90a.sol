// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract Hurricane {
    struct Deposit {
        address depositor;
        uint256 amount;
        string note;
        bool withdrawn;
    }

    address public owner;
    Deposit[] public deposits;
    uint256[] private allowedAmounts;

    event Deposited(address indexed depositor, uint256 amount, string note);
    event Withdrawn(address indexed depositor, uint256 amount, address to, string note);
    event OwnerWithdrawn(address indexed owner, uint256 amount);
    event AllowedAmountAdded(uint256 amount);
    event AllowedAmountRemoved(uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    constructor() {
        owner = msg.sender;
        allowedAmounts = [0.1 ether, 0.25 ether, 0.5 ether, 1 ether, 5 ether, 10 ether, 20 ether, 50 ether, 100 ether]; // Default allowed amounts
    }

    function deposit(string memory _note) external payable {
        require(isAllowedAmount(msg.value), "Invalid deposit amount");

        deposits.push(Deposit(msg.sender, msg.value, _note, false));
        emit Deposited(msg.sender, msg.value, _note);
    }

    function withdraw(uint256 noteId, address payable _to) external {
        require(noteId < deposits.length, "Invalid note ID");
        Deposit storage userDeposit = deposits[noteId];

        require(msg.sender == userDeposit.depositor, "Not your deposit");
        require(!userDeposit.withdrawn, "Already withdrawn");
        require(_to != address(0), "Invalid withdrawal address");

        userDeposit.withdrawn = true;
        _to.transfer(userDeposit.amount);

        emit Withdrawn(msg.sender, userDeposit.amount, _to, userDeposit.note);
    }

    function ownerWithdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds available");

        payable(owner).transfer(balance);
        emit OwnerWithdrawn(owner, balance);
    }

    function addAllowedAmount(uint256 amount) external onlyOwner {
        require(amount > 0, "Amount must be greater than zero");
        require(!isAllowedAmount(amount), "Amount already allowed");

        allowedAmounts.push(amount);
        emit AllowedAmountAdded(amount);
    }

    function removeAllowedAmount(uint256 amount) external onlyOwner {
        for (uint256 i = 0; i < allowedAmounts.length; i++) {
            if (allowedAmounts[i] == amount) {
                allowedAmounts[i] = allowedAmounts[allowedAmounts.length - 1];
                allowedAmounts.pop();
                emit AllowedAmountRemoved(amount);
                return;
            }
        }
        revert("Amount not found");
    }

    function isAllowedAmount(uint256 amount) internal view returns (bool) {
        for (uint256 i = 0; i < allowedAmounts.length; i++) {
            if (allowedAmounts[i] == amount) {
                return true;
            }
        }
        return false;
    }

    function getAllowedAmounts() external view returns (uint256[] memory) {
        return allowedAmounts;
    }

    function getDeposits() external view returns (Deposit[] memory) {
        return deposits;
    }
}