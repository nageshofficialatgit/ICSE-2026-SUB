// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface GoalGenerator {
    function deposit() external payable;
    function withdraw(uint256 amount) external;
    function getOneConnectToken() external view returns (uint256);
}

contract GoalBonding {
    address public owner;
    GoalGenerator public Manage;
    
    constructor(address _Manage) {
        owner = msg.sender;
        Manage = GoalGenerator(_Manage);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    function invest() external payable onlyOwner {
        require(msg.value > 0, "Investment must be greater than zero");
        Manage.deposit{ value: msg.value }();
    }

    function divest(uint256 amount) external onlyOwner {
        Manage.withdraw(amount);
    }

    function getManagedOneConnectToken() external view returns (uint256) {
        return Manage.getOneConnectToken();
    }
}