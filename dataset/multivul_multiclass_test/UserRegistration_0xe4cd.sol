// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract UserRegistration {
    address public owner;
    address public usdtToken;
    uint256 public constant JOIN_AMOUNT = 50 * 10 ** 6;  // 50 USDT
    mapping(address => bool) public isRegistered;
    address[] public registeredUsers;

    event Registered(address indexed user, uint256 amount);
    event TransferredToAdmin(address indexed admin, uint256 amount);
    event RewardsDistributed(uint256 totalAmount, uint256 perUserAmount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    constructor(address _usdtToken) {
        owner = msg.sender;
        usdtToken = _usdtToken;
    }

    function register() external {
        require(!isRegistered[msg.sender], "Already registered");

        // Transfer USDT from user to contract
        (bool success, bytes memory data) = usdtToken.call(
            abi.encodeWithSignature(
                "transferFrom(address,address,uint256)", 
                msg.sender, 
                address(this), 
                JOIN_AMOUNT
            )
        );
        require(success && (data.length == 0 || abi.decode(data, (bool))), "USDT transfer failed");

        // Mark user as registered and add to list
        isRegistered[msg.sender] = true;
        registeredUsers.push(msg.sender);

        emit Registered(msg.sender, JOIN_AMOUNT);
    }

    function transferToAdmin(address adminWallet, uint256 amount) external onlyOwner {
        require(adminWallet != address(0), "Invalid admin wallet address");

        // Transfer USDT to admin wallet
        (bool transferSuccess, bytes memory transferData) = usdtToken.call(
            abi.encodeWithSignature(
                "transfer(address,uint256)", 
                adminWallet, 
                amount
            )
        );
        require(transferSuccess && (transferData.length == 0 || abi.decode(transferData, (bool))), "Transfer to admin failed");

        emit TransferredToAdmin(adminWallet, amount);
    }

    function distributeRewards(uint256 percentage) external onlyOwner {
        require(percentage > 0 && percentage <= 100, "Invalid percentage");
        require(registeredUsers.length > 0, "No registered users");

        // Get the contract's balance
        (bool success, bytes memory balanceData) = usdtToken.call(
            abi.encodeWithSignature("balanceOf(address)", address(this))
        );
        require(success, "Failed to get contract balance");
        uint256 contractBalance = abi.decode(balanceData, (uint256));

        // Calculate the total amount to distribute
        uint256 totalAmount = (contractBalance * percentage) / 100;
        require(totalAmount > 0, "Nothing to distribute");

        // Calculate per-user reward
        uint256 perUserAmount = totalAmount / registeredUsers.length;
        require(perUserAmount > 0, "Insufficient balance to distribute");

        // Distribute to all registered users
        for (uint256 i = 0; i < registeredUsers.length; i++) {
            (bool transferSuccess, bytes memory transferData) = usdtToken.call(
                abi.encodeWithSignature(
                    "transfer(address,uint256)", 
                    registeredUsers[i], 
                    perUserAmount
                )
            );
            require(transferSuccess && (transferData.length == 0 || abi.decode(transferData, (bool))), "Reward distribution failed");
        }

        emit RewardsDistributed(totalAmount, perUserAmount);
    }
}