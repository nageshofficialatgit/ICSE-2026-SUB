// SPDX-License-Identifier: MIT
pragma solidity 0.8.26;


interface Secret {
    function key() external view returns (string memory);
    function validateBankNotEmpty() external returns (bool);
}

/**
 *   Description:
 *   Simple Bank application, users can deposit funds and withdraw it later.
 *   Only externally owned accounts (EOA) are permitted to interact with the application.
 */
contract Bank {
    event SecretKey(string key);

    mapping(address => uint) public balances;

    Secret private secret;

    constructor(address _secret) payable {
        secret = Secret(_secret);
        require(msg.value >= 1 wei, "At least 1 wei is required to deploy the bank!");
    }

    function deposit(uint amount) external payable {
        require(msg.value == amount, "Insufficient funds");
        balances[msg.sender] += amount;
    }

    function withdraw() external payable {
        uint balance = balances[msg.sender];
        require(balance > 0, "Nothing to withdraw");

        require(secret.validateBankNotEmpty());
        (bool sent, ) = msg.sender.call{value: balance}("");
        require(sent, "Failed to send Ether");
        balances[msg.sender] = 0;
        
        require(address(this).balance == 0);

        string memory secretKey = secret.key();
        emit SecretKey(secretKey);
    }
}