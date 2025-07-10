// SPDX-License-Identifier: Apache.2.0
pragma solidity ^0.8.28;

/// @title Epoche - A Blockchain Resource for Information Transfer Without Economic Value
/// @notice Epoche encodes text into 20-digit message blocks, allowing up to 3 blocks per transaction.
/// It uses unlimited minting and omits 'approve' and 'transferFrom' functions to prevent economic
/// value and exchange listing. Designed for app-internal use.
/// @author S.E.
/// @dev For more information or to report issues, please visit: https://github.com/sewing848/epoche

contract Epoche {
    // Public decriptive constants
    string public constant RESOURCE_NAME = "Epoche";
    string public constant RESOURCE_SYMBOL = "EPOCHE";

    // Message blocks are encoded with digits without reference to a decimal point
    uint8 public constant RESOURCE_DECIMALS = 0;

    // Total supply is capped only by the value (2^256 - 1) but distribution rate is limited by faucet settings
    uint256 public totalSupply;

    // Mapping to store balances of each address
    mapping(address => uint256) public balanceOf;

    // The owner of the contract, with special privileges
    address public owner;

    // Variable that sets a maximum transfer amount
    uint256 public currentMaxTransferLimit;

    // Constant that sets a lower bound for the maximum transfer amount restriction
    uint256 public constant MIN_MAX_TRANSFER_AMOUNT = 10**20 - 1;

    // Constant that sets the upper bound for the maximum transfer amount restriction
    // This allows transfers of up to 72 decimal digits
    uint256 public constant MAX_MAX_TRANSFER_AMOUNT = 10**72 - 1;

    // Variables related to the faucet functionality
    uint256 public faucetAmount;
    uint256 public constant MAX_FAUCET_AMOUNT = 10**64;
    
    uint256 public faucetWaitingPeriod = 1 days;

    // Mapping to store the last request time of each address
    mapping(address => uint256) public lastRequestTime;

    // Events to log various contract activities
    event Transfer(address indexed from, address indexed to, uint256 value);
    event OwnershipTransferred(address indexed from, address indexed to, uint256 timestamp);
    event FaucetWaitingPeriodChanged(uint256 oldPeriod, uint256 newPeriod, uint256 timestamp);
    event FaucetAmountChanged(uint256 oldAmount, uint256 newAmount, uint256 timestamp);
    event MaxTransferLimitChanged(uint256 oldLimit, uint256 newLimit, uint256 timestamp);

    constructor() {
        owner = msg.sender;

        // Initial supply is zero.
        totalSupply = 0;

        // Set initially to allow for up to three 20-digit message blocks per transaction.
        currentMaxTransferLimit = 10**60 - 1;

        // Set initially to allow one faucet draw to provide enough Epoche for at least 1,000 message blocks.
        faucetAmount = 10**63;
    }

    // Allows some functions to be accessible only to the contract owner.
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner.");
        _;
    }

    /// @notice Transfer Epoche to another address.
    /// @dev Intentionally omits 'approve' and 'transferFrom' to prevent exchange listing.
    /// @param to The address to transfer Epoche to.
    /// @param amount The amount of Epoche to transfer.
    /// @return bool Returns true if the transfer was successful.
    function transfer(address to, uint256 amount) public returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        require(amount <= currentMaxTransferLimit, "Transfer amount exceeds max");
        require(to != msg.sender, "Cannot transfer to self");
        require(to != address(this), "Cannot transfer to Epoche contract");
 
        unchecked {
            balanceOf[msg.sender] -= amount;
            balanceOf[to] += amount;
        }

        emit Transfer(msg.sender, to, amount);
        return true;
    }

    /// @notice Request Epoche from the faucet.
    /// @dev Unlimited supply is intended to prevent economic value appreciation.
    function faucet() public {
        require(block.timestamp >= lastRequestTime[msg.sender] + faucetWaitingPeriod, "Waiting period not yet passed");
        unchecked {
            totalSupply += faucetAmount;
            balanceOf[msg.sender] += faucetAmount;
        }
        lastRequestTime[msg.sender] = block.timestamp;
        emit Transfer(address(0), msg.sender, faucetAmount); // Transfer event with address(0) to indicate minting.
    }

    /// @notice Set the waiting period for faucet requests.
    /// @param _period The new waiting period in seconds.
    function setFaucetWaitingPeriod(uint256 _period) public onlyOwner {
        uint256 oldPeriod = faucetWaitingPeriod;
        faucetWaitingPeriod = _period;
        emit FaucetWaitingPeriodChanged(oldPeriod, _period, block.timestamp);
    }

    /// @notice Set the amount of Epoche distributed by the faucet.
    /// @param newFaucetAmount The new amount to be distributed by the faucet.
    function setFaucetAmount(uint256 newFaucetAmount) public onlyOwner {
        require(newFaucetAmount <= MAX_FAUCET_AMOUNT, "Cannot exceed 10^64");
        uint256 oldAmount = faucetAmount;
        faucetAmount = newFaucetAmount;
        emit FaucetAmountChanged(oldAmount, newFaucetAmount, block.timestamp);
    }

    /// @notice Set the maximum transfer limit.
    /// @param newMaxTransferLimit The new maximum transfer limit.
    function setMaxTransferLimit(uint256 newMaxTransferLimit) public onlyOwner {
        require(newMaxTransferLimit >= MIN_MAX_TRANSFER_AMOUNT, "Must be at least 10^20 - 1");
        require(newMaxTransferLimit <= MAX_MAX_TRANSFER_AMOUNT, "Cannot exceed 10^72 - 1");
        uint256 oldLimit = currentMaxTransferLimit;
        currentMaxTransferLimit = newMaxTransferLimit;
        emit MaxTransferLimitChanged(oldLimit, newMaxTransferLimit, block.timestamp);
    }

    /// @notice Transfer ownership of the contract.
    /// @param newOwner The address of the new owner.
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Cannot use zero address");
        owner = newOwner;
        emit OwnershipTransferred(msg.sender, newOwner, block.timestamp);
    }
}

/*

 Copyright 2024 S.E.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/