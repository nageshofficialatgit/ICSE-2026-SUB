// SPDX-License-Identifier: MIT
pragma solidity ^0.8.6;

contract LOGGER {
    uint256 public transferCount =0;
    event log(address, address, address);
    function sweepTokenWithFee(
        address token,
        uint256 amountMinimum,
        address recipient,
        uint256 feeBips,
        address feeRecipient
    ) public returns(bool){
        emit log(token, recipient, feeRecipient);
        transferCount++;
        return true;
    }
}