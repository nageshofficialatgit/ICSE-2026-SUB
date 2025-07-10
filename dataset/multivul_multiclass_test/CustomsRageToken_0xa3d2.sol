// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CustomsRageToken {
    string public name = "Customs Rage";
    string public symbol = "TARIFF";
    uint8 public decimals = 18;
    uint public totalSupply = 10000000 * 10 ** uint(decimals);
    mapping(address => uint) public balanceOf;
    address public owner;
    address public customsVault;

    constructor(address _customsVault) {
        owner = msg.sender;
        customsVault = _customsVault;
        balanceOf[msg.sender] = totalSupply;
    }

    function transfer(address to, uint amount) public returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Not enough tokens");

        uint feeBurn = amount * 3 / 100;
        uint feeVault = amount * 2 / 100;
        uint totalFee = feeBurn + feeVault;
        uint amountAfterFee = amount - totalFee;

        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amountAfterFee;
        balanceOf[customsVault] += feeVault;
        totalSupply -= feeBurn;

        return true;
    }
}