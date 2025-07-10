// SPDX-License-Identifier: MIT
pragma solidity ^0.8.7;

contract MijnContract {
    address public eigenaar;
    uint256 public ontvangenBedrag;

    // Payable constructor om Ether te ontvangen tijdens de deployment
    constructor() payable {
        eigenaar = msg.sender;
        ontvangenBedrag = msg.value;
    }

    // Functie om het saldo van het contract op te vragen
    function getSaldo() public view returns (uint256) {
        return address(this).balance;
    }

    // Functie om Ether over te maken naar een specifiek adres
    function stuurEther(address payable ontvanger, uint256 bedrag) public {
        require(msg.sender == eigenaar, "Alleen de eigenaar kan Ether versturen");
        require(bedrag <= address(this).balance, "Onvoldoende saldo in het contract");
        ontvanger.transfer(bedrag);
    }
}