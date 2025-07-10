/**
 *Submitted for verification at Etherscan.io on 2025-02-14
*/

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CoinREP {
    string public name = "CoinREP"; 
    string public symbol = "REP+";  
    uint8 public decimals = 18;     
    uint256 public totalSupply = 5000000 * 10 ** uint256(decimals); 
    
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    // Besitzer des Contracts
    address public owner;

    // Event für die Übertragung des Besitzes
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    // Konstruktor: Setze den Ersteller des Contracts als Besitzer
    constructor() {
        owner = msg.sender;  // Setze die Adresse, die den Vertrag bereitstellt, als Owner
        balanceOf[msg.sender] = totalSupply; // Zuweisung aller Tokens an den Ersteller des Contracts
    }

    // Modifikator: Nur der Besitzer des Contracts kann bestimmte Funktionen aufrufen
    modifier onlyOwner() {
        require(msg.sender == owner, "You are not the owner");
        _;
    }

    // Modifikator: Verhindert Reentrancy-Angriffe
    modifier nonReentrant() {
        bool _entered = false;
        require(!_entered, "Reentrancy detected!");
        _entered = true;
        _;
        _entered = false;
    }

    // Funktion zur Übertragung des Besitzes
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    // Transfer von Tokens: Überprüft, dass der Absender genug Tokens hat
    function transfer(address to, uint256 amount) public nonReentrant returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        
        // Subtrahiere vom Sender und addiere zum Empfänger
        balanceOf[msg.sender] -= amount; 
        balanceOf[to] += amount; 
        
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    // Genehmigung von einem Absender für einen Spender, um Tokens zu übertragen
    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    // Transfer von Tokens im Namen eines anderen (über die Genehmigung)
    function transferFrom(address from, address to, uint256 amount) public nonReentrant returns (bool) {
        require(balanceOf[from] >= amount, "Insufficient balance");
        require(allowance[from][msg.sender] >= amount, "Allowance exceeded");

        // Subtrahiere vom Sender und addiere zum Empfänger
        balanceOf[from] -= amount; 
        balanceOf[to] += amount; 
        
        // Subtrahiere die Genehmigung
        allowance[from][msg.sender] -= amount; 
        
        emit Transfer(from, to, amount);
        return true;
    }
}