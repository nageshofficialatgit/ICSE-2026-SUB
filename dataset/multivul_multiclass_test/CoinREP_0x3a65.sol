// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CoinREP {
    string public name = "CoinREP";    // Name deines Tokens
    string public symbol = "REP+";     // Symbol deines Tokens
    uint8 public decimals = 18;        // Dezimalstellen (Standardwert für ERC-20)
    uint256 public totalSupply = 5000000 * 10 ** uint256(decimals);  // Gesamtangebot: 5 Millionen

    mapping(address => uint256) public balanceOf;  // Balances der Adressen
    mapping(address => mapping(address => uint256)) public allowance;  // Genehmigungen (für Delegationen)

    constructor() {
        balanceOf[msg.sender] = totalSupply;  // Zuweisung aller Tokens an den Ersteller des Contracts
    }

    // Überweisung von Tokens
    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[msg.sender] >= _value, "Nicht genug Balance");
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    // Genehmigung für eine Adresse, eine bestimmte Anzahl von Tokens zu übertragen
    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    // Überweisung von Tokens im Auftrag eines anderen
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[_from] >= _value, "Nicht genug Balance");
        require(allowance[_from][msg.sender] >= _value, "Nicht erlaubt");
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}