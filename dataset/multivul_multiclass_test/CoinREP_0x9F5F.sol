// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CoinREP {
    string public name = "CoinREP";    // Name des Tokens
    string public symbol = "REP+";     // Symbol des Tokens
    uint8 public decimals = 18;        // Dezimalstellen (Standardwert für ERC-20)
    uint256 public totalSupply = 5000000 * 10 ** uint256(decimals);  // Gesamtangebot: 5 Millionen
    
    // Mapping von Adressen zu Kontoständen
    mapping(address => uint256) public balanceOf;
    // Mapping von Adressen zu zugelassenen Ausgaben (Allowance)
    mapping(address => mapping(address => uint256)) public allowance;
    
    // Der Eigentümer des Vertrags
    address public owner;
    
    // Event für Token-Transfer
    event Transfer(address indexed from, address indexed to, uint256 value);
    // Event für Genehmigungen
    event Approval(address indexed owner, address indexed spender, uint256 value);
    // Event für das Ändern der Allowance
    event AllowanceUpdated(address indexed owner, address indexed spender, uint256 oldAllowance, uint256 newAllowance);
    
    // Modifier zur Verhinderung von Reentrancy-Angriffen
    bool private _locked;
    modifier nonReentrant() {
        require(!_locked, "No reentrancy allowed");
        _locked = true;
        _;
        _locked = false;
    }

    // Modifier, um nur dem Eigentümer Zugriff zu gewähren
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }
    
    // Constructor: Initialisiert den Vertrag und weist alle Tokens dem Eigentümer zu
    constructor() {
        owner = msg.sender;
        balanceOf[msg.sender] = totalSupply;  // Zuweisung aller Tokens an den Ersteller des Contracts
    }

    // Übertragungsfunktion
    function transfer(address _to, uint256 _value) public nonReentrant returns (bool success) {
        require(_value > 0, "Amount must be greater than 0");
        require(_to != address(0), "Invalid address");
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");

        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;

        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    // Funktion zur Genehmigung eines Ausgebers (approve)
    function approve(address _spender, uint256 _value) public nonReentrant returns (bool success) {
        require(_spender != address(0), "Invalid spender address");
        
        uint256 oldAllowance = allowance[msg.sender][_spender];
        allowance[msg.sender][_spender] = _value;

        emit Approval(msg.sender, _spender, _value);
        emit AllowanceUpdated(msg.sender, _spender, oldAllowance, _value);  // Event für die Allowance-Änderung
        return true;
    }

    // Funktion zur Übertragung von Tokens im Namen des Besitzers (transferFrom)
    function transferFrom(address _from, address _to, uint256 _value) public nonReentrant returns (bool success) {
        require(_from != address(0) && _to != address(0), "Invalid address");
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Allowance exceeded");

        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;

        emit Transfer(_from, _to, _value);
        return true;
    }
    
    // Funktion zur Überprüfung der Allowance
    function allowanceOf(address _owner, address _spender) public view returns (uint256) {
        return allowance[_owner][_spender];
    }
}