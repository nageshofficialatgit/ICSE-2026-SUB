// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract BESQ27 {
    // Informações básicas da moeda
    string public name = "BESQ27";
    string public symbol = "BESQ";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    // Endereço do administrador (criador do contrato)
    address public admin;

    // Mapeamento de saldos e permissões
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    // Eventos
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    // Informações do lastro
    string public swiftCode = "BARCGB22XXX";
    string public receiverBank = "CAIXA ECONOMICA FEDERAL";
    string public receiverBankAddress = "AV. PAULISTA 750, SAO PAULO, BRAZIL";
    string public beneficiary = "PATRIMONIAL BESERRA QUEIROZ LTDA";
    string public mur = "AXW4770783157504";
    uint256 public backingAmount = 10_000_000_000 * 10**18; // 10 bilhões de dólares, ajustado para 18 casas decimais

    constructor() {
        admin = msg.sender; // Define o criador do contrato como administrador
        totalSupply = 10_000_000_000 * 10**uint256(decimals); // Suprimento inicial de 10 bilhões de tokens
        balanceOf[msg.sender] = totalSupply; // Todo o suprimento inicial vai para o administrador
    }

    // Função para transferir tokens
    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(_to != address(0), "Invalid address");
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");

        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    // Função para aprovar um gasto por outro endereço
    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    // Função para transferir tokens de outro endereço (com permissão)
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        require(_from != address(0), "Invalid address");
        require(_to != address(0), "Invalid address");
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Allowance exceeded");

        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }

    // Função para verificar os detalhes do lastro
    function getBackingDetails() public view returns (
        string memory, 
        string memory, 
        string memory, 
        string memory, 
        string memory, 
        uint256
    ) {
        return (
            swiftCode,
            receiverBank,
            receiverBankAddress,
            beneficiary,
            mur,
            backingAmount
        );
    }

    // Função para ajustar o valor do lastro (somente administrador)
    function updateBackingAmount(uint256 _newAmount) public {
        require(msg.sender == admin, "Only admin can update the backing amount");
        backingAmount = _newAmount;
    }
}