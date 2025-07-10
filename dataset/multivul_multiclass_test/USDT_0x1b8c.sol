// SPDX-License-Identifier: MIT
pragma solidity ^0.8.7;

contract USDT {
    // Información del token
    string public constant name = "USDT";
    string public constant symbol = "USDT";
    uint8 public constant decimals = 6;
    uint256 public totalSupply;
    
    // Mappings para almacenar balances y asignaciones
    mapping(address => uint256) private balances;
    mapping(address => mapping(address => uint256)) private allowed;
    
    // Eventos
    event Transfer(address indexed from, address indexed to, uint256 amount);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    constructor() {
        totalSupply = 67_532_951 * (10 ** uint256(decimals));
        balances[msg.sender] = totalSupply;
        emit Transfer(address(0), msg.sender, totalSupply);
    }

    // Modificador para verificar saldo suficiente
    modifier enoughBalance(address wallet, uint256 amount) {
        require(balances[wallet] >= amount, "Saldo insuficiente");
        _;
    }

    // Modificador para verificar asignación suficiente
    modifier enoughAllowance(address owner, address spender, uint256 amount) {
        require(allowed[owner][spender] >= amount, "Asignacion insuficiente");
        _;
    }

    // Función para consultar balance
    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }

    // Función para consultar asignación
    function allowance(address owner, address spender) public view returns (uint256) {
        return allowed[owner][spender];
    }

    // Función para transferir tokens
    function transfer(address recipient, uint256 amount) 
        public 
        enoughBalance(msg.sender, amount) 
        returns (bool) 
    {
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    // Función para aprobar asignación
    function approve(address spender, uint256 amount) public returns (bool) {
        allowed[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    // Función para transferir desde otra dirección
    function transferFrom(address owner, address recipient, uint256 amount) 
        public 
        enoughBalance(owner, amount) 
        enoughAllowance(owner, msg.sender, amount) 
        returns (bool) 
    {
        balances[owner] -= amount;
        allowed[owner][msg.sender] -= amount;
        balances[recipient] += amount;
        emit Transfer(owner, recipient, amount);
        return true;
    }
}