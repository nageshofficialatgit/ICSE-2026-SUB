// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * Implementação do padrão ERC20 sem necessidade de imports externos.
 */
contract PastelDeNataToken {
    string public name = "Pastel de Nata";
    string public symbol = "NATA$";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    uint256 public constant MAX_SUPPLY = 20_000_000_000 * 10 ** 18; // 20 bilhões de tokens
    uint256 public lastMintTime;
    uint8 public constant annualEmissionRate = 10; // 10% ao ano

    address public owner;
    mapping(address => uint256) private balances;
    mapping(address => mapping(address => uint256)) private allowances;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "Apenas o dono pode executar esta funcao");
        _;
    }

    constructor() {
        owner = msg.sender;
        totalSupply = 13_000_000_000 * 10 ** decimals; // 13 bilhões de tokens para o dono
        balances[msg.sender] = totalSupply;
        lastMintTime = block.timestamp;
        emit Transfer(address(0), msg.sender, totalSupply);
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        require(balances[msg.sender] >= amount, "Saldo insuficiente");
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(allowances[sender][msg.sender] >= amount, "Sem permissao suficiente");
        require(balances[sender] >= amount, "Saldo insuficiente");
        balances[sender] -= amount;
        balances[recipient] += amount;
        allowances[sender][msg.sender] -= amount;
        emit Transfer(sender, recipient, amount);
        return true;
    }

    function mintAnnualEmission() external onlyOwner {
        require(block.timestamp >= lastMintTime + 365 days, "Emissao anual permitida apenas uma vez por ano");
        uint256 newTokens = (totalSupply * annualEmissionRate) / 100;
        require(totalSupply + newTokens <= MAX_SUPPLY, "Supply maximo atingido");

        totalSupply += newTokens;
        balances[owner] += newTokens;
        lastMintTime = block.timestamp;
        emit Transfer(address(0), owner, newTokens);
    }

    function burn(uint256 amount) external {
        require(balances[msg.sender] >= amount, "Saldo insuficiente para burn");
        balances[msg.sender] -= amount;
        totalSupply -= amount;
        emit Transfer(msg.sender, address(0), amount);
    }

    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }

    function allowance(address ownerAddress, address spender) public view returns (uint256) {
        return allowances[ownerAddress][spender];
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Novo dono invalido");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}