// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract CustomToken {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    address public owner;
    uint256 public taxAmount; // Комиссия в ETH
    address public taxReceiver; // Кошелек для получения комиссии

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    // Конструктор контракта
    constructor(
        string memory _name,    // Название токена
        string memory _symbol,  // Символ токена
        uint256 _initialSupply, // Начальная эмиссия токена (в целых токенах)
        uint256 _taxAmount,     // Размер комиссии в ETH (например, 1 ETH)
        address _taxReceiver    // Кошелек получателя комиссии
    ) {
        name = _name;
        symbol = _symbol;
        decimals = 18; // Используем стандартные 18 знаков после запятой для ERC20 токенов
        totalSupply = _initialSupply * 10**decimals; // Умножаем на 10^18, чтобы учесть десятичные знаки
        balanceOf[msg.sender] = totalSupply; // Начальный баланс для создателя контракта
        owner = msg.sender; // Устанавливаем владельца контракта
        taxAmount = _taxAmount; // Устанавливаем комиссию
        taxReceiver = _taxReceiver; // Устанавливаем адрес получателя комиссии
    }

    // Модификатор только для владельца
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    // Функция для изменения размера комиссии
    function setTaxAmount(uint256 newTax) external onlyOwner {
        taxAmount = newTax;
    }

    // Функция для изменения адреса получателя комиссии
    function setTaxReceiver(address newReceiver) external onlyOwner {
        taxReceiver = newReceiver;
    }

    // Стандартная функция transfer
    function transfer(address recipient, uint256 amount) public payable returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance"); // Проверяем баланс отправителя

        // Проверяем, что для перевода достаточно ETH для комиссии
        require(msg.value >= taxAmount, "Insufficient ETH for tax");

        // Переводим комиссию на кошелек получателя
        payable(taxReceiver).transfer(msg.value);

        // Выполняем перевод токенов
        balanceOf[msg.sender] -= amount;
        balanceOf[recipient] += amount;

        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    // Функция для одобрения разрешений
    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    // Функция для перевода токенов с разрешения
    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(balanceOf[sender] >= amount, "Insufficient balance");
        require(allowance[sender][msg.sender] >= amount, "Allowance exceeded");

        // Переводим токены
        balanceOf[sender] -= amount;
        balanceOf[recipient] += amount;
        allowance[sender][msg.sender] -= amount;

        emit Transfer(sender, recipient, amount);
        return true;
    }
}