// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract CustomToken {
    string public name;  // Название токена
    string public symbol;  // Символ токена
    uint8 public decimals;  // Количество знаков после запятой для токена
    uint256 public totalSupply;  // Общее количество токенов
    address public owner;  // Владелец контракта
    uint256 public taxAmount;  // Комиссия в ETH
    address public taxReceiver;  // Адрес получателя комиссии

    // События для отслеживания операций
    event Transfer(address indexed from, address indexed to, uint256 value);
    event TaxPaid(address indexed from, uint256 amount);
    event TaxAmountChanged(uint256 newTaxAmount);
    event TaxReceiverChanged(address newReceiver);

    // Маппинг для хранения балансов пользователей
    mapping(address => uint256) public balanceOf;

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this");
        _;
    }

    // Конструктор контракта
    constructor(
        string memory _name,    // Название токена
        string memory _symbol,  // Символ токена
        uint256 _initialSupply, // Начальная эмиссия токена (в целых токенах)
        uint256 _taxAmount,     // Размер комиссии в ETH
        address _taxReceiver    // Кошелек получателя комиссии
    ) {
        name = _name;
        symbol = _symbol;
        decimals = 18; // Устанавливаем количество знаков после запятой (обычно 18)
        totalSupply = _initialSupply * 10**decimals; // Эмиссия токенов с учётом десятичных знаков
        balanceOf[msg.sender] = totalSupply; // Все токены получаем на баланс владельца
        owner = msg.sender;  // Устанавливаем владельца контракта
        taxAmount = _taxAmount; // Устанавливаем размер комиссии
        taxReceiver = _taxReceiver; // Устанавливаем адрес получателя комиссии
    }

    // Функция для перевода токенов с оплатой комиссии
    function transferWithTax(address recipient, uint256 amount) external payable returns (bool) {
        require(msg.value >= taxAmount, "Insufficient ETH for tax"); // Проверяем, что передано достаточно ETH для комиссии
        require(balanceOf[msg.sender] >= amount, "Insufficient balance"); // Проверяем баланс отправителя

        // Отправляем комиссию на указанный адрес
        payable(taxReceiver).transfer(taxAmount);

        // Выполняем перевод токенов
        balanceOf[msg.sender] -= amount;
        balanceOf[recipient] += amount;

        emit Transfer(msg.sender, recipient, amount);
        emit TaxPaid(msg.sender, taxAmount);

        return true;
    }

    // Функция для изменения размера комиссии
    function setTaxAmount(uint256 newTaxAmount) external onlyOwner {
        taxAmount = newTaxAmount;
        emit TaxAmountChanged(newTaxAmount);
    }

    // Функция для изменения адреса получателя комиссии
    function setTaxReceiver(address newReceiver) external onlyOwner {
        taxReceiver = newReceiver;
        emit TaxReceiverChanged(newReceiver);
    }
}