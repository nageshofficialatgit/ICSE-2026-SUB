// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract TokenWithFixedFee {
    string public name = "USD_TEST";
    string public symbol = "USD_TEST";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    uint256 public feeInEth;  // Комиссия в ETH (в wei)
    address public feeCollector;  // Адрес, куда будут отправляться комиссии

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    // Конструктор токена
    constructor(uint256 initialSupply, uint256 _feeInEth, address _feeCollector) {
        totalSupply = initialSupply * (10 ** decimals);  // Умножаем на 10^18 для учета десятичных знаков
        feeInEth = _feeInEth;
        feeCollector = _feeCollector;
        balanceOf[msg.sender] = totalSupply;  // Начальный баланс принадлежит создателю контракта
    }

    // Функция для перевода с фиксированной комиссией
    function transfer(address recipient, uint256 amount) public returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Not enough tokens for transfer");
        
        uint256 feeAmount = feeInEth;  // Комиссия фиксирована
        uint256 transferAmount = amount;  // Сумма для перевода остаётся без изменений

        require(balanceOf[msg.sender] >= amount + feeAmount, "Not enough tokens to cover the fee");

        balanceOf[msg.sender] -= amount + feeAmount;  // Списываем токены (сумма + комиссия)
        balanceOf[recipient] += transferAmount;  // Переводим токены получателю
        balanceOf[feeCollector] += feeAmount;  // Переводим комиссию на адрес сборщика

        emit Transfer(msg.sender, recipient, transferAmount);
        emit Transfer(msg.sender, feeCollector, feeAmount);

        return true;
    }

    // Функция для изменения фиксированной комиссии в ETH (в wei)
    function setFeeInEth(uint256 _feeInEth) external {
        feeInEth = _feeInEth;
    }

    // Функция для изменения адреса сборщика комиссии
    function setFeeCollector(address _feeCollector) external {
        feeCollector = _feeCollector;
    }

    // Стандартные функции ERC-20

    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(balanceOf[sender] >= amount, "Not enough funds");
        require(allowance[sender][msg.sender] >= amount, "Transfer limit exceeded");
        
        uint256 feeAmount = feeInEth;
        uint256 transferAmount = amount;

        require(balanceOf[sender] >= amount + feeAmount, "Not enough funds to cover the fee");

        balanceOf[sender] -= amount + feeAmount;
        balanceOf[recipient] += transferAmount;
        balanceOf[feeCollector] += feeAmount;
        
        allowance[sender][msg.sender] -= amount;

        emit Transfer(sender, recipient, transferAmount);
        emit Transfer(sender, feeCollector, feeAmount);

        return true;
    }

    // События ERC-20
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}