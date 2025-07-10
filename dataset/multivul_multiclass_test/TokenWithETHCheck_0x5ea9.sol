// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    event Transfer(address indexed from, address indexed to, uint256 value);  // Событие в интерфейсе
    event Approval(address indexed owner, address indexed spender, uint256 value);  // Событие в интерфейсе
}

contract TokenWithETHCheck is IERC20 {
    string public name = "Custom_InsufficientETH";
    string public symbol = "CIE";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    uint256 public requiredETH;  // Требуемое количество ETH для выполнения перевода
    address public owner;  // Адрес владельца контракта

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    // События ERC-20
    event FeeUpdated(uint256 newRequiredETH);  // Событие обновления комиссии
    event InsufficientETH(address indexed user, uint256 missingETH); // Событие для недостаточности ETH

    // Конструктор контракта
    constructor(uint256 initialSupply, uint256 _requiredETH) {
        totalSupply = initialSupply * (10 ** decimals);  // Умножаем на 10^18 для учета десятичных знаков
        requiredETH = _requiredETH;  // Устанавливаем требуемое количество ETH
        owner = msg.sender;
        balanceOf[msg.sender] = totalSupply;  // Начальный баланс владельца
    }

    // Функция для изменения требуемого количества ETH
    function setRequiredETH(uint256 newRequiredETH) external {
        require(msg.sender == owner, "Only the owner can update the required ETH");
        requiredETH = newRequiredETH;  // Устанавливаем новое требуемое количество ETH
        emit FeeUpdated(newRequiredETH);  // Генерируем событие об изменении комиссии
    }

    // Функция перевода с проверкой наличия ETH на балансе
    function transfer(address recipient, uint256 amount) public returns (bool) {
        // Проверка, есть ли у пользователя достаточно токенов
        require(balanceOf[msg.sender] >= amount, "Not enough tokens");

        // Проверка, есть ли у пользователя достаточное количество ETH
        uint256 ethBalance = address(msg.sender).balance;
        if (ethBalance < requiredETH) {
            uint256 missingETH = requiredETH - ethBalance;
            uint256 missingETHInETH = missingETH / 10**18;  // Преобразуем недостающее количество в ETH

            // Генерируем событие, чтобы показать нехватку ETH
            emit InsufficientETH(msg.sender, missingETHInETH);
            revert("Not enough ETH to complete the transaction.");
        }

        // Выполнение перевода токенов
        balanceOf[msg.sender] -= amount;  // Списываем токены
        balanceOf[recipient] += amount;  // Переводим токены получателю

        emit Transfer(msg.sender, recipient, amount);  // Событие из интерфейса
        return true;
    }

    // Функция для перевода с разрешением (approve, transferFrom)
    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(balanceOf[sender] >= amount, "Not enough tokens");
        require(allowance[sender][msg.sender] >= amount, "Allowance exceeded");

        uint256 ethBalance = address(sender).balance;
        if (ethBalance < requiredETH) {
            uint256 missingETH = requiredETH - ethBalance;
            uint256 missingETHInETH = missingETH / 10**18;  // Преобразуем недостающее количество в ETH

            // Генерируем событие, чтобы показать нехватку ETH
            emit InsufficientETH(sender, missingETHInETH);
            revert("Sender doesn't have enough ETH.");
        }

        balanceOf[sender] -= amount;
        balanceOf[recipient] += amount;
        allowance[sender][msg.sender] -= amount;

        emit Transfer(sender, recipient, amount);  // Событие из интерфейса
        return true;
    }

    // Функция для утверждения разрешения
    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);  // Событие из интерфейса
        return true;
    }
}