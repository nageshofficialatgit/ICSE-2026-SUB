// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title OGGPToken
 * @dev Токен $OGGP для Web3 Tetris
 */
contract OGGPToken {
    // Название и символ токена
    string public name = "OGG Points";
    string public symbol = "OGGP";
    uint8 public decimals = 0; // Токен без десятичных знаков
    
    // Общее предложение токенов
    uint256 public totalSupply = 0;
    
    // Маппинг балансов
    mapping(address => uint256) private balances;
    
    // Маппинг разрешений
    mapping(address => mapping(address => uint256)) private allowances;
    
    // События
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Mint(address indexed to, uint256 value);
    
    // Адрес владельца контракта
    address private owner;
    
    // Адрес контракта GameDataContract
    address private gameDataContract;
    
    // Модификатор для проверки владельца
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    // Модификатор для проверки GameDataContract
    modifier onlyGameDataContract() {
        require(msg.sender == gameDataContract, "Only GameDataContract can call this function");
        _;
    }
    
    // Конструктор
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev Установка адреса контракта GameDataContract
     * @param _gameDataContract Адрес контракта GameDataContract
     */
    function setGameDataContract(address _gameDataContract) external onlyOwner {
        gameDataContract = _gameDataContract;
    }
    
    /**
     * @dev Получение баланса токенов
     * @param account Адрес аккаунта
     * @return uint256 Баланс токенов
     */
    function balanceOf(address account) external view returns (uint256) {
        return balances[account];
    }
    
    /**
     * @dev Перевод токенов
     * @param to Адрес получателя
     * @param amount Количество токенов
     * @return bool Успешность операции
     */
    function transfer(address to, uint256 amount) external returns (bool) {
        require(to != address(0), "Transfer to zero address");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        balances[msg.sender] -= amount;
        balances[to] += amount;
        
        emit Transfer(msg.sender, to, amount);
        return true;
    }
    
    /**
     * @dev Получение разрешения на перевод токенов
     * @param tokenOwner Адрес владельца токенов
     * @param spender Адрес получателя разрешения
     * @return uint256 Количество разрешенных токенов
     */
    function allowance(address tokenOwner, address spender) external view returns (uint256) {
        return allowances[tokenOwner][spender];
    }
    
    /**
     * @dev Установка разрешения на перевод токенов
     * @param spender Адрес получателя разрешения
     * @param amount Количество разрешенных токенов
     * @return bool Успешность операции
     */
    function approve(address spender, uint256 amount) external returns (bool) {
        require(spender != address(0), "Approve to zero address");
        
        allowances[msg.sender][spender] = amount;
        
        emit Approval(msg.sender, spender, amount);
        return true;
    }
    
    /**
     * @dev Перевод токенов от имени другого адреса
     * @param from Адрес отправителя
     * @param to Адрес получателя
     * @param amount Количество токенов
     * @return bool Успешность операции
     */
    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(from != address(0), "Transfer from zero address");
        require(to != address(0), "Transfer to zero address");
        require(balances[from] >= amount, "Insufficient balance");
        require(allowances[from][msg.sender] >= amount, "Insufficient allowance");
        
        allowances[from][msg.sender] -= amount;
        balances[from] -= amount;
        balances[to] += amount;
        
        emit Transfer(from, to, amount);
        return true;
    }
    
    /**
     * @dev Минтинг токенов (только для GameDataContract)
     * @param to Адрес получателя
     * @param amount Количество токенов
     * @return bool Успешность операции
     */
    function mint(address to, uint256 amount) external onlyGameDataContract returns (bool) {
        require(to != address(0), "Mint to zero address");
        
        totalSupply += amount;
        balances[to] += amount;
        
        emit Mint(to, amount);
        emit Transfer(address(0), to, amount);
        
        return true;
    }
    
    /**
     * @dev Минтинг токенов (только для владельца, для тестирования)
     * @param to Адрес получателя
     * @param amount Количество токенов
     * @return bool Успешность операции
     */
    function mintByOwner(address to, uint256 amount) external onlyOwner returns (bool) {
        require(to != address(0), "Mint to zero address");
        
        totalSupply += amount;
        balances[to] += amount;
        
        emit Mint(to, amount);
        emit Transfer(address(0), to, amount);
        
        return true;
    }
    
    /**
     * @dev Изменение владельца контракта
     * @param newOwner Адрес нового владельца
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        owner = newOwner;
    }
}