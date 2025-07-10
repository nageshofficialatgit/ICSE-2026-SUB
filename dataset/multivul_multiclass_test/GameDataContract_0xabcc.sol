// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Интерфейс для OGGPToken
interface IOGGPToken {
    function mint(address to, uint256 amount) external returns (bool);
}

/**
 * @title GameDataContract
 * @dev Контракт для отслеживания игровых данных в Web3 Tetris
 */
contract GameDataContract {
    // Структура для хранения игровых данных
    struct GameData {
        uint256 highScore;
        uint256 totalScore;
        uint256 gamesPlayed;
        uint256 totalLinesCleared;
        uint256 oggPoints;
        uint256 lastUpdated;
    }
    
    // Маппинг адресов кошельков к игровым данным
    mapping(address => GameData) private playerData;
    
    // Массив адресов игроков для отслеживания лидерборда
    address[] private players;
    
    // Маппинг для проверки, существует ли игрок
    mapping(address => bool) private playerExists;
    
    // Адрес токена OGGP
    address private oggpTokenAddress;
    
    // События
    event GameDataUpdated(address indexed player, uint256 highScore, uint256 totalScore, uint256 gamesPlayed, uint256 totalLinesCleared, uint256 oggPoints);
    event OggPointsAwarded(address indexed player, uint256 amount, string reason);
    event GameResultMinted(address indexed player, uint256 score, uint256 linesCleared, uint256 oggPoints);
    
    // Модификатор для проверки, что вызывающий - владелец контракта
    address private owner;
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    // Конструктор
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev Установка адреса токена OGGP
     * @param _oggpTokenAddress Адрес токена OGGP
     */
    function setOggpTokenAddress(address _oggpTokenAddress) external onlyOwner {
        oggpTokenAddress = _oggpTokenAddress;
    }
    
    /**
     * @dev Получение игровых данных игрока
     * @param player Адрес кошелька игрока
     * @return GameData Игровые данные
     */
    function getPlayerData(address player) public view returns (GameData memory) {
        return playerData[player];
    }
    
    /**
     * @dev Обновление игровых данных после игровой сессии
     * @param player Адрес кошелька игрока
     * @param score Счет в игровой сессии
     * @param linesCleared Количество очищенных линий
     */
    function updateGameData(address player, uint256 score, uint256 linesCleared) public onlyOwner {
        // Если игрок новый, добавляем его в массив
        if (!playerExists[player]) {
            players.push(player);
            playerExists[player] = true;
        }
        
        GameData storage data = playerData[player];
        
        // Обновление данных
        if (score > data.highScore) {
            data.highScore = score;
        }
        data.totalScore += score;
        data.gamesPlayed += 1;
        data.totalLinesCleared += linesCleared;
        
        // Начисление OGG-поинтов (10 за каждую очищенную линию)
        uint256 newPoints = linesCleared * 10;
        data.oggPoints += newPoints;
        
        // Обновление времени последнего обновления
        data.lastUpdated = block.timestamp;
        
        // Вызов события
        emit GameDataUpdated(player, data.highScore, data.totalScore, data.gamesPlayed, data.totalLinesCleared, data.oggPoints);
        emit OggPointsAwarded(player, newPoints, "Game session completed");
    }
    
    /**
     * @dev Минтинг результата игры (вызывается игроком после завершения игры)
     * @param score Счет в игровой сессии
     * @param linesCleared Количество очищенных линий
     * @return bool Успешность операции
     */
    function mintGameResult(uint256 score, uint256 linesCleared) external returns (bool) {
        require(linesCleared > 0, "No lines cleared");
        
        // Расчет OGG-поинтов (10 за каждую очищенную линию)
        uint256 oggPoints = linesCleared * 10;
        
        // Если адрес токена OGGP установлен, минтим токены
        if (oggpTokenAddress != address(0)) {
            IOGGPToken oggpToken = IOGGPToken(oggpTokenAddress);
            oggpToken.mint(msg.sender, oggPoints);
        }
        
        // Обновляем данные игрока
        if (!playerExists[msg.sender]) {
            players.push(msg.sender);
            playerExists[msg.sender] = true;
        }
        
        GameData storage data = playerData[msg.sender];
        
        // Обновление данных
        if (score > data.highScore) {
            data.highScore = score;
        }
        data.totalScore += score;
        data.gamesPlayed += 1;
        data.totalLinesCleared += linesCleared;
        data.oggPoints += oggPoints;
        data.lastUpdated = block.timestamp;
        
        // Вызов события
        emit GameResultMinted(msg.sender, score, linesCleared, oggPoints);
        
        return true;
    }
    
    /**
     * @dev Начисление OGG-поинтов игроку
     * @param player Адрес кошелька игрока
     * @param amount Количество поинтов
     * @param reason Причина начисления
     */
    function awardOggPoints(address player, uint256 amount, string memory reason) public onlyOwner {
        // Если игрок новый, добавляем его в массив
        if (!playerExists[player]) {
            players.push(player);
            playerExists[player] = true;
            playerData[player].lastUpdated = block.timestamp;
        }
        
        // Начисление поинтов
        playerData[player].oggPoints += amount;
        
        // Если адрес токена OGGP установлен, минтим токены
        if (oggpTokenAddress != address(0)) {
            IOGGPToken oggpToken = IOGGPToken(oggpTokenAddress);
            oggpToken.mint(player, amount);
        }
        
        // Вызов события
        emit OggPointsAwarded(player, amount, reason);
    }
    
    /**
     * @dev Списание OGG-поинтов у игрока (например, при минтинге NFT)
     * @param player Адрес кошелька игрока
     * @param amount Количество поинтов
     * @param reason Причина списания
     */
    function spendOggPoints(address player, uint256 amount, string memory reason) public onlyOwner {
        require(playerExists[player], "Player does not exist");
        require(playerData[player].oggPoints >= amount, "Insufficient OGG points");
        
        // Списание поинтов
        playerData[player].oggPoints -= amount;
        
        // Вызов события
        emit OggPointsAwarded(player, amount, reason);
    }
    
    /**
     * @dev Получение топ-N игроков по очкам
     * @param n Количество игроков для возврата
     * @return addresses Массив адресов
     * @return scores Массив очков
     */
    function getTopPlayers(uint256 n) public view returns (address[] memory addresses, uint256[] memory scores) {
        // Определение размера массива для возврата
        uint256 length = n;
        if (players.length < n) {
            length = players.length;
        }
        
        // Создание временных массивов для сортировки
        address[] memory tempAddresses = new address[](players.length);
        uint256[] memory tempScores = new uint256[](players.length);
        
        // Заполнение временных массивов
        for (uint256 i = 0; i < players.length; i++) {
            tempAddresses[i] = players[i];
            tempScores[i] = playerData[players[i]].highScore;
        }
        
        // Сортировка (простой пузырьковый алгоритм)
        for (uint256 i = 0; i < players.length; i++) {
            for (uint256 j = i + 1; j < players.length; j++) {
                if (tempScores[j] > tempScores[i]) {
                    // Обмен очков
                    uint256 tempScore = tempScores[i];
                    tempScores[i] = tempScores[j];
                    tempScores[j] = tempScore;
                    
                    // Обмен адресов
                    address tempAddress = tempAddresses[i];
                    tempAddresses[i] = tempAddresses[j];
                    tempAddresses[j] = tempAddress;
                }
            }
        }
        
        // Создание массивов для возврата
        addresses = new address[](length);
        scores = new uint256[](length);
        
        // Заполнение массивов для возврата
        for (uint256 i = 0; i < length; i++) {
            addresses[i] = tempAddresses[i];
            scores[i] = tempScores[i];
        }
        
        return (addresses, scores);
    }
    
    /**
     * @dev Получение количества игроков
     * @return uint256 Количество игроков
     */
    function getPlayerCount() public view returns (uint256) {
        return players.length;
    }
    
    /**
     * @dev Изменение владельца контракта
     * @param newOwner Адрес нового владельца
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        owner = newOwner;
    }
}