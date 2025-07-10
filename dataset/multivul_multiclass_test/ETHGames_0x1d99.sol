// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IUniswapV2Pair {
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
    function token0() external view returns (address);
    function token1() external view returns (address);
}

/**
 * @title ETHGames
 * @dev A collection of mini-games that pay out ETH rewards when solved
 */
contract ETHGames {
    address public owner;
    
    // Constants
    uint256 public constant SMALL_REWARD = 0.00025 ether;
    uint256 public constant LARGE_REWARD = 0.00075 ether;
    address public constant UNISWAP_V2_ETH_USDC_PAIR = 0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc;
    address public constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address public constant USDC = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48;
    
    // Game state tracking
    struct GameState {
        uint256 winCount;
        uint256 maxWins;
        bool enabled;
    }
    
    mapping(uint256 => GameState) public games;
    mapping(address => uint256) public depositBlocks;
    
    // Reentrancy guard
    bool private _locked;
    
    // Events
    event GameWon(uint256 indexed gameId, address winner, uint256 reward);
    event GameEnabled(uint256 indexed gameId);
    event AllGamesEnabled();
    event Deposited(address depositor, uint256 amount);
    event Withdrawn(address withdrawer, uint256 amount);
    
    modifier nonReentrant() {
        require(!_locked, "ReentrancyGuard: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "ETHGames: caller is not the owner");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        
        // Initialize games
        games[1] = GameState({winCount: 0, maxWins: 2, enabled: true}); // 2+2=4 game
        games[2] = GameState({winCount: 0, maxWins: 2, enabled: true}); // Price proximity game
        games[3] = GameState({winCount: 0, maxWins: 2, enabled: true}); // x^3=125 game
        games[4] = GameState({winCount: 0, maxWins: 2, enabled: true}); // Deposit/withdraw game
    }
    
    /**
     * @dev Game 1: Solve 2+2
     * @param answer The answer to submit
     */
    function solveAddition(uint256 answer) external nonReentrant {
        require(games[1].enabled, "Game 1 is not enabled");
        require(games[1].winCount < games[1].maxWins, "Game 1 has reached max wins");
        
        require(answer == 2+2, "Wrong answer to 2+2");
        
        // Update game state
        games[1].winCount++;
        if (games[1].winCount >= games[1].maxWins) {
            games[1].enabled = false;
        }
        
        // Pay reward
        (bool success, ) = msg.sender.call{value: SMALL_REWARD}("");
        require(success, "ETHGames: reward transfer failed");
        
        emit GameWon(1, msg.sender, SMALL_REWARD);
    }
    
    /**
     * @dev Game 2: Proximity to Uniswap ETH/USDC price
     * @param priceGuess The price guess in USDC per ETH (scaled by 1e6, as USDC has 6 decimals)
     */
    function guessPrice(uint256 priceGuess) external nonReentrant {
        require(games[2].enabled, "Game 2 is not enabled");
        require(games[2].winCount < games[2].maxWins, "Game 2 has reached max wins");
        
        // Get current ETH/USDC price from Uniswap V2
        uint256 currentPrice = getEthUsdcPrice();
        
        // Calculate percentage difference (scaled by 1e18)
        uint256 diff;
        if (priceGuess > currentPrice) {
            diff = priceGuess - currentPrice;
        } else {
            diff = currentPrice - priceGuess;
        }
        
        uint256 percentDiff = (diff * 1e18) / currentPrice;
        
        // Calculate reward based on proximity
        // If perfectly accurate: 100% of LARGE_REWARD
        // Formula: reward = (100% - |distance %|) * LARGE_REWARD
        uint256 percentAccuracy = 1e18 - percentDiff;
        if (percentAccuracy < 0) {
            percentAccuracy = 0;
        }
        
        uint256 reward = (LARGE_REWARD * percentAccuracy) / 1e18;
        
        // Update game state
        games[2].winCount++;
        if (games[2].winCount >= games[2].maxWins) {
            games[2].enabled = false;
        }
        
        // Pay reward
        (bool success, ) = msg.sender.call{value: reward}("");
        require(success, "ETHGames: reward transfer failed");
        
        emit GameWon(2, msg.sender, reward);
    }
    
    /**
     * @dev Game 3: Solve x^3 = 125
     * @param answer The answer to submit
     */
    function solveCube(uint256 answer) external nonReentrant {
        require(games[3].enabled, "Game 3 is not enabled");
        require(games[3].winCount < games[3].maxWins, "Game 3 has reached max wins");
        
        require(answer * answer * answer == 125, "Wrong answer to x^3=125");
        
        // Update game state
        games[3].winCount++;
        if (games[3].winCount >= games[3].maxWins) {
            games[3].enabled = false;
        }
        
        // Pay reward
        (bool success, ) = msg.sender.call{value: SMALL_REWARD}("");
        require(success, "ETHGames: reward transfer failed");
        
        emit GameWon(3, msg.sender, SMALL_REWARD);
    }
    
    /**
     * @dev Game 4: Deposit 0.00025 ETH, withdraw 0.00075 ETH in the next block
     */
    function depositETH() external payable nonReentrant {
        require(games[4].enabled, "Game 4 is not enabled");
        require(games[4].winCount < games[4].maxWins, "Game 4 has reached max wins");
        
        require(msg.value == SMALL_REWARD, "Must deposit exactly 0.00025 ETH");
        
        // Record the block number of the deposit
        depositBlocks[msg.sender] = block.number;
        
        emit Deposited(msg.sender, msg.value);
    }
    
    /**
     * @dev Game 4 (continuation): Withdraw after depositing
     */
    function withdrawETH() external nonReentrant {
        require(games[4].enabled, "Game 4 is not enabled");
        require(depositBlocks[msg.sender] > 0, "No deposit found");
        require(block.number >= depositBlocks[msg.sender] + 1, "Must withdraw after at least one block");
        
        // Reset deposit block record
        depositBlocks[msg.sender] = 0;
        
        // Update game state
        games[4].winCount++;
        if (games[4].winCount >= games[4].maxWins) {
            games[4].enabled = false;
        }
        
        // Pay reward
        (bool success, ) = msg.sender.call{value: LARGE_REWARD}("");
        require(success, "ETHGames: reward transfer failed");
        
        emit GameWon(4, msg.sender, LARGE_REWARD);
        emit Withdrawn(msg.sender, LARGE_REWARD);
    }
    
    /**
     * @dev Enable a specific game
     * @param gameId The ID of the game to enable
     */
    function enableGame(uint256 gameId) external onlyOwner {
        require(gameId >= 1 && gameId <= 4, "Invalid game ID");
        
        games[gameId].enabled = true;
        games[gameId].winCount = 0;
        
        emit GameEnabled(gameId);
    }
    
    /**
     * @dev Enable all games
     */
    function enableAllGames() external onlyOwner {
        for (uint256 i = 1; i <= 4; i++) {
            games[i].enabled = true;
            games[i].winCount = 0;
        }
        
        emit AllGamesEnabled();
    }
    
    /**
     * @dev Get the current ETH/USDC price from Uniswap V2
     * @return price in USDC per ETH, scaled by 1e6 (USDC decimals)
     */
    function getEthUsdcPrice() public view returns (uint256) {
        IUniswapV2Pair pair = IUniswapV2Pair(UNISWAP_V2_ETH_USDC_PAIR);
        
        // Get token order
        address token0 = pair.token0();
        
        // Get reserves
        (uint112 reserve0, uint112 reserve1, ) = pair.getReserves();
        
        // Calculate price based on token order
        if (token0 == WETH) {
            // Price = USDC / WETH (scaled by 1e6 for USDC decimals)
            return (uint256(reserve1) * 1e6) / uint256(reserve0);
        } else {
            // Price = USDC / WETH (scaled by 1e6 for USDC decimals)
            return (uint256(reserve0) * 1e6) / uint256(reserve1);
        }
    }
    
    /**
     * @dev Check if a specific game is currently enabled
     * @param gameId The ID of the game to check
     * @return True if the game is enabled, false otherwise
     */
    function isGameEnabled(uint256 gameId) external view returns (bool) {
        require(gameId >= 1 && gameId <= 4, "Invalid game ID");
        return games[gameId].enabled;
    }
    
    /**
     * @dev Get current win count for a specific game
     * @param gameId The ID of the game to check
     * @return Number of times the game has been won
     */
    function getGameWinCount(uint256 gameId) external view returns (uint256) {
        require(gameId >= 1 && gameId <= 4, "Invalid game ID");
        return games[gameId].winCount;
    }
    
    /**
     * @dev Allow the contract to receive ETH
     */
    receive() external payable {}
}