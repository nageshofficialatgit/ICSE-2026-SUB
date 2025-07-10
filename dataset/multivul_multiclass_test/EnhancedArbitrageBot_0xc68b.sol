pragma solidity ^0.8.20;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
}

interface ICurvePool {
    function exchange(int128 i, int128 j, uint256 dx, uint256 min_dy) external returns (uint256);
}

interface IBalancerVault {
    function swap(
        bytes32 poolId,
        uint8 kind,
        address tokenIn,
        address tokenOut,
        uint256 amount,
        bytes memory userData
    ) external returns (uint256);
}

/// @title Enhanced Arbitrage Bot
/// @notice Executes arbitrage trades across multiple DEX protocols
contract EnhancedArbitrageBot {
    // Immutable state variables
    address public immutable owner;
    uint256 public immutable DEADLINE_EXTENSION = 300;
    uint256 public immutable BASIS_POINTS = 10000;
    
    // State variables
    bool private _paused;
    bool public circuitBroken;
    bool public isRunning;  // New running status
    uint256 public maxDailyLoss;
    uint256 public dailyLoss;
    uint256 public lastResetTime;
    uint256 public startTime;  // Track when bot started
    uint256 public totalRuns;  // Track number of running sessions
    
    // Packed structs for gas optimization
    struct ProtocolType {
        bool isUniswap;
        bool isCurve;
        bool isBalancer;
    }
    
    struct ProtocolConfig {
        ProtocolType protocolType;
        address router;
        bool active;
        uint96 maxSlippage;
        bytes32 poolId;
    }
    
    struct RiskParams {
        uint96 maxTradeSize;
        uint96 minProfit;
        uint32 maxSlippage;
        uint32 cooldownPeriod;
    }
    
    struct Analytics {
        uint128 totalVolume;
        uint96 profitLoss;
        uint32 tradeCount;
    }
    
    // Mappings
    mapping(address => ProtocolConfig) public protocols;
    mapping(address => RiskParams) public riskParams;
    mapping(address => Analytics) public analytics;
    mapping(address => uint256) public lastTradeTimestamp;
    
    // Events
    event TradeExecuted(
        address indexed token0,
        address indexed token1,
        uint256 profit,
        uint256 gasUsed
    );
    
    event TokenWithdrawn(
        address indexed token, 
        uint256 amount, 
        address indexed to
    );
    
    event EthWithdrawn(
        uint256 amount, 
        address indexed to
    );
    
    event EmergencyWithdrawal(
        address indexed token, 
        uint256 amount
    );
    
    event BotStarted(
        uint256 timestamp,
        address indexed starter
    );
    
    event BotStopped(
        uint256 timestamp,
        address indexed stopper,
        string reason
    );
    
    event BotStatus(
        bool isRunning,
        uint256 totalRuns,
        uint256 runningTime
    );
    
    // Custom errors for gas optimization
    error NotOwner();
    error ContractPaused();
    error CircuitBreakerActive();
    error InvalidProtocol();
    error ExceedsTradeSize();
    error CooldownNotMet();
    error InsufficientProfit();
    error InvalidAmount();
    error TransferFailed();
    error NoBalance();
    
    /// @notice Constructor sets the owner and initial max daily loss
    /// @param _maxDailyLoss Maximum allowed daily loss in wei
    constructor(uint256 _maxDailyLoss) {
        if (_maxDailyLoss == 0) revert InvalidAmount();
        owner = msg.sender;
        maxDailyLoss = _maxDailyLoss;
        lastResetTime = block.timestamp;
    }
    
    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }
    
    modifier whenNotPaused() {
        if (_paused) revert ContractPaused();
        _;
    }
    
    function executeArbitrage(
        address sourceProtocol,
        address targetProtocol,
        address token0,
        address token1,
        uint256 amount
    ) external whenNotPaused {
        require(isRunning, "Bot is not running");
        if (circuitBroken) revert CircuitBreakerActive();
        
        // Check risk parameters
        RiskParams storage risk = riskParams[token0];
        if (amount > risk.maxTradeSize) revert ExceedsTradeSize();
        if (block.timestamp < lastTradeTimestamp[token0] + risk.cooldownPeriod) 
            revert CooldownNotMet();
        
        uint256 gasStart = gasleft();
        uint256 balanceBefore = IERC20(token0).balanceOf(address(this));
        
        // Execute trades
        uint256 midAmount = _executeProtocolTrade(sourceProtocol, token0, token1, amount);
        uint256 finalAmount = _executeProtocolTrade(targetProtocol, token1, token0, midAmount);
        
        uint256 profit = finalAmount > balanceBefore ? finalAmount - balanceBefore : 0;
        if (profit < risk.minProfit) revert InsufficientProfit();
        
        // Update state
        _updateAnalytics(token0, amount, profit);
        
        emit TradeExecuted(
            token0,
            token1,
            profit,
            gasStart - gasleft()
        );
    }
    
    function _executeProtocolTrade(
        address protocol,
        address tokenIn,
        address tokenOut,
        uint256 amount
    ) private returns (uint256) {
        ProtocolConfig storage config = protocols[protocol];
        if (!config.active) revert InvalidProtocol();
        
        IERC20(tokenIn).approve(config.router, amount);
        
        if (config.protocolType.isUniswap) {
            return _executeUniswapTrade(config.router, tokenIn, tokenOut, amount);
        } else if (config.protocolType.isCurve) {
            return _executeCurveTrade(config.router, tokenIn, tokenOut, amount);
        } else if (config.protocolType.isBalancer) {
            return _executeBalancerTrade(config.router, config.poolId, tokenIn, tokenOut, amount);
        }
        revert InvalidProtocol();
    }
    
    function _executeUniswapTrade(
        address router,
        address tokenIn,
        address tokenOut,
        uint256 amount
    ) private returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        uint256[] memory amounts = IUniswapV2Router(router).swapExactTokensForTokens(
            amount,
            0,
            path,
            address(this),
            block.timestamp + DEADLINE_EXTENSION
        );
        
        return amounts[1];
    }
    
    function _executeCurveTrade(
        address pool,
        address,  // tokenIn (unused)
        address,  // tokenOut (unused)
        uint256 amount
    ) private returns (uint256) {
        return ICurvePool(pool).exchange(0, 1, amount, 0);
    }
    
    function _executeBalancerTrade(
        address vault,
        bytes32 poolId,
        address tokenIn,
        address tokenOut,
        uint256 amount
    ) private returns (uint256) {
        return IBalancerVault(vault).swap(
            poolId,
            1,
            tokenIn,
            tokenOut,
            amount,
            ""
        );
    }
    
    function _updateAnalytics(
        address token,
        uint256 amount,
        uint256 profit
    ) private {
        Analytics storage stats = analytics[token];
        stats.totalVolume += uint128(amount);
        stats.profitLoss += uint96(profit);
        stats.tradeCount += 1;
        lastTradeTimestamp[token] = block.timestamp;
        
        if (block.timestamp - lastResetTime >= 1 days) {
            dailyLoss = 0;
            lastResetTime = block.timestamp;
        }
        
        if (profit == 0) {
            dailyLoss += amount;
            if (dailyLoss > maxDailyLoss) {
                circuitBroken = true;
            }
        }
    }
    
    // Admin functions
    function addProtocol(
        address protocolAddress,
        bool isUniswap,
        bool isCurve,
        bool isBalancer,
        uint96 maxSlippage,
        bytes32 poolId
    ) external onlyOwner {
        protocols[protocolAddress] = ProtocolConfig({
            protocolType: ProtocolType(isUniswap, isCurve, isBalancer),
            router: protocolAddress,
            active: true,
            maxSlippage: maxSlippage,
            poolId: poolId
        });
    }
    
    function setRiskParameters(
        address token,
        uint96 maxTradeSize,
        uint96 minProfit,
        uint32 maxSlippage,
        uint32 cooldownPeriod
    ) external onlyOwner {
        riskParams[token] = RiskParams({
            maxTradeSize: maxTradeSize,
            minProfit: minProfit,
            maxSlippage: maxSlippage,
            cooldownPeriod: cooldownPeriod
        });
    }
    
    // Withdrawal functions
    function withdrawToken(
        address token,
        uint256 amount,
        address to
    ) external onlyOwner {
        uint256 balance = IERC20(token).balanceOf(address(this));
        if (balance == 0) revert NoBalance();
        
        uint256 withdrawAmount = amount == 0 ? balance : amount;
        if (withdrawAmount > balance) withdrawAmount = balance;
        
        bool success = IERC20(token).transfer(to, withdrawAmount);
        if (!success) revert TransferFailed();
        
        emit TokenWithdrawn(token, withdrawAmount, to);
    }
    
    function withdrawETH(
        uint256 amount,
        address payable to
    ) external onlyOwner {
        uint256 balance = address(this).balance;
        if (balance == 0) revert NoBalance();
        
        uint256 withdrawAmount = amount == 0 ? balance : amount;
        if (withdrawAmount > balance) withdrawAmount = balance;
        
        (bool success, ) = to.call{value: withdrawAmount}("");
        if (!success) revert TransferFailed();
        
        emit EthWithdrawn(withdrawAmount, to);
    }
    
    function emergencyWithdraw(address[] calldata tokens) external onlyOwner {
        // Withdraw all ETH
        uint256 ethBalance = address(this).balance;
        if (ethBalance > 0) {
            (bool success, ) = owner.call{value: ethBalance}("");
            if (!success) revert TransferFailed();
            emit EmergencyWithdrawal(address(0), ethBalance);
        }
        
        // Withdraw all tokens
        for (uint i = 0; i < tokens.length; i++) {
            uint256 balance = IERC20(tokens[i]).balanceOf(address(this));
            if (balance > 0) {
                bool success = IERC20(tokens[i]).transfer(owner, balance);
                if (!success) revert TransferFailed();
                emit EmergencyWithdrawal(tokens[i], balance);
            }
        }
    }
    
    // Start/Stop functions
    function startBot() external onlyOwner {
        require(!isRunning, "Bot is already running");
        require(!_paused, "Bot is paused");
        require(!circuitBroken, "Circuit breaker is active");
        
        isRunning = true;
        startTime = block.timestamp;
        totalRuns += 1;
        
        emit BotStarted(block.timestamp, msg.sender);
        emit BotStatus(isRunning, totalRuns, 0);
    }
    
    function stopBot(string calldata reason) external onlyOwner {
        require(isRunning, "Bot is not running");
        
        isRunning = false;
        uint256 runningTime = block.timestamp - startTime;
        
        emit BotStopped(block.timestamp, msg.sender, reason);
        emit BotStatus(isRunning, totalRuns, runningTime);
    }
    
    function emergencyStop() external onlyOwner {
        if (isRunning) {
            isRunning = false;
            _paused = true;
            circuitBroken = true;
            
            emit BotStopped(block.timestamp, msg.sender, "Emergency stop triggered");
            emit BotStatus(isRunning, totalRuns, block.timestamp - startTime);
        }
    }
    
    function getBotStats() external view returns (
        bool _isRunning,
        bool _isPaused,
        bool _isCircuitBroken,
        uint256 _totalRuns,
        uint256 _currentRunTime,
        uint256 _dailyLoss
    ) {
        _currentRunTime = isRunning ? block.timestamp - startTime : 0;
        
        return (
            isRunning,
            _paused,
            circuitBroken,
            totalRuns,
            _currentRunTime,
            dailyLoss
        );
    }
    
    function togglePause() external onlyOwner {
        _paused = !_paused;
        if (_paused && isRunning) {
            isRunning = false;
            emit BotStopped(block.timestamp, msg.sender, "Paused by owner");
        }
    }
    
    function toggleCircuitBreaker() external onlyOwner {
        circuitBroken = !circuitBroken;
        if (circuitBroken && isRunning) {
            isRunning = false;
            emit BotStopped(block.timestamp, msg.sender, "Circuit breaker triggered");
        }
    }
    
    receive() external payable {}
}