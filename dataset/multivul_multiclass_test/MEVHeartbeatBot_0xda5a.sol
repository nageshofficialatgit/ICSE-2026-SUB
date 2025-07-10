// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IOpenOracle {
    function createReportInstance(
        address token1Address,
        address token2Address,
        uint256 exactToken1Report,
        uint256 feePercentage,
        uint256 multiplier,
        uint256 settlementTime,
        uint256 escalationHalt,
        uint256 disputeDelay,
        uint256 protocolFee,
        uint256 settlerReward
    ) external payable returns (uint256 reportId);
    
    function nextReportId() external view returns (uint256);
    
    function reportMeta(uint256 reportId) external view returns (
        address token1,
        address token2,
        uint256 feePercentage,
        uint256 multiplier,
        uint256 settlementTime,
        uint256 exactToken1Report,
        uint256 fee,
        uint256 escalationHalt,
        uint256 disputeDelay,
        uint256 protocolFee,
        uint256 settlerReward,
        uint256 requestBlock
    );
}

interface AggregatorV3Interface {
    function latestRoundData() external view returns (
        uint80 roundId,
        int256 answer,
        uint256 startedAt,
        uint256 updatedAt,
        uint80 answeredInRound
    );
}

/**
 * @title MEVHeartbeatBot
 * @dev Contract that creates report instances for the openOracle if there hasn't been one in 4 hours
 */
contract MEVHeartbeatBot {
    // Reentrancy guard
    bool private _locked;
    
    // Oracle contract

    IOpenOracle public immutable oracle;
    
    // Hardcoded token addresses
    address public constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address public constant USDC = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48;
    address public constant FAST_GAS_PRICE_FEED = 0x169E633A2D1E6c10dD91238Ba11c4A708dfEF37C;
    
    // Constants
    uint256 public constant HEARTBEAT_INTERVAL = 4 hours;
    // Block time estimate for time calculations
    uint256 public constant BLOCK_TIME_ESTIMATE = 12 seconds; // 12 seconds per block on Ethereum
    uint256 public constant CREATE_GAS_ESTIMATE = 282000;
    uint256 public constant GAS_BUFFER_PERCENT = 40;
    uint256 public constant MAX_REWARD = 0.005 ether;
    uint256 public constant REWARD_BASE_PERCENT = 60; // Not used anymore but kept for compatibility
    uint256 public constant REWARD_INCREMENT_PERCENT = 30; // Not used anymore but kept for compatibility
    uint256 public constant BLOCKS_PER_INCREMENT = 5; // Not used anymore but kept for compatibility
    
    // Report parameters from the screenshot
    uint256 public constant EXACT_TOKEN1_REPORT = 11111111111111112; // 0.011... ETH
    uint256 public constant FEE_PERCENTAGE = 2222; // 0.2222% in basis points
    uint256 public constant MULTIPLIER = 115; // 1.15x
    uint256 public constant SETTLEMENT_TIME = 48; // seconds
    uint256 public constant ESCALATION_HALT = 250000000000000000; // 0.25 ETH
    uint256 public constant DISPUTE_DELAY = 0; // no delay
    uint256 public constant PROTOCOL_FEE = 2222; // 0.2222% in basis points
    
    // Gas estimates
    uint256 public constant SETTLE_GAS_COST = 200000; // settle through beacon
    uint256 public constant REPORT_GAS_COST = 400000; // submit initial report through beacon
    
    // Heartbeat state tracking
    uint256 public lastHeartbeatTime;
    uint256 public heartbeatEligibleFrom;
    uint256 public eligibilityBlock;
    uint256 public lastHeartbeatBlock;

    // Events
    event HeartbeatTriggered(uint256 indexed reportId, uint256 gasPrice, uint256 reward, address caller);
    event CustomHeartbeatTriggered(uint256 indexed reportId, uint256 value, address caller);
    event HeartbeatEligibilityUpdated(uint256 timestamp, uint256 eligibilityBlock);
    
    // Modifier to prevent reentrancy
    modifier nonReentrant() {
        require(!_locked, "ReentrancyGuard: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }
    
    /**
     * @dev Constructor
     * @param _oracle Address of the openOracle contract
     */
    constructor(address _oracle) {
        oracle = IOpenOracle(_oracle);
        lastHeartbeatTime = block.timestamp;
        heartbeatEligibleFrom = block.timestamp + HEARTBEAT_INTERVAL;
    }
    uint256 settlerRewardReport;
    /**
     * @dev Trigger a heartbeat if conditions are met - inputless function anyone can call
     */
    function triggerHeartbeat() external nonReentrant {
        // Check timing and active state requirements
        require(block.timestamp >= heartbeatEligibleFrom, "Heartbeat not due yet");
        require(block.number != lastHeartbeatBlock);

        // Set eligibility block if not already set
        if (eligibilityBlock == 0) {
            eligibilityBlock = block.number;
            emit HeartbeatEligibilityUpdated(block.timestamp, eligibilityBlock);
        }
        
        // Check if there's a need for a non-economical report
        require(isReportNeeded(), "No heartbeat needed");
        
        // Get current gas price from Chainlink and its updatedAt time
        (uint256 gasPrice, uint256 updatedAt) = getCurrentGasPriceAndTimestamp();
        
        // Calculate the total value needed for the report
        uint256 totalGasCost = (SETTLE_GAS_COST + REPORT_GAS_COST) * gasPrice;
        uint256 valueForReport = totalGasCost + (totalGasCost * GAS_BUFFER_PERCENT / 100);
        settlerRewardReport = SETTLE_GAS_COST*gasPrice + ((SETTLE_GAS_COST*gasPrice) * GAS_BUFFER_PERCENT / 100);
        
        // Cap settlerRewardReport at 0.005 ETH instead of reverting
        if (settlerRewardReport > 5000000000000000) {
            settlerRewardReport = 5000000000000000; // 0.005 ETH
        }
        
        // Convert updatedAt timestamp to approximate block number
        uint256 updatedAtBlock = (updatedAt / BLOCK_TIME_ESTIMATE);
        
        // Calculate blocks passed considering the gas price feed update time
        uint256 reward = calculateDynamicReward(gasPrice, updatedAtBlock);
        
        // Ensure contract has enough balance
        require(address(this).balance >= valueForReport + reward, "Insufficient balance");
        
        // Create the report and handle rewards in a separate function to avoid stack too deep
        createHeartbeatReport(valueForReport, gasPrice, reward);
    }
    
    /**
     * @dev Helper function to create heartbeat report and update state
     * @param valueForReport Value to send to the oracle
     * @param gasPrice Current gas price
     * @param reward Reward for the caller
     */
    function createHeartbeatReport(
        uint256 valueForReport,
        uint256 gasPrice,
        uint256 reward
    ) private {
        // Create the report instance
        uint256 reportId = oracle.createReportInstance{value: valueForReport}(
            WETH,
            USDC,
            EXACT_TOKEN1_REPORT,
            FEE_PERCENTAGE,
            MULTIPLIER,
            SETTLEMENT_TIME,
            ESCALATION_HALT,
            DISPUTE_DELAY,
            PROTOCOL_FEE,
            settlerRewardReport
        );
        
        // Update state after successful call
        lastHeartbeatTime = block.timestamp;
        heartbeatEligibleFrom = block.timestamp + HEARTBEAT_INTERVAL;
        eligibilityBlock = 0;
        lastHeartbeatBlock = block.number;
        settlerRewardReport = 0;

        // Send reward to caller
        (bool sent, ) = payable(msg.sender).call{value: reward}("");
        require(sent, "Failed to send reward");
        
        emit HeartbeatTriggered(reportId, gasPrice, reward, msg.sender);
    }
    
    /**
     * @dev Trigger a custom heartbeat with custom value - bypasses wait time
     */
    function triggerCustomHeartbeat(uint256 _settlerReward) external payable nonReentrant {
        
        // Calculate value for the report (msg.value minus reward for caller)
        uint256 valueForReport = msg.value;
        
        // Create report instance with custom value
        uint256 reportId = oracle.createReportInstance{value: valueForReport}(
            WETH,
            USDC,
            EXACT_TOKEN1_REPORT,
            FEE_PERCENTAGE,
            MULTIPLIER,
            SETTLEMENT_TIME,
            ESCALATION_HALT,
            DISPUTE_DELAY,
            PROTOCOL_FEE,
            _settlerReward
        );
        

        emit CustomHeartbeatTriggered(reportId, msg.value, msg.sender);
    }
    
    /**
     * @dev Check if a new report is needed
     * @return True if a new report is needed (last report was > 4 hours ago)
     */
    function isReportNeeded() public view returns (bool) {
        uint256 nextId = oracle.nextReportId();
        
        // If no reports yet, we need one
        if (nextId <= 1) {
            return true;
        }
        
        // Check when the most recent report was requested
        (
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            uint256 requestBlock
        ) = oracle.reportMeta(nextId - 1);
        
        // Estimate timestamp from block number (approximate)
        uint256 blockTime = 12; // seconds per block, approximate
        uint256 blocksSince = block.number - requestBlock;
        uint256 timeSince = blocksSince * blockTime;
        
        // If the last report was created more than 4 hours ago, we need a new one
        return timeSince >= HEARTBEAT_INTERVAL;
    }
    
    /**
     * @dev Calculate dynamic reward that increases with blocks passed
     * @param gasPrice Current gas price in wei
     * @param updatedAtBlock Approximate block number when the gas price was updated
     * @return Dynamic reward amount capped at MAX_REWARD
     */
    function calculateDynamicReward(uint256 gasPrice, uint256 updatedAtBlock) public view returns (uint256) {
        // Get current block number
        uint256 currentBlock = block.number;
        
        // Base cost is the gas cost of the create report operation
        uint256 baseCost = CREATE_GAS_ESTIMATE * gasPrice;
        
        // Base reward is now 100% of the gas cost
        uint256 baseReward = baseCost;
        
        // Choose the more recent of eligibility block or gas price update block
        uint256 startingBlock = eligibilityBlock;
        if (updatedAtBlock > startingBlock) {
            startingBlock = updatedAtBlock;
        }
        
        // Calculate how many blocks passed since the starting block
        uint256 effectiveBlocksPassed = 0;
        if (currentBlock > startingBlock) {
            effectiveBlocksPassed = currentBlock - startingBlock;
        }
        
        // Calculate increments: each 10 blocks = 10% increase
        uint256 percentageIncrease = (effectiveBlocksPassed / 10) * 10;
        
        // Calculate additional reward based on percentage increase
        uint256 additionalReward = (baseCost * percentageIncrease) / 100;
        
        // Total reward is base + additional
        uint256 totalReward = baseReward + additionalReward;
        
        // Cap at maximum reward
        return totalReward > MAX_REWARD ? MAX_REWARD : totalReward;
    }
    
    /**
     * @dev Get current gas price and update timestamp from Chainlink
     * @return Current gas price in wei and the timestamp when it was updated
     */
    function getCurrentGasPriceAndTimestamp() public view returns (uint256, uint256) {
        AggregatorV3Interface gasPriceFeed = AggregatorV3Interface(FAST_GAS_PRICE_FEED);
        (, int256 answer, , uint256 updatedAt, ) = gasPriceFeed.latestRoundData();
        
        // Ensure the price feed is fresh (within last 3 hours) and value is positive
        require(block.timestamp - updatedAt < 3 hours && answer > 0, "Invalid gas price data");
        
        return (uint256(answer), updatedAt);
    }
    
    /**
     * @dev Get current gas price from Chainlink (kept for compatibility)
     * @return Current gas price in wei
     */
    function getCurrentGasPrice() public view returns (uint256) {
        (uint256 gasPrice, ) = getCurrentGasPriceAndTimestamp();
        return gasPrice;
    }
    
    /**
     * @dev Check contract balance
     * @return Contract balance in wei
     */
    function getBalance() external view returns (uint256) {
        return address(this).balance;
    }
    
    /**
     * @dev Receive function to allow contract to receive ETH
     */
    receive() external payable {}
}