// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}

interface AggregatorV3Interface {
    function latestRoundData() external view returns (
        uint80 roundId,
        int256 answer,
        uint256 startedAt,
        uint256 updatedAt,
        uint80 answeredInRound
    );
    function decimals() external view returns (uint8);
}

interface IOpenOracle {
    function submitInitialReport(uint256 reportId, uint256 amount1, uint256 amount2) external;
    
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
    
    function reportStatus(uint256 reportId) external view returns (
        uint256 currentAmount1,
        uint256 currentAmount2,
        address payable currentReporter,
        address payable initialReporter,
        uint256 reportTimestamp,
        uint256 settlementTimestamp,
        uint256 price,
        bool isSettled,
        bool disputeOccurred,
        bool isDistributed,
        uint256 lastDisputeBlock
    );
    
    function nextReportId() external view returns (uint256);
}

/**
 * @title initialReportBot
 * @dev A contract that uses Chainlink price feeds to submit initial reports
 * to the openOracle. 
 */
contract initialReportBot {
    // Simple reentrancy guard
    bool private _locked;
    
    // Simple ownership
    address public owner;
    
    // Constants
    uint256 public constant MIN_REPORT_FEE_PERCENT = 100; // Min fee must be 1% of 2*exactToken1Amount (in basis points)
    uint256 public constant MIN_ETH_AMOUNT = 11111111111111111; // Min 0.011111... ETH (to ensure 0.01 ETH after fees)
    uint256 public constant ETH_USD_MAX_PRICE_FEED_DELAY = 66 minutes; // Maximum allowed staleness of ETH/USD price feed (1.1 hours)
    uint256 public constant USDC_USD_MAX_PRICE_FEED_DELAY = 25 hours; // Maximum allowed staleness of USDC/USD price feed (1 day and 1 hour)
    uint256 public constant MAX_REPORTS_TO_CHECK = 3; // Check the 3 most recent reports
    uint256 public constant MAX_SETTLEMENT_TIME = 2 minutes; // Maximum settlement time allowed
    uint256 public constant GAS_PREMIUM_PERCENT = 25; // Premium added to gas cost (25%)
    uint256 public constant FINALIZATION_GAS_ESTIMATE = 250000; // Buffer for remaining operations
    uint256 public constant SETTLE_GAS_ESTIMATE = 400000; // Estimated gas usage for the settle function
    uint256 public constant GAS_PRICE_STALENESS = 185 minutes; // Gas price staleness threshold (unsure of exact heartbeat on this)
    uint256 public constant REWARD_INCREMENT_PERCENT = 30; // Increase reward by 30% per block
    
    // Hardcoded token addresses
    address public constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address public constant USDC = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48;
    
    // Hardcoded Chainlink price feeds
    address public constant ETH_USD_PRICE_FEED = 0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419;
    address public constant USDC_USD_PRICE_FEED = 0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6;
    address public constant FAST_GAS_PRICE_FEED = 0x169E633A2D1E6c10dD91238Ba11c4A708dfEF37C; // Chainlink Fast Gas Price Feed
    
    // State variables
    IOpenOracle public immutable oracle;
    AggregatorV3Interface public immutable ethUsdPriceFeed;
    AggregatorV3Interface public immutable usdcUsdPriceFeed;
    AggregatorV3Interface public immutable gasPriceFeed;
    
    // Track submitted reports
    mapping(uint256 => bool) public isReportSubmitted;
    
    // Events
    event ReportSubmitted(uint256 indexed reportId, address submitter, uint256 amount1, uint256 amount2, uint256 reward, uint256 gasCost, uint256 blocksSinceRequest);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event UnprofitableOperation(uint256 reportId, uint256 gasCost, uint256 reward, uint256 settlerReward);
    
    // Struct to cache report metadata
    struct ReportData {
        address token1;
        address token2;
        uint256 settlementTime;
        uint256 exactToken1Amount;
        uint256 fee;
        uint256 settlerReward;
        address payable currentReporter;
        uint256 requestBlock;
    }
    
    // Struct to cache price feed data
    struct PriceFeedData {
        int256 ethUsdPrice;
        uint256 ethUsdUpdatedAt;
        uint8 ethDecimals;
        int256 usdcUsdPrice;
        uint256 usdcUsdUpdatedAt;
        uint8 usdcDecimals;
        uint256 gasPrice;
        uint256 gasPriceUpdatedAt;
    }
    
    // Modifier to prevent reentrancy
    modifier nonReentrant() {
        require(!_locked, "ReentrancyGuard: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }
    
    // Modifier to restrict access to owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Ownable: caller is not the owner");
        _;
    }
    
    /**
     * @dev Constructor
     * @param _oracle Address of the openOracle contract
     */
    constructor(address _oracle) {
        owner = msg.sender;
        oracle = IOpenOracle(_oracle);
        ethUsdPriceFeed = AggregatorV3Interface(ETH_USD_PRICE_FEED);
        usdcUsdPriceFeed = AggregatorV3Interface(USDC_USD_PRICE_FEED);
        gasPriceFeed = AggregatorV3Interface(FAST_GAS_PRICE_FEED);
    }
    
    /**
     * @dev Transfer ownership to a new address
     * @param newOwner The address of the new owner
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
    
    /**
     * @dev Inputless function to find and submit reports using Chainlink price
     * Checks the 3 most recent reports and submits the first valid one
     */
    function submitReport() external nonReentrant {
        // Store initial gas to measure consumption
        uint256 startGas = gasleft();
        
        // Fetch price feed data once for the entire transaction
        PriceFeedData memory priceFeedData = _fetchPriceFeedData();
        
        uint256 nextId = oracle.nextReportId();
        require(nextId > 1, "No reports created yet");
        
        uint256 startFromId = nextId > MAX_REPORTS_TO_CHECK ? nextId - MAX_REPORTS_TO_CHECK : 1;
        
        for (uint256 i = nextId - 1; i >= startFromId; i--) {
            // Skip if already submitted
            if (isReportSubmitted[i]) {
                continue;
            }
            
            // Fetch and cache report metadata for this iteration - reduces multiple external calls
            ReportData memory reportData = _fetchReportData(i);
            
            // Check if this report is valid for submission
            if (_isValidForSubmission(reportData, priceFeedData)) {

                    _submitReportWithChainlinkPrice(i, reportData, priceFeedData, startGas);
                    return; // Exit after finding the first valid report assuming this function doesnt revert

            }
            
            // Prevent underflow
            if (i == 1) break;
        }
        
        revert("No valid reports found for submission");
    }
    
    /**
     * @dev Fetch report metadata and status in a single function to avoid multiple calls
     */
    function _fetchReportData(uint256 reportId) internal view returns (ReportData memory data) {
        // Get report metadata (single external call)
        (
            data.token1,
            data.token2,
            ,  // feePercentage
            ,  // multiplier
            data.settlementTime,
            data.exactToken1Amount,
            data.fee,
            ,  // escalationHalt
            ,  // disputeDelay
            ,  // protocolFee
            data.settlerReward,
            data.requestBlock
        ) = oracle.reportMeta(reportId);
        
        // Get current reporter (single external call)
        (, , data.currentReporter, , , , , , , ,) = oracle.reportStatus(reportId);
        
        return data;
    }
    
    /**
     * @dev Fetch all price feed data in a single function to avoid multiple calls
     */
    function _fetchPriceFeedData() internal view returns (PriceFeedData memory data) {
        // Get ETH/USD price data (single external call)
        {
            (uint80 roundId1, int256 ethPrice, uint256 startedAt1, uint256 updatedAt1, uint80 answeredInRound1) = ethUsdPriceFeed.latestRoundData();
            data.ethUsdPrice = ethPrice;
            data.ethUsdUpdatedAt = updatedAt1;
            data.ethDecimals = ethUsdPriceFeed.decimals();
        }
        
        // Get USDC/USD price data (single external call)
        {
            (uint80 roundId2, int256 usdcPrice, uint256 startedAt2, uint256 updatedAt2, uint80 answeredInRound2) = usdcUsdPriceFeed.latestRoundData();
            data.usdcUsdPrice = usdcPrice;
            data.usdcUsdUpdatedAt = updatedAt2;
            data.usdcDecimals = usdcUsdPriceFeed.decimals();
        }
        
        // Get gas price data (single external call)
        {
            (uint80 roundId3, int256 gasAnswer, uint256 startedAt3, uint256 updatedAt3, uint80 answeredInRound3) = gasPriceFeed.latestRoundData();
            data.gasPriceUpdatedAt = updatedAt3;
            
            // If feed is fresh (within last 33 minutes) and value is positive
            if (block.timestamp - data.gasPriceUpdatedAt < GAS_PRICE_STALENESS && gasAnswer > 0) {
                data.gasPrice = uint256(gasAnswer);
            } else {
                revert("Cannot obtain reliable gas price");
            }
        }
        
        return data;
    }
    
    /**
     * @dev Check if a report is valid for submission using cached data
     */
    function _isValidForSubmission(ReportData memory data, PriceFeedData memory priceFeedData) internal view returns (bool) {
        // Skip if already reported
        if (data.currentReporter != address(0)) {
            return false;
        }
        
        // Check token types
        if (data.token1 != WETH || data.token2 != USDC) {
            return false;
        }
        
        // Validate minimum ETH amount (0.011111... ETH)
        if (data.exactToken1Amount < MIN_ETH_AMOUNT) {
            return false;
        }
        
        // Validate fee is at least 1% of 2*exactToken1Amount
        uint256 minRequiredFee = (2 * data.exactToken1Amount * MIN_REPORT_FEE_PERCENT) / 10000;
        if (data.fee < minRequiredFee) {
            return false;
        }
        
        // Check if settlement time is not too long
        if (data.settlementTime > MAX_SETTLEMENT_TIME) {
            return false;
        }
        
        // Check if price feeds are fresh using data passed in
        if (!_arePriceFeedsActive(priceFeedData)) {
            return false;
        }
        
        // Validate that the settler reward is sufficient to cover gas costs
        uint256 estimatedSettleGasCost = SETTLE_GAS_ESTIMATE * priceFeedData.gasPrice;
        if (data.settlerReward < estimatedSettleGasCost) {
            return false; // Settler reward too low to cover gas costs
        }
        
        return true;
    }
    
    /**
     * @dev Verify that Chainlink price feeds are recent enough
     * Modified to use cached data instead of making external calls
     */
    function _arePriceFeedsActive(PriceFeedData memory data) internal view returns (bool) {
        // Use unchecked for timestamp comparisons (safe in practice)
        unchecked {
            return (
                block.timestamp - data.ethUsdUpdatedAt < ETH_USD_MAX_PRICE_FEED_DELAY &&
                block.timestamp - data.usdcUsdUpdatedAt < USDC_USD_MAX_PRICE_FEED_DELAY
            );
        }
    }

    /**
     * @dev Get the Chainlink price for the WETH/USDC pair using cached price feed data
     * @param wethAmount The amount of WETH
     * @param data The cached price feed data
     * @return The equivalent amount of USDC based on Chainlink price
     */
    function getChainlinkPrice(uint256 wethAmount, PriceFeedData memory data) public pure returns (uint256) {
        // Validate price feed data
        require(data.ethUsdPrice > 0, "Invalid ETH/USD price");
        require(data.usdcUsdPrice > 0, "Invalid USDC/USD price");
        
        // Calculate WETH/USDC price
        // WETH/USDC = (ETH/USD) / (USDC/USD)
        uint256 scaledEthPrice = uint256(data.ethUsdPrice) * (10 ** (18 - data.ethDecimals));
        uint256 scaledUsdcPrice = uint256(data.usdcUsdPrice) * (10 ** (18 - data.usdcDecimals));
        
        // WETH has 18 decimals, USDC has 6
        uint256 usdcAmount = (wethAmount * scaledEthPrice * (10 ** 6)) / (scaledUsdcPrice * (10 ** 18));
        
        return usdcAmount;
    }
    
    /**
     * @dev Calculate the dynamic reward based on blocks passed since request
     * @param reportData The cached report data
     * @return Dynamic reward that scales with blocks passed since request
     */
    /**
     * @dev Calculate the dynamic reward based on blocks passed since request
     * @param reportData The cached report data
     * @param baseCost The base gas cost that will be scaled
     * @return Dynamic reward that scales with blocks passed since request
     */
    function calculateDynamicReward(ReportData memory reportData, uint256 baseCost) public view returns (uint256) {
        // Calculate blocks passed since the report was requested
        uint256 blocksPassed = block.number - reportData.requestBlock;
        
        // Cap at 5 blocks for the calculation
        if (blocksPassed > 5) {
            blocksPassed = 5;
        }
        
        // Calculate the maximum reward (90% of initial reporter fee)
        uint256 maxReward = (reportData.fee * 90) / 100;
        
        // If no blocks passed, return the base cost
        if (blocksPassed == 0) {
            return baseCost;
        }
        
        // Calculate directly: baseCost * (1.3^blocksPassed)
        // Using integer math: baseCost * (130/100)^blocksPassed
        uint256 dynamicReward = baseCost;
        
        // Calculate (130/100)^blocksPassed using efficient exponentiation
        uint256 multiplier = integerPower(130, blocksPassed);
        uint256 divisor = integerPower(100, blocksPassed);
        
        // Apply the calculated multiplier
        dynamicReward = (dynamicReward * multiplier) / divisor;
        
        // Ensure reward covers base costs at minimum
        dynamicReward = dynamicReward < baseCost ? baseCost : dynamicReward;
        
        // Cap the reward at 90% of the initial reporter fee
        return dynamicReward > maxReward ? maxReward : dynamicReward;
    }
    
    /**
     * @dev Efficiently calculate base^exponent using integer math
     * @param base The base value
     * @param exponent The exponent
     * @return The result of base^exponent
     */
    function integerPower(uint256 base, uint256 exponent) internal pure returns (uint256) {
        if (exponent == 0) {
            return 1;
        }
        if (exponent == 1) {
            return base;
        }
        
        uint256 result = 1;
        while (exponent > 0) {
            if (exponent % 2 == 1) {
                result = (result * base);
            }
            base = (base * base);
            exponent = exponent / 2;
        }
        return result;
    }
    
    /**
     * @dev Submit a report with the Chainlink-derived price
     * @param reportId The report ID to submit
     * @param reportData The cached report data
     * @param priceFeedData The cached price feed data
     * @param startGas The initial gas at function start for gas calculation
     */
    function _submitReportWithChainlinkPrice(
        uint256 reportId, 
        ReportData memory reportData,
        PriceFeedData memory priceFeedData,
        uint256 startGas
    ) internal {
        // Get price from Chainlink using cached data
        uint256 token2Amount = getChainlinkPrice(reportData.exactToken1Amount, priceFeedData);
        
        // Ensure the contract has enough tokens
        require(IERC20(WETH).balanceOf(address(this)) >= reportData.exactToken1Amount, "Insufficient WETH balance");
        require(IERC20(USDC).balanceOf(address(this)) >= token2Amount, "Insufficient USDC balance");
        
        // Approve oracle to spend tokens
        IERC20(WETH).approve(address(oracle), reportData.exactToken1Amount);
        IERC20(USDC).approve(address(oracle), token2Amount);
        
        // Submit the report
        oracle.submitInitialReport(reportId, reportData.exactToken1Amount, token2Amount);
        
        // Mark as submitted
        isReportSubmitted[reportId] = true;
        
        // Calculate gas cost
        uint256 finalGasCost = calculateGasCost(startGas, priceFeedData.gasPrice);
        
        // Calculate dynamic reward using the helper function
        uint256 dynamicReward = calculateDynamicReward(reportData, finalGasCost);
        
        // Calculate maximum reward (90% of initial reporter fee)
        uint256 maxReward = (reportData.fee * 90) / 100;
        
        // Ensure the reward doesn't exceed the maximum (90% of fee)
        require(dynamicReward <= maxReward, "Reward exceeds maximum allowed");
        
        // Send reward to caller
        (bool sentETH, ) = payable(msg.sender).call{value: dynamicReward}("");
        require(sentETH, "ETH transfer failed");
        
        // Calculate blocks passed for the event
        uint256 blocksPassed = block.number - reportData.requestBlock;
        
        emit ReportSubmitted(reportId, msg.sender, reportData.exactToken1Amount, token2Amount, dynamicReward, finalGasCost, blocksPassed);
    }
    
    /**
     * @dev Withdraw tokens from the contract (owner only)
     * @param token The token address
     * @param amount The amount to withdraw
     */
    function withdrawTokens(address token, uint256 amount) external onlyOwner nonReentrant {
        bool success = IERC20(token).transfer(msg.sender, amount);
        require(success, "Token transfer failed");
    }
    
    /**
     * @dev Withdraw ETH from the contract (owner only)
     * @param amount The amount to withdraw
     */
    function withdrawETH(uint256 amount) external onlyOwner nonReentrant {
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "ETH transfer failed");
    }
    
    /**
     * @dev Estimate remaining gas cost with premium
     * @param startGas The initial gas at the start of execution
     * @param gasPrice Cached gas price to avoid additional external calls
     * @return Total cost with premium in wei
     */
    function calculateGasCost(uint256 startGas, uint256 gasPrice) public view returns (uint256) {
        uint256 gasUsed;
        uint256 gasCost;
        uint256 totalCost;
        
        // Use unchecked block for gas math (won't overflow in practice)
        unchecked {
            // Calculate gas used + remaining buffer
            gasUsed = startGas - gasleft() + FINALIZATION_GAS_ESTIMATE;
            
            // Calculate total cost
            gasCost = gasUsed * gasPrice;
            
            // Add premium (x * 125 / 100 = x * 1.25)
            totalCost = (gasCost * (100 + GAS_PREMIUM_PERCENT)) / 100;
        }
        
        return totalCost;
    }
    
    /**
     * @dev Receive ETH
     */
    receive() external payable {}
}