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
        uint256 settlerReward
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
 * @title SimplifiedChainlinkReporter
 * @dev A contract that uses Chainlink price feeds to submit initial reports
 * to openOracle.
 */
contract initialReportBot {
    // Simple reentrancy guard
    bool private _locked;
    
    // Simple ownership
    address public owner;
    
    // Constants
    uint256 public constant MIN_REPORT_FEE_PERCENT = 100; // Min fee must be 1% of 2*exactToken1Amount (in basis points)
    uint256 public constant MIN_ETH_AMOUNT = 11111111111111111; // Min 0.011111... ETH (to ensure 0.01 ETH after fees)
    uint256 public constant MAX_PRICE_FEED_DELAY = 66 minutes; // Maximum allowed staleness of price feed (1.1 hours)
    uint256 public constant MAX_REPORTS_TO_CHECK = 3; // Check the 3 most recent reports
    uint256 public constant BOT_REWARD_PERCENT = 20; // MEV bot gets 20% of the fee upfront
    
    // Hardcoded token addresses
    address public constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address public constant USDC = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48;
    
    // Hardcoded Chainlink price feeds
    address public constant ETH_USD_PRICE_FEED = 0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419;
    address public constant USDC_USD_PRICE_FEED = 0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6;
    
    // State variables
    IOpenOracle public immutable oracle;
    AggregatorV3Interface public immutable ethUsdPriceFeed;
    AggregatorV3Interface public immutable usdcUsdPriceFeed;
    
    // Track submitted reports
    mapping(uint256 => bool) public isReportSubmitted;
    
    // Events
    event ReportSubmitted(uint256 indexed reportId, address submitter, uint256 amount1, uint256 amount2, uint256 reward);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
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
        uint256 nextId = oracle.nextReportId();
        require(nextId > 1, "No reports created yet");
        
        uint256 startFromId = nextId > MAX_REPORTS_TO_CHECK ? nextId - MAX_REPORTS_TO_CHECK : 1;
        
        for (uint256 i = nextId - 1; i >= startFromId; i--) {
            // Skip if already submitted
            if (isReportSubmitted[i]) {
                continue;
            }
            
            // Check if this report is valid for submission
            if (_isValidForSubmission(i)) {
                _submitReportWithChainlinkPrice(i);
                return; // Exit after finding the first valid report
            }
            
            // Prevent underflow
            if (i == 1) break;
        }
        
        revert("No valid reports found for submission");
    }
    
    /**
     * @dev Check if a report is valid for submission
     */
    function _isValidForSubmission(uint256 reportId) internal view returns (bool) {
        try oracle.reportMeta(reportId) returns (
            address token1,
            address token2,
            uint256 ,
            uint256 ,
            uint256 ,
            uint256 exactToken1Amount,
            uint256 fee,
            uint256 ,
            uint256 ,
            uint256 ,
            uint256 
        ) {
            address payable currentReporter;
            (, , currentReporter, , , , , , , ,) = oracle.reportStatus(reportId);
            
            // Skip if already reported
            if (currentReporter != address(0)) {
                return false;
            }
            
            // Check token types
            if (token1 != WETH || token2 != USDC) {
                return false;
            }
            
            // Validate minimum ETH amount (0.011111... ETH)
            if (exactToken1Amount < MIN_ETH_AMOUNT) {
                return false;
            }
            
            // Validate fee is at least 1% of 2*exactToken1Amount
            uint256 minRequiredFee = (2 * exactToken1Amount * MIN_REPORT_FEE_PERCENT) / 10000;
            if (fee < minRequiredFee) {
                return false;
            }
            
            // Check if price feeds are fresh
            if (!_arePriceFeedsActive()) {
                return false;
            }
            
            return true;
        } catch {
            return false;
        }
    }
    
    /**
     * @dev Verify that Chainlink price feeds are recent enough
     */
    function _arePriceFeedsActive() internal view returns (bool) {
        (, , , uint256 ethUsdUpdatedAt, ) = ethUsdPriceFeed.latestRoundData();
        (, , , uint256 usdcUsdUpdatedAt, ) = usdcUsdPriceFeed.latestRoundData();
        
        return (
            block.timestamp - ethUsdUpdatedAt < MAX_PRICE_FEED_DELAY &&
            block.timestamp - usdcUsdUpdatedAt < MAX_PRICE_FEED_DELAY
        );
    }

    /**
     * @dev Get the Chainlink price for the WETH/USDC pair
     * @param wethAmount The amount of WETH
     * @return The equivalent amount of USDC based on Chainlink price
     */
    function getChainlinkPrice(uint256 wethAmount) public view returns (uint256) {
        // Get ETH/USD price
        (, int256 ethUsdPrice, , uint256 ethUsdUpdatedAt, ) = ethUsdPriceFeed.latestRoundData();
        uint8 ethDecimals = ethUsdPriceFeed.decimals();
        
        // Get USDC/USD price
        (, int256 usdcUsdPrice, , uint256 usdcUsdUpdatedAt, ) = usdcUsdPriceFeed.latestRoundData();
        uint8 usdcDecimals = usdcUsdPriceFeed.decimals();
        
        // Validate price feed data
        require(ethUsdPrice > 0, "Invalid ETH/USD price");
        require(usdcUsdPrice > 0, "Invalid USDC/USD price");
        require(block.timestamp - ethUsdUpdatedAt < MAX_PRICE_FEED_DELAY, "ETH/USD price stale");
        require(block.timestamp - usdcUsdUpdatedAt < MAX_PRICE_FEED_DELAY, "USDC/USD price stale");
        
        // Calculate WETH/USDC price
        // WETH/USDC = (ETH/USD) / (USDC/USD)
        uint256 scaledEthPrice = uint256(ethUsdPrice) * (10 ** (18 - ethDecimals));
        uint256 scaledUsdcPrice = uint256(usdcUsdPrice) * (10 ** (18 - usdcDecimals));
        
        // WETH has 18 decimals, USDC has 6
        uint256 usdcAmount = (wethAmount * scaledEthPrice * (10 ** 6)) / (scaledUsdcPrice * (10 ** 18));
        
        return usdcAmount;
    }
    
    /**
     * @dev Submit a report with the Chainlink-derived price
     */
    function _submitReportWithChainlinkPrice(uint256 reportId) internal {
        // Get report metadata
        (
            , 
            , 
            , 
            , 
            , 
            uint256 exactToken1Amount,
            uint256 fee,
            , 
            , 
            , 
            
        ) = oracle.reportMeta(reportId);
        
        // Get price from Chainlink
        uint256 token2Amount = getChainlinkPrice(exactToken1Amount);
        
        // Transfer tokens to this contract (caller must have pre-approved)
        bool success1 = IERC20(WETH).transferFrom(msg.sender, address(this), exactToken1Amount);
        require(success1, "WETH transfer failed");
        
        bool success2 = IERC20(USDC).transferFrom(msg.sender, address(this), token2Amount);
        require(success2, "USDC transfer failed");
        
        // Approve oracle to spend tokens
        IERC20(WETH).approve(address(oracle), exactToken1Amount);
        IERC20(USDC).approve(address(oracle), token2Amount);
        
        // Submit the report
        oracle.submitInitialReport(reportId, exactToken1Amount, token2Amount);
        
        // Mark as submitted
        isReportSubmitted[reportId] = true;
        
        // Calculate and send partial reward to caller (20%)
        uint256 botReward = (fee * BOT_REWARD_PERCENT) / 100;
        (bool sentETH, ) = payable(msg.sender).call{value: botReward}("");
        require(sentETH, "ETH transfer failed");
        
        emit ReportSubmitted(reportId, msg.sender, exactToken1Amount, token2Amount, botReward);
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
     * @dev Receive ETH
     */
    receive() external payable {}
}