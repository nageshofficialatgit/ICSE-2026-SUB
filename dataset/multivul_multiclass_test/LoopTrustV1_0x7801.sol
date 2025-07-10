// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @notice Chainlink ETH/USD price feed interface (8 decimals)
interface AggregatorV3Interface {
    function decimals() external view returns (uint8);
    function description() external view returns (string memory);
    function version() external view returns (uint256);
    function latestRoundData()
        external
        view
        returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );
}

/// @title LoopTrustV1 - Dynamic Fee Contract
/// @author You
/// @notice Automatically charges either a flat fee or percentage fee based on order size.
/// @dev Chainlink ETH/USD price feed is used to dynamically convert USD fees to ETH.
contract LoopTrustV1 {
    address public owner;
    AggregatorV3Interface internal priceFeed;

    // Constants
    uint256 public constant FLAT_FEE_USD = 100;       // $1 fee, but in cents ($1 * 100 cents)
    uint256 public constant ORDER_THRESHOLD_USD = 10000; // $100 order threshold (in cents)
    uint256 public constant PERCENT_FEE_BPS = 200;    // 2% fee, in basis points (bps = 1/100 of 1%)

    // Custom Errors
    error NotOwner();
    error FeeTooLow();
    error InvalidPriceFeed();
    error WithdrawFailed();
    error InvalidPriceData();

    // Events
    event FeePaid(address indexed user, uint256 amountETH, uint256 timestamp);
    event Withdraw(address indexed by, uint256 amount);

    constructor() {
        owner = msg.sender;
        priceFeed = AggregatorV3Interface(
            0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419 // Chainlink ETH/USD mainnet feed
        );
    }

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    /// @notice Main function for payments. Auto calculates fee.
    /// @param orderValueUSD Order value in **cents** (e.g., $99.99 = 9999)
    function pay(uint256 orderValueUSD) external payable {
        uint256 requiredFee = calculateFee(orderValueUSD);

        if (msg.value < requiredFee) revert FeeTooLow();

        emit FeePaid(msg.sender, msg.value, block.timestamp);
    }

    /// @notice Calculates ETH fee based on order size
    /// @param orderValueUSD Order amount in cents (e.g., $100 = 10000)
    function calculateFee(uint256 orderValueUSD) public view returns (uint256) {
        (, int256 price, , , ) = priceFeed.latestRoundData();
        if (price <= 0) revert InvalidPriceData();

        uint256 ethPrice = uint256(price); // ETH/USD price with 8 decimals

        uint256 usdFeeCents;

        if (orderValueUSD <= ORDER_THRESHOLD_USD) {
            // Flat fee for orders <= $100
            usdFeeCents = FLAT_FEE_USD * 100; // Convert dollars to cents
        } else {
            // 2% fee for orders > $100
            usdFeeCents = (orderValueUSD * PERCENT_FEE_BPS) / 10000;
        }

        // Convert USD cents to ETH (Chainlink price feed has 8 decimals, ETH is 18 decimals)
        uint256 usdFee = usdFeeCents * 1e8; // Convert to Chainlink format
        uint256 ethFee = (usdFee * 1e18) / ethPrice;

        return ethFee;
    }

    /// @notice Owner can withdraw collected ETH
    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        (bool success, ) = payable(owner).call{value: balance}("");
        if (!success) revert WithdrawFailed();

        emit Withdraw(owner, balance);
    }

    /// @notice Owner can update Chainlink price feed address if needed
    function updatePriceFeed(address newFeed) external onlyOwner {
        if (newFeed == address(0)) revert InvalidPriceFeed();
        require(newFeed.code.length > 0, "Address must be a contract");
        priceFeed = AggregatorV3Interface(newFeed);
    }

    /// @notice Helper function to check contract balance
    function getContractBalance() external view returns (uint256) {
        return address(this).balance;
    }
}