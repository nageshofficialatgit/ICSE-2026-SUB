// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @notice Inline Chainlink ETH/USD price feed interface (8 decimals)
interface AggregatorV3Interface {
    function decimals() external view returns (uint8);
    function description() external view returns (string memory);
    function version() external view returns (uint256);
    function getRoundData(uint80 _roundId)
        external
        view
        returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );
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

contract LoopTrustV1 {
    address public owner;
    AggregatorV3Interface internal priceFeed;

    uint256 public lastCallTimestamp;

    // Constants
    uint256 public constant INACTIVITY_THRESHOLD = 2 hours;
    uint256 public constant FEE_PERCENT_BPS = 700; // 7% = 700 bps
    uint256 public constant USD_FLAT_FEE = 50;     // 50 cents

    // Custom Errors
    error NotOwner();
    error FeeTooLow();
    error InvalidPriceFeed();
    error FeeTransferFailed();
    error WithdrawFailed();
    error InvalidPriceData();

    // Events
    event FeePaid(address indexed user, uint256 amountETH, uint256 timestamp);
    event Withdraw(address indexed by, uint256 amount);

    constructor() {
        owner = msg.sender;
        priceFeed = AggregatorV3Interface(
            0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419
        );  
    }   

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    function payWithFlatFee() external payable {
        uint256 ethFee = getFlatFeeInETH();
        if (msg.value < ethFee) revert FeeTooLow();

        lastCallTimestamp = block.timestamp;
        emit FeePaid(msg.sender, msg.value, block.timestamp);
    }

    function getFlatFeeInETH() public view returns (uint256) {
        (, int256 price,,,) = priceFeed.latestRoundData();
        if (price <= 0) revert InvalidPriceData();

        uint256 ethPrice = uint256(price); // Chainlink returns 8 decimals
        uint256 usdAmount = USD_FLAT_FEE * 1e8; // Convert cents to 8 decimals
        return (usdAmount * 1e18) / ethPrice;
    }

    function payWithPercent(uint256 transactionValue) external payable {
        uint256 requiredFee = (transactionValue * FEE_PERCENT_BPS) / 10000;
        if (msg.value < requiredFee) revert FeeTooLow();

        lastCallTimestamp = block.timestamp;
        emit FeePaid(msg.sender, msg.value, block.timestamp);
    }

    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        (bool sent, ) = payable(owner).call{value: balance}("");
        if (!sent) revert WithdrawFailed();
        emit Withdraw(owner, balance);
    }

    function timeSinceLastCall() external view returns (uint256) {
        return block.timestamp - lastCallTimestamp;
    }

    function updatePriceFeed(address newFeed) external onlyOwner {
        if (newFeed == address(0)) revert InvalidPriceFeed();
        require(newFeed.code.length > 0, "Address must be a contract");
        priceFeed = AggregatorV3Interface(newFeed);
    }
}