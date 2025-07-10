// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @dev Minimal ERC20 interface for token rescue.
interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

/// @dev Chainlink aggregator interface for price data.
interface AggregatorV3Interface {
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

/**
 * @title Qrz112Reader
 *
 * This contract is supported by HypeLoot.com
 * Stay connected with us:
 * - Website: https://hypeloot.com/
 * - X (formerly Twitter): https://x.com/HypeLootCom
 * - Telegram Channel: https://t.me/Hypelootcom
 * - Instagram: https://instagram.com/hypelootcom
 *
 * For platform support, please email: support[at]hypeloot.com
 * We are continuously expanding our activities.
 *
 * @notice The Qrz112Reader contract integrates with Chainlink oracles to retrieve BTC/USD and ETH/USD price data,
 *         ensuring the data is both valid and fresh before performing computations. Its primary function, `getRatio()`,
 *         calculates and returns the integer square root (using the Babylonian method) of the average of the two price feeds.
 * 
 *         Key features of the contract include:
 *         - Secure storage and management of Chainlink oracle addresses for BTC and ETH, with built‐in checks to confirm that
 *           the provided prices are positive and the data is current.
 *         - Configurable data freshness control via a maximum delay parameter (maxDelay), which prevents the use of stale oracle data.
 *         - Additional functions that not only return individual BTC and ETH prices, but also compute their integer ratio,
 *           and pack two fixed-point (scaled by 1e18) ratio values into a single bytes32 output for efficient on-chain access.
 *         - A robust ownership model that restricts sensitive operations—such as updating oracle addresses, modifying parameters,
 *           transferring ownership, and rescuing stuck ERC20 tokens or ETH—to the contract owner.
 *         - Emission of detailed events to log important state changes including updates to oracle addresses, the maxDelay value,
 *           ownership transfers, and rescue operations.
 */
contract Qrz112Reader {
    //--------------------------------------------------------------------------------
    // Events
    //--------------------------------------------------------------------------------

    /** 
     * @dev Emitted when the contract owner is changed.
     * @param previousOwner The address of the previous owner.
     * @param newOwner The address of the new owner.
     */
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /** 
     * @dev Emitted when the BTC oracle address is updated.
     * @param oldBtcOracle The old BTC oracle address.
     * @param newBtcOracle The new BTC oracle address.
     */
    event BtcOracleSet(address indexed oldBtcOracle, address indexed newBtcOracle);

    /** 
     * @dev Emitted when the ETH oracle address is updated.
     * @param oldEthOracle The old ETH oracle address.
     * @param newEthOracle The new ETH oracle address.
     */
    event EthOracleSet(address indexed oldEthOracle, address indexed newEthOracle);

    /** 
     * @dev Emitted when the maxDelay value is updated.
     * @param oldDelay The old maxDelay value (in seconds).
     * @param newDelay The new maxDelay value (in seconds).
     */
    event MaxDelaySet(uint256 oldDelay, uint256 newDelay);

    /**
     * @dev Emitted when stuck ERC20 tokens are rescued by the owner.
     * @param token The address of the ERC20 token rescued.
     * @param amount The amount of tokens rescued.
     */
    event ERC20Rescued(address indexed token, uint256 amount);

    /**
     * @dev Emitted when stuck ETH is rescued by the owner.
     * @param amount The amount of ETH rescued (in wei).
     */
    event ETHRescued(uint256 amount);

    //--------------------------------------------------------------------------------
    // Ownership
    //--------------------------------------------------------------------------------

    /**
     * @dev The owner of this contract, assigned at construction time.
     */
    address public owner;

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(msg.sender == owner, "Qrz112Reader: caller is not the owner");
        _;
    }

    /**
     * @dev Sets the deployer as the initial owner upon contract creation and emits an event.
     */
    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }

    /**
     * @dev Transfers ownership of the contract to a new address.
     *      Cannot be the zero address.
     * @param newOwner The address that will become the new owner.
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Qrz112Reader: new owner is zero address");
        address oldOwner = owner;
        owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }

    //--------------------------------------------------------------------------------
    // Oracle Addresses
    //--------------------------------------------------------------------------------

    /**
     * @dev The address of the BTC/USD Chainlink aggregator contract.
     */
    address public btcOracle;

    /**
     * @dev The address of the ETH/USD Chainlink aggregator contract.
     */
    address public ethOracle;

    /**
     * @dev Sets the address of the BTC/USD oracle. Only callable by the owner.
     *      Also checks that the price at the time of setting is positive.
     *      Emits BtcOracleSet event.
     * @param _btcOracle The Chainlink aggregator address for BTC/USD.
     */
    function setBtcOracle(address _btcOracle) external onlyOwner {
        require(_btcOracle != address(0), "Qrz112Reader: invalid btc oracle address");

        AggregatorV3Interface aggregator = AggregatorV3Interface(_btcOracle);
        (, int256 price, , , ) = aggregator.latestRoundData();
        require(price > 0, "Qrz112Reader: BTC price is zero or negative");

        address oldBtcOracle = btcOracle;
        btcOracle = _btcOracle;
        emit BtcOracleSet(oldBtcOracle, _btcOracle);
    }

    /**
     * @dev Sets the address of the ETH/USD oracle. Only callable by the owner.
     *      Also checks that the price at the time of setting is positive.
     *      Emits EthOracleSet event.
     * @param _ethOracle The Chainlink aggregator address for ETH/USD.
     */
    function setEthOracle(address _ethOracle) external onlyOwner {
        require(_ethOracle != address(0), "Qrz112Reader: invalid eth oracle address");

        AggregatorV3Interface aggregator = AggregatorV3Interface(_ethOracle);
        (, int256 price, , , ) = aggregator.latestRoundData();
        require(price > 0, "Qrz112Reader: ETH price is zero or negative");

        address oldEthOracle = ethOracle;
        ethOracle = _ethOracle;
        emit EthOracleSet(oldEthOracle, _ethOracle);
    }

    //--------------------------------------------------------------------------------
    // Price Freshness Control
    //--------------------------------------------------------------------------------

    /**
     * @dev Maximum allowed delay (in seconds) for the oracle data. If the `updatedAt` 
     *      of the oracle data is older than `block.timestamp - maxDelay`, the data is considered stale.
     */
    uint256 public maxDelay;

    /**
     * @dev Sets the maximum allowed delay for the oracle data. Only callable by the owner.
     *      Delay in seconds must not exceed 48 hours (172800 seconds).
     *      Emits MaxDelaySet event.
     * @param _maxDelay The new maximum delay in seconds.
     */
    function setMaxDelay(uint256 _maxDelay) external onlyOwner {
        require(_maxDelay <= 172800, "Qrz112Reader: maxDelay cannot exceed 48 hours");
        uint256 oldDelay = maxDelay;
        maxDelay = _maxDelay;
        emit MaxDelaySet(oldDelay, _maxDelay);
    }

    //--------------------------------------------------------------------------------
    // Main Calculation: getRatio
    //--------------------------------------------------------------------------------

    /**
     * @dev Retrieves BTC and ETH prices from their respective Chainlink oracles, 
     *      verifies that the data is not stale, calculates the average of the two prices, 
     *      and then returns the square root of that average.
     *
     *      Checks performed:
     *       - Oracles must be set (non-zero addresses).
     *       - Each oracle's latest price must be positive (int256 answer > 0).
     *       - `answeredInRound >= roundId` indicating valid aggregator data.
     *       - `updatedAt >= block.timestamp - maxDelay` if `maxDelay > 0`, ensuring fresh data.
     *
     * @return The integer square root of the average BTC/USD and ETH/USD price, as a uint256.
     */
    function getRatio() external view returns (uint256) {
        require(btcOracle != address(0), "Qrz112Reader: btc oracle not set");
        require(ethOracle != address(0), "Qrz112Reader: eth oracle not set");

        // Fetch latest BTC price data
        (
            uint80 btcRoundId,
            int256 btcAnswer,
            ,
            uint256 btcUpdatedAt,
            uint80 btcAnsweredInRound
        ) = AggregatorV3Interface(btcOracle).latestRoundData();

        // Fetch latest ETH price data
        (
            uint80 ethRoundId,
            int256 ethAnswer,
            ,
            uint256 ethUpdatedAt,
            uint80 ethAnsweredInRound
        ) = AggregatorV3Interface(ethOracle).latestRoundData();

        // Validate BTC data
        require(btcAnswer > 0, "Qrz112Reader: BTC price is zero or negative");
        require(btcAnsweredInRound >= btcRoundId, "Qrz112Reader: invalid BTC round data");
        if (maxDelay > 0) {
            require(btcUpdatedAt >= block.timestamp - maxDelay, "Qrz112Reader: BTC price is stale");
        }

        // Validate ETH data
        require(ethAnswer > 0, "Qrz112Reader: ETH price is zero or negative");
        require(ethAnsweredInRound >= ethRoundId, "Qrz112Reader: invalid ETH round data");
        if (maxDelay > 0) {
            require(ethUpdatedAt >= block.timestamp - maxDelay, "Qrz112Reader: ETH price is stale");
        }

        // Calculate the average
        uint256 avg = (uint256(btcAnswer) + uint256(ethAnswer)) / 2;

        // Return the square root of the average
        return _sqrt(avg);
    }

    //--------------------------------------------------------------------------------
    // Additional Functions: getPricesAndRatio & getRatiosFixedPoint
    //--------------------------------------------------------------------------------

    /**
     * @dev Returns three values:
     *       1. The current BTC/USD price (as uint256).
     *       2. The current ETH/USD price (as uint256).
     *       3. The integer ratio (BTC price / ETH price).
     *
     *      Similar checks for price staleness and validity apply here as in `getRatio()`.
     *
     * @return btcPrice    Current BTC/USD price as uint256.
     * @return ethPrice    Current ETH/USD price as uint256.
     * @return btcEthRatio The integer division result of (btcPrice / ethPrice).
     */
    function getPricesAndRatio()
        external
        view
        returns (
            uint256 btcPrice,
            uint256 ethPrice,
            uint256 btcEthRatio
        )
    {
        require(btcOracle != address(0), "Qrz112Reader: btc oracle not set");
        require(ethOracle != address(0), "Qrz112Reader: eth oracle not set");

        // Fetch latest BTC price data
        (
            uint80 btcRoundId,
            int256 btcAnswer,
            ,
            uint256 btcUpdatedAt,
            uint80 btcAnsweredInRound
        ) = AggregatorV3Interface(btcOracle).latestRoundData();

        // Fetch latest ETH price data
        (
            uint80 ethRoundId,
            int256 ethAnswer,
            ,
            uint256 ethUpdatedAt,
            uint80 ethAnsweredInRound
        ) = AggregatorV3Interface(ethOracle).latestRoundData();

        // Validate BTC data
        require(btcAnswer > 0, "Qrz112Reader: BTC price is zero or negative");
        require(btcAnsweredInRound >= btcRoundId, "Qrz112Reader: invalid BTC round data");
        if (maxDelay > 0) {
            require(btcUpdatedAt >= block.timestamp - maxDelay, "Qrz112Reader: BTC price is stale");
        }

        // Validate ETH data
        require(ethAnswer > 0, "Qrz112Reader: ETH price is zero or negative");
        require(ethAnsweredInRound >= ethRoundId, "Qrz112Reader: invalid ETH round data");
        if (maxDelay > 0) {
            require(ethUpdatedAt >= block.timestamp - maxDelay, "Qrz112Reader: ETH price is stale");
        }

        // Convert to uint256
        btcPrice = uint256(btcAnswer);
        ethPrice = uint256(ethAnswer);

        // Calculate ratio as integer division
        btcEthRatio = btcPrice / ethPrice;
    }

    /**
     * @dev Returns a single bytes32 value containing two fixed-point (wad, 1e18) ratio values.
     * The most significant 128 bits represent the square root of the average BTC/ETH price (sqrtAvgRatioFP),
     * and the least significant 128 bits represent the BTC/ETH ratio (btcEthRatioFP).
     * This packing scheme combines both 128-bit values into one 256-bit value for efficient storage and transfer.
     */
    function getRatiosFixedPoint() external view returns (bytes32 packedRatios) {
        // Retrieve the integer square root of the average price (validated inside getRatio())
        uint256 sqrtAvgRatio = this.getRatio();
        
        // Retrieve the (btcPrice, ethPrice, ratio) (validated inside getPricesAndRatio())
        ( , , uint256 rawBtcEthRatio) = this.getPricesAndRatio();
        
        // Scale values to fixed-point format (wad, 1e18)
        uint256 sqrtAvgRatioFP = sqrtAvgRatio * 1e18;
        uint256 btcEthRatioFP = rawBtcEthRatio * 1e18;
        
        // Ensure that both values fit in 128 bits
        require(sqrtAvgRatioFP < 2**128, "sqrtAvgRatioFP exceeds 128 bits");
        require(btcEthRatioFP < 2**128, "btcEthRatioFP exceeds 128 bits");
        
        // Pack the two 128-bit values into one bytes32:
        // - The upper 128 bits store sqrtAvgRatioFP.
        // - The lower 128 bits store btcEthRatioFP.
        packedRatios = bytes32((sqrtAvgRatioFP << 128) | btcEthRatioFP);
    }

    //--------------------------------------------------------------------------------
    // Rescue / Claim Functions (onlyOwner)
    //--------------------------------------------------------------------------------

    /**
     * @dev Allows the contract owner to rescue (transfer out) all ERC20 tokens held by this contract.
     *      Emits an ERC20Rescued event.
     * @param token The ERC20 token contract address.
     */
    function rescueERC20(address token) external onlyOwner {
        require(token != address(0), "Qrz112Reader: invalid token address");

        uint256 balance = IERC20(token).balanceOf(address(this));
        require(balance > 0, "Qrz112Reader: no tokens to rescue");

        bool success = IERC20(token).transfer(owner, balance);
        require(success, "Qrz112Reader: ERC20 transfer failed");

        emit ERC20Rescued(token, balance);
    }

    /**
     * @dev Allows the owner to rescue (transfer out) stuck ETH.
     *      Emits ETHRescued event.
     */
    function rescueETH() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "Qrz112Reader: no ETH to rescue");

        (bool success, ) = owner.call{value: balance}("");
        require(success, "Qrz112Reader: ETH transfer failed");

        emit ETHRescued(balance);
    }

    //--------------------------------------------------------------------------------
    // Internal square root function
    //--------------------------------------------------------------------------------

    /**
     * @dev Internal integer square root function using the Babylonian method.
     *      Returns the floor of the square root of x.
     * @param x The number to compute the square root for.
     * @return The largest integer less than or equal to the square root of x.
     */
    function _sqrt(uint256 x) internal pure returns (uint256) {
        if (x == 0) {
            return 0;
        }

        uint256 z = (x + 1) / 2;
        uint256 y = x;

        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
        return y;
    }
}