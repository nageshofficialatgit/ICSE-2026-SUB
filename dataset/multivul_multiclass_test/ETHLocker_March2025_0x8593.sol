//SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.28;

/**
 * @notice If you discover any vulnerabilities in this contract, please reach out to me at ethlocker.parade164@passinbox.com.
 *
 * Vulnerability reports will be rewarded with a bounty equal to 10% of the contract's funds.
 *
 */

interface I_chainlink {
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

    function description() external view returns (string memory);
 
    function latestTimestamp() external view returns (uint256);

    function getRoundData(uint80 roundId)
        external
        view
        returns (
            uint80,
            int256,
            uint256,
            uint256,
            uint80
        );
}

contract ETHLocker_March2025 {

    address payable public owner;
    uint256 public nextUnlockTime;
    uint256 public init_time;
    int256 public max_eth_usdt;
    int256 public min_eth_usdt;
    uint256 public hard_unlock_time;
   
    modifier onlyOwner() {
        require(
            msg.sender == owner,
            "Not authorized owner"
        );
        _;
    }

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event UnlockTimeUpdated(uint256 newUnlockTime);
    
    constructor() {
        owner = payable(msg.sender);
        max_eth_usdt = 3900 * 10**8;
        min_eth_usdt = 1200 * 10**8;
        require(
            max_eth_usdt > min_eth_usdt,
            "max target price should be greater than min"
        );

        hard_unlock_time = 1767225600; // hard unlock at Thursday, January 1, 2026 12:00:00 AM

        // Set nextUnlockTime to 2 months (approximated as 60 days) from now
        nextUnlockTime = block.timestamp + 60 days;
        init_time = block.timestamp;
    }

    function get_ETH_price(uint80 _roundId) public view returns (int256 price) {
        I_chainlink ChainLink = I_chainlink(
            0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419
        );
        if (_roundId != 0) {
            // Get historical round data and enforce a timestamp check
            (
                ,
                int256 historicalPrice,
                ,
                uint256 historicalTimestamp,
                // The returned answeredInRound is omitted
            ) = ChainLink.getRoundData(_roundId);
            require(
                historicalTimestamp > init_time,
                "historical data timestamp should be greater than init_time"
            );
            return historicalPrice;
        }

        // Get latest round data
        (, int256 latestPrice, , , ) = ChainLink.latestRoundData();
        return latestPrice;
    }

    /// @notice Updates the unlock time.
    /// @dev The new unlock time must be in the future and greater than the current nextUnlockTime.
    function setUnlockTime(uint256 newUnlockTime) external onlyOwner {
        require(
            newUnlockTime > block.timestamp,
            "Unlock time must be in the future"
        );
        require(
            newUnlockTime > nextUnlockTime,
            "New unlock time must be greater than current unlock time"
        );
        require(
            newUnlockTime < hard_unlock_time,
            "New unlock time must be less than hard unlock time"
        );
        nextUnlockTime = newUnlockTime;
        init_time = block.timestamp;

        emit UnlockTimeUpdated(newUnlockTime);
    }

    function canUnlock(uint80 _round_id) public view returns (bool) {
        if (block.timestamp > hard_unlock_time) return true;

        int256 ethPrice = get_ETH_price(_round_id);
        if (ethPrice == 0) return true;
        return (ethPrice > max_eth_usdt ||
            ethPrice < min_eth_usdt ||
            block.timestamp > nextUnlockTime);
    }

    function withdraw(uint80 _round_id) external onlyOwner {
        require(canUnlock(_round_id), "Unlock conditions not met");
        uint256 balance = address(this).balance;
        require(balance > 0, "No balance to withdraw");

        (bool sent, ) = owner.call{value: balance}("");
        require(sent, "Transfer failed");

        emit Withdrawn(owner, balance);
    }

    // Function to receive ETH
    receive() external payable {
        emit Deposited(msg.sender, msg.value);
    }

    // Fallback in case receive is not triggered
    fallback() external payable {
        emit Deposited(msg.sender, msg.value);
    }

    function backup_withdraw_2() external onlyOwner {
        // Ensure the withdrawal is allowed only after the hard unlock time.
        require(
            block.timestamp > hard_unlock_time,
            "current time should be > hard_unlock_time"
        );

        uint256 balance = address(this).balance;
        owner.transfer(balance);
    }
}