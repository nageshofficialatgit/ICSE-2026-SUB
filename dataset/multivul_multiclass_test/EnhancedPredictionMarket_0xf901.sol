/**
 *Submitted for verification at Etherscan.io on 2025-03-31
*/

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
}

contract EnhancedPredictionMarket {
    address public owner;
    IERC20 public immutable x10funToken;

    struct Prediction {
        uint256 id;
        string title;
        uint256 createdAt;
        uint256 endTime;
        uint256 yesPool;
        uint256 noPool;
        bool resolved;
        bool outcome;
        uint256 totalParticipants;
    }

    struct UserBet {
        uint256 yesAmount;
        uint256 noAmount;
        bool claimed;
    }

    mapping(uint256 => Prediction) public predictions;
    mapping(uint256 => mapping(address => UserBet)) public userBets;
    uint256 public nextPredictionId = 1;
    uint256 public platformFee = 20;
    uint256 public totalFees;
    uint256 public constant MAX_FEE = 50;
    uint256 public minBetAmount = 1000 * 1e18;
    bool private locked;

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier nonReentrant() {
        require(!locked, "Reentrant call");
        locked = true;
        _;
        locked = false;
    }

    event PredictionCreated(uint256 indexed id, string title, uint256 createdAt, uint256 endTime);
    event BetPlaced(uint256 indexed predictionId, address indexed user, bool choice, uint256 amount);
    event PredictionResolved(uint256 indexed predictionId, bool outcome);
    event WinningsClaimed(uint256 indexed predictionId, address indexed user, uint256 amount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event MinBetChanged(uint256 newMinBet);

    constructor(address _x10funToken) {
        owner = msg.sender;
        x10funToken = IERC20(_x10funToken);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid owner");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function createPrediction(string memory _title, uint256 _duration) external onlyOwner {
        predictions[nextPredictionId] = Prediction({
            id: nextPredictionId,
            title: _title,
            createdAt: block.timestamp,
            endTime: block.timestamp + _duration,
            yesPool: 0,
            noPool: 0,
            resolved: false,
            outcome: false,
            totalParticipants: 0
        });
        emit PredictionCreated(nextPredictionId, _title, block.timestamp, block.timestamp + _duration);
        nextPredictionId++;
    }

    function placeBet(uint256 _predictionId, bool _choice, uint256 _amount) external nonReentrant {
        require(_predictionId < nextPredictionId, "Invalid prediction");
        Prediction storage p = predictions[_predictionId];
        require(!p.resolved, "Prediction resolved");
        require(block.timestamp < p.endTime, "Prediction ended");
        require(_amount >= minBetAmount, "Total amount below minimum");

        uint256 fee = (_amount * platformFee) / 1000;
        uint256 betAmount = _amount - fee;

        require(x10funToken.transferFrom(msg.sender, address(this), _amount), "Transfer failed");
        totalFees += fee;

        UserBet storage bet = userBets[_predictionId][msg.sender];
        if (_choice) {
            p.yesPool += betAmount;
            bet.yesAmount += betAmount;
        } else {
            p.noPool += betAmount;
            bet.noAmount += betAmount;
        }

        if (bet.yesAmount + bet.noAmount == betAmount) {
            p.totalParticipants++;
        }

        emit BetPlaced(_predictionId, msg.sender, _choice, betAmount);
    }

    function resolvePrediction(uint256 _predictionId, bool _outcome) external onlyOwner {
        Prediction storage p = predictions[_predictionId];
        require(!p.resolved, "Already resolved");
        require(block.timestamp >= p.endTime, "Not ended");
        p.resolved = true;
        p.outcome = _outcome;
        emit PredictionResolved(_predictionId, _outcome);
    }

    function claimWinnings(uint256 _predictionId) external nonReentrant {
        Prediction storage p = predictions[_predictionId];
        UserBet storage bet = userBets[_predictionId][msg.sender];
        require(p.resolved, "Not resolved");
        require(!bet.claimed, "Already claimed");

        uint256 winningAmount;
        if (p.outcome) {
            require(bet.yesAmount > 0, "No winning bet");
            winningAmount = (bet.yesAmount * (p.yesPool + p.noPool)) / p.yesPool;
        } else {
            require(bet.noAmount > 0, "No winning bet");
            winningAmount = (bet.noAmount * (p.yesPool + p.noPool)) / p.noPool;
        }

        bet.claimed = true;
        require(x10funToken.transfer(msg.sender, winningAmount), "Transfer failed");
        emit WinningsClaimed(_predictionId, msg.sender, winningAmount);
    }

    function withdrawFees() external onlyOwner {
        uint256 amount = totalFees;
        totalFees = 0;
        require(x10funToken.transfer(owner, amount), "Transfer failed");
    }

    function setPlatformFee(uint256 _newFee) external onlyOwner {
        require(_newFee <= MAX_FEE, "Fee too high");
        platformFee = _newFee;
    }

    function setMinBetAmount(uint256 _newMin) external onlyOwner {
        minBetAmount = _newMin;
        emit MinBetChanged(_newMin);
    }

    function getPredictionsPaginated(uint256 page, uint256 pageSize) external view returns (Prediction[] memory) {
        uint256 totalPredictions = nextPredictionId - 1;
        uint256 start = page * pageSize;
        if (start >= totalPredictions) return new Prediction[](0);
        
        uint256 end = start + pageSize;
        if (end > totalPredictions) end = totalPredictions;
        
        Prediction[] memory result = new Prediction[](end - start);
        for (uint256 i = 0; i < end - start; i++) {
            result[i] = predictions[start + i + 1];
        }
        return result;
    }

    function getAllPredictions() external view returns (Prediction[] memory) {
        Prediction[] memory all = new Prediction[](nextPredictionId - 1);
        for (uint256 i = 1; i < nextPredictionId; i++) {
            all[i-1] = predictions[i];
        }
        return all;
    }

    function getUserBet(uint256 _predictionId, address _user) external view returns (UserBet memory) {
        return userBets[_predictionId][_user];
    }
}