pragma solidity ^0.4.19;
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    assert(c / a == b);
    return c;
  }
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a / b;
    return c;
  }
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }
  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}
contract Ownable {
  address public owner;
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
  function Ownable() public {
    owner = msg.sender;
  }
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }
  function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }
}
contract Claimable is Ownable {
  address public pendingOwner;
  modifier onlyPendingOwner() {
    require(msg.sender == pendingOwner);
    _;
  }
  function transferOwnership(address newOwner) onlyOwner public {
    pendingOwner = newOwner;
  }
  function claimOwnership() onlyPendingOwner public {
    OwnershipTransferred(owner, pendingOwner);
    owner = pendingOwner;
    pendingOwner = address(0);
  }
}
contract Pausable is Ownable {
  event Pause();
  event Unpause();
  bool public paused = false;
  modifier whenNotPaused() {
    require(!paused);
    _;
  }
  modifier whenPaused() {
    require(paused);
    _;
  }
  function pause() onlyOwner whenNotPaused public {
    paused = true;
    Pause();
  }
  function unpause() onlyOwner whenPaused public {
    paused = false;
    Unpause();
  }
}
contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}
contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender) public view returns (uint256);
  function transferFrom(address from, address to, uint256 value) public returns (bool);
  function approve(address spender, uint256 value) public returns (bool);
  event Approval(address indexed owner, address indexed spender, uint256 value);
}
library SafeERC20 {
  function safeTransfer(ERC20Basic token, address to, uint256 value) internal {
    assert(token.transfer(to, value));
  }
  function safeTransferFrom(ERC20 token, address from, address to, uint256 value) internal {
    assert(token.transferFrom(from, to, value));
  }
  function safeApprove(ERC20 token, address spender, uint256 value) internal {
    assert(token.approve(spender, value));
  }
}
contract CanReclaimToken is Ownable {
  using SafeERC20 for ERC20Basic;
  function reclaimToken(ERC20Basic token) external onlyOwner {
    uint256 balance = token.balanceOf(this);
    token.safeTransfer(owner, balance);
  }
}
contract BurnupGameAccessControl is Claimable, Pausable, CanReclaimToken {
    mapping (address => bool) public cfo;
    function BurnupGameAccessControl() public {
        cfo[msg.sender] = true;
    }
    modifier onlyCFO() {
        require(cfo[msg.sender]);
        _;
    }
    function setCFO(address addr, bool set) external onlyOwner {
        require(addr != address(0));
        if (!set) {
            delete cfo[addr];
        } else {
            cfo[addr] = true;
        }
    }
}
contract BurnupGameBase is BurnupGameAccessControl {
    using SafeMath for uint256;
    event ActiveTimes(uint256[] from, uint256[] to);
    event AllowStart(bool allowStart);
    event NextGame(
        uint256 rows,
        uint256 cols,
        uint256 initialActivityTimer,
        uint256 finalActivityTimer,
        uint256 numberOfFlipsToFinalActivityTimer,
        uint256 timeoutBonusTime,
        uint256 unclaimedTilePrice,
        uint256 buyoutReferralBonusPercentage,
        uint256 firstBuyoutPrizePoolPercentage,
        uint256 buyoutPrizePoolPercentage,
        uint256 buyoutDividendPercentage,
        uint256 buyoutFeePercentage,
        uint256 buyoutPriceIncreasePercentage
    );
    event Start(
        uint256 indexed gameIndex,
        address indexed starter,
        uint256 timestamp,
        uint256 prizePool
    );
    event End(uint256 indexed gameIndex, address indexed winner, uint256 indexed identifier, uint256 x, uint256 y, uint256 timestamp, uint256 prize);
    event Buyout(
        uint256 indexed gameIndex,
        address indexed player,
        uint256 indexed identifier,
        uint256 x,
        uint256 y,
        uint256 timestamp,
        uint256 timeoutTimestamp,
        uint256 newPrice,
        uint256 newPrizePool
    );
    event LastTile(
        uint256 indexed gameIndex,
        uint256 indexed identifier,
        uint256 x,
        uint256 y
    );
    event PenultimateTileTimeout(
        uint256 indexed gameIndex,
        uint256 timeoutTimestamp
    );
    event SpiceUpPrizePool(uint256 indexed gameIndex, address indexed spicer, uint256 spiceAdded, string message, uint256 newPrizePool);
    struct GameSettings {
        uint256 rows; 
        uint256 cols; 
        uint256 initialActivityTimer; 
        uint256 finalActivityTimer; 
        uint256 numberOfFlipsToFinalActivityTimer; 
        uint256 timeoutBonusTime; 
        uint256 unclaimedTilePrice; 
        uint256 buyoutReferralBonusPercentage; 
        uint256 firstBuyoutPrizePoolPercentage; 
        uint256 buyoutPrizePoolPercentage; 
        uint256 buyoutDividendPercentage; 
        uint256 buyoutFeePercentage; 
        uint256 buyoutPriceIncreasePercentage;
    }
    struct GameState {
        bool gameStarted;
        uint256 gameStartTimestamp;
        mapping (uint256 => address) identifierToOwner;
        mapping (uint256 => uint256) identifierToTimeoutTimestamp;
        mapping (uint256 => uint256) identifierToBuyoutPrice;
        mapping (address => uint256) addressToNumberOfTiles;
        uint256 numberOfTileFlips;
        uint256 lastTile;
        uint256 penultimateTileTimeout;
        uint256 prizePool;
    }
    mapping (uint256 => GameState) public gameStates;
    uint256 public gameIndex = 0;
    GameSettings public gameSettings;
    GameSettings public nextGameSettings;
    uint256[] public activeTimesFrom;
    uint256[] public activeTimesTo;
    bool public allowStart;
    function BurnupGameBase() public {
        setNextGameSettings(
            4, 
            5, 
            300, 
            150, 
            5, 
            30, 
            0.01 ether, 
            750, 
            40000, 
            10000, 
            5000, 
            2500, 
            150000 
        );
    }
    function validCoordinate(uint256 x, uint256 y) public view returns(bool) {
        return x < gameSettings.cols && y < gameSettings.rows;
    }
    function coordinateToIdentifier(uint256 x, uint256 y) public view returns(uint256) {
        require(validCoordinate(x, y));
        return (y * gameSettings.cols) + x + 1;
    }
    function identifierToCoordinate(uint256 identifier) public view returns(uint256 x, uint256 y) {
        y = (identifier - 1) / gameSettings.cols;
        x = (identifier - 1) - (y * gameSettings.cols);
    }
    function setNextGameSettings(
        uint256 rows,
        uint256 cols,
        uint256 initialActivityTimer,
        uint256 finalActivityTimer,
        uint256 numberOfFlipsToFinalActivityTimer,
        uint256 timeoutBonusTime,
        uint256 unclaimedTilePrice,
        uint256 buyoutReferralBonusPercentage,
        uint256 firstBuyoutPrizePoolPercentage,
        uint256 buyoutPrizePoolPercentage,
        uint256 buyoutDividendPercentage,
        uint256 buyoutFeePercentage,
        uint256 buyoutPriceIncreasePercentage
    )
        public
        onlyCFO
    {
        require(2000 <= buyoutDividendPercentage && buyoutDividendPercentage <= 12500);
        require(buyoutFeePercentage <= 5000);
        if (numberOfFlipsToFinalActivityTimer == 0) {
            require(initialActivityTimer == finalActivityTimer);
        }
        nextGameSettings = GameSettings({
            rows: rows,
            cols: cols,
            initialActivityTimer: initialActivityTimer,
            finalActivityTimer: finalActivityTimer,
            numberOfFlipsToFinalActivityTimer: numberOfFlipsToFinalActivityTimer,
            timeoutBonusTime: timeoutBonusTime,
            unclaimedTilePrice: unclaimedTilePrice,
            buyoutReferralBonusPercentage: buyoutReferralBonusPercentage,
            firstBuyoutPrizePoolPercentage: firstBuyoutPrizePoolPercentage,
            buyoutPrizePoolPercentage: buyoutPrizePoolPercentage,
            buyoutDividendPercentage: buyoutDividendPercentage,
            buyoutFeePercentage: buyoutFeePercentage,
            buyoutPriceIncreasePercentage: buyoutPriceIncreasePercentage
        });
        NextGame(
            rows,
            cols,
            initialActivityTimer,
            finalActivityTimer,
            numberOfFlipsToFinalActivityTimer,
            timeoutBonusTime,
            unclaimedTilePrice,
            buyoutReferralBonusPercentage, 
            firstBuyoutPrizePoolPercentage,
            buyoutPrizePoolPercentage,
            buyoutDividendPercentage,
            buyoutFeePercentage,
            buyoutPriceIncreasePercentage
        );
    }
    function setActiveTimes(uint256[] _from, uint256[] _to) external onlyCFO {
        require(_from.length == _to.length);
        activeTimesFrom = _from;
        activeTimesTo = _to;
        ActiveTimes(_from, _to);
    }
    function setAllowStart(bool _allowStart) external onlyCFO {
        allowStart = _allowStart;
        AllowStart(_allowStart);
    }
    function canStart() public view returns (bool) {
        uint256 timeOfWeek = (block.timestamp - 345600) % 604800;
        uint256 windows = activeTimesFrom.length;
        if (windows == 0) {
            return true;
        }
        for (uint256 i = 0; i < windows; i++) {
            if (timeOfWeek >= activeTimesFrom[i] && timeOfWeek <= activeTimesTo[i]) {
                return true;
            }
        }
        return false;
    }
    function calculateBaseTimeout() public view returns(uint256) {
        uint256 _numberOfTileFlips = gameStates[gameIndex].numberOfTileFlips;
        if (_numberOfTileFlips >= gameSettings.numberOfFlipsToFinalActivityTimer || gameSettings.numberOfFlipsToFinalActivityTimer == 0) {
            return gameSettings.finalActivityTimer;
        } else {
            if (gameSettings.finalActivityTimer <= gameSettings.initialActivityTimer) {
                uint256 difference = gameSettings.initialActivityTimer - gameSettings.finalActivityTimer;
                uint256 decrease = difference.mul(_numberOfTileFlips).div(gameSettings.numberOfFlipsToFinalActivityTimer);
                return (gameSettings.initialActivityTimer - decrease);
            } else {
                difference = gameSettings.finalActivityTimer - gameSettings.initialActivityTimer;
                uint256 increase = difference.mul(_numberOfTileFlips).div(gameSettings.numberOfFlipsToFinalActivityTimer);
                return (gameSettings.initialActivityTimer + increase);
            }
        }
    }
    function tileTimeoutTimestamp(uint256 identifier, address player) public view returns (uint256) {
        uint256 bonusTime = gameSettings.timeoutBonusTime.mul(gameStates[gameIndex].addressToNumberOfTiles[player]);
        uint256 timeoutTimestamp = block.timestamp.add(calculateBaseTimeout()).add(bonusTime);
        uint256 currentTimeoutTimestamp = gameStates[gameIndex].identifierToTimeoutTimestamp[identifier];
        if (currentTimeoutTimestamp == 0) {
            currentTimeoutTimestamp = gameStates[gameIndex].gameStartTimestamp.add(gameSettings.initialActivityTimer);
        }
        if (timeoutTimestamp >= currentTimeoutTimestamp) {
            return timeoutTimestamp;
        } else {
            return currentTimeoutTimestamp;
        }
    }
    function _setGameSettings() internal {
        if (gameSettings.rows != nextGameSettings.rows) {
            gameSettings.rows = nextGameSettings.rows;
        }
        if (gameSettings.cols != nextGameSettings.cols) {
            gameSettings.cols = nextGameSettings.cols;
        }
        if (gameSettings.initialActivityTimer != nextGameSettings.initialActivityTimer) {
            gameSettings.initialActivityTimer = nextGameSettings.initialActivityTimer;
        }
        if (gameSettings.finalActivityTimer != nextGameSettings.finalActivityTimer) {
            gameSettings.finalActivityTimer = nextGameSettings.finalActivityTimer;
        }
        if (gameSettings.numberOfFlipsToFinalActivityTimer != nextGameSettings.numberOfFlipsToFinalActivityTimer) {
            gameSettings.numberOfFlipsToFinalActivityTimer = nextGameSettings.numberOfFlipsToFinalActivityTimer;
        }
        if (gameSettings.timeoutBonusTime != nextGameSettings.timeoutBonusTime) {
            gameSettings.timeoutBonusTime = nextGameSettings.timeoutBonusTime;
        }
        if (gameSettings.unclaimedTilePrice != nextGameSettings.unclaimedTilePrice) {
            gameSettings.unclaimedTilePrice = nextGameSettings.unclaimedTilePrice;
        }
        if (gameSettings.buyoutReferralBonusPercentage != nextGameSettings.buyoutReferralBonusPercentage) {
            gameSettings.buyoutReferralBonusPercentage = nextGameSettings.buyoutReferralBonusPercentage;
        }
        if (gameSettings.firstBuyoutPrizePoolPercentage != nextGameSettings.firstBuyoutPrizePoolPercentage) {
            gameSettings.firstBuyoutPrizePoolPercentage = nextGameSettings.firstBuyoutPrizePoolPercentage;
        }
        if (gameSettings.buyoutPrizePoolPercentage != nextGameSettings.buyoutPrizePoolPercentage) {
            gameSettings.buyoutPrizePoolPercentage = nextGameSettings.buyoutPrizePoolPercentage;
        }
        if (gameSettings.buyoutDividendPercentage != nextGameSettings.buyoutDividendPercentage) {
            gameSettings.buyoutDividendPercentage = nextGameSettings.buyoutDividendPercentage;
        }
        if (gameSettings.buyoutFeePercentage != nextGameSettings.buyoutFeePercentage) {
            gameSettings.buyoutFeePercentage = nextGameSettings.buyoutFeePercentage;
        }
        if (gameSettings.buyoutPriceIncreasePercentage != nextGameSettings.buyoutPriceIncreasePercentage) {
            gameSettings.buyoutPriceIncreasePercentage = nextGameSettings.buyoutPriceIncreasePercentage;
        }
    }
}
contract BurnupGameOwnership is BurnupGameBase {
    event Transfer(address indexed from, address indexed to, uint256 indexed deedId);
    function name() public pure returns (string _deedName) {
        _deedName = "Burnup Tiles";
    }
    function symbol() public pure returns (string _deedSymbol) {
        _deedSymbol = "BURN";
    }
    function _owns(address _owner, uint256 _identifier) internal view returns (bool) {
        return gameStates[gameIndex].identifierToOwner[_identifier] == _owner;
    }
    function _transfer(address _from, address _to, uint256 _identifier) internal {
        gameStates[gameIndex].identifierToOwner[_identifier] = _to;
        if (_from != 0x0) {
            gameStates[gameIndex].addressToNumberOfTiles[_from] = gameStates[gameIndex].addressToNumberOfTiles[_from].sub(1);
        }
        gameStates[gameIndex].addressToNumberOfTiles[_to] = gameStates[gameIndex].addressToNumberOfTiles[_to].add(1);
        Transfer(_from, _to, _identifier);
    }
    function ownerOf(uint256 _identifier) external view returns (address _owner) {
        _owner = gameStates[gameIndex].identifierToOwner[_identifier];
        require(_owner != address(0));
    }
    function transfer(address _to, uint256 _identifier) external whenNotPaused {
        require(_owns(msg.sender, _identifier));
        _transfer(msg.sender, _to, _identifier);
    }
}
contract PullPayment {
  using SafeMath for uint256;
  mapping(address => uint256) public payments;
  uint256 public totalPayments;
  function withdrawPayments() public {
    address payee = msg.sender;
    uint256 payment = payments[payee];
    require(payment != 0);
    require(this.balance >= payment);
    totalPayments = totalPayments.sub(payment);
    payments[payee] = 0;
    assert(payee.send(payment));
  }
  function asyncSend(address dest, uint256 amount) internal {
    payments[dest] = payments[dest].add(amount);
    totalPayments = totalPayments.add(amount);
  }
}
contract BurnupHoldingAccessControl is Claimable, Pausable, CanReclaimToken {
    address public cfoAddress;
    mapping (address => bool) burnupGame;
    function BurnupHoldingAccessControl() public {
        cfoAddress = msg.sender;
    }
    modifier onlyCFO() {
        require(msg.sender == cfoAddress);
        _;
    }
    modifier onlyBurnupGame() {
        require(burnupGame[msg.sender]);
        _;
    }
    function setCFO(address _newCFO) external onlyOwner {
        require(_newCFO != address(0));
        cfoAddress = _newCFO;
    }
    function addBurnupGame(address addr) external onlyOwner {
        burnupGame[addr] = true;
    }
    function removeBurnupGame(address addr) external onlyOwner {
        delete burnupGame[addr];
    }
}
contract BurnupHoldingReferral is BurnupHoldingAccessControl {
    event SetReferrer(address indexed referral, address indexed referrer);
    mapping (address => address) addressToReferrerAddress;
    function referrerOf(address player) public view returns (address) {
        return addressToReferrerAddress[player];
    }
    function _setReferrer(address playerAddr, address referrerAddr) internal {
        addressToReferrerAddress[playerAddr] = referrerAddr;
        SetReferrer(playerAddr, referrerAddr);
    }
}
contract BurnupHoldingCore is BurnupHoldingReferral, PullPayment {
    using SafeMath for uint256;
    address public beneficiary1;
    address public beneficiary2;
    function BurnupHoldingCore(address _beneficiary1, address _beneficiary2) public {
        cfoAddress = msg.sender;
        beneficiary1 = _beneficiary1;
        beneficiary2 = _beneficiary2;
    }
    function payBeneficiaries() external payable {
        uint256 paymentHalve = msg.value.div(2);
        uint256 otherPaymentHalve = msg.value.sub(paymentHalve);
        asyncSend(beneficiary1, paymentHalve);
        asyncSend(beneficiary2, otherPaymentHalve);
    }
    function setBeneficiary1(address addr) external onlyCFO {
        beneficiary1 = addr;
    }
    function setBeneficiary2(address addr) external onlyCFO {
        beneficiary2 = addr;
    }
    function setReferrer(address playerAddr, address referrerAddr) external onlyBurnupGame whenNotPaused returns(bool) {
        if (referrerOf(playerAddr) == address(0x0) && playerAddr != referrerAddr) {
            _setReferrer(playerAddr, referrerAddr);
            return true;
        }
        return false;
    }
}
contract BurnupGameFinance is BurnupGameOwnership, PullPayment {
    BurnupHoldingCore burnupHolding;
    function BurnupGameFinance(address burnupHoldingAddress) public {
        burnupHolding = BurnupHoldingCore(burnupHoldingAddress);
    }