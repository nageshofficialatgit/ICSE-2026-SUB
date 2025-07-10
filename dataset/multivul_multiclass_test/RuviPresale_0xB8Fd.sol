// Ruvi AI Presale
//SPDX-License-Identifier: MIT Licensed
pragma solidity ^0.8.29;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    /**
     * @dev Initializes the contract setting the deployer as the initial owner.
     */
    constructor() {
        _transferOwnership(_msgSender());
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(
            newOwner != address(0),
            "Ownable: new owner is the zero address"
        );
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

interface IERC20 {
    function name() external view returns (string memory);

    function symbol() external view returns (string memory);

    function decimals() external view returns (uint8);

    function totalSupply() external view returns (uint256);

    function balanceOf(address owner) external view returns (uint256);

    function allowance(address owner, address spender)
        external
        view
        returns (uint256);

    function approve(address spender, uint256 value) external;

    function transfer(address to, uint256 value) external;

    function transferFrom(
        address from,
        address to,
        uint256 value
    ) external;

    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
    event Transfer(address indexed from, address indexed to, uint256 value);
}

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

contract RuviPresale is Ownable {
    IERC20 public mainToken;
    IERC20 public USDT = IERC20(0xdAC17F958D2ee523a2206206994597C13D831ec7);
    IERC20 public USDC = IERC20(0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48);

    AggregatorV3Interface public priceFeed;

    struct Phase {
        uint256 tokensToSell;
        uint256 totalSoldTokens;
        uint256 tokenPerUsdPrice;
    }
    mapping(uint256 => Phase) public phases;

    uint256 public totalStages;
    uint256 public currentStage;
    uint256 public soldToken;
    uint256 public amountRaised;
    uint256 public amountRaisedUSDT;
    uint256 public amountRaisedUSDC;
    uint256 public amountRaisedOverall;
    uint256 public uniqueBuyers;

    uint256[] public tokenPerUsdPrice = [
        100000000000000000000,
        66666666666666666666,
        50000000000000000000,
        35714285714285714285,
        28571428571428571428,
        20000000000000000000,
        14285714285714285714
    ];

    uint256[] public tokensToSell = [
        100_000_000 * 10**18,
        150_000_000 * 10**18,
        200_000_000 * 10**18,
        225_000_000 * 10**18,
        250_000_000 * 10**18,
        275_000_000 * 10**18,
        300_000_000 * 10**18
    ];

    address payable public fundReceiver;

    bool public presaleStatus;
    bool public isPresaleEnded;
    uint256 public claimStartTime;

    address[] public UsersAddresses;
    struct User {
        uint256 native_balance;
        uint256 ethToUsd_balance;
        uint256 usdt_balance;
        uint256 usdc_balance;
        uint256 token_balance;
    }

    mapping(address => User) public users;
    mapping(address => bool) public isExist;

    event BuyToken(address indexed _user, uint256 indexed _amount);
    event ClaimToken(address indexed _user, uint256 indexed _amount);
    event UpdatePrice(uint256 _oldPrice, uint256 _newPrice);

    constructor(IERC20 _token, address _fundReceiver) {
        mainToken = _token;
        fundReceiver = payable(_fundReceiver);
        priceFeed = AggregatorV3Interface(
            0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419
        );
        for (uint256 i = 0; i < tokensToSell.length; i++) {
            phases[i].tokensToSell = tokensToSell[i];
            phases[i].tokenPerUsdPrice = tokenPerUsdPrice[i];
        }
        totalStages = tokensToSell.length;
    }

    function updatePresale(
        uint256 _phaseId,
        uint256 _tokensToSell,
        uint256 _tokenPerUsdPrice
    ) public onlyOwner {
        require(phases[_phaseId].tokensToSell > 0, "presale doesn't exist");
        phases[_phaseId].tokensToSell = _tokensToSell;
        phases[_phaseId].tokenPerUsdPrice = _tokenPerUsdPrice;
    }

    function getLatestPrice() public view returns (uint256) {
        (, int256 price, , , ) = priceFeed.latestRoundData();
        return uint256(price);
    }

    function buyToken() public payable {
        require(!isPresaleEnded, "Presale ended.");
        require(presaleStatus, " Presale is currently Paused");
        if (!isExist[msg.sender]) {
            isExist[msg.sender] = true;
            uniqueBuyers++;
            UsersAddresses.push(msg.sender);
        }
        fundReceiver.transfer(msg.value);
        uint256 numberOfTokens;
        uint256 ethToUsdConverted;
        numberOfTokens = nativeToToken(msg.value, currentStage);
        require(
            phases[currentStage].totalSoldTokens + numberOfTokens <=
                phases[currentStage].tokensToSell,
            "Phase Limit Reached"
        );
        mainToken.transfer(msg.sender, numberOfTokens);
        soldToken = soldToken + (numberOfTokens);
        amountRaised = amountRaised + (msg.value);
        ethToUsdConverted = nativeToUsd(msg.value);
        amountRaisedOverall = amountRaisedOverall + ethToUsdConverted;

        users[msg.sender].native_balance += (msg.value);
        users[msg.sender].ethToUsd_balance += ethToUsdConverted;
        users[msg.sender].token_balance += numberOfTokens;
        phases[currentStage].totalSoldTokens += numberOfTokens;
    }

    function buyTokenUSDC(uint256 amount) public {
        require(!isPresaleEnded, "Presale ended.");
        require(presaleStatus, " Presale is currently Paused");
        if (!isExist[msg.sender]) {
            isExist[msg.sender] = true;
            uniqueBuyers++;
            UsersAddresses.push(msg.sender);
        }
        USDC.transferFrom(msg.sender, fundReceiver, amount);
        uint256 numberOfTokens;
        numberOfTokens = usdtToToken(amount, currentStage);
        require(
            phases[currentStage].totalSoldTokens + numberOfTokens <=
                phases[currentStage].tokensToSell,
            "Phase Limit Reached"
        );
        mainToken.transfer(msg.sender, numberOfTokens);
        soldToken = soldToken + numberOfTokens;
        amountRaisedUSDT = amountRaisedUSDT + amount;
        amountRaisedOverall = amountRaisedOverall + amount;

        users[msg.sender].usdc_balance += amount;
        users[msg.sender].token_balance += numberOfTokens;
        phases[currentStage].totalSoldTokens += numberOfTokens;
    }

    function buyTokenUSDT(uint256 amount) public {
        require(!isPresaleEnded, "Presale ended");
        require(presaleStatus, " Presale is currently Paused");
        if (!isExist[msg.sender]) {
            isExist[msg.sender] = true;
            uniqueBuyers++;
            UsersAddresses.push(msg.sender);
        }
        USDT.transferFrom(msg.sender, fundReceiver, amount);

        uint256 numberOfTokens;
        numberOfTokens = usdtToToken(amount, currentStage);
        require(
            phases[currentStage].totalSoldTokens + numberOfTokens <=
                phases[currentStage].tokensToSell,
            "Phase Limit Reached"
        );
        mainToken.transfer(msg.sender, numberOfTokens);
        soldToken = soldToken + numberOfTokens;
        amountRaisedUSDC = amountRaisedUSDC + amount;
        amountRaisedOverall = amountRaisedOverall + amount;

        users[msg.sender].usdt_balance += amount;
        users[msg.sender].token_balance += numberOfTokens;
        phases[currentStage].totalSoldTokens += numberOfTokens;
    }

    function getPhaseDetail(uint256 phaseInd)
        external
        view
        returns (
            uint256 tokenToSell,
            uint256 soldTokens,
            uint256 priceUsd
        )
    {
        Phase memory phase = phases[phaseInd];
        return (
            phase.tokensToSell,
            phase.totalSoldTokens,
            phase.tokenPerUsdPrice
        );
    }

    function setPresaleStatus(bool _status) external onlyOwner {
        presaleStatus = _status;
    }

    function endPresale(bool _status) external onlyOwner {
        isPresaleEnded = _status;
    }

    function nativeToToken(uint256 _amount, uint256 phaseId)
        public
        view
        returns (uint256)
    {
        uint256 ethToUsd = (_amount * (getLatestPrice())) / (1 ether);
        uint256 numberOfTokens = (ethToUsd * phases[phaseId].tokenPerUsdPrice) /
            (1e8);
        return numberOfTokens;
    }

    function usdtToToken(uint256 _amount, uint256 phaseId)
        public
        view
        returns (uint256)
    {
        uint256 numberOfTokens = (_amount * phases[phaseId].tokenPerUsdPrice) /
            (1e6);
        return numberOfTokens;
    }

    function nativeToUsd(uint256 _amount) public view returns (uint256) {
        uint256 nativeTousd = (_amount * (getLatestPrice())) / (1e20);
        return nativeTousd;
    }

    function updateToken(address _token) external onlyOwner {
        mainToken = IERC20(_token);
    }

    function updateStableTokens(IERC20 _USDT, IERC20 _USDC) external onlyOwner {
        USDT = IERC20(_USDT);
        USDC = IERC20(_USDC);
    }

    function initiateTransfer(uint256 _value) external onlyOwner {
        fundReceiver.transfer(_value);
    }

    function changeFundReciever(address _addr) external onlyOwner {
        fundReceiver = payable(_addr);
    }

    function setCurrentStage(uint256 _stageNum) public onlyOwner {
        currentStage = _stageNum;
    }

    function updatePriceFeed(AggregatorV3Interface _priceFeed)
        external
        onlyOwner
    {
        priceFeed = _priceFeed;
    }

    function transferTokens(IERC20 token, uint256 _value) external onlyOwner {
        token.transfer(msg.sender, _value);
    }

    function totalUsersCount() external view returns (uint256) {
        return UsersAddresses.length;
    }
}