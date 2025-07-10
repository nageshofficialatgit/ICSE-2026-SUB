// SPDX-License-Identifier: MIT
// File: @openzeppelin/contracts/security/ReentrancyGuard.sol

// OpenZeppelin Contracts v4.9.0 (security/ReentrancyGuard.sol)

pragma solidity ^0.8.0;

/**
 * @dev Contract module that helps prevent reentrant calls to a function.
 *
 * Inheriting from `ReentrancyGuard` will make the {nonReentrant} modifier
 * available, which can be applied to functions to make sure there are no nested
 * (reentrant) calls to them.
 *
 * Note that because there is a single `nonReentrant` guard, functions marked as
 * `nonReentrant` may not call one another. This can be worked around by making
 * those functions `private`, and then adding `external` `nonReentrant` entry
 * points to them.
 *
 * TIP: If you would like to learn more about reentrancy and alternative ways
 * to protect against it, check out our blog post
 * https://blog.openzeppelin.com/reentrancy-after-istanbul/
 */
abstract contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
    }

    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
    }

    function _nonReentrantAfter() private {
        _status = _NOT_ENTERED;
    }

    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == _ENTERED;
    }
}

// File: CSWEPPresale.sol

pragma solidity ^0.8.19;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

interface AggregatorV3Interface {
    function latestRoundData() external view returns (
        uint80, int, uint, uint, uint80
    );
}

contract CSWEPPresale is ReentrancyGuard {
    address public owner;
    uint256 public totalRaisedETH;
    uint256 public totalRaisedUSDC;
    uint256 public totalRaisedUSDT;
    uint256 public totalRaisedDAI;
    address public tokenAddress;
    uint256 public claimUnlockTime;
    bool public saleEnded = false;
    bool public paused = false;

    uint256 public fallbackPrice = 0;
    bool public useFallback = false;

    mapping(address => uint256) public tokensOwed;
    mapping(address => uint256) public ethContributions;
    mapping(address => uint256) public usdcContributions;
    mapping(address => uint256) public usdtContributions;
    mapping(address => uint256) public daiContributions;

    IERC20 public USDC;
    IERC20 public USDT;
    IERC20 public DAI;

    AggregatorV3Interface public ethUsdPriceFeed;

    uint256 public constant usdUnit = 1e18;

    uint256[] public phaseThresholds = [
        0 * usdUnit,
        25000 * usdUnit,
        50000 * usdUnit,
        75000 * usdUnit,
        100000 * usdUnit,
        150000 * usdUnit,
        200000 * usdUnit,
        300000 * usdUnit,
        400000 * usdUnit,
        500000 * usdUnit,
        650000 * usdUnit,
        800000 * usdUnit,
        1000000 * usdUnit
    ];

    uint256[] public tokenPrices = [
        1e15, 2e15, 3e15, 4e15, 5e15,
        6e15, 7e15, 8e15, 9e15, 1e16,
        11e15, 12e15, 13e15
    ];

    event Contribution(address indexed buyer, uint256 usdValue, uint256 tokensAllocated);
    event TokensClaimed(address indexed buyer, uint256 amount);
    event TokenAddressSet(address token);
    event ClaimUnlockTimeSet(uint256 timestamp);
    event SaleEnded();
    event Paused();
    event Unpaused();

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier notPaused() {
        require(!paused, "Presale paused");
        _;
    }

    constructor(address _usdc, address _usdt, address _dai, address _priceFeed) {
        owner = msg.sender;
        USDC = IERC20(_usdc);
        USDT = IERC20(_usdt);
        DAI = IERC20(_dai);
        ethUsdPriceFeed = AggregatorV3Interface(_priceFeed);
    }

    receive() external payable notPaused {
        require(!saleEnded, "Presale ended");
        uint256 usdAmount = getEthInUsd(msg.value);
        uint256 price = getCurrentPrice();
        uint256 tokenAmount = (usdAmount * 1e18) / price;

        totalRaisedETH += usdAmount;
        ethContributions[msg.sender] += usdAmount;
        tokensOwed[msg.sender] += tokenAmount;

        emit Contribution(msg.sender, usdAmount, tokenAmount);

        if (getTotalRaisedUSD() >= phaseThresholds[phaseThresholds.length - 1]) {
            saleEnded = true;
            emit SaleEnded();
        }
    }

    function contributeStable(address token, uint256 amount) internal notPaused {
        require(!saleEnded, "Presale ended");
        require(IERC20(token).transferFrom(msg.sender, address(this), amount), "Transfer failed");

        uint256 usdAmount = amount;
        if (token == address(USDC)) totalRaisedUSDC += usdAmount;
        if (token == address(USDT)) totalRaisedUSDT += usdAmount;
        if (token == address(DAI)) totalRaisedDAI += usdAmount;

        uint256 price = getCurrentPrice();
        uint256 tokenAmount = (usdAmount * 1e18) / price;

        tokensOwed[msg.sender] += tokenAmount;
        emit Contribution(msg.sender, usdAmount, tokenAmount);

        if (getTotalRaisedUSD() >= phaseThresholds[phaseThresholds.length - 1]) {
            saleEnded = true;
            emit SaleEnded();
        }
    }

    function contributeUSDC(uint256 amount) external {
        contributeStable(address(USDC), amount);
    }

    function contributeUSDT(uint256 amount) external {
        contributeStable(address(USDT), amount);
    }

    function contributeDAI(uint256 amount) external {
        contributeStable(address(DAI), amount);
    }

    function claim() external nonReentrant {
        require(tokenAddress != address(0), "Token not set");
        require(block.timestamp >= claimUnlockTime, "Claim not unlocked");
        uint256 amount = tokensOwed[msg.sender];
        require(amount > 0, "No tokens owed");

        tokensOwed[msg.sender] = 0;
        require(IERC20(tokenAddress).transfer(msg.sender, amount), "Transfer failed");
        emit TokensClaimed(msg.sender, amount);
    }

    function getEthInUsd(uint256 ethAmount) internal view returns (uint256) {
        if (useFallback) return (ethAmount * fallbackPrice) / 1e8;
        (, int price,,,) = ethUsdPriceFeed.latestRoundData();
        require(price > 0, "Invalid price");
        return (ethAmount * uint256(price)) / 1e8;
    }

    function getCurrentPrice() public view returns (uint256) {
        uint256 totalUSD = getTotalRaisedUSD();
        for (uint256 i = phaseThresholds.length; i > 0; i--) {
            if (totalUSD >= phaseThresholds[i - 1]) {
                return tokenPrices[i - 1];
            }
        }
        return tokenPrices[0];
    }

    function getTotalRaisedUSD() public view returns (uint256) {
        return totalRaisedETH + totalRaisedUSDC + totalRaisedUSDT + totalRaisedDAI;
    }

    function setTokenAddress(address _token) external onlyOwner {
        tokenAddress = _token;
        emit TokenAddressSet(_token);
    }

    function setClaimUnlockTime(uint256 _timestamp) external onlyOwner {
        claimUnlockTime = _timestamp;
        emit ClaimUnlockTimeSet(_timestamp);
    }

    function manualEndPresale() external onlyOwner {
        require(!saleEnded, "Presale already ended");
        saleEnded = true;
        emit SaleEnded();
    }

    function withdrawToken(address token) external onlyOwner {
        uint256 balance = IERC20(token).balanceOf(address(this));
        require(IERC20(token).transfer(owner, balance), "Withdraw failed");
    }

    function withdrawETH() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }

    function pause() external onlyOwner {
        paused = true;
        emit Paused();
    }

    function unpause() external onlyOwner {
        paused = false;
        emit Unpaused();
    }

    function setFallbackPrice(uint256 price) external onlyOwner {
        fallbackPrice = price;
    }

    function toggleFallback(bool use) external onlyOwner {
        useFallback = use;
    }
}