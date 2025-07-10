// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IAggregator {
    function latestRoundData()
        external
        view
        returns (
            uint80,
            int256 answer,
            uint256,
            uint256 updatedAt,
            uint80
        );
}

interface IReachToken {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function decimals() external view returns (uint8);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
}

contract ReachTokenSeller {
    address public owner;
    address public treasury;
    IAggregator public priceFeed;
    IReachToken public reachToken;

    bool public paused = false;
    uint256 public floorPrice = 27 * 1e18;
    uint256 private constant ONE_ETHER = 1e18;

    mapping(address => uint256) private lastCall;

    event TokensSold(address indexed seller, uint256 tokenAmount, uint256 ethAmount, uint256 price, uint256 timestamp);
    event TokensBought(address indexed buyer, uint256 ethAmount, uint256 tokenAmount, uint256 price, uint256 timestamp);
    event FloorPriceUpdated(uint256 oldPrice, uint256 newPrice, uint256 timestamp);
    event TreasuryUpdated(address oldTreasury, address newTreasury, uint256 timestamp);
    event PauseToggled(bool isPaused, uint256 timestamp);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier nonReentrant() {
        require(lastCall[msg.sender] < block.timestamp, "Reentrant call");
        lastCall[msg.sender] = block.timestamp;
        _;
    }

    constructor(address _priceFeed, address _token, address _treasury) {
        require(_priceFeed != address(0), "Invalid price feed");
        require(_token != address(0), "Invalid token");
        require(_treasury != address(0), "Invalid treasury");

        owner = msg.sender;
        priceFeed = IAggregator(_priceFeed);
        reachToken = IReachToken(_token);
        treasury = _treasury;
    }

    receive() external payable {}

    function togglePause() external onlyOwner {
        paused = !paused;
        emit PauseToggled(paused, block.timestamp);
    }

    function getLatestPrice() public view returns (uint256) {
        (, int256 answer, , uint256 updatedAt, ) = priceFeed.latestRoundData();
        require(answer > 0, "Invalid price");
        require(block.timestamp - updatedAt < 1 hours, "Stale price");
        return uint256(answer) * 1e10;
    }

    function buyTokens(uint256 minTokens) external payable nonReentrant {
        require(!paused, "Paused");
        require(msg.value > 0, "No ETH");

        uint256 price = getLatestPrice();
        if (price < floorPrice) price = floorPrice;

        uint256 decimals = reachToken.decimals();
        uint256 tokensToSend = (msg.value * price * (10 ** decimals)) / ONE_ETHER / ONE_ETHER;
        require(tokensToSend >= minTokens, "Slippage exceeded");

        require(reachToken.balanceOf(treasury) >= tokensToSend, "Insufficient token supply");

        bool sent = reachToken.transferFrom(treasury, msg.sender, tokensToSend);
        require(sent, "Token transfer failed");

        emit TokensBought(msg.sender, msg.value, tokensToSend, price, block.timestamp);
    }

    function sellTokens(uint256 tokenAmount, uint256 maxEth) external nonReentrant {
        require(!paused, "Paused");
        require(tokenAmount > 0, "Token amount required");

        uint256 price = getLatestPrice();
        if (price < floorPrice) price = floorPrice;

        uint256 decimals = reachToken.decimals();
        uint256 ethAmount = (tokenAmount * ONE_ETHER) / price / (10 ** decimals);

        require(ethAmount <= maxEth, "Slippage exceeded");
        require(address(this).balance >= ethAmount, "Insufficient ETH");

        require(reachToken.allowance(msg.sender, address(this)) >= tokenAmount, "Allowance too low");

        bool success = reachToken.transferFrom(msg.sender, treasury, tokenAmount);
        require(success, "Token transfer failed");

        (bool sent, ) = payable(msg.sender).call{value: ethAmount}("");
        require(sent, "ETH transfer failed");

        emit TokensSold(msg.sender, tokenAmount, ethAmount, price, block.timestamp);
    }

    function setTreasury(address _treasury) external onlyOwner {
        require(_treasury != address(0), "Zero address");
        emit TreasuryUpdated(treasury, _treasury, block.timestamp);
        treasury = _treasury;
    }

    function updateFloorPrice(uint256 newPrice) external onlyOwner {
        require(newPrice > 0, "Price must be > 0");
        emit FloorPriceUpdated(floorPrice, newPrice, block.timestamp);
        floorPrice = newPrice;
    }

    function withdrawETH(uint256 amount) external onlyOwner nonReentrant {
        require(amount <= address(this).balance, "Insufficient balance");
        (bool sent, ) = payable(owner).call{value: amount}("");
        require(sent, "Withdraw failed");
    }
}