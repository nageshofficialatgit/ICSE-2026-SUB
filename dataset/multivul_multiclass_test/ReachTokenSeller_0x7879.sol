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
            uint256,
            uint80
        );
}

interface IReachToken {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function decimals() external view returns (uint8);
}

contract ReachTokenSeller {
    address public owner;
    address public treasury;
    IAggregator public priceFeed;
    IReachToken public reachToken;

    uint256 public constant FLOOR_PRICE = 27 * 1e18; // $27

    event TokensSold(address indexed seller, uint256 tokenAmount, uint256 ethAmount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor(address _priceFeed, address _token, address _treasury) {
        require(_priceFeed != address(0), "Invalid price feed");
        require(_token != address(0), "Invalid token address");
        require(_treasury != address(0), "Invalid treasury address");
        owner = msg.sender;
        priceFeed = IAggregator(_priceFeed);
        reachToken = IReachToken(_token);
        treasury = _treasury;
    }

    receive() external payable {}

    function getLatestPrice() public view returns (uint256) {
        (, int256 answer,,,) = priceFeed.latestRoundData();
        require(answer > 0, "Invalid price");
        return uint256(answer) * 1e10; // Adjust to 18 decimals
    }

    function sellTokens(uint256 tokenAmount) external {
        require(tokenAmount > 0, "Amount must be greater than zero");

        uint256 price = getLatestPrice();
        if (price < FLOOR_PRICE) {
            price = FLOOR_PRICE;
        }

        uint256 decimals = reachToken.decimals();
        uint256 ethAmount = (tokenAmount * price) / (10 ** decimals) / 1e18;
        require(address(this).balance >= ethAmount, "Insufficient ETH in contract");

        bool success = reachToken.transferFrom(msg.sender, treasury, tokenAmount);
        require(success, "Token transfer failed");

        payable(msg.sender).transfer(ethAmount);
        emit TokensSold(msg.sender, tokenAmount, ethAmount);
    }

    function setTreasury(address _treasury) external onlyOwner {
        require(_treasury != address(0), "Zero address");
        treasury = _treasury;
    }

    function withdrawETH(uint256 amount) external onlyOwner {
        payable(owner).transfer(amount);
    }
}