// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

interface IUniswapV2Pair {
    function getReserves() external view returns (uint112, uint112, uint32);
    function token0() external view returns (address);
}

interface IChainlinkPriceFeed {
    function latestRoundData() external view returns (
        uint80, int256, uint256, uint256, uint80
    );
}

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

contract PeezyOTC {
    address public owner;
    address public peezyToken = 0x698b1d54E936b9F772b8F58447194bBc82EC1933;
    address public peezyPair = 0x1D91389b2Aa45C388C4d02eB39a7726d02a71d18;
    address public receiverWallet;
    address public ethUsdPriceFeed = 0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419;

    uint256 public discountPercent = 15;
    uint256 public cooldownPeriod = 23.5 hours;
    uint256 public maxAmountUSD = 500 * 1e18; // Start at $500
    uint256 public tolerancePercent = 5;

    mapping(address => bool) public whitelist;
    mapping(address => uint256) public lastTransactionTime;

    event Swapped(address indexed user, uint256 ethSent, uint256 peezyReceived, uint256 ethRefunded);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyWhitelist() {
        require(whitelist[msg.sender], "Not whitelisted");
        _;
    }

    modifier cooldown() {
        require(block.timestamp >= lastTransactionTime[msg.sender] + cooldownPeriod, "Cooldown active");
        _;
    }

    constructor(address _receiverWallet) {
        owner = msg.sender;
        receiverWallet = _receiverWallet;
    }

    function getPeezyPrice() public view returns (uint256) {
        IUniswapV2Pair pair = IUniswapV2Pair(peezyPair);
        (uint112 reserve0, uint112 reserve1, ) = pair.getReserves();
        address token0 = pair.token0();

        return (token0 == peezyToken) ? (reserve1 * 1e27 / reserve0) : (reserve0 * 1e27 / reserve1);
    }

    function getEthUsdPrice() public view returns (uint256) {
        (, int256 price, , , ) = IChainlinkPriceFeed(ethUsdPriceFeed).latestRoundData();
        require(price > 0, "Invalid Chainlink price");
        return uint256(price);
    }

    receive() external payable onlyWhitelist cooldown {
        swap();
    }

    function swap() public payable onlyWhitelist cooldown {
        uint256 peezyPrice = getPeezyPrice();
        uint256 discountedPrice = peezyPrice * (100 - discountPercent) / 100;
        uint256 peezyAmount = msg.value * 1e18 / discountedPrice;

        uint256 ethPriceUSD = getEthUsdPrice();
        uint256 ethValueUSD = msg.value * ethPriceUSD / 1e8;
        uint256 maxAllowedUSD = maxAmountUSD * (100 + tolerancePercent) / 100;
        uint256 refundAmount = 0;

        if (ethValueUSD > maxAllowedUSD) {
            uint256 maxEth = (maxAllowedUSD * 1e8 / ethPriceUSD) + 1;
            peezyAmount = maxEth * 1e18 / discountedPrice;
            refundAmount = msg.value - maxEth;
        }

        lastTransactionTime[msg.sender] = block.timestamp;
        transferTokens(peezyToken, msg.sender, peezyAmount);
        payable(receiverWallet).transfer(msg.value - refundAmount);

        if (refundAmount > 0) {
            payable(msg.sender).transfer(refundAmount);
        }

        emit Swapped(msg.sender, msg.value, peezyAmount, refundAmount);
    }

    function withdrawETH() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }

    function updateMaxAmountUSD(uint256 _newMaxAmountUSD) external onlyOwner {
        maxAmountUSD = _newMaxAmountUSD;
    }

    function updateDiscount(uint256 _newDiscount) external onlyOwner {
        discountPercent = _newDiscount;
    }

    function updateCooldown(uint256 _newCooldown) external onlyOwner {
        cooldownPeriod = _newCooldown;
    }

    function addToWhitelist(address _wallet) external onlyOwner {
        whitelist[_wallet] = true;
    }

    function removeFromWhitelist(address _wallet) external onlyOwner {
        whitelist[_wallet] = false;
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner is zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function transferTokens(address token, address recipient, uint256 amount) internal {
        require(IERC20(token).transfer(recipient, amount), "Transfer failed");
    }

    fallback() external payable {
        revert("Not allowed to send ETH here.");
    }
}