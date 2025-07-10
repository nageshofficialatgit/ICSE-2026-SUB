// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

interface IUniswapV2Pair {
    function getReserves() external view returns (uint112, uint112, uint32);
    function token0() external view returns (address);
}

interface IChainlinkPriceFeed {
    function latestRoundData() external view returns (
        uint80, int256, uint256, uint256, uint80
    );
}

contract PeezyOTC {
    address public owner;
    address public peezyToken = 0x698b1d54E936b9F772b8F58447194bBc82EC1933;
    address public peezyPair = 0x1D91389b2Aa45C388C4d02eB39a7726d02a71d18;
    address public receiverWallet;
    address public ethUsdPriceFeed = 0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419;

    // Discount percent applied to the Peezy token price.
    uint256 public discountPercent = 15;
    // Maximum USD amount (in 18 decimals) allowed per swap.
    uint256 public maxAmountUSD = 500 * 1e18;
    // Tolerance percent used in ETH value calculation.
    uint256 public tolerancePercent = 5;

    mapping(address => bool) public whitelist;

    event Swapped(address indexed user, uint256 ethSent, uint256 peezyReceived);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event MaxAmountUSDUpdated(uint256 newMaxAmountUSD);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyWhitelist() {
        require(whitelist[msg.sender], "Not whitelisted");
        _;
    }

    constructor(address _receiverWallet) {
        owner = msg.sender;
        receiverWallet = _receiverWallet;
    }

    // Fetch the Peezy price from the Uniswap pair.
    function getPeezyPrice() public view returns (uint256) {
        IUniswapV2Pair pair = IUniswapV2Pair(peezyPair);
        (uint112 reserve0, uint112 reserve1, ) = pair.getReserves();
        address token0 = pair.token0();
        // Adjust decimals: Peezy token has 9 decimals and WETH has 18 decimals.
        if (token0 == peezyToken) {
            return uint256(reserve1) * 1e9 / uint256(reserve0); // Price in wei per Peezy token
        } else {
            return uint256(reserve0) * 1e9 / uint256(reserve1); // Price in wei per Peezy token
        }
    }

    // Fetch the ETH/USD price from Chainlink.
    function getEthUsdPrice() public view returns (uint256) {
        (, int256 price, , , ) = IChainlinkPriceFeed(ethUsdPriceFeed).latestRoundData();
        require(price > 0, "Invalid Chainlink price");
        return uint256(price);
    }

    // Dedicated swap function for whitelisted addresses.
    function swap() external payable onlyWhitelist {
        uint256 ethPriceUSD = getEthUsdPrice();
        // Calculate the maximum allowed ETH in wei from the USD limit.
        uint256 maxAllowedETH = (maxAmountUSD * 1e8) / ethPriceUSD;
        maxAllowedETH = maxAllowedETH * (100 + tolerancePercent) / 100;
        require(msg.value <= maxAllowedETH, "ETH value exceeds allowed maximum.");

        uint256 peezyPrice = getPeezyPrice();
        // Apply the discount to the Peezy price.
        uint256 discountedPrice = peezyPrice * (100 - discountPercent) / 100;
        require(discountedPrice > 0, "Discounted price is zero");

        // Calculate the amount of Peezy tokens to send.
        // Adjusting multiplier to 1e9 to reflect Peezy token's 9 decimals.
        uint256 peezyAmount = msg.value * 1e9 / discountedPrice;

        // Deduct a fixed gas fee before sending ETH to the receiver.
        uint256 estimatedGasFee = 0.001 ether;
        require(msg.value > estimatedGasFee, "Insufficient ETH to cover gas fee.");
        uint256 ethToSend = msg.value - estimatedGasFee;

        // Check that the contract holds enough Peezy tokens.
        uint256 tokenBalance = IERC20(peezyToken).balanceOf(address(this));
        require(tokenBalance >= peezyAmount, "Insufficient token balance in contract");

        // Transfer Peezy tokens to the sender.
        require(IERC20(peezyToken).transfer(msg.sender, peezyAmount), "Token transfer failed");

        // Transfer the remaining ETH to the receiver wallet.
        payable(receiverWallet).transfer(ethToSend);

        emit Swapped(msg.sender, msg.value, peezyAmount);
    }

    // Fallback functions simply accept ETH without triggering swap logic.
    receive() external payable {}
    fallback() external payable {}

    // Administrative functions:

    function withdrawETH() external onlyOwner {
        payable(owner).transfer(address(this).balance);
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

    function updateMaxAmountUSD(uint256 newMaxAmountUSD) external onlyOwner {
        maxAmountUSD = newMaxAmountUSD;
        emit MaxAmountUSDUpdated(newMaxAmountUSD);
    }

    function withdrawTokens(address tokenAddress) external onlyOwner {
        uint256 tokenBalance = IERC20(tokenAddress).balanceOf(address(this));
        require(IERC20(tokenAddress).transfer(owner, tokenBalance), "Token transfer failed");
    }
}