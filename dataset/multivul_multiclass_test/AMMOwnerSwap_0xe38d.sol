// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
}

interface AggregatorV3Interface {
    function latestRoundData() external view returns (
        uint80 roundId,
        int256 answer,
        uint256 startedAt,
        uint256 updatedAt,
        uint80 answeredInRound
    );
}

contract AMMOwnerSwap {
    address public immutable owner;
    IERC20 public usdtToken;
    AggregatorV3Interface public priceFeed;

    uint256 public reserve0; // ETH Reserve
    uint256 public reserve1; // USDT Reserve
    uint256 public minEthReserve; // Dynamic 2% buffer

    event LiquidityAdded(address indexed user, uint256 amount0, uint256 amount1);
    event Swap(address indexed user, uint256 amountIn, uint256 amountOut, bool isEthToUsdt);
    event Withdrawal(address indexed owner, uint256 amount, bool isETH);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    constructor(address _usdtToken, address _priceFeed) {
        require(_usdtToken != address(0), "Invalid USDT token address");
        require(_priceFeed != address(0), "Invalid price feed address");
        owner = msg.sender;
        usdtToken = IERC20(_usdtToken);
        priceFeed = AggregatorV3Interface(_priceFeed);
        minEthReserve = 0; // Initialize with zero, updated when liquidity is added
    }

    function updateMinEthReserve() internal {
        minEthReserve = reserve0 * 2 / 100; // 2% buffer from reserve0
    }

    function addLiquidity(uint256 amount0, uint256 amount1) external payable onlyOwner {
        require(amount0 > 0 || amount1 > 0, "At least one value must be greater than 0");

        if (amount0 > 0) {
            require(msg.value == amount0, "ETH amount mismatch");
            reserve0 += amount0;
        }

        if (amount1 > 0) {
            uint256 allowance = usdtToken.allowance(msg.sender, address(this));
            require(allowance >= amount1, "USDT allowance too low");
            require(usdtToken.transferFrom(msg.sender, address(this), amount1), "USDT transfer failed");
            reserve1 += amount1;
        }

        updateMinEthReserve();
        emit LiquidityAdded(msg.sender, amount0, amount1);
    }

    function getMarketPrice() public view returns (uint256) {
        (, int256 price,,,) = priceFeed.latestRoundData();
        require(price > 0, "Invalid price data");
        return uint256(price); // Price in USD with 8 decimals
    }

    function swap(bool isEthToUsdt) external onlyOwner {
        require(reserve0 > 0 && reserve1 > 0, "Insufficient reserves");
        uint256 marketPrice = getMarketPrice();

        uint256 amountIn;
        uint256 amountOut;

        if (isEthToUsdt) {
            amountIn = reserve0;
            amountOut = (amountIn * marketPrice) / 1e8; // Adjust for price decimals
            reserve0 = 0;
            reserve1 += amountOut;
            require(usdtToken.transfer(msg.sender, amountOut), "USDT transfer failed");
        } else {
            amountIn = reserve1;
            amountOut = (amountIn * 1e8) / marketPrice; // Adjust for price decimals
            require(amountOut <= address(this).balance - minEthReserve, "Not enough ETH for swap");
            reserve1 = 0;
            reserve0 += amountOut;
            (bool sent, ) = msg.sender.call{value: amountOut}("");
            require(sent, "ETH transfer failed");
        }

        emit Swap(msg.sender, amountIn, amountOut, isEthToUsdt);
    }

    function withdraw(uint256 amount, bool isETH) external onlyOwner {
        if (isETH) {
            require(amount <= reserve0 - minEthReserve, "Not enough ETH reserves");
            reserve0 -= amount;
            (bool sent, ) = msg.sender.call{value: amount}("");
            require(sent, "ETH transfer failed");
        } else {
            require(amount <= reserve1, "Not enough USDT reserves");
            reserve1 -= amount;
            require(usdtToken.transfer(msg.sender, amount), "USDT transfer failed");
        }
        emit Withdrawal(msg.sender, amount, isETH);
    }

    receive() external payable {}
    fallback() external payable {}
}