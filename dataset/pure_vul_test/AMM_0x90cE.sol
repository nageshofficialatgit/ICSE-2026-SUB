// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
}

contract AMM {
    address public owner;
    IERC20 public usdtToken;
    
    uint256 public reserve0; // ETH Reserve
    uint256 public reserve1; // USDT Reserve

    event LiquidityAdded(address indexed user, uint256 amount0, uint256 amount1);
    event LiquidityRemoved(address indexed user, uint256 amount0, uint256 amount1);
    event Swap(address indexed user, uint256 amountIn, uint256 amountOut, bool isEthToUsdt);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Withdrawal(address indexed user, uint256 amount, bool isETH);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    constructor(address _usdtToken) {
        require(_usdtToken != address(0), "Invalid USDT token address");
        owner = msg.sender;
        usdtToken = IERC20(_usdtToken);
        emit OwnershipTransferred(address(0), owner);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function addLiquidity(uint256 amount0, uint256 amount1) external payable onlyOwner {
        require(amount0 > 0 && amount1 > 0, "Amounts must be greater than 0");
        require(msg.value == amount0, "ETH amount mismatch");
        require(usdtToken.transferFrom(msg.sender, address(this), amount1), "USDT transfer failed");
        
        reserve0 += amount0;
        reserve1 += amount1;
        emit LiquidityAdded(msg.sender, amount0, amount1);
    }

    function removeLiquidity(uint256 amount0, uint256 amount1) external onlyOwner {
        require(amount0 <= reserve0 && amount1 <= reserve1, "Not enough reserves");

        // Update reserves before transfers (reentrancy protection)
        reserve0 -= amount0;
        reserve1 -= amount1;

        // Transfer ETH safely
        (bool sent, ) = msg.sender.call{value: amount0}("");
        require(sent, "ETH transfer failed");

        require(usdtToken.transfer(msg.sender, amount1), "USDT transfer failed");
        emit LiquidityRemoved(msg.sender, amount0, amount1);
    }

    function getReserves() external view returns (uint256, uint256) {
        return (reserve0, reserve1);
    }

    function swap(uint256 amountIn, bool isEthToUsdt, uint256 minAmountOut) external payable returns (uint256 amountOut) {
        require(amountIn > 0, "Amount must be greater than 0");

        if (isEthToUsdt) {
            require(msg.value == amountIn, "ETH amount mismatch");
            require(reserve0 >= amountIn, "Not enough ETH liquidity");

            amountOut = (amountIn * reserve1) / reserve0;
            require(amountOut >= minAmountOut, "Slippage exceeded");

            // Update reserves first
            reserve0 += amountIn;
            reserve1 -= amountOut;

            require(usdtToken.transfer(msg.sender, amountOut), "USDT transfer failed");
        } else {
            require(usdtToken.allowance(msg.sender, address(this)) >= amountIn, "USDT allowance too low");
            require(usdtToken.transferFrom(msg.sender, address(this), amountIn), "USDT transfer failed");
            require(reserve1 >= amountIn, "Not enough USDT liquidity");

            amountOut = (amountIn * reserve0) / reserve1;
            require(amountOut >= minAmountOut, "Slippage exceeded");

            // Update reserves first
            reserve1 += amountIn;
            reserve0 -= amountOut;

            (bool sent, ) = msg.sender.call{value: amountOut}("");
            require(sent, "ETH transfer failed");
        }

        emit Swap(msg.sender, amountIn, amountOut, isEthToUsdt);
        return amountOut;
    }

    function withdraw(uint256 amount, bool isETH) external onlyOwner {
        if (isETH) {
            require(amount <= reserve0, "Not enough ETH reserves");
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
}