// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IUniswapV2Router {
    function swapExactETHForTokensSupportingFeeOnTransferTokens(
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable;
}

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
}

contract AsbubaSniperBot {
    address public owner;
    IUniswapV2Router public constant router = IUniswapV2Router(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
    
    address public constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    uint256 public slippageTolerance = 5; // Default slippage 2%

    event TokenPurchased(address indexed buyer, address indexed token, uint256 amount);
    event Withdrawal(address indexed recipient, uint256 amount);
    event OwnershipTransferred(address indexed oldOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function buyToken(address _token) external payable onlyOwner {
        require(msg.value > 0, "ETH required for purchase");

         address[] memory path = new address[](2);
            path[0] = WETH;
            path[1] = _token;

        require(slippageTolerance <= 100, "Invalid slippage tolerance");
        uint256 amountOutMin = (msg.value * (100 - slippageTolerance)) / 100;

        router.swapExactETHForTokensSupportingFeeOnTransferTokens{value: msg.value}(
            amountOutMin,
            path,
            address(this),
            block.timestamp + 300
        );

        emit TokenPurchased(msg.sender, _token, IERC20(_token).balanceOf(address(this)));
    }

        function buyTokenWithPercentage(address _token, uint256 percentage) external onlyOwner {
        require(percentage > 0 && percentage <= 100, "Invalid percentage");
        uint256 contractBalance = address(this).balance;
        require(contractBalance > 0, "No ETH available in contract");

        uint256 ethAmount = (contractBalance * percentage) / 100;
        require(ethAmount > 0, "ETH amount too low");

        address[] memory path = new address[](2);
            path[0] = WETH;
            path[1] = _token;
            
        uint256 amountOutMin = (ethAmount * (100 - slippageTolerance)) / 100;

        router.swapExactETHForTokensSupportingFeeOnTransferTokens{value: ethAmount}(
            amountOutMin,
            path,
            address(this),
            block.timestamp + 300
        );

        emit TokenPurchased(msg.sender, _token, IERC20(_token).balanceOf(address(this)));
    }


    function withdrawTokens(address _token) external onlyOwner {
        uint256 balance = IERC20(_token).balanceOf(address(this));
        require(balance > 0, "No tokens to withdraw");
        require(IERC20(_token).transfer(owner, balance), "Token transfer failed");

        emit Withdrawal(owner, balance);
    }

    function withdrawETH() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH to withdraw");
        payable(owner).transfer(balance);

        emit Withdrawal(owner, balance);
    }

    function setSlippage(uint256 _slippage) external onlyOwner {
        require(_slippage <= 29, "Slippage too high"); // 
        slippageTolerance = _slippage;
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid new owner");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    receive() external payable {}
}