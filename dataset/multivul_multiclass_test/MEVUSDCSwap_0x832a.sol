// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
}

/**
 * @title MEV USDC Swap
 * @dev A simplified contract that allows swapping ETH for USDC at a 50% favorable rate
 * to test MEV bot response and atomic swap capabilities
 */
contract MEVUSDCSwap {
    // Constants
    address public constant USDC = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48; // Mainnet USDC
    uint256 public constant ETH_PRICE = 2000000000; // $2000 with 6 decimals
    uint256 public constant BONUS_MULTIPLIER = 200; // 2.0x favorable rate (100% bonus)
    
    // State variables
    address public owner;
    bool private _locked; // Reentrancy guard
    
    // Events
    event Swap(address indexed user, uint256 ethAmount, uint256 usdcAmount);
    
    constructor() {
        owner = msg.sender;
    }
    
    // Modifier to prevent reentrancy attacks
    modifier nonReentrant() {
        require(!_locked, "ReentrancyGuard: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }
    
    /**
     * @dev Swap ETH for USDC at a 50% favorable rate
     * No parameters needed, uses msg.value
     */
    function swap() external payable nonReentrant {
        // Validate input
        require(msg.value > 0, "Must send ETH");
        
        // Calculate USDC amount (with 50% bonus)
        // ETH has 18 decimals, USDC has 6 decimals, ETH_PRICE has 6 decimals
        // Formula: (ETH amount * ETH price in USD * bonus multiplier) / 1e20
        uint256 usdcAmount = (msg.value * ETH_PRICE * BONUS_MULTIPLIER) / 1e20;
        
        // Ensure contract has enough USDC
        require(IERC20(USDC).balanceOf(address(this)) >= usdcAmount, "Insufficient USDC");
        
        // Transfer USDC to sender
        bool success = IERC20(USDC).transfer(msg.sender, usdcAmount);
        require(success, "USDC transfer failed");
        
        emit Swap(msg.sender, msg.value, usdcAmount);
    }
    
    /**
     * @dev Calculate the amount of USDC that would be received for a given amount of ETH
     * @param ethAmount The amount of ETH in wei
     * @return The amount of USDC that would be received (6 decimals)
     */
    function getUsdcAmount(uint256 ethAmount) external pure returns (uint256) {
        return (ethAmount * ETH_PRICE * BONUS_MULTIPLIER) / 1e20;
    }
    
    /**
     * @dev Withdraw ETH from the contract (only owner)
     */
    function withdrawETH() external nonReentrant {
        require(msg.sender == owner, "Not owner");
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH to withdraw");
        
        (bool success, ) = owner.call{value: balance}("");
        require(success, "ETH transfer failed");
    }
    
    /**
     * @dev Withdraw USDC from the contract (only owner)
     */
    function withdrawUSDC() external nonReentrant {
        require(msg.sender == owner, "Not owner");
        uint256 balance = IERC20(USDC).balanceOf(address(this));
        require(balance > 0, "No USDC to withdraw");
        
        bool success = IERC20(USDC).transfer(owner, balance);
        require(success, "USDC transfer failed");
    }
    
    // Function to receive ETH
    receive() external payable {}
}