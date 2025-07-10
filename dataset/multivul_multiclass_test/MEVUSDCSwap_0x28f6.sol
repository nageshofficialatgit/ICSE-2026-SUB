// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
}

/**
 * @title MEV USDC Swap (WETH Version)
 * @dev A simplified contract that allows swapping WETH for USDC at a 50% favorable rate
 * to test MEV bot response and atomic swap capabilities.
 * This version uses WETH instead of native ETH to make it more easily simulatable.
 */
contract MEVUSDCSwap {
    // Constants
    address public constant USDC = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48; // Mainnet USDC
    address public constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2; // Mainnet WETH
    uint256 public constant ETH_PRICE = 2000000000; // $2000 with 6 decimals
    uint256 public constant BONUS_MULTIPLIER = 200; // 2.0x favorable rate (100% bonus)
    uint256 public constant FIXED_WETH_AMOUNT = 777000000000000; // 0.000777 WETH (7.77 * 10^14 wei)
    
    // State variables
    address public owner;
    bool private _locked; // Reentrancy guard
    
    // Events
    event Swap(address indexed user, uint256 wethAmount, uint256 usdcAmount);
    
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
     * @dev Swap a fixed amount of WETH (0.01) for USDC at a 50% favorable rate
     * No parameters needed - uses the FIXED_WETH_AMOUNT constant
     */
    function swap() external nonReentrant {
        // Calculate USDC amount (with 50% bonus)
        // WETH has 18 decimals, USDC has 6 decimals, ETH_PRICE has 6 decimals
        // Formula: (WETH amount * ETH price in USD * bonus multiplier) / 1e20
        uint256 usdcAmount = (FIXED_WETH_AMOUNT * ETH_PRICE * BONUS_MULTIPLIER) / 1e20;
        
        // Ensure contract has enough USDC
        require(IERC20(USDC).balanceOf(address(this)) >= usdcAmount, "Insufficient USDC");
        
        // Transfer WETH from sender to contract
        require(
            IERC20(WETH).transferFrom(msg.sender, address(this), FIXED_WETH_AMOUNT),
            "WETH transfer failed"
        );
        
        // Transfer USDC to sender
        bool success = IERC20(USDC).transfer(msg.sender, usdcAmount);
        require(success, "USDC transfer failed");
        
        emit Swap(msg.sender, FIXED_WETH_AMOUNT, usdcAmount);
    }
    
    /**
     * @dev Calculate the amount of USDC that would be received for a given amount of WETH
     * @param wethAmount The amount of WETH in wei
     * @return The amount of USDC that would be received (6 decimals)
     */
    function getUsdcAmount(uint256 wethAmount) external pure returns (uint256) {
        return (wethAmount * ETH_PRICE * BONUS_MULTIPLIER) / 1e20;
    }
    
    /**
     * @dev Withdraw WETH from the contract (only owner)
     */
    function withdrawWETH() external nonReentrant {
        require(msg.sender == owner, "Not owner");
        uint256 balance = IERC20(WETH).balanceOf(address(this));
        require(balance > 0, "No WETH to withdraw");
        
        bool success = IERC20(WETH).transfer(owner, balance);
        require(success, "WETH transfer failed");
    }
    
    /**
     * @dev Withdraw ETH from the contract (only owner)
     * This is kept in case native ETH is accidentally sent to the contract
     */
    function withdrawETH() external nonReentrant {
        require(msg.sender == owner, "Not owner");
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH to withdraw");
        
        (bool success, ) = owner.call{value: balance}("");
        require(success, "ETH transfer failed");
    }
    
    // Function to receive ETH (kept for compatibility)
    receive() external payable {}
}