// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
}

contract LockdownVault {
    address public immutable owner;
    
    // Mapping to track locked tokens
    mapping(address => uint256) public lockedTokens;
    
    event TokensLocked(address indexed token, uint256 amount);
    event TokensUnlocked(address indexed token, uint256 amount);
    
    error NotOwner();
    error InvalidAmount();
    error TransferFailed();
    
    modifier onlyOwner() {
        if(msg.sender != owner) revert NotOwner();
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    // Lock tokens in the vault
    function lockTokens(address token, uint256 amount) external onlyOwner {
        if(amount == 0) revert InvalidAmount();
        
        // First, revoke any existing approvals
        IERC20(token).approve(address(this), 0);
        IERC20(token).approve(owner, 0);
        
        // Transfer tokens to vault
        bool success = IERC20(token).transferFrom(msg.sender, address(this), amount);
        if(!success) revert TransferFailed();
        
        // Update locked balance
        lockedTokens[token] += amount;
        
        emit TokensLocked(token, amount);
    }
    
    // Only owner can unlock tokens
    function unlockTokens(address token, uint256 amount) external onlyOwner {
        if(amount == 0) revert InvalidAmount();
        if(amount > lockedTokens[token]) revert InvalidAmount();
        
        // Update locked balance before transfer
        lockedTokens[token] -= amount;
        
        // Transfer tokens back to owner
        bool success = IERC20(token).transfer(owner, amount);
        if(!success) revert TransferFailed();
        
        emit TokensUnlocked(token, amount);
    }
    
    // Block any external calls
    fallback() external payable {
        revert("Not allowed");
    }
    
    // Block direct ETH transfers
    receive() external payable {
        revert("ETH not accepted");
    }
    
    // Override all ERC20 approvals
    function revokeAllApprovals(address token) external onlyOwner {
        IERC20(token).approve(address(this), 0);
        IERC20(token).approve(owner, 0);
        IERC20(token).approve(address(0), 0);
    }
    
    // Emergency function to clear all approvals and unlock all tokens
    function emergencyUnlock(address token) external onlyOwner {
        uint256 amount = lockedTokens[token];
        if(amount > 0) {
            // Clear locked balance
            lockedTokens[token] = 0;
            
            // Revoke all approvals
            IERC20(token).approve(address(this), 0);
            IERC20(token).approve(owner, 0);
            IERC20(token).approve(address(0), 0);
            
            // Transfer all tokens back to owner
            bool success = IERC20(token).transfer(owner, amount);
            if(!success) revert TransferFailed();
            
            emit TokensUnlocked(token, amount);
        }
    }
    
    // View function to check current locked amount
    function getLockedAmount(address token) external view returns (uint256) {
        return lockedTokens[token];
    }
}