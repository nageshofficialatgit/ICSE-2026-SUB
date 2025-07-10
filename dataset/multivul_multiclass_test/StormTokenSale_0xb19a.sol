// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title StormTokenSale
 * @dev Minimal contract to collect ETH for STORM token allocations at $1 each
 */
contract StormTokenSale {
    address public owner = 0x68356B9D503F36addc4Bf8e45ef5B04AFB08572D; // Your wallet address
    uint256 public tokenPrice = 500000000000000; // 0.0005 ETH (assuming ETH = $2000)
    uint256 public tokensSold;
    uint256 public constant TOTAL_TOKENS = 1500000;
    
    mapping(address => uint256) public purchases;
    
    event Purchase(address buyer, uint256 amount);
    event FailedTransfer(address receiver, uint256 amount);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    function buyTokens() public payable {
        require(msg.value > 0, "Send ETH to buy tokens");
        require(tokensSold < TOTAL_TOKENS, "All tokens sold");
        
        uint256 tokenAmount = (msg.value * 1 ether) / tokenPrice;
        require(tokensSold + tokenAmount <= TOTAL_TOKENS, "Not enough tokens left");
        
        // Update state before transfer to prevent reentrancy
        tokensSold += tokenAmount;
        purchases[msg.sender] += tokenAmount;
        
        // Safely transfer funds to owner
        _safeTransferETH(owner, msg.value);
        
        emit Purchase(msg.sender, tokenAmount);
    }
    
    /**
     * @dev Safely transfer ETH to the specified address
     * @param to Recipient address
     * @param amount Amount of ETH to send
     */
    function _safeTransferETH(address to, uint256 amount) private {
        // Using call is the recommended way to send ETH
        (bool success, ) = to.call{value: amount}("");
        
        // If transfer fails, log the event but don't revert the transaction
        // This ensures the buyer's purchase is still recorded
        if (!success) {
            emit FailedTransfer(to, amount);
        }
    }
    
    function setPrice(uint256 _priceInWei) external onlyOwner {
        tokenPrice = _priceInWei;
    }
    
    function getTokenAmount(uint256 ethAmount) public view returns (uint256) {
        return (ethAmount * 1 ether) / tokenPrice;
    }
    
    /**
     * @dev Owner can withdraw any ETH that might be stuck in the contract
     */
    function withdrawETH() external onlyOwner {
        uint256 balance = address(this).balance;
        if (balance > 0) {
            _safeTransferETH(owner, balance);
        }
    }
    
    // Allow ETH to be sent directly to contract
    receive() external payable {
        buyTokens();
    }
}