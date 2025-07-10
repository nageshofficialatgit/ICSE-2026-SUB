// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract FomoFluffICO {
    string public name = "FomoFluff";
    string public symbol = "FOMO";
    uint256 public totalSupply = 86000000000 * 10**9; // 86 billion with 9 decimal places
    uint256 public tokenPrice = 0.00000001 ether; // Price per token (0.00000001 ETH)
    uint256 public startTime; // ICO start time
    uint256 public endTime; // ICO end time
    address public owner;
    string private _tokenURI; // Token image URI
    address public rewardAddress = 0x6185137BC8aCf79d7Ef837cF6c1BCe4a70BE8857; // Address to receive distribution rewards

    mapping(address => uint256) public balanceOf;

    // Minimum and maximum contribution limits
    uint256 public minContribution = 0.00053 ether; // Minimum contribution (0.00053 ETH)
    uint256 public maxContribution = 5 ether; // Maximum contribution (5 ETH)

    event TokensPurchased(address indexed buyer, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can execute this");
        _;
    }

    modifier icoActive() {
        require(block.timestamp >= startTime && block.timestamp <= endTime, "ICO is not active");
        _;
    }

    constructor() {
        owner = msg.sender;
        startTime = block.timestamp + 1 days; // ICO starts tomorrow
        endTime = startTime + 60 days; // ICO ends 60 days after the start
        _tokenURI = "https://ipfs.io/ipfs/bafkreig57c5ezypfu45v2dlayvutymbnnmphxtze4loekksr5ne6wswvnm"; // Default token image URI
    }

    function buyTokens() external payable icoActive {
        require(msg.value >= minContribution && msg.value <= maxContribution, "Contribution out of range");

        uint256 tokenAmount = msg.value / tokenPrice;
        require(totalSupply >= tokenAmount, "Not enough tokens available");

        balanceOf[msg.sender] += tokenAmount;
        totalSupply -= tokenAmount;

        // Distribute 3% of the purchase to the reward address
        uint256 rewardAmount = tokenAmount * 3 / 100;
        balanceOf[rewardAddress] += rewardAmount;

        emit TokensPurchased(msg.sender, tokenAmount);
    }

    // Withdraw ICO funds (only the owner)
    function withdraw() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }

    // Get remaining ICO time
    function timeRemaining() external view returns (uint256) {
        if (block.timestamp >= endTime) {
            return 0;
        }
        return endTime - block.timestamp;
    }

    // Get the current token image URI
    function tokenURI() public view returns (string memory) {
        return _tokenURI;
    }

    // ✅ Function to update the token image URI (only the owner can change it)
    function setTokenURI(string memory newTokenURI) external onlyOwner {
        _tokenURI = newTokenURI;
    }

    // Function to transfer tokens between users
    function transfer(address to, uint256 amount) external {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
    }

    // Function to handle direct payments (reverts)
    receive() external payable {
        revert("Direct payments not allowed. Use buyTokens.");
    }

    fallback() external payable {
        revert("Invalid transaction.");
    }

    // Function for the owner to reclaim unsold tokens after ICO
    function reclaimUnsoldTokens() external onlyOwner {
        require(block.timestamp > endTime, "ICO not yet ended");
        uint256 unsoldTokens = totalSupply;
        totalSupply = 0;
        balanceOf[owner] += unsoldTokens;
    }
}