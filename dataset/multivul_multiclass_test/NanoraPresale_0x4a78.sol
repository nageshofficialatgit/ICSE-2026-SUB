/**
 *Submitted for verification at testnet.bscscan.com on 2024-12-26
*/

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/*
$$\   $$\  $$$$$$\  $$\   $$\  $$$$$$\  $$$$$$$\   $$$$$$\  
$$$\  $$ |$$  __$$\ $$$\  $$ |$$  __$$\ $$  __$$\ $$  __$$\ 
$$$$\ $$ |$$ /  $$ |$$$$\ $$ |$$ /  $$ |$$ |  $$ |$$ /  $$ |
$$ $$\$$ |$$$$$$$$ |$$ $$\$$ |$$ |  $$ |$$$$$$$  |$$$$$$$$ |
$$ \$$$$ |$$  __$$ |$$ \$$$$ |$$ |  $$ |$$  __$$< $$  __$$ |
$$ |\$$$ |$$ |  $$ |$$ |\$$$ |$$ |  $$ |$$ |  $$ |$$ |  $$ |
$$ | \$$ |$$ |  $$ |$$ | \$$ | $$$$$$  |$$ |  $$ |$$ |  $$ |
\__|  \__|\__|  \__|\__|  \__| \______/ \__|  \__|\__|  \__|
                                                            
                                                            
Redefining the future of blockchain

Website: https://nanora.org/
Telegram: https://t.me/nanora
X/Twitter: https://x.com/nanoraofficial
*/


contract NanoraPresale {
    address public owner;
    address public tokenAddress; // Address of the token contract
    uint256 public tokenPrice; // Price of one token in wei
    uint256 public hardCap; // Maximum amount of funds to be raised
    uint256 public totalContributions; // Total funds raised so far
    uint256 public individualCap; // Maximum contribution per wallet
    bool public isPresaleActive; // Presale status
    uint256 public referralPercentage; // Percentage of contribution sent to referrer

    mapping(address => uint256) public contributions; // Tracks individual contributions
    mapping(address => uint256) public referralRewards; // Tracks referral rewards for each address
    address[] public contributorsList; // List of contributors for iteration

    event PresaleStarted();
    event PresaleStopped();
    event TokensPurchased(address indexed buyer, uint256 amount);
    event TokensDistributed(address indexed recipient, uint256 amount);
    event ReferralReward(address indexed referrer, address indexed buyer, uint256 reward);
    event BatchProcessed(uint256 startIndex, uint256 endIndex); // Added missing event

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can perform this action");
        _;
    }

    modifier whenPresaleActive() {
        require(isPresaleActive, "Presale is not active");
        _;
    }

    constructor(uint256 _tokenPrice, uint256 _hardCap, uint256 _individualCap, uint256 _referralPercentage) {
        owner = msg.sender;
        tokenPrice = _tokenPrice;
        hardCap = _hardCap;
        individualCap = _individualCap;
        referralPercentage = _referralPercentage;
        isPresaleActive = false;
    }

    function startPresale() external onlyOwner {
        require(!isPresaleActive, "Presale is already active");
        isPresaleActive = true;
        emit PresaleStarted();
    }

    function stopPresale() external onlyOwner {
        require(isPresaleActive, "Presale is not active");
        isPresaleActive = false;
        emit PresaleStopped();
    }

    function setTokenPrice(uint256 _newPrice) external onlyOwner {
        require(_newPrice > 0, "Token price must be greater than zero");
        tokenPrice = _newPrice;
    }

    function setTokenAddress(address _tokenAddress) external onlyOwner {
        require(_tokenAddress != address(0), "Invalid token address");
        tokenAddress = _tokenAddress;
    }

    function setReferralPercentage(uint256 _newPercentage) external onlyOwner {
        require(_newPercentage <= 100, "Percentage cannot exceed 100");
        referralPercentage = _newPercentage;
    }

    function buyTokens(address referrer) external payable whenPresaleActive {
        require(msg.value > 0, "Contribution must be greater than zero");
        require(totalContributions + msg.value <= hardCap, "Hard cap exceeded");
        require(contributions[msg.sender] + msg.value <= individualCap, "Individual cap exceeded");
        require(referrer != msg.sender, "Referrer cannot be the buyer");

        uint256 tokensToPurchase = (msg.value * 10**18) / tokenPrice;
        if (contributions[msg.sender] == 0) {
            contributorsList.push(msg.sender);
        }
        contributions[msg.sender] += msg.value;
        totalContributions += msg.value;

        if (referrer != address(0)) {
            uint256 reward = (msg.value * referralPercentage) / 100;
            referralRewards[referrer] += reward;
            emit ReferralReward(referrer, msg.sender, reward);
        }

        emit TokensPurchased(msg.sender, tokensToPurchase);
    }

    function withdrawFunds() external onlyOwner {
        require(!isPresaleActive, "Cannot withdraw funds while presale is active");
        payable(owner).transfer(address(this).balance);
    }

    function distributeTokens(uint256 startIndex, uint256 batchSize) external onlyOwner {
        require(!isPresaleActive, "Cannot distribute tokens during active presale");
        require(totalContributions > 0, "No contributions to distribute tokens for");
        require(tokenAddress != address(0), "Token address not set");
        require(startIndex < contributorsList.length, "Invalid start index");

        uint256 totalTokensToDistribute = (totalContributions * 10**18) / tokenPrice;
        uint256 endIndex = startIndex + batchSize;

        if (endIndex > contributorsList.length) {
            endIndex = contributorsList.length;
        }

        for (uint256 i = startIndex; i < endIndex; i++) {
            address contributor = contributorsList[i];
            uint256 contribution = contributions[contributor];

            if (contribution > 0) {
                uint256 tokensToDistribute = (contribution * totalTokensToDistribute) / totalContributions;
                IERC20(tokenAddress).transfer(contributor, tokensToDistribute);
                contributions[contributor] = 0;
                emit TokensDistributed(contributor, tokensToDistribute);
            }
        }

        emit BatchProcessed(startIndex, endIndex);
    }

    function claimReferralReward() external {
        uint256 reward = referralRewards[msg.sender];
        require(reward > 0, "No referral rewards to claim");

        referralRewards[msg.sender] = 0;
        payable(msg.sender).transfer(reward);
    }
}

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
}