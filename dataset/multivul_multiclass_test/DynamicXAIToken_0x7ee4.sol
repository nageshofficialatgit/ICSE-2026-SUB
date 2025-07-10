// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract DynamicXAIToken {
    string public name = "DynamicX.AIToken";
    string public symbol = "xAI";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    uint256 public tokenPrice;
    uint256 public priceFactor = 1;  
    uint256 public totalBought;
    uint256 public totalSold;
    uint256 public minTokenPrice = 0.0001 ether;
    uint256 public feePercent = 5;
    bool public sellingAllowed = false;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    address public owner;
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event TokensBought(address indexed buyer, uint256 amount);
    event TokensSold(address indexed seller, uint256 amount, uint256 payout);
    event TradingStatusChanged(bool allowed);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier nonReentrant() {
        require(!locked, "Reentrancy detected");
        locked = true;
        _;
        locked = false;
    }
    
    bool private locked = false;

    constructor(address initialOwner) {
        owner = initialOwner;
        totalSupply = 1_000_000 * (10 ** decimals);
        balanceOf[owner] = totalSupply;
        tokenPrice = 0.0001 ether;
    }

    function transfer(address recipient, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool) {
        require(balanceOf[sender] >= amount, "Insufficient balance");
        require(allowance[sender][msg.sender] >= amount, "Allowance exceeded");
        
        allowance[sender][msg.sender] -= amount;
        _transfer(sender, recipient, amount);
        return true;
    }

    function buyTokens() external payable nonReentrant {
        require(msg.value > 0, "Send ETH to buy tokens");
        require(tokenPrice > 0, "Token price not set");

        uint256 amountToBuy = msg.value / tokenPrice;
        require(balanceOf[owner] >= amountToBuy, "Not enough tokens");

        _transfer(owner, msg.sender, amountToBuy);
        totalBought += amountToBuy;
        _updateTokenPrice(true);

        emit TokensBought(msg.sender, amountToBuy);
    }

    function sellTokens(uint256 amount) external nonReentrant {
        require(sellingAllowed, "Selling disabled");
        require(balanceOf[msg.sender] >= amount, "Not enough tokens");

        uint256 ethToReturn = (amount * tokenPrice) / (10 ** decimals);
        require(address(this).balance >= ethToReturn, "Not enough ETH");

        uint256 fee = (ethToReturn * feePercent) / 1000;
        uint256 payout = ethToReturn - fee;

        _transfer(msg.sender, owner, amount);
        payable(msg.sender).transfer(payout);

        totalSold += amount;
        _updateTokenPrice(false);
        emit TokensSold(msg.sender, amount, payout);
    }

    function _updateTokenPrice(bool isBuying) internal {
        if (isBuying) {
            tokenPrice += (priceFactor * totalBought / 1000);
        } else {
            uint256 newPrice = tokenPrice - (priceFactor * totalSold / 1000);
            tokenPrice = newPrice > minTokenPrice ? newPrice : minTokenPrice;
        }
    }

    function setPriceFactor(uint256 newFactor) external onlyOwner {
        priceFactor = newFactor;
    }

    function setFeePercent(uint256 newFee) external onlyOwner {
        require(newFee <= 50, "Fee too high");
        feePercent = newFee;
    }

    function setSellingAllowed(bool allowed) external onlyOwner {
        sellingAllowed = allowed;
        emit TradingStatusChanged(allowed);
    }

    function withdrawETH() external onlyOwner {
        require(address(this).balance > 0, "No ETH to withdraw");
        payable(owner).transfer(address(this).balance);
    }

    function getContractBalance() external view returns (uint256) {
        return address(this).balance;
    }

    function _transfer(address from, address to, uint256 amount) internal {
        require(to != address(0), "Invalid recipient");
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        emit Transfer(from, to, amount);
    }
}