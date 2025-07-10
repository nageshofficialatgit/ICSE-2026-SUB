// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

interface IUniswapV2Router02 {
    function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);
}

contract tGtPresale {
    address public owner;
    IERC20 public token;
    uint256 public constant rate = 26666666; // 1 ETH = 26,666,666 tGt
    uint256 public constant totalTokensForSale = 400_000_000 * 10**18; // 400M tGt
    uint256 public constant liquidityReserve = 200_000_000 * 10**18; // 200M tGt Uniswap likidite için

    uint256 public constant softcap = 9 ether;
    uint256 public constant hardcap = 14.9 ether;
    uint256 public constant minContribution = 0.003 ether;
    uint256 public constant maxContribution = 0.15 ether;
    uint256 public totalRaised;

    uint256 public startTime;
    uint256 public endTime;
    bool public liquidityAdded = false;

    address public constant uniswapRouter = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D; // Uniswap V2 Router (Ethereum için)
    mapping(address => uint256) public contributions;
    mapping(address => uint256) public lockedTokens;
    mapping(address => uint256) public claimTimes;

    event TokensPurchased(address indexed buyer, uint256 ethAmount, uint256 tokenAmount);
    event TokensClaimed(address indexed user, uint256 amount);
    event LiquidityAdded(uint256 amountToken, uint256 amountETH);
    event Withdrawn(address indexed owner, uint256 amount);
    event SoftcapReached(uint256 totalRaised);
    event PresaleEnded(uint256 totalETH, uint256 totalTokens);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    modifier isActive() {
        require(block.timestamp >= startTime && block.timestamp <= endTime, "Presale is not active");
        _;
    }

    constructor(address _tokenAddress, uint256 _startTime, uint256 _endTime) {
        require(_startTime < _endTime, "Start time must be before end time");
        owner = msg.sender;
        token = IERC20(_tokenAddress);
        startTime = _startTime;
        endTime = _endTime;
    }

    function buyTokens() public payable isActive {
        require(msg.value >= minContribution && msg.value <= maxContribution, "Contribution out of range");
        require(totalRaised + msg.value <= hardcap, "Hardcap reached");

        uint256 totalTokenAmount = msg.value * rate;
        uint256 immediateTokens = totalTokenAmount / 2;
        uint256 lockedAmount = totalTokenAmount - immediateTokens;

        require(token.balanceOf(address(this)) >= totalTokenAmount, "Not enough tokens in contract");

        // Önce token transferini yap
        require(token.transfer(msg.sender, immediateTokens), "Token transfer failed");

        // Sonra state değişkenlerini güncelle
        contributions[msg.sender] += msg.value;
        lockedTokens[msg.sender] += lockedAmount;
        claimTimes[msg.sender] = endTime;
        totalRaised += msg.value;

        emit TokensPurchased(msg.sender, msg.value, totalTokenAmount);

        if (totalRaised >= softcap) {
            emit SoftcapReached(totalRaised);
        }
    }

    function claimTokens() public {
        require(lockedTokens[msg.sender] > 0, "No tokens to claim");

        uint256 totalLocked = lockedTokens[msg.sender];
        uint256 timeSinceListing = block.timestamp - endTime;
        uint256 claimableAmount = 0;

        if (timeSinceListing >= 540 days) {
            claimableAmount = totalLocked;
        } else if (timeSinceListing >= 180 days) {
            claimableAmount = totalLocked / 2;
        } else {
            revert("Tokens are still locked");
        }

        require(token.balanceOf(address(this)) >= claimableAmount, "Not enough tokens in contract");

        lockedTokens[msg.sender] -= claimableAmount;
        require(token.transfer(msg.sender, claimableAmount), "Token claim failed");

        emit TokensClaimed(msg.sender, claimableAmount);
    }

    function finalizePresale() external onlyOwner {
        require(block.timestamp > endTime, "Presale is not yet finished");
        require(totalRaised >= softcap, "Softcap not reached");
        require(!liquidityAdded, "Liquidity already added");

        uint256 remainingTokens = token.balanceOf(address(this));
        uint256 totalLiquidityTokens = remainingTokens + liquidityReserve;

        require(token.balanceOf(address(this)) >= totalLiquidityTokens, "Not enough tokens in contract");

        require(token.approve(uniswapRouter, totalLiquidityTokens), "Token approval failed");

        IUniswapV2Router02(uniswapRouter).addLiquidityETH{value: totalRaised}(
            address(token),
            totalLiquidityTokens,
            0,
            0,
            owner,
            block.timestamp + 300
        );

        liquidityAdded = true;
        emit LiquidityAdded(totalLiquidityTokens, totalRaised);
        emit PresaleEnded(totalRaised, totalLiquidityTokens);
    }

    function withdrawTokens() external onlyOwner {
        require(totalRaised < softcap, "Softcap reached, cannot withdraw");
        uint256 contractBalance = token.balanceOf(address(this));
        require(contractBalance > 0, "No tokens to withdraw");

        require(token.transfer(owner, contractBalance), "Token transfer failed");

        emit Withdrawn(owner, contractBalance);
    }

    function withdrawETH() external onlyOwner {
        require(totalRaised < softcap, "Cannot withdraw ETH after softcap reached");
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH to withdraw");

        (bool success, ) = payable(owner).call{value: balance}("");
        require(success, "Withdraw failed");

        emit Withdrawn(owner, balance);
    }

    receive() external payable {
        buyTokens();
    }
}