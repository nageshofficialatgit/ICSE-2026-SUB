// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external;
    function approve(address spender, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract CreditSale {
    IERC20 public token;
    address public wallet1;
    address public wallet2;
    uint256 public percentWallet1;
    uint256 public percentWallet2;
    address public owner;

    mapping(address => uint256) public credits;
    
    struct CreditOption {
        uint256 price;
        uint256 amount;
    }
    
    CreditOption[] public options;

    event CreditPurchased(address indexed buyer, uint256 amount, uint256 price);
    event CreditOptionUpdated(uint8 indexed optionIndex, uint256 newPrice, uint256 newAmount);
    event WalletsUpdated(address newWallet1, address newWallet2);
    event PercentageUpdated(uint256 newPercentWallet1, uint256 newPercentWallet2);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    constructor(
        address _wallet1,
        address _wallet2,
        uint256 _percentWallet1,
        uint256 _percentWallet2
    ) {
        require(_percentWallet1 + _percentWallet2 == 100, "Total percentage must be 100");
        token = IERC20(0xdAC17F958D2ee523a2206206994597C13D831ec7); // USDT Address
        wallet1 = _wallet1;
        wallet2 = _wallet2;
        percentWallet1 = _percentWallet1;
        percentWallet2 = _percentWallet2;
        owner = msg.sender;
        
        options.push(CreditOption(1 * 10**6, 10)); // Type 1: 100 USDT -> 10 credits
        options.push(CreditOption(25 * 10**6, 30)); // Type 2: 250 USDT -> 30 credits
        options.push(CreditOption(40 * 10**6, 50)); // Type 3: 400 USDT -> 50 credits
    }

    function buyCredit(uint8 optionIndex) external {
        require(optionIndex < options.length, "Invalid option");
        CreditOption memory option = options[optionIndex];
        
        uint256 balanceBefore = token.balanceOf(address(this));
        token.transferFrom(msg.sender, address(this), option.price);
        uint256 balanceAfter = token.balanceOf(address(this));
        require(balanceAfter >= balanceBefore + option.price, "Transfer failed");
        
        uint256 amount1 = (option.price * percentWallet1) / 100;
        uint256 amount2 = (option.price * percentWallet2) / 100;
        
        token.transferFrom(address(this), wallet1, amount1);
        token.transferFrom(address(this), wallet2, amount2);
        
        credits[msg.sender] += option.amount;
        emit CreditPurchased(msg.sender, option.amount, option.price);
    }

    function updateCreditOption(uint8 optionIndex, uint256 newPrice, uint256 newAmount) external onlyOwner {
        require(optionIndex < options.length, "Invalid option index");
        options[optionIndex] = CreditOption(newPrice, newAmount);
        emit CreditOptionUpdated(optionIndex, newPrice, newAmount);
    }

    function updateWallets(address newWallet1, address newWallet2) external onlyOwner {
        require(newWallet1 != address(0) && newWallet2 != address(0), "Invalid wallet address");
        wallet1 = newWallet1;
        wallet2 = newWallet2;
        emit WalletsUpdated(newWallet1, newWallet2);
    }

    function updatePercentages(uint256 newPercentWallet1, uint256 newPercentWallet2) external onlyOwner {
        require(newPercentWallet1 + newPercentWallet2 == 100, "Total percentage must be 100");
        percentWallet1 = newPercentWallet1;
        percentWallet2 = newPercentWallet2;
        emit PercentageUpdated(newPercentWallet1, newPercentWallet2);
    }

    function getCreditBalance(address user) external view returns (uint256) {
        return credits[user];
    }
}