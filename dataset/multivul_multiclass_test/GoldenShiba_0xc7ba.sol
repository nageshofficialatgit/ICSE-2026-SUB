// SPDX-License-Identifier: MIT
pragma solidity 0.8.27;

contract GoldenShiba {
    // Token details
    string private constant _name = "GOLDENSHIBA";
    string private constant _symbol = "GOL";
    uint8 private constant _decimals = 18;
    uint256 private constant _totalSupply = 1_000_000_000 * 10**_decimals;

    // Owner state
    address public owner;

    // Tax details
    uint256 public constant buyTax = 5;   // 5%
    uint256 public constant sellTax = 5;  // 5%
    address public constant taxWallet = 0x50Ee2d1d768398081CC9c4f425709cAe3C2F4711;

    // Trading toggle
    bool public tradingOpen = false;

    // Mappings for balances and allowances
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    // Mapping to track which addresses are designated as DEX pairs
    mapping(address => bool) public isPair;

    // Events
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event TradingOpened(bool enabled);
    event PairUpdated(address pair, bool status);

    constructor() {
        owner = msg.sender;
        _balances[msg.sender] = _totalSupply;
        emit Transfer(address(0), msg.sender, _totalSupply);
    }

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Not contract owner");
        _;
    }

    // Basic ERC20 Functions

    function name() external pure returns (string memory) {
        return _name;
    }

    function symbol() external pure returns (string memory) {
        return _symbol;
    }

    function decimals() external pure returns (uint8) {
        return _decimals;
    }

    function totalSupply() external pure returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) external view returns (uint256) {
        return _balances[account];
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }

    function allowance(address holder, address spender) external view returns (uint256) {
        return _allowances[holder][spender];
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) external returns (bool) {
        uint256 currentAllowance = _allowances[from][msg.sender];
        require(currentAllowance >= amount, "Transfer > allowance");
        
        _approve(from, msg.sender, currentAllowance - amount);
        _transfer(from, to, amount);
        return true;
    }

    // Owner can transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner is zero addr");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    // Toggle trading
    function openTrading(bool _open) external onlyOwner {
        tradingOpen = _open;
        emit TradingOpened(_open);
    }

    // Mark or unmark an address as a DEX pair
    function setPair(address pair, bool status) external onlyOwner {
        isPair[pair] = status;
        emit PairUpdated(pair, status);
    }

    // Internal helpers
    function _approve(
        address holder,
        address spender,
        uint256 amount
    ) internal {
        require(holder != address(0), "Approve from zero addr");
        require(spender != address(0), "Approve to zero addr");

        _allowances[holder][spender] = amount;
        emit Approval(holder, spender, amount);
    }

    function _transfer(
        address from,
        address to,
        uint256 amount
    ) internal {
        require(from != address(0), "Transfer from zero addr");
        require(to != address(0), "Transfer to zero addr");

        // Check trading is open or sender is owner
        if (!tradingOpen) {
            require(from == owner, "Trading not opened");
        }

        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "Insufficient balance");

        // Determine if buy or sell to apply tax
        bool takeTax = false;
        uint256 taxAmount = 0;

        if (isPair[from]) {
            // from = DEX Pair => BUY
            takeTax = true;
            taxAmount = (amount * buyTax) / 100;
        } else if (isPair[to]) {
            // to = DEX Pair => SELL
            takeTax = true;
            taxAmount = (amount * sellTax) / 100;
        }

        _balances[from] = fromBalance - amount;

        if (takeTax && taxAmount > 0) {
            // Send tax to taxWallet
            uint256 transferAmount = amount - taxAmount;
            _balances[taxWallet] += taxAmount;
            _balances[to] += transferAmount;

            emit Transfer(from, taxWallet, taxAmount);
            emit Transfer(from, to, transferAmount);
        } else {
            // No tax scenario
            _balances[to] += amount;
            emit Transfer(from, to, amount);
        }
    }
}