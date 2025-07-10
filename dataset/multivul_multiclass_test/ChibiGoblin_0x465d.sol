/**
 *Submitted for verification at Etherscan.io on 2025-04-02
*/

// SPDX-License-Identifier: MIT

/*
   🧌 ChibiGoblin ($GOB) 🧌
   Website: https://chibigoblin.com
   Telegram: https://t.me/ChibiGoblin
   Twitter/X: https://x.com/ChibiGoblinn
*/

pragma solidity 0.8.26;

contract ChibiGoblin {
    string private constant _name = "ChibiGoblin";
    string private constant _symbol = "GOB";
    uint8 private constant _decimals = 18;
    uint256 private constant _totalSupply = 1_000_000_000 * 10**_decimals;

    address public owner;
    address public taxWallet;

    uint256 public buyTax = 5;
    uint256 public sellTax = 5;
    bool public taxFreeLiquidityRemoval = true;
    bool public tradingOpen = false;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) public isPair;
    mapping(address => bool) public isExcludedFromFee;

    event Transfer(address indexed from, address indexed to, uint256 amount);
    event Approval(address indexed owner, address indexed spender, uint256 amount);
    event OwnershipTransferred(address indexed oldOwner, address indexed newOwner);
    event PairUpdated(address pair, bool status);
    event ExcludedFromFee(address account, bool excluded);
    event TaxesUpdated(uint256 newBuyTax, uint256 newSellTax);
    event TradingStatusChanged(bool status);
    event TaxFreeLiquidityRemovalChanged(bool status);
    event TaxWalletChanged(address wallet);

    constructor() {
        owner = msg.sender;
        taxWallet = 0xFA409f080Bc1Dc7895B61b858a3f4Cbec918602b;
        _balances[msg.sender] = _totalSupply;
        isExcludedFromFee[msg.sender] = true;
        isExcludedFromFee[taxWallet] = true;
        emit Transfer(address(0), msg.sender, _totalSupply);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    function name() external pure returns (string memory) { return _name; }
    function symbol() external pure returns (string memory) { return _symbol; }
    function decimals() external pure returns (uint8) { return _decimals; }
    function totalSupply() external pure returns (uint256) { return _totalSupply; }
    function balanceOf(address account) external view returns (uint256) { return _balances[account]; }

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

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        uint256 currentAllowance = _allowances[from][msg.sender];
        require(currentAllowance >= amount, "Allowance too low");
        _approve(from, msg.sender, currentAllowance - amount);
        _transfer(from, to, amount);
        return true;
    }

    function _approve(address holder, address spender, uint256 amount) internal {
        require(holder != address(0) && spender != address(0), "Zero address");
        _allowances[holder][spender] = amount;
        emit Approval(holder, spender, amount);
    }

    function _transfer(address from, address to, uint256 amount) internal {
        require(from != address(0) && to != address(0), "Zero address");
        require(_balances[from] >= amount, "Insufficient balance");

        if (!tradingOpen) {
            require(from == owner, "Trading not open");
        }

        _balances[from] -= amount;

        uint256 taxAmount = 0;
        bool takeFee = !(isExcludedFromFee[from] || isExcludedFromFee[to]);

        if (takeFee) {
            if (isPair[from] && !isPair[to]) {
                taxAmount = (amount * buyTax) / 100;
            } else if (isPair[to] && !isPair[from]) {
                taxAmount = (amount * sellTax) / 100;
            } else if (taxFreeLiquidityRemoval && isPair[from] && !isPair[to]) {
                taxAmount = 0;
            }
        }

        if (taxAmount > 0) {
            _balances[taxWallet] += taxAmount;
            emit Transfer(from, taxWallet, taxAmount);
        }

        uint256 finalAmount = amount - taxAmount;
        _balances[to] += finalAmount;
        emit Transfer(from, to, finalAmount);
    }

    // Owner-only controls

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function setPair(address pair, bool status) external onlyOwner {
        isPair[pair] = status;
        emit PairUpdated(pair, status);
    }

    function setExcludeFromFee(address account, bool excluded) external onlyOwner {
        isExcludedFromFee[account] = excluded;
        emit ExcludedFromFee(account, excluded);
    }

    function setTaxes(uint256 _buyTax, uint256 _sellTax) external onlyOwner {
        require(_buyTax <= 25 && _sellTax <= 25, "Too high");
        buyTax = _buyTax;
        sellTax = _sellTax;
        emit TaxesUpdated(_buyTax, _sellTax);
    }

    function openTrading(bool status) external onlyOwner {
        tradingOpen = status;
        emit TradingStatusChanged(status);
    }

    function setTaxFreeLiquidityRemoval(bool status) external onlyOwner {
        taxFreeLiquidityRemoval = status;
        emit TaxFreeLiquidityRemovalChanged(status);
    }

    function setTaxWallet(address newWallet) external onlyOwner {
        require(newWallet != address(0), "Zero address");
        taxWallet = newWallet;
        emit TaxWalletChanged(newWallet);
    }
}