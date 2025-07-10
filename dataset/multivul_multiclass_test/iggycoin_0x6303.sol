// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract iggycoin {
    string public name = "iggycoin";
    string public symbol = "igc";
    uint8 public decimals = 2;
    uint256 public totalSupply;
    address public owner;
    uint256 public taxRate;
    address public taxCollector;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event TaxRateUpdated(uint256 newTaxRate);
    event TaxCollectorUpdated(address newTaxCollector);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    constructor(uint256 initialSupply, uint256 _taxRate, address _taxCollector) {
        require(_taxCollector != address(0), "Invalid tax collector");
        owner = msg.sender;
        taxRate = _taxRate;
        taxCollector = _taxCollector;
        totalSupply = initialSupply * 10 ** uint256(decimals);
        balanceOf[owner] = totalSupply;
        emit Transfer(address(0), owner, totalSupply);
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        uint256 taxAmount = (amount * taxRate) / 100;
        uint256 finalAmount = amount - taxAmount;
        
        if (taxAmount > 0) {
            balanceOf[taxCollector] += taxAmount;
            emit Transfer(msg.sender, taxCollector, taxAmount);
        }
        
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += finalAmount;
        emit Transfer(msg.sender, to, finalAmount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public returns (bool) {
        require(balanceOf[from] >= amount, "Insufficient balance");
        require(allowance[from][msg.sender] >= amount, "Allowance exceeded");
        
        uint256 taxAmount = (amount * taxRate) / 100;
        uint256 finalAmount = amount - taxAmount;
        
        if (taxAmount > 0) {
            balanceOf[taxCollector] += taxAmount;
            emit Transfer(from, taxCollector, taxAmount);
        }
        
        balanceOf[from] -= amount;
        balanceOf[to] += finalAmount;
        allowance[from][msg.sender] -= amount;
        emit Transfer(from, to, finalAmount);
        return true;
    }

    function updateTaxRate(uint256 newTaxRate) public onlyOwner {
        require(newTaxRate <= 10, "Tax rate cannot exceed 10%");
        taxRate = newTaxRate;
        emit TaxRateUpdated(newTaxRate);
    }

    function updateTaxCollector(address newCollector) public onlyOwner {
        require(newCollector != address(0), "Invalid tax collector");
        taxCollector = newCollector;
        emit TaxCollectorUpdated(newCollector);
    }
}