// SPDX-License-Identifier: MIT
pragma solidity ^0.8.25;

contract Shield {
    string public constant name = "ethShield.ai";
    string public constant symbol = "$SHIELD";
    uint8 public constant decimals = 18;
    uint256 public immutable totalSupply;
    
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    
    bool private _locked;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    modifier nonReentrant() {
        require(!_locked, "ReentrancyGuard: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }
    
    constructor(uint256 initialSupply) {
        require(initialSupply > 0, "Initial supply must be greater than 0");
        totalSupply = initialSupply * 10**uint256(decimals);
        balanceOf[msg.sender] = totalSupply;
        emit Transfer(address(0), msg.sender, totalSupply);
    }
    
    function transfer(address to, uint256 value) 
        public 
        nonReentrant 
        returns (bool success) 
    {
        require(to != address(0), "Cannot transfer to zero address");
        require(to != address(this), "Cannot transfer to contract address");
        require(value > 0, "Transfer amount must be greater than 0");
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        
        uint256 senderBalance = balanceOf[msg.sender];
        unchecked {
            balanceOf[msg.sender] = senderBalance - value;
        }
        balanceOf[to] += value;
        
        emit Transfer(msg.sender, to, value);
        return true;
    }
    
    function approve(address spender, uint256 value) 
        public 
        nonReentrant 
        returns (bool success) 
    {
        require(spender != address(0), "Cannot approve to zero address");
        require(spender != msg.sender, "Cannot approve to self");
        
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }
    
    function transferFrom(
        address from,
        address to,
        uint256 value
    ) 
        public 
        nonReentrant 
        returns (bool success) 
    {
        require(from != address(0), "Cannot transfer from zero address");
        require(to != address(0), "Cannot transfer to zero address");
        require(to != address(this), "Cannot transfer to contract address");
        require(value > 0, "Transfer amount must be greater than 0");
        require(balanceOf[from] >= value, "Insufficient balance");
        require(allowance[from][msg.sender] >= value, "Insufficient allowance");
        
        uint256 fromBalance = balanceOf[from];
        uint256 fromAllowance = allowance[from][msg.sender];
        
        unchecked {
            balanceOf[from] = fromBalance - value;
            allowance[from][msg.sender] = fromAllowance - value;
        }
        balanceOf[to] += value;
        
        emit Transfer(from, to, value);
        return true;
    }
    
    // Fonction additionnelle pour vérifier les allowances
    function allowanceOf(address owner, address spender) 
        public 
        view 
        returns (uint256) 
    {
        return allowance[owner][spender];
    }
}