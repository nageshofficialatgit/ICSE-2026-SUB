/**
The intent is that this token automatically distributes .5% reflections in Kishu Coin
to all holders with each transaction and that it is can be paired in a uniswap V4 pool.  
Assuming the code works as intended, LP will be burned.  
If it does not we will be pulling the LP and rewriting the contract.  Please
take that into consideration whereas this is an experimental code.
*/

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimplifiedKishu {
    string public name = "Kishu Coin";
    string public symbol = "Kishu";
    uint8 public decimals = 9;
    uint256 public totalSupply = 1000000000 * 10**decimals;
    
    mapping(address => uint256) private _rOwned;
    mapping(address => uint256) private _tOwned;
    mapping(address => mapping(address => uint256)) private _allowances;
    
    uint256 private _rTotal = (type(uint256).max - (type(uint256).max % totalSupply));
    uint256 private _tFeeTotal;
    uint256 public reflectionFee = 5; 
    
    address public owner;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    constructor() {
        owner = msg.sender;
        _rOwned[msg.sender] = _rTotal;
        emit Transfer(address(0), msg.sender, totalSupply);
    }
    
    function balanceOf(address account) public view returns (uint256) {
        return tokenFromReflection(_rOwned[account]);
    }
    
    function transfer(address recipient, uint256 amount) public returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }
    
    function approve(address spender, uint256 amount) public returns (bool) {
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }
    
    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(_allowances[sender][msg.sender] >= amount, "Allowance exceeded");
        _allowances[sender][msg.sender] -= amount;
        _transfer(sender, recipient, amount);
        return true;
    }
    
    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0) && recipient != address(0), "Invalid address");
        require(balanceOf(sender) >= amount, "Insufficient balance");
        
        uint256 currentRate = _getRate();
        uint256 rAmount = amount * currentRate;
        uint256 rFee = (rAmount * reflectionFee) / 1000;
        uint256 rTransferAmount = rAmount - rFee;
        
        _rOwned[sender] -= rAmount;
        _rOwned[recipient] += rTransferAmount;
        _reflectFee(rFee, amount * reflectionFee / 1000);
        
        emit Transfer(sender, recipient, amount - (amount * reflectionFee / 1000));
    }
    
    function _reflectFee(uint256 rFee, uint256 tFee) private {
        _rTotal -= rFee;
        _tFeeTotal += tFee;
    }
    
    function tokenFromReflection(uint256 rAmount) public view returns (uint256) {
        require(rAmount <= _rTotal, "Amount must be less than total reflections");
        return rAmount / _getRate();
    }
    
    function _getRate() private view returns (uint256) {
        return _rTotal / totalSupply;
    }
}