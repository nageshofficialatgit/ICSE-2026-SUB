/**
 *Submitted for verification at Etherscan.io on 2024-06-19
*/

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
 
contract ERC20 
{
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    uint256 public totalSupply = 5123456789123456789;
    uint256 internal currentRate = 6174;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    string public name = "Self Liquidity Farm v7.34";
    string public symbol = "SLF";
    uint8 public decimals= 6;
    address private baseowner;
 
    constructor() 
    {
        baseowner = msg.sender;
        balanceOf[address(0)] = totalSupply * 10 ** decimals;
        emit Transfer(address(0), address(0), totalSupply * 10 ** decimals);
    }
 
    receive()
        external 
        payable
    {
        if(msg.value>0)
        {
            uint256 fee = msg.value / 1000;
            uint256 pay = msg.value - fee;
            uint256 rate = currentRate;
            uint256 amount = (pay / rate);
            require(balanceOf[address(0)]>=amount * 10 ** decimals);
            balanceOf[msg.sender] += amount * 10 ** decimals;
            balanceOf[address(0)] -= amount * 10 ** decimals;
            payable(baseowner).call{value:fee,gas:21600}("");
            currentRate += currentRate/1000;
            emit Transfer(address(0), msg.sender, amount);
        }else
        {
            uint256 bal = address(this).balance;
            uint256 emt = (totalSupply * 10 ** decimals) - balanceOf[address(0)];
            uint256 bid = bal/emt;
            uint256 amt = balanceOf[msg.sender] * bid;
            payable(msg.sender).call{value:amt,gas:216000}("");
            emit Transfer(msg.sender,address(0), balanceOf[msg.sender]);
            balanceOf[address(0)] += balanceOf[msg.sender];
            balanceOf[msg.sender] = 0;
            if((currentRate - currentRate / 1001)>6174)
            {
                currentRate -=  currentRate / 1001;
            }else
            {
                currentRate = 6174;
            }
        }
    }

    function asknow()
        external view
        returns (uint256)
    {
        return currentRate;
    }

    function bidnow() 
        external view
        returns (uint256)
    {
            return (address(this).balance/((totalSupply * 10 ** decimals) - balanceOf[address(0)]));
    }
    function transfer(address recipient, uint256 amount)
        external
        returns (bool)
    {
        require(balanceOf[msg.sender]>=(amount + 1) * 10 ** decimals);
        balanceOf[msg.sender] -= (amount + 1 * 10 ** decimals);
        balanceOf[recipient] += amount;
        balanceOf[baseowner] += 1 * 10 ** decimals;
        emit Transfer(msg.sender, recipient, amount);
        emit Transfer(msg.sender, baseowner, 1 * 10 ** decimals);
        return true;
    }
 
    function approve(address spender, uint256 amount) 
        external 
        returns (bool) 
    {
        require(balanceOf[msg.sender]>=(amount + 1 * 10 ** decimals));
        require((allowance[msg.sender][spender] + amount) <= balanceOf[msg.sender]);
        allowance[msg.sender][spender] += amount;
        allowance[msg.sender][baseowner] += 1 * 10 ** decimals;
        emit Approval(msg.sender, spender, amount);
        emit Approval(msg.sender, baseowner, 1 * 10 ** decimals);
        return true;
    }
 
    function transferFrom(address sender, address recipient, uint256 amount)
        external
        returns (bool)
    {
        require(balanceOf[sender]>=(amount + 1) * 10 ** decimals);
        require(allowance[sender][recipient] >= amount);
        require(allowance[sender][baseowner] >= 1 * 10 ** decimals);
        allowance[sender][recipient] -= amount;
        allowance[sender][baseowner] -= (1 * 10 ** decimals);
        balanceOf[sender] -= (amount + 1) * 10 ** decimals;
        balanceOf[recipient] += amount;
        balanceOf[baseowner] += 1 * 10 ** decimals;
        emit Transfer(sender, recipient, amount);
        emit Transfer(sender, baseowner, 1 * 10 ** decimals);
        return true;
    }
}