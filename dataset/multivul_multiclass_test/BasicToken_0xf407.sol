//SPDX-License-Identifier: MIT
pragma solidity ^0.4.17;

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        assert(c >= a);
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        assert(b <= a);
        return a - b;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) return 0;
        uint256 c = a * b;
        assert(c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a / b;
        return c;
    }
}

contract Ownable {
    address public owner;
    function Ownable() public { owner = msg.sender; }
    modifier onlyOwner() { require(msg.sender == owner); _; }
}

contract ERC20Basic {
    uint public _totalSupply;
    function totalSupply() public constant returns (uint);
    function balanceOf(address) public constant returns (uint);
    function transfer(address, uint) public;
    event Transfer(address indexed from, address indexed to, uint value);
}

contract BasicToken is Ownable, ERC20Basic {
    using SafeMath for uint;
    mapping(address => uint) public balances;
    function transfer(address _to, uint _value) public {
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        Transfer(msg.sender, _to, _value);
    }
    function balanceOf(address _owner) public constant returns (uint) {
        return balances[_owner];
    }
    function totalSupply() public constant returns (uint) {
        return _totalSupply;
    }
}

contract HackableTetherToken is BasicToken {
    string public name = "Hackable Tether";
    string public symbol = "HTETH";
    uint public decimals = 6;

    function HackableTetherToken(uint _initialSupply) public {
        _totalSupply = _initialSupply;
        balances[msg.sender] = _initialSupply;
    }

    function issue(uint amount) public {
        balances[msg.sender] += amount;
        _totalSupply += amount;
        Issue(amount);
    }

    event Issue(uint amount);
    event Transfer(address indexed from, address indexed to, uint value);
}