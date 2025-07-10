// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract OmniToken {
    string public name = "Omni";
    string public symbol = "OMNI";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    address public owner;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    constructor(uint256 _initialSupply) {
        owner = msg.sender;
        totalSupply = _initialSupply * (10 ** uint256(decimals));
        balanceOf[owner] = totalSupply;
        emit Transfer(address(0), owner, totalSupply);
    }

    function transfer(address _to, uint256 _value) public returns (bool) {
        require(balanceOf[msg.sender] >= _value, "Not enough balance");
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public returns (bool) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        require(balanceOf[_from] >= _value, "Not enough balance");
        require(allowance[_from][msg.sender] >= _value, "Not approved");
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }

    function mint(uint256 _amount) public onlyOwner {
        uint256 mintedAmount = _amount * (10 ** uint256(decimals));
        totalSupply += mintedAmount;
        balanceOf[owner] += mintedAmount;
        emit Transfer(address(0), owner, mintedAmount);
    }

    function burn(uint256 _amount) public {
        uint256 burnedAmount = _amount * (10 ** uint256(decimals));
        require(balanceOf[msg.sender] >= burnedAmount, "Not enough to burn");
        balanceOf[msg.sender] -= burnedAmount;
        totalSupply -= burnedAmount;
        emit Transfer(msg.sender, address(0), burnedAmount);
    }
}