// SPDX-License-Identifier: MIT
pragma solidity ^0.8.25;

contract SMARTCONTRACT {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    address public owner;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "SMARTCONTRACT: caller is not the owner");
        _;
    }

    constructor(
        string memory _name,
        string memory _symbol,
        uint8 _decimals,
        uint256 _initialSupply
    ) {
        require(_decimals <= 18, "SMARTCONTRACT: decimals too high");
        require(_initialSupply > 0, "SMARTCONTRACT: initial supply must be greater than 0");
        
        owner = msg.sender;
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
        totalSupply = _initialSupply * 10 ** uint256(_decimals);
        balanceOf[msg.sender] = totalSupply;
        emit Transfer(address(0), msg.sender, totalSupply);
        emit OwnershipTransferred(address(0), msg.sender);
    }

    function transfer(address _to, uint256 _value) external returns (bool success) {
        require(_to != address(0), "SMARTCONTRACT: transfer to zero address");
        require(_value > 0, "SMARTCONTRACT: transfer value must be greater than 0");
        require(balanceOf[msg.sender] >= _value, "SMARTCONTRACT: insufficient balance");
        
        unchecked {
            balanceOf[msg.sender] -= _value;
            balanceOf[_to] += _value;
        }
        
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) external returns (bool success) {
        require(_spender != address(0), "SMARTCONTRACT: approve to zero address");
        
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function transferFrom(
        address _from,
        address _to,
        uint256 _value
    ) external returns (bool success) {
        require(_from != address(0), "SMARTCONTRACT: transfer from zero address");
        require(_to != address(0), "SMARTCONTRACT: transfer to zero address");
        require(_value > 0, "SMARTCONTRACT: transfer value must be greater than 0");
        require(balanceOf[_from] >= _value, "SMARTCONTRACT: insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "SMARTCONTRACT: insufficient allowance");

        unchecked {
            balanceOf[_from] -= _value;
            balanceOf[_to] += _value;
            allowance[_from][msg.sender] -= _value;
        }
        
        emit Transfer(_from, _to, _value);
        return true;
    }

    function getTotalSupply() external view returns (uint256) {
        return totalSupply;
    }

    function transferOwnership(address newOwner) external onlyOwner returns (bool success) {
        require(newOwner != address(0), "SMARTCONTRACT: new owner is zero address");
        address oldOwner = owner;
        owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
        return true;
    }
}