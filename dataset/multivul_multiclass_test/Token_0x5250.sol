/*
SPDX-License-Identifier: Apache-2.0
*/
pragma solidity ^0.8.29;

contract Token {
	//状态变量（注意，如果对合约升级，新合约不得改变原合约的状态变量，包括名称、类型和顺序，但是可以追加新的状态变量）
    string public name = "ETron";      //  token name
    string public symbol = "ETR";           //  token symbol
    uint256 public constant decimals = 6;            //  token digit

    mapping (address => uint256) public balanceOf;
    mapping (address => mapping (address => uint256)) public allowance;

    uint256 public totalSupply = 0;  // 初始总供应量为0
    bool public stopped = false;

    address owner = address(0x0);

    modifier isOwner {
        assert(owner == msg.sender);
        _;
    }

    modifier isRunning {
        assert (!stopped);
        _;
    }

    modifier validAddress(address _addr) {
        require(_addr != address(0), "Zero address not allowed");
        _;
    }

    constructor() {
        owner = msg.sender;  // 只设置合约拥有者
    }

    // 添加铸造功能，只有owner可以调用
    function mint(address _to, uint256 _value) public isOwner validAddress(_to) returns (bool success) {
        require(_to != address(0), "Invalid address");
        totalSupply += _value;
        balanceOf[_to] += _value;
        emit Transfer(address(0x0), _to, _value);
        return true;
    }

    function transfer(address _to, uint256 _value) public isRunning validAddress(_to) returns (bool success) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public isRunning validAddress(_to) returns (bool success) {
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Insufficient allowance");
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public isRunning validAddress(_spender) returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function stop() public isOwner {
        stopped = true;
        emit Stopped(msg.sender);
    }

    function start() public isOwner {
        stopped = false;
        emit Started(msg.sender);
    }

    function setName(string memory _name) public isOwner {
        string memory oldName = name;
        name = _name;
        emit NameChanged(oldName, _name);
    }

    function burn(uint256 _value) public {
        require(balanceOf[msg.sender] >= _value);
        balanceOf[msg.sender] -= _value;
        balanceOf[address(0x0)] += _value;
        emit Transfer(msg.sender, address(0x0), _value);
        emit Burned(msg.sender, _value);
    }

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
    event Stopped(address indexed by);
    event Started(address indexed by);
    event NameChanged(string oldName, string newName);
    event Burned(address indexed burner, uint256 value);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    function transferOwnership(address newOwner) public isOwner {
        require(newOwner != address(0), "Invalid owner address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}