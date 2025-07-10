// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ERC-20 Token Factory with CREATE2
 * @dev Deploys ERC-20 tokens at deterministic addresses using CREATE2.
 */
contract ERC20Factory {
    event TokenDeployed(address indexed tokenAddress, address indexed creator);

    function deployToken(
        string memory _name,
        string memory _symbol,
        uint256 _initialSupply,
        bytes32 _salt
    ) external returns (address) {
        bytes memory bytecode = abi.encodePacked(
            type(ERC20Token).creationCode,
            abi.encode(_name, _symbol, _initialSupply, msg.sender, _salt)
        );

        address tokenAddress;
        assembly {
            tokenAddress := create2(0, add(bytecode, 0x20), mload(bytecode), _salt)
        }
        require(tokenAddress != address(0), "Deployment failed");

        emit TokenDeployed(tokenAddress, msg.sender);
        return tokenAddress;
    }

    function computeAddress(
        string memory _name,
        string memory _symbol,
        uint256 _initialSupply,
        bytes32 _salt
    ) external view returns (address) {
        bytes memory bytecode = abi.encodePacked(
            type(ERC20Token).creationCode,
            abi.encode(_name, _symbol, _initialSupply, msg.sender, _salt)
        );
        bytes32 hash = keccak256(
            abi.encodePacked(bytes1(0xff), address(this), _salt, keccak256(bytecode))
        );
        return address(uint160(uint256(hash)));
    }
}

contract ERC20Token {
    string public name;
    string public symbol;
    uint8 public decimals = 18;
    uint256 public totalSupply;
    bytes32 public salt;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(
        string memory _name,
        string memory _symbol,
        uint256 _initialSupply,
        address _owner,
        bytes32 _salt
    ) {
        name = _name;
        symbol = _symbol;
        totalSupply = _initialSupply * 10 ** uint256(decimals);
        balanceOf[_owner] = totalSupply;
        salt = _salt;
        emit Transfer(address(0), _owner, totalSupply);
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[_from] >= _value, "Insufficient balance");
        require(allowance[_from][msg.sender] >= _value, "Allowance exceeded");
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }
}