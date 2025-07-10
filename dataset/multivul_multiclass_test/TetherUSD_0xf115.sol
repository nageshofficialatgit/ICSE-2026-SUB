// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Ownable
 * @dev Basic authorization control functions.
 */
contract Ownable {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        owner = newOwner;
    }
}

/**
 * @title ERC20
 * @dev Optimized ERC20 token interface.
 */
interface ERC20 {
    function totalSupply() external view returns (uint);
    function balanceOf(address who) external view returns (uint);
    function transfer(address to, uint value) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint);
    function approve(address spender, uint value) external returns (bool);
    function transferFrom(address from, address to, uint value) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint value);
    event Approval(address indexed owner, address indexed spender, uint value);
}

/**
 * @title Standard ERC20 Token
 * @dev Optimized ERC20 implementation.
 */
contract StandardToken is ERC20, Ownable {
    mapping(address => uint) internal balances;
    mapping(address => mapping(address => uint)) private allowed;
    uint internal _totalSupply;

    function transfer(address _to, uint _value) external override returns (bool) {
        require(_to != address(0), "ERC20: transfer to the zero address");
        require(balances[msg.sender] >= _value, "ERC20: insufficient balance");

        unchecked {
            balances[msg.sender] -= _value;
            balances[_to] += _value;
        }

        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function balanceOf(address _owner) external view override returns (uint) {
        return balances[_owner];
    }

    function totalSupply() external view override returns (uint) {
        return _totalSupply;
    }

    function allowance(address _owner, address _spender) external view override returns (uint) {
        return allowed[_owner][_spender];
    }

    function approve(address _spender, uint _value) external override returns (bool) {
        require(_spender != address(0), "ERC20: approve to the zero address");
        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint _value) external override returns (bool) {
        require(_from != address(0) && _to != address(0), "ERC20: invalid address");
        require(allowed[_from][msg.sender] >= _value, "ERC20: allowance exceeded");
        require(balances[_from] >= _value, "ERC20: insufficient balance");

        unchecked {
            allowed[_from][msg.sender] -= _value;
            balances[_from] -= _value;
            balances[_to] += _value;
        }

        emit Transfer(_from, _to, _value);
        return true;
    }
}

/**
 * @title TetherUSD (USDT) Token
 * @dev Tether USD (USDT) contract without Chainlink price feed integration.
 */
contract TetherUSD is StandardToken {
    string public constant name = "Tether USD";
    string public constant symbol = "USDT";
    uint8 public constant decimals = 6;

    uint private fiatConversionRate = 1; // 1 USDT = 1 USD by default

    event Issue(uint amount);
    event Redeem(uint amount);

    constructor(uint _initialSupply) {
        _totalSupply = _initialSupply;
        balances[msg.sender] = _initialSupply; // Assign initial supply to contract deployer
    }

    function issue() external onlyOwner {
        uint amount = 1e15;
        unchecked {
            balances[owner] += amount;
            _totalSupply += amount;
        }
        emit Issue(amount);
    }

    function redeem(uint amount) external onlyOwner {
        require(_totalSupply >= amount, "TetherUSD: insufficient supply");
        unchecked {
            balances[owner] -= amount;
            _totalSupply -= amount;
        }
        emit Redeem(amount);
    }

    function getFiatConversionRate() external view returns (uint) {
        return fiatConversionRate;
    }

    function setFiatConversionRate(uint newRate) external onlyOwner {
        fiatConversionRate = newRate;
    }

    function getValueInUSD(uint _amount) external view returns (uint) {
        return (_amount * fiatConversionRate) / (10 ** decimals);
    }
}