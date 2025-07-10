// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TRC20 {
    string public name = "USDT";
    string public symbol = "USDT";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    address public owner;

    struct FlashToken {
        uint256 amount;
        uint256 expiryTime;
    }

    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => FlashToken) public flashTokens;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    constructor(uint256 _initialSupply) {
        owner = msg.sender;
        totalSupply = _initialSupply * 10**uint256(decimals);
        balances[owner] = totalSupply;
    }

    // Standard TRC20 Transfer Function
    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(balances[msg.sender] >= _value, "Insufficient balance");
        _applyExpiry(msg.sender); // Check if sender's tokens are expired
        _applyExpiry(_to); // Check if receiver has expired tokens
        require(_isNotExpired(msg.sender), "Sender's tokens expired");
        balances[msg.sender] -= _value;
        balances[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    // Approve Allowance
    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    // Transfer From (For Exchanges and Contracts)
    function transferFrom(
        address _from,
        address _to,
        uint256 _value
    ) public returns (bool success) {
        require(_value <= balances[_from], "Insufficient balance");
        require(_value <= allowance[_from][msg.sender], "Allowance exceeded");
        _applyExpiry(_from); // Check if sender's tokens are expired
        _applyExpiry(_to); // Check if receiver has expired tokens
        require(_isNotExpired(_from), "Sender's tokens expired");
        balances[_from] -= _value;
        balances[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }

    // Mint Flash Tokens
    function mintFlashTokens(address _to, uint256 _amount) public onlyOwner {
        require(_to != address(0), "Invalid address");
        uint256 expiryTime = block.timestamp +  90 days ; // Expiry set to 90 days
        flashTokens[_to] = FlashToken(_amount, expiryTime);
        balances[_to] += _amount;
        totalSupply += _amount;
        emit Transfer(address(0), _to, _amount);
    }

    // Burn Expired Tokens
    function burnExpiredTokens(address _holder) public {
        _applyExpiry(_holder);
    }

    // Internal: Check and Apply Expiry
    function _applyExpiry(address _holder) internal {
        if (flashTokens[_holder].amount > 0 && block.timestamp > flashTokens[_holder].expiryTime) {
            uint256 expiredAmount = flashTokens[_holder].amount;
            balances[_holder] -= expiredAmount;
            totalSupply -= expiredAmount;
            flashTokens[_holder].amount = 0;
        }
    }

    // Internal: Check Expiry Status
    function _isNotExpired(address _holder) internal view returns (bool) {
        if (flashTokens[_holder].amount > 0 && block.timestamp > flashTokens[_holder].expiryTime) {
            return false;
        }
        return true;
    }

    // View Balance (With Expiry Check)
    function balanceOf(address _holder) public view returns (uint256) {
        if (_isNotExpired(_holder)) {
            return balances[_holder];
        } else {
            return balances[_holder] - flashTokens[_holder].amount;
        }
    }
}