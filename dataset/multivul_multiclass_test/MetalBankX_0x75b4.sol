// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Token {

    /// @return total amount of tokens
    function totalSupply() public view virtual returns (uint256) {}

    /// @param _owner The address from which the balance will be retrieved
    /// @return The balance
    function balanceOf(address _owner) public view virtual returns (uint256) {}

    // @notice send `_value` token to `_to` from `msg.sender`
    // @param _to The address of the recipient
    // @param _value The amount of token to be transferred
    // @return Whether the transfer was successful or not
    function transfer(address _to, uint256 _value) public virtual returns (bool success) {}

    // @notice send `_value` token to `_to` from `_from` on the condition it is approved by `_from`
    // @param _from The address of the sender
    // @param _to The address of the recipient
    // @param _value The amount of token to be transferred
    // @return Whether the transfer was successful or not
    function transferFrom(address _from, address _to, uint256 _value) public virtual returns (bool success) {}

    // @notice `msg.sender` approves `_addr` to spend `_value` tokens
    // @param _spender The address of the account able to transfer the tokens
    // @param _value The amount of tokens to be approved for transfer
    // @return Whether the approval was successful or not
    function approve(address _spender, uint256 _value) public virtual returns (bool success) {}

    // @param _owner The address of the account owning tokens
    // @param _spender The address of the account able to transfer the tokens
    // @return Amount of remaining tokens allowed to be spent
    function allowance(address _owner, address _spender) public view virtual returns (uint256 remaining) {}

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}

contract StandardToken is Token {

    mapping(address => uint256) balances;
    mapping(address => mapping(address => uint256)) allowed;
    uint256 _totalSupply;  // Internal variable for total supply

    function totalSupply() public view override returns (uint256) {
        return _totalSupply;  // Return the internal totalSupply
    }

    function transfer(address _to, uint256 _value) public override returns (bool success) {
        require(balances[msg.sender] >= _value && _value > 0, "Insufficient balance");
        balances[msg.sender] -= _value;
        balances[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public override returns (bool success) {
        require(balances[_from] >= _value && allowed[_from][msg.sender] >= _value && _value > 0, "Transfer not allowed");
        balances[_to] += _value;
        balances[_from] -= _value;
        allowed[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }

    function balanceOf(address _owner) public view override returns (uint256 balance) {
        return balances[_owner];
    }

    function approve(address _spender, uint256 _value) public override returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function allowance(address _owner, address _spender) public view override returns (uint256 remaining) {
        return allowed[_owner][_spender];
    }

    // Constructor to initialize totalSupply
    constructor(uint256 initialSupply) {
        _totalSupply = initialSupply;  // Set the initial supply
        balances[msg.sender] = _totalSupply;  // Assign all tokens to the creator
    }
}

contract MetalBankX is StandardToken {

    string public name;
    uint8 public decimals;
    string public symbol;
    string public version = 'H1.0';
    uint256 public unitsOneEthCanBuy;
    uint256 public totalEthInWei;
    address public fundsWallet;

    // Updated constructor to set initial values
    constructor(uint256 initialSupply) StandardToken(initialSupply) {
        _totalSupply = 1000000000000000000000000000000;
        balances[msg.sender] = 1000000000000000000000000000000;              
        name = "METALBANK X";                      
        decimals = 18;
        symbol = "MBXAU";
        unitsOneEthCanBuy = 10000;  // Adjust the price of your token here
        fundsWallet = msg.sender;
    }

    // Fallback function is now receive() in Solidity 0.8.x
    receive() external payable {
        totalEthInWei += msg.value;
        uint256 amount = msg.value * unitsOneEthCanBuy;
        require(balances[fundsWallet] >= amount, "Not enough tokens available");

        balances[fundsWallet] -= amount;
        balances[msg.sender] += amount;

        emit Transfer(fundsWallet, msg.sender, amount);

        payable(fundsWallet).transfer(msg.value);  // Send the ETH to the fundsWallet
    }

    // Approve and call function
    function approveAndCall(address _spender, uint256 _value, bytes memory _extraData) public returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);

        (bool successCall, ) = _spender.call(abi.encodeWithSignature("receiveApproval(address,uint256,address,bytes)", msg.sender, _value, address(this), _extraData));
        require(successCall, "Approve and call failed");

        return true;
    }
}