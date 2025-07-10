pragma solidity ^0.4.0;
contract Token {
    function totalSupply() public constant returns (uint256 supply) {}
    function balanceOf(address _owner) public constant returns (uint256 balance) {}
    function transfer(address _to, uint256 _value) public returns (bool success) {}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {}
    function approve(address _spender, uint256 _value) public returns (bool success) {}
    function allowance(address _owner, address _spender) public constant returns (uint256 remaining) {}
    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}
contract StandardToken is Token {
    function transfer(address _to, uint256 _value) public returns (bool success) {
        if (balances[msg.sender] >= _value && _value > 0) {
            balances[msg.sender] -= _value;
            balances[_to] += _value;
            Transfer(msg.sender, _to, _value);
            return true;
        } else { return false; }
    }
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        if (balances[_from] >= _value && allowed[_from][msg.sender] >= _value && _value > 0) {
            balances[_to] += _value;
            balances[_from] -= _value;
            allowed[_from][msg.sender] -= _value;
            Transfer(_from, _to, _value);
            return true;
        } else { return false; }
    }
    function balanceOf(address _owner) public constant returns (uint256 balance) {
        return balances[_owner];
    }
    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        Approval(msg.sender, _spender, _value);
        return true;
    }
    function allowance(address _owner, address _spender) public constant returns (uint256 remaining) {
      return allowed[_owner][_spender];
    }
    mapping (address => uint256) balances;
    mapping (address => mapping (address => uint256)) allowed;
    uint256 public totalSupply;
}
contract SeneroToken is StandardToken {
    function () public {
        revert();
    }
    string public name;                   
    uint8 public decimals;                
    string public symbol;                 
    string public version = 'H1.0';       
    function SeneroToken(
        ) public {
        balances[0xAf468Bcc3B923C6d5588b9f3032a042Eb4ca4F60] = 259600; 
        balances[0x6ddD1c854EbAfFdb4bFf8CF8334871612A314285] = 178200; 
        balances[0x1B355DEB18bE4B579E4F1272cFc9958b2B0C5d49] = 209000; 
        balances[0xD0cc81a97737E02F9466eF51E4DDEf6D02D30a75] = 259600; 
        balances[0x96fa4CBb4869eFdFEC0C97f1178CA02da4CFe084] = 209000; 
        balances[0x143efe51524f53274fE120e69C761774a3e5d570] = 169400; 
        balances[0x0b1ae337a9d0f62c9026effcfdcee442e3ce31e6] = 178200; 
        balances[0xc62f738afab6fbce2cbc7f0fd274858cdc4a1448] = 158400; 
        balances[0xa510184cB3C83021253d7DD48FD28035ccEB4af4] = 270600; 
        balances[msg.sender] = 17800000;           
        totalSupply = 20000000;                    
        name = "Senero";                           
        decimals = 18;                             
        symbol = "SEN";                            
    }
    function approveAndCall(address _spender, uint256 _value, bytes _extraData) public returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        Approval(msg.sender, _spender, _value);
        if(!_spender.call(bytes4(bytes32(keccak256("receiveApproval(address,uint256,address,bytes)"))), msg.sender, _value, this, _extraData)) { revert(); }
        return true;
    }
}