pragma solidity ^0.4.11;
contract ForeignToken {
    function balanceOf(address _owner) constant returns (uint256);
    function transfer(address _to, uint256 _value) returns (bool);
}
contract EliteToken { 
    string public name;
    string public symbol;
    uint8 public decimals;
    address owner;
    mapping (address => uint256) public balanceOf;
    event Transfer(address indexed from, address indexed to, uint256 value);
    function EliteToken() {
        balanceOf[this] = 100;
        name = "EliteToken";     
        symbol = "ELT";
        owner = msg.sender;
        decimals = 0;
    }
    function transfer(address _to, uint256 _value) {
        if (balanceOf[msg.sender] < _value) throw;
        if (balanceOf[_to] + _value < balanceOf[_to]) throw;
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        Transfer(msg.sender, _to, _value);
    }
    function() payable {
        if (msg.value == 0) { return; }
        owner.transfer(msg.value);
        uint256 amount = msg.value / 1000000000000000000;  
        if (balanceOf[this] < amount) throw;               
        balanceOf[msg.sender] += amount;                   
        balanceOf[this] -= amount;                         
        Transfer(this, msg.sender, amount);                
    }
    function WithdrawForeign(address _tokenContract) returns (bool) {
        if (msg.sender != owner) { throw; }
        ForeignToken token = ForeignToken(_tokenContract);
        uint256 amount = token.balanceOf(address(this));
        return token.transfer(owner, amount);
    }
}