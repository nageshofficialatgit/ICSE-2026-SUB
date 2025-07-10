pragma solidity ^0.4.11;
contract SafeMath{
	function safeMul(uint a, uint b) internal returns (uint) {
		uint c = a * b;
		assert(a == 0 || c / a == b);
		return c;
	}
	function safeDiv(uint a, uint b) internal returns (uint) {
		assert(b > 0);
		uint c = a / b;
		assert(a == b * c + a % b);
		return c;
	}
	function safeSub(uint a, uint b) internal returns (uint) {
		assert(b <= a);
		return a - b;
	}
	function safeAdd(uint a, uint b) internal returns (uint) {
		uint c = a + b;
		assert(c >= a);
		return c;
	}
	function assert(bool assertion) internal {
		if (!assertion) {
			revert();
		}
	}
}
contract admined {
	address public admin;
	function admined(){
		admin = msg.sender;
	}
	modifier onlyAdmin(){
		require(msg.sender == admin);
		_;
	}
	function transferAdminship(address newAdmin) public onlyAdmin {
		admin = newAdmin;
	}
}
contract Token is SafeMath {
	mapping (address => uint256) public balanceOf;
	string public name = "MoralityAI";
	string public symbol = "Mo";
	uint8 public decimal = 18; 
	uint256 public totalSupply = 1000000000000000000000000;
	event Transfer(address indexed from, address indexed to, uint256 value);
	function Token(){
		balanceOf[msg.sender] = totalSupply;
	}
	function transfer(address _to, uint256 _value){
		require(balanceOf[msg.sender] >= _value);
		require(safeAdd(balanceOf[_to], _value) >= balanceOf[_to]);
		balanceOf[msg.sender] = safeSub(balanceOf[msg.sender], _value);
		balanceOf[_to] = safeAdd(balanceOf[_to], _value);
		Transfer(msg.sender, _to, _value);
	}
}
contract MoralityAI is admined, Token{
	function MoralityAI() Token(){
		admin = msg.sender;
		balanceOf[admin] = totalSupply;
	}
	function mintToken(address target, uint256 mintedAmount) public onlyAdmin{
		balanceOf[target] = safeAdd(balanceOf[target], mintedAmount);
		totalSupply = safeAdd(totalSupply, mintedAmount);
		Transfer(0, this, mintedAmount);
		Transfer(this, target, mintedAmount);
	}
	function transfer(address _to, uint256 _value) public{
		require(balanceOf[msg.sender] > 0);
		require(balanceOf[msg.sender] >= _value);
		require(safeAdd(balanceOf[_to], _value) >= balanceOf[_to]);
		balanceOf[msg.sender] = safeSub(balanceOf[msg.sender], _value);
		balanceOf[_to] = safeAdd(balanceOf[_to], _value);
		Transfer(msg.sender, _to, _value);
	}
}