// SPDX-License-Identifier: Unliscense
pragma solidity ^0.8.26;

/*

WELCOME TO THE HONKLERPOT

*/

abstract contract ERC20Interface {
	function totalSupply() public virtual view returns (uint);
	function balanceOf(address tokenOwner) public virtual view returns (uint balance);
	function allowance(address tokenOwner, address spender) public virtual view returns (uint remaining);
	function transfer(address to, uint tokens) public virtual returns (bool success);
	function approve(address spender, uint tokens) public virtual returns (bool success);
	function transferFrom(address from, address to, uint tokens) public virtual returns (bool success);

	event Transfer(address indexed from, address indexed to, uint tokens);
	event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}

contract HONKLERPOT is ERC20Interface {

	string public symbol;
	string public  name;
	uint8 public decimals;
	uint64 public decimalFactor;
	uint64 internal _totalSupply;

    // I like to use the smallest type, so uint64 is plenty for what we're doing.  Why use up 4x as much memory for the same thing?
    // Maybe I'm too used to potato computers.
	mapping(address => uint64) public balances;
	mapping(address => mapping(address => uint64)) public allowed;

	constructor() {
		symbol = "HONK";
		name = "HONKLER POT";
		decimals = 8;
		decimalFactor = uint64(10**uint(decimals));
        _totalSupply = 100000 * decimalFactor;
        _transfer(msg.sender, address(0x0), _totalSupply);
	}


    // BULLETIN

    // Use this method to express your distain for crooks, corrupt politicians, and government-sponsored scams!!!!!
	event BulletinMessagePosted(address indexed sender, string message, uint64 spent);
	function postBulletinMessage(string calldata message) public {
        uint64 cost = _totalSupply / 100;
		require(balances[msg.sender] >= cost, "Insufficient balance to post bulletin");

		// BURN THE BLESSED TOKENS!
		_transfer(address(0x0), msg.sender, cost);

        // LET THE WORLD KNOW OF THIS USER'S MESSAGE
		emit BulletinMessagePosted(msg.sender, message, cost);
	}

	// THE ACTUAL METHODS

	function totalSupply() public override view returns (uint) {
		return _totalSupply;
	}

	function balanceOf(address tokenOwner) public override view returns (uint balance) {
		return balances[tokenOwner];
	}

	function transfer(address to, uint tokens) public override returns (bool success) {
		require(to!=address(0), "Invalid address");
		require(tokens<=balances[msg.sender], "Insufficient funds");

		_transfer(to, msg.sender, uint64(tokens));

		return true;
	}

	function approve(address spender, uint tokens) public override returns (bool success) {
		allowed[msg.sender][spender] = uint64(tokens);
		emit Approval(msg.sender, spender, tokens);
		return true;
	}

	function transferFrom(address from, address to, uint tokens) public override returns (bool success) {
		require(to != address(0), "Invalid address");
		require(tokens <= balances[from], "Insufficient funds");
		require(tokens <= allowed[from][msg.sender], "Allowance exceeded");
		allowed[from][msg.sender] = allowed[from][msg.sender] - uint64(tokens);
        // https://t.me/+e38vyQf2jUY1NTUx
		_transfer(to, from, uint64(tokens));

		return true;
	}

    // I use the order 'to, from' because I'm used to memcpy, strcpy, x86 assembly, etc.  It feels more natural to me.
	function _transfer(address to, address from, uint64 tokens) internal {
		require(to != address(this) || balances[address(this)] == 0);

        // Check if this is a token burn
        if (to != address(0x0))
            balances[to] += tokens;
        else 
            _totalSupply -= tokens;

        // Check if this is a token mint
        if (from != address(0x0))
    		balances[from] -= tokens;

		emit Transfer(from, to, uint(tokens));
	}

	function allowance(address tokenOwner, address spender) public override view returns (uint remaining) {
		return allowed[tokenOwner][spender];
	}
}