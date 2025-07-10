// SPDX-License-Identifier: Unliscense
pragma solidity ^0.8.26;

/*

ATTENTION:  THIS COIN IS A RUGPULL!  BUT WHO WILL PULL THE RUG?

Telegram group: https://t.me/ +s0E9cyJ9cGwxY2Jh  (remove the space to join, just preventing botspam)
Website: http://rugpull.llms-are-retarded.lol/en-us/index.html

HOW IT WORKS:

1. I create the contract, granting me 20% of the supply and granting the contract 80%
2. I offer most of my holding as locked liquidity
3. People buy their share of the meme, to participate in the joke, and NOT as an investment
4. After 8 hours, the contract enables the 'rugpull' function - except absolutely anyone can execute the function (not just me)
5. Somebody calls 'rugpull', costing them some ethereum, but granting them 80% of the entire coin supply (which the contract was holding)

At this point, whoever rugpulled may sell their coins, use them to change the token name and post bulletins, or whatever else.  Up to them.

Then, after this point, the rugpull function becomes enabled any time that the contract gains 60% of the supply or more.  Why would this happen?
Well, people can change the name of the token or post bulletins, but this costs OpenRugPull2 tokens.  This way, this token is REUSABLE, that the
rug can be pulled HUNDREDS or THOUSANDS of times.

Now, there are two notes for the 'rugpull' function:
1. You must pay 0.01 eth to call the rugpull function.  This goes to me, as compensation for my work, and as extra inscentive to keep the liquidity pool up.
2. The rugpuller must hold some nonzero amount of OpenRugPull2.

This meme coin was made by the Nuclear Man.  Hopefully more to come, if I have time...
Also, fair warning, I am a serial memester of coin making.  If that concerns you, then DON'T BUY.

It should be obvious that buying this coin is very likely a bad financial decision.  This coin is not intended as an investment,
but as an elaborate joke.  I, the creator of the coin, disclaim all liability, and disclaim any intent of making this a good investment.
Furthermore, I may remove liquidity, sell, or otherwise leave at my discretion.

LEGAL DISCLAIMER:

**NO INVESTMENT VALUE – FOR ENTERTAINMENT PURPOSES ONLY**  

The OpenRugPull2 (ORP) token is explicitly designed as a joke and not as a financial investment. By acquiring ORP, you acknowledge and agree to the following:  

1. **This Token is a Gimmick** – ORP is structured as an experimental meme token where the primary function is an open and unpredictable "rugpull" event. This token has no inherent value, utility, or future promise.  
2. **Financial Risk** – Purchasing ORP carries a high probability of total financial loss. The creator makes no assurances regarding liquidity, price stability, or any potential return.  
3. **Decentralized Rugpull** – The smart contract includes a rugpull function, which can be executed by anyone who meets the conditions. The outcome of this function is not controlled by the creator, nor is it guaranteed to be fair or beneficial to any participant.  
4. **No Liability** – The creator of ORP disclaims all liability arising from the token, its smart contract, and any losses incurred by participants. This project is provided “as-is” without warranties of any kind.  
5. **Regulatory Compliance** – ORP is not a registered security, currency, or financial product. Participants are solely responsible for understanding and complying with applicable laws and regulations in their jurisdiction.  
6. **No Guarantees of Execution** – Due to Ethereum network congestion, gas fee volatility, or other unforeseen factors, the rugpull function may become impractical or impossible to execute.  

By interacting with the OpenRugPull contract, you affirm that you understand these terms and accept full responsibility for any consequences. If you do not agree, do not buy, sell, or engage with this token in any way.  
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

abstract contract StringUtils1 {
    // UTILITY FUNCTIONS

	function stripWhitespace(string memory input) public virtual pure returns (string memory);
	function _toLower(string memory str) public virtual pure returns (string memory);
	function clean(string memory str) public virtual pure returns (string memory);
	function contains(string memory what, string memory where) public virtual pure returns (bool);
	function isBadString(string memory input) public virtual pure returns (bool);
	function sqrt(uint64 x) public virtual pure returns (uint32);
}

contract OpenRugPull is ERC20Interface {

	string public symbol;
	string public  name;
	uint8 public decimals;
	uint64 public decimalFactor;
	StringUtils1 public libStrUtil;
	uint64 internal _totalSupply;
    uint64 public numCirculating;
    address payable public creator;

    address[] internal hodlers;

    uint public rugpullBlock;

	mapping(address => uint64) public balances;
	mapping(address => mapping(address => uint64)) public allowed;

	constructor() {
		symbol = "ORP2";
		name = "OpenRugPull V2 (COME PULL THE RUG!!!)";
		decimals = 8;
		decimalFactor = uint64(10**uint(decimals));
		_totalSupply = 0;
        creator = payable(msg.sender);
	}

	function initialize(address libAdr) public {
		require(address(libStrUtil) == address(0x0), "The contract has already been initialized.");
		require(address(libAdr) != address(0x0), "Bruh");

		libStrUtil = StringUtils1(libAdr);
        _totalSupply = 100000000 * decimalFactor;

        // Set up balances
        uint64 oneFifth = _totalSupply / 5;
        _transfer(msg.sender, address(0x0), oneFifth);
        _transfer(address(this), address(0x0), _totalSupply - oneFifth);

        // Set the countdown!
        rugpullBlock = block.number + uint(8 * 3600 + 600) / 13;
	}


    // THE BIG FUNNY

    function rugpull() public payable {
        // If someone uses the constructor hack here, that would be dumb, because they'd just be wasting gas.
        require(msg.sender.code.length == 0, "Contracts cannot call this function.");
        require(rugpullBlock <= block.number, "You can't rugpull yet, be patient!");
        require(balances[msg.sender] >= decimalFactor, "You must hold some OpenRugPull to rugpull!");
        require(balances[address(this)] >= (_totalSupply*3) / 5, "You can't rugpull yet, be patient!");

        uint requiredFee = 1 ether / 100;

        require(msg.value >= requiredFee, "Insufficient ETH sent for rugpull!");

        // Pay a fee to the creator
        creator.transfer(requiredFee);

        // Refund any extra eth
        payable(msg.sender).transfer(msg.value - requiredFee);

        // Transfer all OpenRugPull tokens from contract to msg.sender
        _transfer(msg.sender, address(this), balances[address(this)]);
    }

	event CurrencyNameChanged(address indexed sender, string newName);

	function renameContract(string calldata newName) public {
		uint64 cost = (_totalSupply / 20);
		require(balances[msg.sender] >= cost, "Insufficient balance to rename the contract");

		// Strip whitespace from the input
		string memory strippedName = libStrUtil.stripWhitespace(newName);

		// Ensure the stripped name is at least 4 characters long
		require(bytes(strippedName).length >= 4, "Name must be at least 4 characters long after stripping");
		require(bytes(strippedName).length <= 32, "Name must be at most 32 characters");

		// Deduct Lyra from the sender's balance
		// Note that this happens BEFORE the name is checked for bad content, so someone who attempts
		// to experiment with circumventing it might spend a lot more.
		// If you want to post stupid stuff then use the bulletin.
		_transfer(address(this), msg.sender, cost);

		// Check for retarded content
		require(!libStrUtil.isBadString(strippedName), "Name contains retard nonsense words.");

		// Update the contract name
		name = strippedName;

		emit CurrencyNameChanged(msg.sender, strippedName); // Emit an event for tracking changes
	}

	event BulletinMessagePosted(address indexed sender, string message);

	function postBulletinMessage(string calldata message) public {
		uint64 cost = (_totalSupply / 200);
		require(balances[msg.sender] >= cost, "Insufficient balance to post bulletin");

		// Deduct Lyra from the sender's balance
		_transfer(address(this), msg.sender, cost);

		emit BulletinMessagePosted(msg.sender, message); // Emit the event!
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
		_transfer(to, from, uint64(tokens));

		return true;
	}

	function _transfer(address to, address from, uint64 tokens) internal {
        require(to != address(0x0));

        if (tokens != 0 && to.code.length == 0 && to != address(0x0))
            numCirculating += tokens;
        if (from != address(0x0) && from.code.length == 0)
            numCirculating -= tokens;

        balances[to] += tokens;
        if (from != address(0x0))
    		balances[from] -= tokens;

		emit Transfer(from, to, uint(tokens));
	}

	function allowance(address tokenOwner, address spender) public override view returns (uint remaining) {
		return allowed[tokenOwner][spender];
	}

	// https://ethereum-magicians.org/t/eip-7054-gas-efficient-square-root-calculation-with-binary-search-approach/14539
	function sqrt(uint x) public pure returns (uint128) {
		if (x == 0) return 0;
		else{
			uint xx = x;
			uint r = 1;
			if (xx >= 0x100000000000000000000000000000000) { xx >>= 128; r <<= 64; }
			if (xx >= 0x10000000000000000) { xx >>= 64; r <<= 32; }
			if (xx >= 0x100000000) { xx >>= 32; r <<= 16; }
			if (xx >= 0x10000) { xx >>= 16; r <<= 8; }
			if (xx >= 0x100) { xx >>= 8; r <<= 4; }
			if (xx >= 0x10) { xx >>= 4; r <<= 2; }
			if (xx >= 0x8) { r <<= 1; }
			r = (r + x / r) >> 1;
			r = (r + x / r) >> 1;
			r = (r + x / r) >> 1;
			r = (r + x / r) >> 1;
			r = (r + x / r) >> 1;
			r = (r + x / r) >> 1;
			r = (r + x / r) >> 1;
			uint r1 = x / r;
			return uint128 (r < r1 ? r : r1);
		}
	}
}