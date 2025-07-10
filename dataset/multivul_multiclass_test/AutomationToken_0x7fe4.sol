// SPDX-License-Identifier: Unliscense
pragma solidity 0.8.26;


/*
 * AUTOMATION TOKEN
 *
 *  Discord server: discord.gg *slash* DYEcm *then* buy2v    (I need to do this to prevent spam, just add the parts of the link together)
 *  My CMC account: https://coinmarketcap.com/community/profile/Nuclear_Man_D/
 *  This contract also has a variable 'notes' which I can update to provide more recent information.
 *
 *  This token allows for automatic transaction execution on the blockchain.  Essentially, this allows you to execute arbitrary
 *  tasks later, without manually creating a transaction off-chain.
 *
 *  This token allows automated market makers, trading bots, and other on-chain automation requiring constant execution.
 *  With AUT token, **contracts can run their code completely autonomously, instead of relying on some external entity to trigger them.**
 *
 *
 * TL;DR: HOW CAN I MAKE MONEY ON THIS?
 *
 *  Due to the way that the token handles transactions, simply sending tokens will sometimes net you a fee in AUT tokens.  For
 *  example, if you buy or sell AUT, since you trigger a transfer, it is possible you will receive a few AUT as a fee for providing
 *  the transaction volume!
 *
 *  You may also find particularly expensive tasks queued in the TaskEngine, and execute them directly for a larger fee.  This can be
 *  done in batches to save on Ethereum transaction fees.
 *
 *  For more information, see "HOW DOES IT WORK?" below.
 *
 *
 * WILL THIS GO UP?
 *
 *  I legally cannot tell you that this will go up, and I legally cannot tell you that it is a security or offered with any promise of
 *  appreciation.  Legally this is a utility token, and nothing here should be taken as investment advice.  See legal disclosures below.
 *  That said...
 *
 *  Contracts using this token will need to buy the token to continue running.  If the contracts make money, then they will presumably
 *  continue buying in perpetuity...  Which is to say, there will always be buying pressure on the coin, if contracts adopt it.  Considering
 *  that I'm actually building this for my own AMM project, well, this token will have a customer from day one: me.
 *
 *  Also, there are incentives to trade the coin, since fees can actually be earned by doing this.
 *
 *  So, it *appears* that there could be incentives to use the coin.  That's all I'll say.  Again, not any investment advice.
 *
 * 
 * HOW DOES IT WORK?
 *
 *  This token is designed to work with a TaskEngine contract.  The two work together to match queued 'tasks' with real transactions,
 *  so that contract code can execute autonomously.  The token provides transaction volume, visibility, and a more fluid marketplace
 *  to determine how much tasks should cost.  The TaskEngine is responsible for organizing and queueing 'tasks', handing fees to
 *  accounts that provide transaction volume, determining what tasks execute when, and protecting the network from malicious tasks.
 *  Think of the TaskEngine as the operating system and the token as the power supply to run the computer.
 *
 *  The AutomationToken contract provides the AUT token, which is used to pay fees to transaction executors.  The token also calls the
 *  task engine from functions like approve, transfer, and transferFrom, in a process I call "transaction piggybacking".  The code is
 *  written to prevent this from increasing the gas costs to insane levels - the TaskEngine only handles up to one task per piggyback,
 *  and it institutes limits on how much gas a task is allowed to spend, to protect AUT owners from excessive transaction fees.  If all
 *  else fails, there are the functions approveFallback, transferFallback, and transferFromFallback, which do not piggyback, so you can
 *  still move your funds if something really does go wrong.  Whoever creates the transaction that executes a task earns the fee for that
 *  task, *even if* all the transaction executor did was call transfer or approve.  Great way to drop tokens to users and reward users
 *  for adoption.
 *
 *  The TaskEngine uses a FIFO to queue tasks.  Each time the task engine is ticked, it scans through some of the FIFO to see what
 *  task needs to be executed next.  If the task isn't ready to execute yet, then the task is sent to the back of the task queue.
 *  As transactions come in and the AutomationToken ticks the TaskEngine, it scans through the queue, executing tasks when they need
 *  to be executed.  For larger tasks, an external executor may select which tasks in the FIFO to execute, in an attempt to earn fees
 *  from the tasks.  Naturally, the executor will select the tasks with the highest paying fees first, allowing users to bid to have
 *  their tasks prioritized.
 *
 *  Each task is a combination of a Checker and Executor.  The Checker determines when the task should execute (after a particular time,
 *  when Eth reaches a certain price, at a random and unpredictable time, etc).  Once it is determined that the task should execute and it
 *  is ready to do so, the Executor is called, and the task is completed.
 *
 *  Read the source of the involved contracts to see the nitty-gritty details.  There are interfaces somewhere near the top of each contract,
 *  which you can use for interacting with these contracts and writing your own contract.  I do not recommend interacting with the TaskEngine
 *  directly, as it may be migrated in the future if I find a bug or it needs new features.  Instead, the AutomationToken contract has the
 *  important functions wrapped so that you can just interact with the contract without worrying about what TaskEngine manages your task.
 *
 *
 * WHAT'S IN IT FOR THE CREATOR?
 *
 *  I get 2% of all task fees from the TaskEngine, and I make a 1% fee on trades to the Uniswap liquidity pools (at least in the beginning,
 *  this is subject to change).  I will also hold 2% of the token supply.  That said, the biggest way I make money is actually by being a
 *  customer of the token itself.  In a few days, I will launch my AMM, which uses the token to manage Uniswap liquidity pools.  This is
 *  probably where most of my profits will come from at first.  If the token moons (which it could), then I could make money on that too.
 *
 *  I've done all the math for profitability and how much it costs to make the token, I don't need to make *that* much for it to be worth doing.
 *  The main thing I need is transaction volume, that's really why I built this.  If my code can execute when I'm sleeping, and without any
 *  oracle...  oh man, I can do a LOT with that.
 *
 *
 * WHO CREATED THIS?
 *
 *  Online I am known as Nuclear Man.  And yes, I do actually love all things nuclear, always have.  This is the first actual legit protocol
 *  I've launched, the rest were meme coins and a lot of experimentation.
 *
 *  Anyway, here's some info for reaching me and for getting updates on the token:
 *  CoinMarketCap account: https://coinmarketcap.com/community/profile/Nuclear_Man_D/
 *
 *  If/when I add a website I'll put the URL in the notes of this contract.  Read the contract on Etherscan or something to get the notes.
 *
 * LEGAL STUFF
 *
 *  This project, including the AutomationToken (AUT) and associated TaskEngine contract, is offered strictly as a utility token and on-chain
 *  automation tool. It is **not** a security, investment vehicle, or financial product. Nothing in this documentation, the codebase, or
 *  associated communication channels (including Discord and CoinMarketCap) constitutes financial advice, a recommendation to invest, or a
 *  solicitation to purchase any security or token.
 *
 *  By interacting with the AutomationToken, the TaskEngine, or any related smart contracts, you acknowledge and agree to the following:
 *   1. There is no guarantee of profit, return, appreciation in value, or even functionality. Smart contracts are inherently risky, may contain
 *      bugs, and may behave unpredictably due to network congestion, external contract interaction, or future protocol changes.
 *   2. You assume **full responsibility** for all actions taken with this token or protocol. This includes (but is not limited to) loss of funds,
 *      failed transactions, high gas fees, contract vulnerabilities, or misuse of the protocol. Always DYOR (Do Your Own Research).
 *   3. This system is decentralized. Once deployed, the contracts cannot be controlled or altered by the creator beyond what is built into the smart
 *      contracts themselves. No party guarantees uptime, access, or success of queued tasks.
 *   4. You are solely responsible for ensuring that your use of the token and interaction with the protocol complies with local laws, regulations,
 *      and tax obligations. The creator makes no warranties that this project complies with the laws of any jurisdiction.
 *   5. The software is provided "as is", without warranty of any kind - express or implied - including but not limited to merchantability, fitness
 *      for a particular purpose, and non-infringement.
 *   6. While the creator (Nuclear Man) is actively developing tools and systems that use the AutomationToken, this does not imply any guarantee of
 *      continued support, future development, or market activity. The community is free to adopt or fork the system at any time.
 *
 *  By participating in this project, you release the creator and all contributors from any and all liability arising from your use of the protocol,
 *  the AUT token, or any derivative thereof.  By interacting with the smart contract you are agreeing to these terms.
*/


abstract contract Context {
	function _msgSender() internal view virtual returns (address) {
		return msg.sender;
	}

	function _msgData() internal view virtual returns (bytes calldata) {
		return msg.data;
	}
}

interface IERC20 {
	function totalSupply() external view returns (uint256);

	function balanceOf(address account) external view returns (uint256);

	function transfer(address recipient, uint256 amount) external returns (bool);

	function decimals() external pure returns (uint8);

	function allowance(
		address owner,
		address spender
	) external view returns (uint256);

	function approve(address spender, uint256 amount) external returns (bool);

	function transferFrom(
		address sender,
		address recipient,
		uint256 amount
	) external returns (bool);

	event Transfer(address indexed from, address indexed to, uint256 value);
	event Approval(address indexed owner, address indexed spender, uint256 value);
}

abstract contract Ownable is Context {
	address private _owner;

	event OwnershipTransferred(
		address indexed previousOwner,
		address indexed newOwner
	);

	constructor() {
		_transferOwnership(_msgSender());
	}

	modifier onlyOwner() {
		_checkOwner();
		_;
	}

	function owner() public view virtual returns (address) {
		return _owner;
	}

	function _checkOwner() internal view virtual {
		require(owner() == _msgSender(), "Ownable: caller is not the owner");
	}

	function renounceOwnership() public virtual onlyOwner {
		_transferOwnership(address(0));
	}

	function transferOwnership(address newOwner) public virtual onlyOwner {
		require(newOwner != address(0), "Ownable: new owner is the zero address");
		_transferOwnership(newOwner);
	}

	function _transferOwnership(address newOwner) internal virtual {
		address oldOwner = _owner;
		_owner = newOwner;
		emit OwnershipTransferred(oldOwner, newOwner);
	}
}


abstract contract IAirdropManager {
	function acceptTokens(IERC20 token, uint256 amount) external virtual;
	function airdropTokens(IERC20 token, address[] calldata addresses, uint256 tokensPerAddress) external virtual;
	function rescueTokens(address tokenAdr) external virtual;
}

interface RunnableTask {
	function execute(uint256 handle) external;
}


interface RunnableChecker {
	function shouldExecute(uint256 handle) external view returns (bool);
}


interface IAutomationToken is IERC20 {
	function getCirculatingSupply() external view returns (uint256);

	function transferFallback(address recipient, uint256 amount) external returns (bool);
	function approveFallback(address spender, uint256 amount) external returns (bool);
	function transferFromFallback(address sender, address recipient, uint256 amount) external returns (bool);

	// Utilities
	function minBidFor(uint256 gasNeeded) external view returns (uint256);
	function marketTaskBid(uint256 gasNeeded) external view returns (uint256);
	function queueTask(RunnableTask task, RunnableChecker checker, uint256 handle, uint256 operationBid, uint256 gasNeeded, uint256 expiration) external returns (uint256);
}


interface ITaskEngine {

	// Ticking and Execution functions
	function piggybackTick() external;
	function tick(uint64 maxExecutions) external;
	function batchExecute(uint256[] calldata taskIndices) external;

	// Queueing functions
	function queueTask(RunnableTask task, RunnableChecker checker, uint256 handle, uint256 operationBid, uint256 gasNeeded, uint256 expiration) external returns (uint256);

	// Getting data about task bid rates
	function minBidFor(uint256 gasNeeded) external view returns (uint256);
	function marketTaskBid(uint256 gasNeeded) external view returns (uint256);

	// Fee collection functions
	function ownerCollectFees() external; 
}


contract AutomationToken is IAutomationToken, Ownable {
	address constant DEAD = 0x000000000000000000000000000000000000dEaD;
	address constant ZERO = 0x0000000000000000000000000000000000000000;

	string constant _name = "Automation Token";
	string constant _symbol = "AUT";
	uint8 constant _decimals = 18;

	uint256 _totalSupply = 5_000_000 * (10 ** _decimals);

	uint256 constant TOKENS_PER_AIRDROPPED_ADDRESS = 195 * (10 ** _decimals);
	uint256 constant TOKENS_FOR_AIRDROP = 3_900_000 * (10 ** _decimals);

	mapping(address => uint256) _balances;
	mapping(address => mapping(address => uint256)) _allowances;

	bool hasTaskEngine = false;
	IAirdropManager public airdropper;
	ITaskEngine public taskEngine;

	// Query this variable to get recent information about the token
	string public notes = "Read comments in contract source code";

	// CONSTRUCTOR AND TOKEN DISTRIBUTION

	constructor(address airdropperAddress) {
		airdropper = IAirdropManager(airdropperAddress);
		_balances[address(0x0)] = _totalSupply;
	}

	function distributeTokens(address poolManager) external onlyOwner {
		// Prepare for the airdrop (78% of the tokens)
		_transfer(address(this), address(0x0), TOKENS_FOR_AIRDROP);
		_allowances[address(this)][address(airdropper)] = TOKENS_FOR_AIRDROP;
		airdropper.acceptTokens(this, TOKENS_FOR_AIRDROP);

		// Send the contract creator 2% of the supply
		_transfer(owner(), address(0x0), _totalSupply / 50);
	
		// Send the liquidity pool manager the remaining 20%
		_transfer(poolManager, address(0x0), _totalSupply / 5);
	}


	// UTILITY AND CONTRACT INTERACTION FUNCTIONS (For convenience)

	function minBidFor(uint256 gasNeeded) external view returns (uint256) {
		return taskEngine.minBidFor(gasNeeded);
	}

	function marketTaskBid(uint256 gasNeeded) external view returns (uint256) {
		return taskEngine.marketTaskBid(gasNeeded);
	}

	function queueTask(RunnableTask task, RunnableChecker checker, uint256 handle, uint256 operationBid, uint256 gasNeeded, uint256 expiration) external returns (uint256) {
		return taskEngine.queueTask(task, checker, handle, operationBid, gasNeeded, expiration);
	}


	// ADMINISTRATION FUNCTIONS

	event AirdroppedTokensEvent(uint16 numAddresses);
	function airdrop(address[] calldata addresses) external onlyOwner {
		airdropper.airdropTokens(this, addresses, TOKENS_PER_AIRDROPPED_ADDRESS);
		emit AirdroppedTokensEvent(uint16(addresses.length));
	}

	function rescueAirdrop(address newAirdropper) external onlyOwner {
		airdropper.rescueTokens(address(this));

		airdropper = IAirdropManager(newAirdropper);
		_allowances[address(this)][newAirdropper] = balanceOf(address(this));
		airdropper.acceptTokens(this, balanceOf(address(this)));
	}

	event NotesUpdatedEvent(string text);
	function setNotes(string calldata newNotes) external onlyOwner {
		notes = newNotes;
		emit NotesUpdatedEvent(newNotes);
	}

	event TaskEngineSetEvent(address newTaskEngine);
	function setTaskEngine(address taskEngineAddress) external onlyOwner {
		require(taskEngineAddress != address(0), "Invalid address");

		taskEngine = ITaskEngine(taskEngineAddress);
		hasTaskEngine = true;

		// This just ensures that there is a valid function here, so that if there isn't, this transaction
		// reverts instead of causing a transaction lockup or other issues.
		// Technically a malicious TaskEngine can still be put here - this is mainly just to prevent accidents.
		taskEngine.piggybackTick();

		emit TaskEngineSetEvent(taskEngineAddress);
	}

	// In case eth gets stuck in the contract (that should never happen but whatever)
	function rescueEth() external onlyOwner {
		payable(owner()).transfer(address(this).balance);
	}

	// BORING IRC20 AND Owned FUNCTIONS

	function totalSupply() external view override returns (uint256) {
		return _totalSupply;
	}

	function decimals() external pure returns (uint8) {
		return _decimals;
	}

	function symbol() external pure returns (string memory) {
		return _symbol;
	}

	function name() external pure returns (string memory) {
		return _name;
	}

	function getOwner() external view returns (address) {
		return owner();
	}

	function balanceOf(address account) public view override returns (uint256) {
		return _balances[account];
	}

	function transfer(address to, uint tokens) public override returns (bool success) {
		success = transferFallback(to, tokens);

		// Do some work on top of the transfer
		if (hasTaskEngine) taskEngine.piggybackTick();

		return success;
	}

	// Just in case there is a problem with the task engine, so people can still move their funds
	function transferFallback(address to, uint tokens) public override returns (bool success) {
		require(to != address(0), "Invalid address");
		require(tokens <= _balances[msg.sender], "Insufficient funds");

		_transfer(to, msg.sender, tokens);

		return true;
	}

	function approve(address spender, uint tokens) public override returns (bool success) {
		success = approveFallback(spender, tokens);

		// Do some work on top of the approval
		if (hasTaskEngine) taskEngine.piggybackTick();

		return success;
	}

	function approveFallback(address spender, uint tokens) public override returns (bool success) {
		_allowances[msg.sender][spender] = tokens;
		emit Approval(msg.sender, spender, tokens);

		return true;
	}

	function allowance(address holder, address spender) external view override returns (uint256) {
		return _allowances[holder][spender];
	}

	function transferFrom(address from, address to, uint tokens) public override returns (bool success) {
		success = transferFromFallback(from, to, tokens);

		// Do some work on top of the approval
		if (hasTaskEngine) taskEngine.piggybackTick();

		return success;
	}

	function transferFromFallback(address from, address to, uint tokens) public override returns (bool success) {
		require(tokens <= _allowances[from][msg.sender], "Allowance exceeded");
		_allowances[from][msg.sender] = _allowances[from][msg.sender] - tokens;
		_transfer(to, from, tokens);

		return true;
	}

	function _transfer(address to, address from, uint256 tokens) internal {
		require(to != address(0x0), "Invalid address");
		require(_balances[from] >= tokens, "Insufficient funds");

		_balances[to] += tokens;
		_balances[from] -= tokens;

		emit Transfer(from, to, uint(tokens));
	}

	function approveMaxAmount(address spender) external returns (bool) {
		return approve(spender, type(uint256).max);
	}

	function getCirculatingSupply() public view returns (uint256) {
		return _totalSupply - (balanceOf(DEAD) + balanceOf(ZERO));
	}
}