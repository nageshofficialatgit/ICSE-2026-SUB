// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

abstract contract ERC20 {
    string public name;
    string public symbol;
    uint8 public decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(string memory _name, string memory _symbol) {
        name = _name;
        symbol = _symbol;
    }

    function transfer(address to, uint256 value) public returns (bool) {
        require(balanceOf[msg.sender] >= value, "ERC20: transfer amount exceeds balance");
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) public returns (bool) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public returns (bool) {
        require(balanceOf[from] >= value, "ERC20: transfer amount exceeds balance");
        require(allowance[from][msg.sender] >= value, "ERC20: transfer amount exceeds allowance");
        allowance[from][msg.sender] -= value;
        balanceOf[from] -= value;
        balanceOf[to] += value;
        emit Transfer(from, to, value);
        return true;
    }

    function _mint(address to, uint256 value) internal {
        totalSupply += value;
        balanceOf[to] += value;
        emit Transfer(address(0), to, value);
    }

    function _burn(address from, uint256 value) internal {
        require(balanceOf[from] >= value, "ERC20: burn amount exceeds balance");
        balanceOf[from] -= value;
        totalSupply -= value;
        emit Transfer(from, address(0), value);
    }

    function _transfer(address from, address to, uint256 value) internal {
        require(balanceOf[from] >= value, "ERC20: transfer amount exceeds balance");
        balanceOf[from] -= value;
        balanceOf[to] += value;
        emit Transfer(from, to, value);
    }
}

/** @dev Ownable Logic **/
abstract contract Ownable {
    address public owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}

/** @dev Reentrancy Protection **/
abstract contract ReentrancyGuard {
    uint256 private _status;
    constructor() {
        _status = 1;
    }
    modifier nonReentrant() {
        require(_status != 2, "ReentrancyGuard: reentrant call");
        _status = 2;
        _;
        _status = 1;
    }
}

/** @dev Chainlink Aggregator Interface **/
interface AggregatorV3Interface {
    function latestRoundData() external view returns (
        uint80 roundId,
        int256 answer,
        uint256 startedAt,
        uint256 updatedAt,
        uint80 answeredInRound
    );
}

/** @title ReachToken - The 9D-RC Token **/
contract ReachToken is ERC20("Reach Token", "9D-RC"), Ownable, ReentrancyGuard {
    uint256 public constant TOTAL_SUPPLY = 18_000_000_000 * 1e18;
    uint256 public floorPrice = 27 * 1e18;
    uint256 public buybackReserve;
    uint256 public buybackAllocation = 50;
    uint256 public stakingAllocation = 30;
    address public buybackWallet;

    AggregatorV3Interface public priceFeed;

    mapping(address => uint256) public stakingBalance;
    mapping(address => uint256) public lastStakeTime;

    struct Proposal {
        uint256 newFloorPrice;
        uint256 voteCount;
        bool executed;
        address creator;
    }

    Proposal[] public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;

    event TokensBought(address indexed buyer, uint256 ethSpent, uint256 tokensReceived);
    event BuybackExecuted(uint256 amount, uint256 price);
    event TokensStaked(address indexed user, uint256 amount);
    event TokensUnstaked(address indexed user, uint256 amount);
    event ProposalCreated(uint256 proposalId, uint256 newFloorPrice);
    event VoteCast(address voter, uint256 proposalId);

    constructor(address _priceFeed, address _buybackWallet) {
        require(_priceFeed != address(0), "Invalid price feed");
        require(_buybackWallet != address(0), "Invalid wallet");

        priceFeed = AggregatorV3Interface(_priceFeed);
        buybackWallet = _buybackWallet;

        _mint(msg.sender, TOTAL_SUPPLY);
    }

    function getLatestPrice() public view returns (uint256) {
        (, int256 price, , ,) = priceFeed.latestRoundData();
        return uint256(price) * 1e10;
    }

    function buyTokens() public payable nonReentrant {
        require(msg.value > 0, "No ETH sent");

        uint256 currentPrice = getLatestPrice();
        if (currentPrice < floorPrice) currentPrice = floorPrice;

        uint256 tokensToBuy = (msg.value * 1e18) / currentPrice;
        _transfer(owner, msg.sender, tokensToBuy);

        uint256 contribution = (msg.value * buybackAllocation) / 100;
        buybackReserve += contribution;

        emit TokensBought(msg.sender, msg.value, tokensToBuy);
    }

    function executeBuyback() public onlyOwner nonReentrant {
        uint256 currentPrice = getLatestPrice();
        require(currentPrice < floorPrice, "Price too high");
        require(buybackReserve > 0, "No reserve");

        uint256 buyAmount = buybackReserve / currentPrice;
        buybackReserve -= buyAmount * currentPrice;
        _mint(address(this), buyAmount);

        emit BuybackExecuted(buyAmount, currentPrice);
    }

    function stakeTokens(uint256 amount) external nonReentrant {
        require(balanceOf[msg.sender] >= amount, "Not enough tokens");
        _transfer(msg.sender, address(this), amount);
        stakingBalance[msg.sender] += amount;
        lastStakeTime[msg.sender] = block.timestamp;
        emit TokensStaked(msg.sender, amount);
    }

    function unstakeTokens() external nonReentrant {
        require(stakingBalance[msg.sender] > 0, "Nothing staked");
        uint256 amount = stakingBalance[msg.sender];
        stakingBalance[msg.sender] = 0;
        _transfer(address(this), msg.sender, amount);
        emit TokensUnstaked(msg.sender, amount);
    }

    function createProposal(uint256 newPrice) external onlyOwner {
        proposals.push(Proposal(newPrice, 0, false, msg.sender));
        emit ProposalCreated(proposals.length - 1, newPrice);
    }

    function vote(uint256 proposalId) external {
        require(proposalId < proposals.length, "Invalid proposal");
        require(!hasVoted[proposalId][msg.sender], "Already voted");

        proposals[proposalId].voteCount++;
        hasVoted[proposalId][msg.sender] = true;
        emit VoteCast(msg.sender, proposalId);
    }

    function executeProposal(uint256 proposalId) external onlyOwner {
        require(proposalId < proposals.length, "Invalid proposal");
        Proposal storage p = proposals[proposalId];
        require(!p.executed, "Already executed");
        require(p.voteCount >= 10, "Not enough votes");

        floorPrice = p.newFloorPrice;
        p.executed = true;
    }
}