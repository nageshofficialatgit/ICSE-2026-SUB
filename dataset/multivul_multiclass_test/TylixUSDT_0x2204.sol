// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/**
 * @title TylixUSDT
 * @dev It manages TylixUSDT, an ERC-20-like token with controlled minting and blacklist security.
 */
contract TylixUSDT {
    string public name = "Tylix USDT";
    string public symbol = "USDT";
    uint256 public totalSupply;
    uint256 public maxSupply = 2000000 * 10 ** 18; // Set a maximum supply limit
    uint8 public decimals = 18;
    address public owner;
    bool private locked;
    bool public paused;
    uint256 public lastMintTime;
    uint256 public mintCooldown = 1 days; // Timelock for minting

    mapping(address => uint256) private balances;
    mapping(address => mapping(address => uint256)) private allowances;
    mapping(address => bool) private blacklist;
    mapping(address => uint256) private nonces;

    struct Log {
        string action;
        address account;
        uint256 amount;
        uint256 timestamp;
    }
    Log[] private logs;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Blacklisted(address indexed account, bool value);
    event Mint(address indexed minter, address indexed recipient, uint256 amount);
    event Burn(address indexed burner, uint256 amount);
    event LogAction(string action, address indexed account, uint256 timestamp);
    event Paused(address account);
    event Unpaused(address account);

    modifier onlyOwner() {
        require(msg.sender == owner, "Ownable: caller is not the owner");
        _;
    }

    modifier nonReentrant() {
        require(!locked, "ReentrancyGuard: reentrant call");
        locked = true;
        _;
        locked = false;
    }

    modifier whenNotPaused() {
        require(!paused, "Error: Contract is paused");
        _;
    }

    constructor() {
        owner = msg.sender;
        totalSupply = 1000000 * 10 ** decimals;
        balances[owner] = totalSupply;
    }

    function pause() external onlyOwner {
        paused = true;
        emit Paused(msg.sender);
    }

    function unpause() external onlyOwner {
        paused = false;
        emit Unpaused(msg.sender);
    }

    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }

    function transfer(address recipient, uint256 amount) public nonReentrant whenNotPaused returns (bool) {
        require(!blacklist[msg.sender], "Error: Sender is blacklisted");
        require(balances[msg.sender] >= amount, "Error: Insufficient balance");
        require(recipient != address(0), "Error: Invalid recipient address");
        
        balances[msg.sender] -= amount;
        balances[recipient] += amount;

        emit Transfer(msg.sender, recipient, amount);
        logAction("Transfer", msg.sender, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public whenNotPaused returns (bool) {
        require(spender != address(0), "Error: Invalid spender address");
        allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public nonReentrant whenNotPaused returns (bool) {
        require(!blacklist[sender], "Error: Sender is blacklisted");
        require(balances[sender] >= amount, "Error: Insufficient balance");
        require(allowances[sender][msg.sender] >= amount, "Error: Allowance exceeded");
        require(recipient != address(0), "Error: Invalid recipient address");

        allowances[sender][msg.sender] -= amount;
        balances[sender] -= amount;
        balances[recipient] += amount;

        emit Transfer(sender, recipient, amount);
        logAction("TransferFrom", sender, amount);
        return true;
    }

    function setBlacklist(address account, bool value) external onlyOwner {
        require(account != address(0), "Error: Invalid account");
        require(blacklist[account] != value, "Error: Account already in this state");
        blacklist[account] = value;
        emit Blacklisted(account, value);
        logAction("Blacklist", account, value ? 1 : 0);
    }

    function mint(address recipient, uint256 amount) external onlyOwner nonReentrant whenNotPaused {
        require(recipient != address(0), "Error: Invalid address");
        require(totalSupply + amount <= maxSupply, "Error: Max supply exceeded");
        require(block.timestamp >= lastMintTime + mintCooldown, "Error: Minting cooldown active");
        
        totalSupply += amount;
        balances[recipient] += amount;
        lastMintTime = block.timestamp;

        emit Mint(msg.sender, recipient, amount);
        emit Transfer(address(0), recipient, amount);
        logAction("Mint", recipient, amount);
    }

    function burn(uint256 amount) external nonReentrant whenNotPaused {
        require(balances[msg.sender] >= amount, "Error: Insufficient balance");
        balances[msg.sender] -= amount;
        totalSupply -= amount;

        emit Burn(msg.sender, amount);
        emit Transfer(msg.sender, address(0), amount);
        logAction("Burn", msg.sender, amount);
    }

    function getLogs() external view returns (Log[] memory) {
        return logs;
    }

    function logAction(string memory action, address account, uint256 amount) internal {
        logs.push(Log(action, account, amount, block.timestamp));
        emit LogAction(action, account, block.timestamp);
    }
}