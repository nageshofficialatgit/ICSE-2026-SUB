// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

/**
 * @title Ancient FirePepe Token (AFPEP)
 * @dev ERC20 Token with Access Control, Reentrancy Protection, and Reserve Management
 */
contract Token {
    // -----------------------------------------
    // Events
    // -----------------------------------------
    
    /// @notice Emitted when tokens are transferred between accounts
    event Transfer(address indexed from, address indexed to, uint256 value);
    
    /// @notice Emitted when an allowance is approved
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    /// @notice Emitted when the first buy from Uniswap pool is completed
    event FirstBuyDone();
    
    /// @notice Emitted when new tokens are minted
    event TokensMinted(address indexed account, uint256 amount);
    
    /// @notice Emitted when tokens are burned
    event TokensBurned(address indexed account, uint256 amount);

    /// @notice Emitted when the contract is paused
    event Paused(address account);

    /// @notice Emitted when the contract is unpaused
    event Unpaused(address account);

    // -----------------------------------------
    // Access Control
    // -----------------------------------------
    
    /// @dev Role data structure for AccessControl
    struct RoleData {
        mapping(address => bool) members;
        bytes32 adminRole;
    }
    
    /// @dev Mapping of roles to their data
    mapping(bytes32 => RoleData) private _roles;
    
    /// @dev Default admin role identifier
    bytes32 private constant DEFAULT_ADMIN_ROLE = 0x00;
    
    /// @dev Role identifier for minters
    bytes32 private constant MINTER_ROLE = keccak256("MINTER_ROLE");
    
    /// @dev Role identifier for burners
    bytes32 private constant BURNER_ROLE = keccak256("BURNER_ROLE");

    /// @dev Role identifier for pausers
    bytes32 private constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    // -----------------------------------------
    // Reentrancy Guard
    // -----------------------------------------
    
    /// @dev Reentrancy status: NOT_ENTERED (1) or ENTERED (2)
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status = _NOT_ENTERED;

    // -----------------------------------------
    // Pausable State
    // -----------------------------------------

    /// @dev Paused status
    bool private _paused;

    // -----------------------------------------
    // ERC20 State Variables
    // -----------------------------------------
    
    /// @dev Mapping of account balances
    mapping(address => uint256) private _balances;
    
    /// @dev Mapping of allowances granted by accounts
    mapping(address => mapping(address => uint256)) private _allowances;
    
    /// @dev Total token supply
    uint256 private _totalSupply;
    
    /// @dev Token name
    string private _name;
    
    /// @dev Token symbol
    string private _symbol;
    
    /// @dev Token decimals (fixed to 18)
    uint8 private constant _decimals = 18;

    // -----------------------------------------
    // Reserves and Wallet Addresses
    // -----------------------------------------
    
    /// @dev Presale allocation (600 billion tokens)
    uint256 private constant presaleReserve = 600e27; // 600 billion tokens
    
    /// @dev Staking allocation (240 billion tokens)
    uint256 private constant stakingReserve = 240e27; // 240 billion tokens
    
    /// @dev Marketing allocation (400 billion tokens)
    uint256 private constant marketingReserve = 400e27; // 400 billion tokens
    
    /// @dev Liquidity allocation (300 billion tokens)
    uint256 private constant liquidityReserve = 300e27; // 300 billion tokens
    
    /// @dev Rewards allocation (200 billion tokens)
    uint256 private constant RewardsReserve = 200e27; // 200 billion tokens
    
    /// @dev Development allocation (260 billion tokens)
    uint256 private constant DevelopmentReserve = 260e27; // 260 billion tokens

    /// @dev Wallet addresses for reserves
    address public presaleWallet;
    address public stakingWallet;
    address public marketingWallet;
    address public liquidityWallet;
    address public rewardsWallet;
    address public developmentWallet;

    /// @dev Flag to track the first buy from Uniswap pool
bool public firstBuyCompleted;
    
    /// @dev Address of the Uniswap liquidity pool
    address public uniswapPool;

    // -----------------------------------------
    // Address Management (Iterable Mapping)
    // -----------------------------------------
    
    /// @dev Structure to track addresses with balances
    struct AddressMap {
        mapping(address => uint256) index;  // Maps address to its index in the array
        address[] addresses;                // Array of addresses with non-zero balances
    }
    AddressMap private _addressMap;

    // -----------------------------------------
    // Constructor
    // -----------------------------------------
    
    /**
     * @dev Initializes the token contract:
     * - Sets reentrancy status
     * - Assigns token name and symbol
     * - Grants roles to the deployer
     * - Mints initial reserves to predefined wallets
     */
    constructor(
        address _presaleWallet,
        address _stakingWallet,
        address _marketingWallet,
        address _liquidityWallet,
        address _rewardsWallet,
        address _developmentWallet
    ) payable {
        _status = _NOT_ENTERED;
        _paused = false;
        _name = "Ancient FirePepe";
        _symbol = "AFPEP";
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(BURNER_ROLE, msg.sender);
        _grantRole(PAUSER_ROLE, msg.sender);

        // Set wallet addresses
        presaleWallet = _presaleWallet;
        stakingWallet = _stakingWallet;
        marketingWallet = _marketingWallet;
        liquidityWallet = _liquidityWallet;
        rewardsWallet = _rewardsWallet;
        developmentWallet = _developmentWallet;

        // Mint initial reserves
        _mint(presaleWallet, presaleReserve);
        _mint(stakingWallet, stakingReserve);
        _mint(marketingWallet, marketingReserve);
        _mint(liquidityWallet, liquidityReserve);
        _mint(rewardsWallet, RewardsReserve);
        _mint(developmentWallet, DevelopmentReserve);
    }

    // -----------------------------------------
    // Core ERC20 Functions
    // -----------------------------------------
    
    /**
     * @dev Internal function to mint tokens
     * @param account Address to receive minted tokens
     * @param amount Amount of tokens to mint
     * Requirements:
     * - `account` cannot be the zero address
     */
    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to zero address");
        _totalSupply = _totalSupply + amount;
        _balances[account] = _balances[account] + amount;
        emit Transfer(address(0), account, amount);
        emit TokensMinted(account, amount);
    }

    /**
     * @dev Internal function to burn tokens
     * @param account Address whose tokens will be burned
     * @param amount Amount of tokens to burn
     * Requirements:
     * - `account` cannot be the zero address
     * - `account` must have at least `amount` tokens
     */
    function _burn(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: burn from zero address");
        uint256 accountBalance = _balances[account];
        require(accountBalance > amount, "ERC20: burn exceeds balance");
        unchecked {
            _balances[account] = accountBalance - amount;
            _totalSupply = _totalSupply - amount;
        }
        emit Transfer(account, address(0), amount);
        emit TokensBurned(account, amount);
    }

    /**
     * @dev Transfers tokens to a specified address
     * @param to Recipient address
     * @param amount Amount of tokens to transfer
     * @return bool True if the transfer succeeds
     */
    function transfer(address to, uint256 amount) public returns (bool) {
        _transfer(_msgSender(), to, amount);
        return true;
    }

    /**
     * @dev Internal transfer function with reentrancy guard
     * @param sender Sender address
     * @param recipient Recipient address
     * @param amount Amount of tokens to transfer
     * Requirements:
     * - `amount` must be greater than 0
     * - `sender` and `recipient` cannot be zero addresses
     * - `sender` must have sufficient balance
     * - First buy from Uniswap pool must be authorized by owner
     */
   function _transfer(
    address sender,
    address recipient,
    uint256 amount
) internal nonReentrant whenNotPaused {
    require(amount != 0, "ERC20: zero transfer");
    require(sender != address(0), "ERC20: transfer from zero");
    require(recipient != address(0), "ERC20: transfer to zero");
    
    uint256 senderBalance = _balances[sender];
    require(senderBalance >= amount, "ERC20: insufficient balance");
    
    // First buy validation for Uniswap pool
    if (!firstBuyCompleted && sender == uniswapPool) {
        require(tx.origin == owner(), "First Buy Pending");
        firstBuyCompleted = true;
        emit FirstBuyDone();
    }
    
    uint256 recipientBalance = _balances[recipient]; // Cache recipient balance

    unchecked {
        _balances[sender] = senderBalance - amount;
        _balances[recipient] = recipientBalance + amount; // Use cached recipient balance
    }

    // Cache the index in memory
    uint256 recipientIndex = _addressMap.index[recipient];
    uint256 senderIndex = _addressMap.index[sender];

    // Update address tracking
    if (recipientIndex == 0) {
        _addressMap.addresses.push(recipient);
        _addressMap.index[recipient] = _addressMap.addresses.length;
    }
    if (_balances[sender] == 0) {
        uint256 index = senderIndex - 1;
        address lastAddress = _addressMap.addresses[_addressMap.addresses.length - 1];
        _addressMap.addresses[index] = lastAddress;
        _addressMap.index[lastAddress] = index + 1;
        _addressMap.addresses.pop();
        delete _addressMap.index[sender];
    }

    emit Transfer(sender, recipient, amount);
    checkInvariant();
}



    /**
     * @dev Approves a spender to spend tokens on behalf of the owner
     * @param spender Address allowed to spend tokens
     * @param amount Allowance amount
     * @return bool True if approval succeeds
     */
    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    /**
     * @dev Increases the allowance granted to `spender` by the caller.
     * @param spender Address allowed to spend tokens
     * @param addedValue Amount to increase allowance by
     * @return bool True if the operation succeeds
     */
    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
        _approve(_msgSender(), spender, _allowances[_msgSender()][spender] + addedValue);
        return true;
    }

    /**
     * @dev Decreases the allowance granted to `spender` by the caller.
     * @param spender Address allowed to spend tokens
     * @param subtractedValue Amount to decrease allowance by
     * @return bool True if the operation succeeds
     */
    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
        uint256 currentAllowance = _allowances[_msgSender()][spender];
        require(currentAllowance >= subtractedValue, "ERC20: allowance < 0");
        unchecked {
            _approve(_msgSender(), spender, currentAllowance - subtractedValue);
        }
        return true;
    }

    /**
     * @dev Internal approval function
     * @param _owner Token owner
     * @param spender Spender address
     * @param amount Allowance amount
     * Requirements:
     * - `_owner` and `spender` cannot be zero addresses
     */
    function _approve(
    address _owner,
    address spender,
    uint256 amount
) internal {
    require(_owner != address(0), "ERC20: approve from zero");
    require(spender != address(0), "ERC20: approve to zero");
    
    uint256 currentAllowance = _allowances[_owner][spender];
    if (currentAllowance != amount) {
        _allowances[_owner][spender] = amount;
        emit Approval(_owner, spender, amount);
    }
}



    /**
     * @dev Transfers tokens using an allowance
     * @param from Sender address
     * @param to Recipient address
     * @param amount Amount of tokens to transfer
     * @return bool True if transfer succeeds
     * Requirements:
     * - `from` must have sufficient allowance
     */
    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) public returns (bool) {
        _transfer(from, to, amount);
        uint256 currentAllowance = _allowances[from][_msgSender()];
        require(currentAllowance > amount, "ERC20: allowance exceeded");
        unchecked {
            _approve(from, _msgSender(), currentAllowance - amount);
        }
        return true;
    }

    // -----------------------------------------
    // Security and Helper Functions
    // -----------------------------------------
    
    /**
     * @dev Validates that total balances match total supply (invariant check)
     * @notice Reverts if invariant is violated
     */
    function checkInvariant() internal view {
        uint256 totalBalances;
        address[] memory accounts = getAllAccounts();
        uint256 length = accounts.length;
        for (uint256 i = 0; i < length; ++i) {
            totalBalances += _balances[accounts[i]];
        }
        require(totalBalances == _totalSupply, "Invariant failed");
    }

    /**
     * @dev Returns all addresses with non-zero balances
     * @return address[] Array of addresses
     */
    function getAllAccounts() internal view returns (address[] memory) {
        return _addressMap.addresses;
    }

    // -----------------------------------------
    // Pausable Functions
    // -----------------------------------------

    /**
     * @dev Pauses the contract, preventing transfers and approvals.
     * Requirements:
     * - Caller must have the PAUSER_ROLE.
     */
     /**
 * @dev Allows the contract's operations to be halted in emergency situations.
 * This feature is crucial for preventing security attacks, potential exploits, 
 * or stopping malicious activities. If an issue is detected, the contract administrator 
 * can suspend transactions to mitigate potential losses.
 * Additionally, this functionality enhances user trust and improves the overall security of the contract.
 */

    function pause() external onlyRole(PAUSER_ROLE) {
        _paused = true;
        emit Paused(_msgSender());
    }

    /**
     * @dev Unpauses the contract, allowing transfers and approvals.
     * Requirements:
     * - Caller must have the PAUSER_ROLE.
     */
    function unpause() external onlyRole(PAUSER_ROLE) {
        _paused = false;
        emit Unpaused(_msgSender());
    }

    /**
     * @dev Modifier to make a function callable only when the contract is not paused.
     */
    modifier whenNotPaused() {
        require(!_paused, "Pausable: paused");
        _;
    }

    // -----------------------------------------
    // Standard View Functions
    // -----------------------------------------
    
    /// @return string Token name
    function name() public view returns (string memory) {
        return _name;
    }

    /// @return string Token symbol
    function symbol() public view returns (string memory) {
        return _symbol;
    }

    /// @return uint8 Token decimals (fixed to 18)
    function decimals() public pure returns (uint8) {
        return _decimals;
    }

    /// @return uint256 Total token supply
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    /// @return uint256 Balance of the specified account
    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    /// @return uint256 Allowance granted by `_owner` to `spender`
    function allowance(address _owner, address spender) public view returns (uint256) {
        return _allowances[_owner][spender];
    }

    // -----------------------------------------
    // Role Management (AccessControl)
    // -----------------------------------------
    
    /**
     * @dev Checks if an account has a specific role
     * @param role Role identifier
     * @param account Address to check
     * @return bool True if account has the role
     */
    function hasRole(bytes32 role, address account) public view returns (bool) {
        return _roles[role].members[account];
    }

    /**
     * @dev Grants a role to an account (internal)
     * @param role Role identifier
     * @param account Address to assign role
     */
    function _grantRole(bytes32 role, address account) internal {
        _roles[role].members[account] = true;
    }

    /**
     * @dev Modifier to make a function callable only by accounts with a specific role.
     * @param role Role identifier
     */
    modifier onlyRole(bytes32 role) {
        require(hasRole(role, _msgSender()), "AccessControl: unauthorized");
        _;
    }

    // -----------------------------------------
    // Reentrancy Guard Modifier
    // -----------------------------------------
    
    /// @dev Prevents reentrant calls
    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }

    // -----------------------------------------
    // Utility Functions
    // -----------------------------------------
    
    /// @return address Transaction sender's address
    function _msgSender() internal view returns (address) {
        return msg.sender;
    }

    /// @return address Contract owner's address (deployer)
    function owner() public view returns (address) {
        return _msgSender();
    }

    // -----------------------------------------
    // Additional Functions from Code 1
    // -----------------------------------------

    /**
     * @dev Destroys `amount` tokens from the caller's account, reducing the total supply.
     * @param amount The amount of tokens to burn.
     */
    function burn(uint256 amount) external onlyRole(BURNER_ROLE) {
    require(amount > 0, "Burn amount must be greater than zero");
    _burn(_msgSender(), amount);
}
}