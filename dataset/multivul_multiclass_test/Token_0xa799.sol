// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

/**
 * @title Ancient Firepepe Token Contract
 * @dev ERC20 implementation with simplified ownership and reentrancy protection
 * @notice This version focuses on audit compliance and simplicity
 */
contract Token {
    // -----------------------------------------
    // Events
    // -----------------------------------------
    
    /// @dev Emitted when tokens are transferred between accounts
    event Transfer(address indexed from, address indexed to, uint256 value);
    
    /// @dev Emitted when an allowance is approved
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    /// @dev Emitted when the first buy from Uniswap pool is completed
    event FirstBuyDone();
    
    /// @dev Emitted when ownership is transferred
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
    /// @dev Emitted when ownership transfer is initiated
    event OwnershipTransferInitiated(address indexed previousOwner, address indexed newOwner);
    
    /// @dev Emitted when ownership transfer is cancelled
    event OwnershipTransferCancelled(address indexed previousOwner, address indexed cancelledOwner);

    // -----------------------------------------
    // Reentrancy Guard
    // -----------------------------------------
    
    /// @dev Reentrancy status: NOT_ENTERED (1) or ENTERED (2)
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    
    /// @dev Current reentrancy status
    uint256 private _status = _NOT_ENTERED;

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
    string private constant _name = "Ancient Firepepe";
    
    /// @dev Token symbol
    string private constant _symbol = "AFPEP";
    
    /// @dev Token decimals (fixed to 18)
    uint8 private constant _decimals = 18;

    // -----------------------------------------
    // Token Reserves (Constants)
    // -----------------------------------------
    
    /// @dev Presale allocation reserve amount
    uint256 public constant presaleReserve = 600_000_000_000 * (10 ** 18);
    
    /// @dev Staking allocation reserve amount
    uint256 public constant stakingReserve = 460_000_000_000 * (10 ** 18);
    
    /// @dev Marketing allocation reserve amount
    uint256 public constant marketingReserve = 360_000_000_000 * (10 ** 18);
    
    /// @dev Liquidity allocation reserve amount
    uint256 public constant liquidityReserve = 300_000_000_000 * (10 ** 18);
    
    /// @dev Rewards allocation reserve amount
    uint256 public constant RewardsReserve = 140_000_000_000 * (10 ** 18);
    
    /// @dev Development allocation reserve amount
    uint256 public constant DevelopmentReserve = 140_000_000_000 * (10 ** 18);

    // -----------------------------------------
    // Wallet Addresses
    // -----------------------------------------
    
    /// @dev Address for presale reserve
    address public presaleWallet;
    
    /// @dev Address for staking reserve
    address public stakingWallet;
    
    /// @dev Address for marketing reserve
    address public marketingWallet;
    
    /// @dev Address for liquidity reserve
    address public liquidityWallet;
    
    /// @dev Address for rewards reserve
    address public rewardsWallet;
    
    /// @dev Address for development reserve
    address public developmentWallet;
    
    /// @dev Flag indicating if first buy from Uniswap is completed
    bool public firstBuyCompleted;
    
    /// @dev Address of the Uniswap liquidity pool
    address public uniswapPool;
    
    /// @dev Current owner of the contract
    address public owner;
    
    /// @dev Pending owner for two-step ownership transfer
    address private _pendingOwner;

    // -----------------------------------------
    // Modifiers
    // -----------------------------------------
    
    /// @dev Throws if called by any account other than the owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;
    }

    // -----------------------------------------
    // Constructor
    // -----------------------------------------
    
    /**
     * @dev Initializes the token contract
     * @param _presaleWallet Address for presale reserve
     * @param _stakingWallet Address for staking reserve
     * @param _marketingWallet Address for marketing reserve
     * @param _liquidityWallet Address for liquidity reserve
     * @param _rewardsWallet Address for rewards reserve
     * @param _developmentWallet Address for development reserve
     * @notice All reserve wallets must be non-zero addresses
     */
    constructor(
        address _presaleWallet,
        address _stakingWallet,
        address _marketingWallet,
        address _liquidityWallet,
        address _rewardsWallet,
        address _developmentWallet
    ) {
        require(_presaleWallet != address(0), "Presale wallet cannot be zero");
        require(_stakingWallet != address(0), "Staking wallet cannot be zero");
        require(_marketingWallet != address(0), "Marketing wallet cannot be zero");
        require(_liquidityWallet != address(0), "Liquidity wallet cannot be zero");
        require(_rewardsWallet != address(0), "Rewards wallet cannot be zero");
        require(_developmentWallet != address(0), "Development wallet cannot be zero");

        // Set contract owner
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);

        // Set reserve wallet addresses
        presaleWallet = _presaleWallet;
        stakingWallet = _stakingWallet;
        marketingWallet = _marketingWallet;
        liquidityWallet = _liquidityWallet;
        rewardsWallet = _rewardsWallet;
        developmentWallet = _developmentWallet;

        // Mint initial reserves to respective wallets
        _mint(presaleWallet, presaleReserve);
        _mint(stakingWallet, stakingReserve);
        _mint(marketingWallet, marketingReserve);
        _mint(liquidityWallet, liquidityReserve);
        _mint(rewardsWallet, RewardsReserve);
        _mint(developmentWallet, DevelopmentReserve);
        
        // Calculate total supply
        _totalSupply = presaleReserve + stakingReserve + marketingReserve + 
                      liquidityReserve + RewardsReserve + DevelopmentReserve;
    }

    // -----------------------------------------
    // ERC20 Core Functions
    // -----------------------------------------
    
    /**
     * @dev Internal function to mint tokens (only used in constructor)
     * @param account Address to receive minted tokens
     * @param amount Amount of tokens to mint
     * Requirements:
     * - `account` cannot be the zero address
     */
    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to zero address");
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    /**
     * @dev Transfers tokens to a specified address
     * @param to Recipient address
     * @param amount Amount of tokens to transfer
     * @return bool True if the transfer succeeds
     */
    function transfer(address to, uint256 amount) public returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }

    /**
     * @dev Internal transfer function with security checks
     * @param sender Sender address
     * @param recipient Recipient address
     * @param amount Amount of tokens to transfer
     */
    function _transfer(
        address sender,
        address recipient,
        uint256 amount
    ) internal nonReentrant {
        require(amount > 0, "ERC20: transfer amount zero");
        require(sender != address(0), "ERC20: transfer from zero address");
        require(recipient != address(0), "ERC20: transfer to zero address");
        
        uint256 senderBalance = _balances[sender];
        require(senderBalance >= amount, "ERC20: insufficient balance");
        
        if (!firstBuyCompleted && sender == uniswapPool) {
            require(msg.sender == owner, "First Buy Pending");
            firstBuyCompleted = true;
            emit FirstBuyDone();
        }
        
        unchecked {
            _balances[sender] = senderBalance - amount;
        }
        _balances[recipient] += amount;
        
        emit Transfer(sender, recipient, amount);
    }

    /**
     * @dev Approves a spender to spend tokens on behalf of the caller
     * @param spender Address allowed to spend tokens
     * @param amount Allowance amount
     * @return bool True if approval succeeds
     */
    function approve(address spender, uint256 amount) public returns (bool) {
        require(spender != address(0), "ERC20: approve to zero address");
        require(amount == 0 || _allowances[msg.sender][spender] == 0, "ERC20: race condition detected");
        _approve(msg.sender, spender, amount);
        return true;
    }

    /**
     * @dev Internal approval function
     * @param _owner Token owner 
     * @param spender Spender address
     * @param amount Allowance amount
     */
    function _approve(
        address _owner,
        address spender,
        uint256 amount
    ) internal {
        require(_owner != address(0), "ERC20: approve from zero address");
        require(spender != address(0), "ERC20: approve to zero address");
        _allowances[_owner][spender] = amount;
        emit Approval(_owner, spender, amount);
    }

    /**
     * @dev Transfers tokens using an allowance
     * @param from Sender address
     * @param to Recipient address
     * @param amount Amount of tokens to transfer
     * @return bool True if transfer succeeds
     */
    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) public returns (bool) {
        _transfer(from, to, amount);
        uint256 currentAllowance = _allowances[from][msg.sender];
        require(currentAllowance >= amount, "ERC20: allowance exceeded");
        unchecked {
            _approve(from, msg.sender, currentAllowance - amount);
        }
        return true;
    }

    // -----------------------------------------
    // View Functions
    // -----------------------------------------
    
    /**
     * @dev Returns the name of the token
     * @return string Token name
     */
    function name() public pure returns (string memory) {
        return _name;
    }

    /**
     * @dev Returns the symbol of the token
     * @return string Token symbol
     */
    function symbol() public pure returns (string memory) {
        return _symbol;
    }

    /**
     * @dev Returns the decimals places of the token
     * @return uint8 Token decimals (fixed to 18)
     */
    function decimals() public pure returns (uint8) {
        return _decimals;
    }

    /**
     * @dev Returns the total token supply
     * @return uint256 Total supply
     */
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    /**
     * @dev Returns the balance of the specified account
     * @param account Address to query
     * @return uint256 Account balance
     */
    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    /**
     * @dev Returns the allowance granted by `_owner` to `spender`
     * @param _owner Owner address 
     * @param spender Spender address
     * @return uint256 Allowance amount
     */
    function allowance(address _owner, address spender) public view returns (uint256) {
        return _allowances[_owner][spender];
    }

    /**
     * @dev Returns the pending owner address
     * @return address Pending owner address
     */
    function pendingOwner() public view returns (address) {
        return _pendingOwner;
    }

    // -----------------------------------------
    // Reentrancy Protection
    // -----------------------------------------
    
    /**
     * @dev Prevents reentrant calls
     */
    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }

    // -----------------------------------------
    // Ownership Management (Fixed)
    // -----------------------------------------
    
    /**
     * @dev Initiates ownership transfer to new account
     * @param newOwner Address of new pending owner
     * Requirements:
     * - Caller must be current owner
     * - `newOwner` cannot be the zero address
     * - `newOwner` must be an externally owned account
     */
    function initiateOwnershipTransfer(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner is zero address");
        uint256 codeSize;
        assembly { codeSize := extcodesize(newOwner) }
        require(codeSize == 0, "Cannot transfer to contract");
        _pendingOwner = newOwner;
        emit OwnershipTransferInitiated(owner, newOwner);
    }

    /**
     * @dev Completes ownership transfer
     * Requirements:
     * - Caller must be pending owner
     */
    function completeOwnershipTransfer() public {
        require(msg.sender == _pendingOwner, "Caller is not pending owner");
        emit OwnershipTransferred(owner, _pendingOwner);
        owner = _pendingOwner;
        _pendingOwner = address(0);
    }

    /**
     * @dev Cancels pending ownership transfer
     * Requirements:
     * - Caller must be current owner
     */
    function cancelOwnershipTransfer() public onlyOwner {
        require(_pendingOwner != address(0), "No pending transfer");
        address cancelledOwner = _pendingOwner;
        _pendingOwner = address(0);
        emit OwnershipTransferCancelled(owner, cancelledOwner);
    }

    /**
     * @dev Sets the Uniswap pool address
     * @param _uniswapPool Address of the Uniswap pool
     * Requirements:
     * - Caller must be current owner
     */
    function setUniswapPool(address _uniswapPool) public onlyOwner {
        uniswapPool = _uniswapPool;
    }

    /**
     * @dev Returns the current owner of the contract
     * @return address Owner address
     */
    function getOwner() public view returns (address) {
        return owner;
    }
}