// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

//----------------------------------------------------------------------------------
// External Interfaces
//----------------------------------------------------------------------------------

/**
 * @dev Interface specifying a function to retrieve a ratio used when converting
 * deposited USDC into this custom token.
 */
interface IQrz112Reader {
    // Returns the current ratio for conversion of USDC to this token.
    function getRatio() external view returns (uint256);
}

/**
 * @dev Minimal ERC20 interface with `transferFrom` used for USDC or other tokens
 * during the deposit process.
 */
interface IERC20 {
    // Transfers `amount` from `sender` to `recipient`, given sufficient allowance.
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    
    // Standard ERC20 approve method (required for the deposit modification).
    function approve(address spender, uint256 amount) external returns (bool);
}

/**
 * @dev Interface for contracts that want to handle logic immediately after
 * an approval is set via `approveAndCall`.
 */
interface IApproveAndCallReceiver {
    // Called by the token contract after an approval is set.
    function onApprove(address owner, uint256 amount, bytes calldata data) external returns (bool);
}

/**
 * @dev Interface for contracts that want to handle logic immediately after
 * a token transfer via `transferAndCall`.
 */
interface ITransferAndCallReceiver {
    // Called by the token contract after a transfer is executed.
    function onTransfer(address sender, uint256 amount, bytes calldata data) external returns (bool);
}

/**
 * @dev Interface for the Treasury contract to handle logic after a deposit.
 */
interface ITreasury {
    // Called by the token contract after a deposit has been made.
    function afterDeposit(
        address depositor, 
        uint256 usdcAmount, 
        uint256 depositCount, 
        uint256 timestamp
    ) external;
}

//----------------------------------------------------------------------------------
// HyperSolverToken Contract
//----------------------------------------------------------------------------------

/**
 * @title HyperSolverToken
  *
 * This contract is supported by HypeLoot.com
 * Stay connected with us:
 * - Website: https://hypeloot.com/
 * - X (formerly Twitter): https://x.com/HypeLootCom
 * - Telegram Channel: https://t.me/Hypelootcom
 * - Instagram: https://instagram.com/hypelootcom
 *
 * For platform support, please email: support[at]hypeloot.com
 * We are continuously expanding our activities.
 *
 * @dev HyperSolverToken (HPS) is an advanced ERC20-compatible token that extends the standard functionality with a wide range of enhanced features designed for improved security, scalability, and interoperability within decentralized financial ecosystems.
 *
 * Key features include:
 *  - A robust role-based access control system that assigns distinct Admin and Minter roles, ensuring secure management of token issuance, administrative privileges, and operational governance.
 *  - A bespoke deposit mechanism which utilizes an external ratio reader (IQrz112Reader) to dynamically calculate deposit conversion ratios, thereby supporting flexible and accurate token conversions in various economic scenarios.
 *  - Integration with a dedicated Treasury contract interface that manages post-deposit logic, enabling efficient fund allocation and streamlined financial operations after deposits are made.
 *  - Advanced tokenomics controls, featuring custom burn allowances and sophisticated burning procedures that allow for controlled reduction of the token supply to manage inflationary pressures.
 *  - Support for intricate transaction patterns including Approve-and-call and Transfer-and-call, which facilitate seamless and efficient interactions with other smart contracts and decentralized protocols.
 *  - A refined permit mechanism inspired by EIP-2612 that allows off-chain signature approvals, incorporating essential parameters such as the token's name, symbol, verifying contract address, and chain ID to ensure secure and transparent permissioning.
 *  - An internally implemented nonReentrant modifier that provides robust protection against reentrancy attacks, ensuring that critical functions execute safely in a mutually exclusive manner.
 */
contract HyperSolverToken {
    //----------------------------------------------------------------------------------
    // Reentrancy Protection
    //----------------------------------------------------------------------------------

    // Indicates whether the contract is currently locked against reentrancy.
    bool private _locked;

    /**
     * @dev Ensures that the function cannot be reentered while it is still executing.
     * If another nonReentrant function calls back into itself, it will fail.
     */
    modifier nonReentrant() {
        require(!_locked, "HyperSolverToken: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }

    //----------------------------------------------------------------------------------
    // Role Management Data and Events
    //----------------------------------------------------------------------------------

    // Mapping that indicates which addresses hold a particular role.
    // The outer key is the role's bytes32 identifier, the inner key is the address.
    mapping(bytes32 => mapping(address => bool)) private _hasRole;

    // Mapping that indicates whether a given address has ever held a particular role.
    mapping(bytes32 => mapping(address => bool)) private _roleHistory;

    // Emitted when an address gains a specific role (e.g. ADMIN_ROLE).
    event RoleGranted(bytes32 indexed role, address indexed account);

    // Emitted when an address loses a specific role (e.g. ADMIN_ROLE).
    event RoleRevoked(bytes32 indexed role, address indexed account);

    // keccak256("admin") role
    bytes32 public constant ADMIN_ROLE = keccak256("admin");

    // keccak256("minter") role
    bytes32 public constant MINTER_ROLE = keccak256("minter");

    /**
     * @dev Modifier that allows only addresses with the admin role to invoke the function.
     * This role manages minting, treasury settings, ratio readers, etc.
     */
    modifier onlyAdmin() {
        require(_hasRole[ADMIN_ROLE][msg.sender], "HyperSolverToken: caller is not an admin");
        _;
    }

    //----------------------------------------------------------------------------------
    // ERC20 Basic Storage
    //----------------------------------------------------------------------------------

    // Stores each address's balance of the token.
    mapping(address => uint256) private _balances;

    // Tracks how many tokens an owner allows a spender to manage on their behalf.
    mapping(address => mapping(address => uint256)) private _allowances;

    // A separate allowance mechanism for burning tokens. 
    // An owner can approve another address to burn tokens from the owner's balance.
    mapping(address => mapping(address => uint256)) private _burnAllowances;

    // Token name
    string public name;

    // Token symbol
    string public symbol;

    // Token decimals
    uint8 public decimals;

    // Total supply of the token, updated upon minting/burning.
    uint256 public totalSupply;

    //----------------------------------------------------------------------------------
    // Additional Addresses and Configuration
    //----------------------------------------------------------------------------------

    // Address that will receive all USDC deposited into this contract.
    address public treasury;

    // The address of a contract that provides the current ratio for
    // converting deposited USDC into HyperSolverToken. 
    address public qrz112Reader;

    // Mapping for the denominator used in token minting calculations.
    // This value serves as the divisor in both deposit and previewDeposit functions.
    uint256 public denominator;

    // A simple flag to ensure the contract can be initialized only once.
    bool private _initialized;

    //----------------------------------------------------------------------------------
    // Deposit Counters
    //----------------------------------------------------------------------------------

    // Tracks the number of deposit operations performed by each user.
    // Incremented every time the user calls `deposit`.
    mapping(address => uint256) private _depositCounts;

    //----------------------------------------------------------------------------------
    // Custom Permit (EIP-2612-like) Storage
    //----------------------------------------------------------------------------------

    // Each address has a nonce used during permit signatures. Nonces increment with
    // each successful `permit` call, preventing replay attacks.
    mapping(address => uint256) public nonces;

    // A custom typehash that includes the token name, token symbol,
    // verifying contract address, chain ID, and the usual Permit fields.
    //
    // The final structure being hashed is:
    // keccak256(
    //   "Permit(string name,string symbol,address verifyingContract,uint256 chainId,address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)"
    // )
    bytes32 public constant PERMIT_TYPEHASH = keccak256(
        "Permit(string name,string symbol,address verifyingContract,uint256 chainId,address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)"
    );

    // Although we are using a custom PERMIT_TYPEHASH, we still keep a domain separator
    // in EIP-712 style (name, version, chainId, verifyingContract).
    bytes32 public DOMAIN_SEPARATOR;

    //----------------------------------------------------------------------------------
    // ERC20 Events
    //----------------------------------------------------------------------------------

    // Standard ERC20 event emitted when `value` tokens move from one address to another.
    event Transfer(address indexed from, address indexed to, uint256 value);

    // Standard ERC20 event emitted when the allowance of `spender` for an `owner` is set.
    event Approval(address indexed owner, address indexed spender, uint256 value);

    //----------------------------------------------------------------------------------
    // Custom Events
    //----------------------------------------------------------------------------------

    // Emitted when `transferAndCall` completes successfully.
    event TransferAndCallExecuted(address indexed from, address indexed to, uint256 amount, bytes data);

    // Emitted when `approveAndCall` completes successfully.
    event ApproveAndCallExecuted(address indexed owner, address indexed spender, uint256 amount, bytes data);

    // Event emitted when the denominator value is updated.
    event DenominatorUpdated(uint256 newDenominator);

    // Custom event for burn.
    event Burned(address indexed burner, uint256 amount);

    // Custom event for approveBurn.
    event BurnApproval(address indexed owner, address indexed spender, uint256 amount);

    // Custom event for burnFrom.
    event BurnedFrom(address indexed operator, address indexed from, uint256 amount);

    // Custom event emitted on deposit with a deposit count.
    event DepositMade(address indexed user, uint256 usdcAmount, uint256 depositCount, uint256 timestamp);

    //----------------------------------------------------------------------------------
    // Initialization
    //----------------------------------------------------------------------------------

    /**
     * @notice Initializes the contract with critical parameters and roles.
     * It must be called exactly once, otherwise the `require(!_initialized)` check will fail.
     *
     * @param _admin The address that will receive both ADMIN_ROLE and MINTER_ROLE.
     * @param _treasury The address where all deposited USDC is sent.
     * @param _qrz112Reader The address of a contract implementing IQrz112Reader, providing a ratio for deposits.
     */
    function initialize(
        address _admin,
        address _treasury,
        address _qrz112Reader
    ) external {
        require(!_initialized, "HyperSolverToken: already initialized");
        _initialized = true;

        // Set the token name, symbol, and decimals.
        name = "HyperSolver";
        symbol = "HPS";
        decimals = 18;

        // Grant roles to the specified admin address.
        _grantRole(ADMIN_ROLE, _admin);
        _grantRole(MINTER_ROLE, _admin);

        // Configure treasury and ratio reader.
        treasury = _treasury;
        qrz112Reader = _qrz112Reader;

        // Mint an initial supply of 100 million tokens (with 18 decimals) to the admin address.
        uint256 initialSupply = 100_000_000 * (10 ** decimals);
        _mint(_admin, initialSupply);

        // Set up the EIP-712 domain separator.
        DOMAIN_SEPARATOR = keccak256(
            abi.encode(
                keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
                keccak256(bytes(name)),      // token name
                keccak256(bytes("1")),       // version
                block.chainid,
                address(this)
            )
        );
    }

    //----------------------------------------------------------------------------------
    // Role Management
    //----------------------------------------------------------------------------------

    /**
     * @notice Grants a specific role (e.g., ADMIN_ROLE, MINTER_ROLE) to a chosen account.
     * @dev Only an address with the ADMIN_ROLE may call this function.
     */
    function grantRole(bytes32 role, address account) external onlyAdmin {
        _grantRole(role, account);
    }

    /**
     * @notice Revokes a specific role from a chosen account.
     * @dev Only an address with the ADMIN_ROLE may call this function.
     */
    function revokeRole(bytes32 role, address account) external onlyAdmin {
        _revokeRole(role, account);
    }

    /**
     * @notice Checks if a given address currently holds a specific role.
     */
    function hasRole(bytes32 role, address account) external view returns (bool) {
        return _hasRole[role][account];
    }

    /**
     * @notice Checks if a given address ever held a specific role at any point in the past.
     */
    function hadRole(bytes32 role, address account) external view returns (bool) {
        return _roleHistory[role][account];
    }

    //----------------------------------------------------------------------------------
    // Token Accounting & Allowances
    //----------------------------------------------------------------------------------

    /**
     * @notice Returns the number of deposit operations performed by the specified user.
     * @param user The address of the user.
     * @return count The number of deposits for the given user.
     */
    function getDepositCount(address user) external view returns (uint256 count) {
        return _depositCounts[user];
    }

    /**
     * @notice Returns the token balance of a given address.
     * @param account The address of the account.
     * @return balance The token balance.
     */
    function balanceOf(address account) external view returns (uint256 balance) {
        return _balances[account];
    }

    /**
     * @notice Returns the remaining number of tokens that the spender is allowed to spend on behalf of the owner.
     * @param owner The address of the token owner.
     * @param spender The address of the spender.
     * @return remaining The remaining token allowance.
     */
    function allowance(address owner, address spender) external view returns (uint256 remaining) {
        return _allowances[owner][spender];
    }

    /**
     * @notice Returns the burn allowance that an owner has granted to a burner.
     * @param owner The address of the token owner.
     * @param burner The address allowed to burn tokens from the owner's balance.
     * @return burnAllowance The burn allowance amount.
     */
    function getBurnAllowance(address owner, address burner) external view returns (uint256 burnAllowance) {
        return _burnAllowances[owner][burner];
    }

    //----------------------------------------------------------------------------------
    // USDC Deposit Logic
    //----------------------------------------------------------------------------------

    /**
     * @notice Allows a user to deposit a specified amount of USDC to the treasury address.
     * The function initiates by securely transferring the USDC from the caller to this contract,
     * ensuring that the deposited funds are received prior to processing any further logic. Following
     * the transfer, the depositor's count of deposits is incremented to maintain an accurate record of
     * deposit occurrences. The contract then approves the treasury to spend the transferred USDC, and
     * calls the treasury's `afterDeposit` function to execute additional post-deposit processing, which
     * may include fund allocation or further internal bookkeeping.
     *
     * The number of HyperSolverTokens to be minted is subsequently calculated based on the deposited
     * USDC amount, adjusted by an external ratio provided by the `qrz112Reader`. Importantly, instead of
     * using a hardcoded division by 10000, the calculation divides by a configurable `denominator`
     * value, which can only be set by the contract owner, thereby allowing flexible adjustment of token
     * minting dynamics over time.
     *
     * @param usdcAmount The amount of USDC (expressed in USDC decimals) that the user intends to deposit.
     */
    function deposit(uint256 usdcAmount) external nonReentrant {
        require(usdcAmount > 0, "HyperSolverToken: deposit amount must be > 0");
        require(treasury != address(0), "HyperSolverToken: treasury not set");
        require(qrz112Reader != address(0), "HyperSolverToken: ratio reader not set");

        // Transfer the specified USDC amount from the caller to this contract.
        IERC20(getUsdcAddress()).transferFrom(msg.sender, address(this), usdcAmount);

        // Increase the depositor's deposit count to track the number of deposits made.
        _depositCounts[msg.sender]++;
        uint256 currentDepositCount = _depositCounts[msg.sender];

        // Approve the treasury to spend the transferred USDC amount from this contract.
        IERC20(getUsdcAddress()).approve(treasury, usdcAmount);

        // Call the treasury's afterDeposit function with the depositor's address, deposited amount,
        // current deposit count, and the current block timestamp for further processing.
        ITreasury(treasury).afterDeposit(
            msg.sender,
            usdcAmount,
            currentDepositCount,
            block.timestamp
        );

        // Calculate the number of HyperSolverTokens to mint.
        // The calculation multiplies the USDC amount (normalized to 18 decimals) by the external ratio,
        // then divides by the configurable `denominator` to determine the minting amount.
        uint256 mintAmount = (usdcAmount * (10 ** 12) * IQrz112Reader(qrz112Reader).getRatio()) / denominator;

        // Mint the calculated HyperSolverTokens to the caller.
        _mint(msg.sender, mintAmount);

        // Emit an event to record the deposit, including the depositor's address, the USDC amount deposited,
        // the current deposit count, and the timestamp of the deposit.
        emit DepositMade(msg.sender, usdcAmount, currentDepositCount, block.timestamp);
    }

    /**
     * @notice Updates the denominator value used for token minting calculations.
     * This function is restricted to admin usage.
     *
     * @param newDenom The new denominator value to be set.
     */
    function setDenominator(uint256 newDenom) external onlyAdmin {
        require(newDenom != denominator, "New denominator must be different from the current one");
        denominator = newDenom;
        emit DenominatorUpdated(newDenom);
    }

    /**
     * @notice Provides a read-only preview of the HyperSolverTokens that would be minted if a deposit were made.
     * This function enables users to simulate the minting outcome by inputting a prospective USDC deposit amount.
     * The preview calculation converts the USDC amount to an 18-decimal format, multiplies it by the current
     * external ratio obtained from the `qrz112Reader`, and then divides by a configurable `denominator` (settable
     * only by the contract owner) rather than a fixed value. This design ensures flexibility in the minting process
     * and allows adjustments to the token issuance dynamics as needed.
     *
     * @param usdcAmount The amount of USDC (in USDC decimals) being considered for deposit.
     * @return The estimated amount of HyperSolverTokens that would be minted, based on the current ratio and the
     *         owner-configurable denominator.
     */
    function previewDeposit(uint256 usdcAmount) external view returns (uint256) {
        if (usdcAmount == 0 || qrz112Reader == address(0)) {
            return 0;
        }

        uint256 currentRatio = IQrz112Reader(qrz112Reader).getRatio();
        uint256 usdcIn18 = usdcAmount * (10 ** 12);

        return (usdcIn18 * currentRatio) / denominator;
    }

    /**
     * @dev Helper function that returns the USDC contract address utilized in the deposit process.
     * This function centralizes the reference to the USDC token, ensuring that the same address is
     * consistently used throughout the contract for all USDC-related operations.
     */
    function getUsdcAddress() public pure returns (address) {
        return 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48;
    }

    //----------------------------------------------------------------------------------
    // Minting (Minter Role)
    //----------------------------------------------------------------------------------

    /**
     * @notice Allows addresses with the MINTER_ROLE to mint new tokens to a specified address.
     * @param to The recipient of the newly minted tokens.
     * @param amount The quantity of tokens (18 decimals) to be minted.
     */
    function mint(address to, uint256 amount) external {
        require(_hasRole[MINTER_ROLE][msg.sender], "HyperSolverToken: caller is not a minter");
        _mint(to, amount);
    }

    //----------------------------------------------------------------------------------
    // Standard ERC20-Like Functions
    //----------------------------------------------------------------------------------

    /**
     * @notice Transfers `amount` tokens from the caller's address to `recipient`.
     * @param recipient The address that will receive the tokens.
     * @param amount The number of tokens to be transferred.
     * @return True if the transfer succeeds, otherwise reverts.
     */
    function transfer(address recipient, uint256 amount) public returns (bool) {
        require(recipient != address(0), "HyperSolverToken: transfer to zero address");
        require(_balances[msg.sender] >= amount, "HyperSolverToken: transfer amount exceeds balance");

        _balances[msg.sender] -= amount;
        _balances[recipient] += amount;

        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    /**
     * @notice Sets `amount` as the allowance of `spender` over the caller's tokens.
     * @param spender The address allowed to transfer the tokens from the caller's balance.
     * @param amount The number of tokens `spender` may spend.
     * @return True if the approval succeeds.
     */
    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    /**
     * @notice Transfers `amount` tokens from `sender` to `recipient`, using the allowance
     * mechanism. The caller must have an allowance for `sender` to execute this transfer.
     * @param sender The address from which tokens are transferred.
     * @param recipient The address receiving the tokens.
     * @param amount The number of tokens to transfer.
     * @return True if the transfer succeeds, otherwise reverts.
     */
    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(sender != address(0), "HyperSolverToken: transfer from zero address");
        require(recipient != address(0), "HyperSolverToken: transfer to zero address");
        require(_balances[sender] >= amount, "HyperSolverToken: transfer amount exceeds balance");
        require(_allowances[sender][msg.sender] >= amount, "HyperSolverToken: transfer amount exceeds allowance");

        _allowances[sender][msg.sender] -= amount;
        _balances[sender] -= amount;
        _balances[recipient] += amount;

        emit Transfer(sender, recipient, amount);
        return true;
    }

    /**
     * @notice Increases the allowance of `spender` by `addedValue`.
     * @param spender The address whose allowance is being increased.
     * @param addedValue The number of tokens by which the allowance is increased.
     * @return True if the increase succeeds.
     */
    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
        uint256 currentAllowance = _allowances[msg.sender][spender];
        _approve(msg.sender, spender, currentAllowance + addedValue);
        return true;
    }

    /**
     * @notice Decreases the allowance of `spender` by `subtractedValue`.
     * @param spender The address whose allowance is being decreased.
     * @param subtractedValue The number of tokens by which the allowance is decreased.
     * @return True if the decrease succeeds, otherwise reverts if it goes below zero.
     */
    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
        uint256 currentAllowance = _allowances[msg.sender][spender];
        require(currentAllowance >= subtractedValue, "HyperSolverToken: decreased allowance below zero");

        _approve(msg.sender, spender, currentAllowance - subtractedValue);
        return true;
    }

    //----------------------------------------------------------------------------------
    // Burning Functions
    //----------------------------------------------------------------------------------

    /**
     * @notice Burns `amount` of the caller's tokens, reducing the total supply.
     * @param amount The number of tokens to burn.
     * @return Returns true upon successful burn.
     */
    function burn(uint256 amount) external returns (bool) {
        require(_balances[msg.sender] >= amount, "HyperSolverToken: burn amount exceeds balance");
        _burn(msg.sender, amount);
        emit Burned(msg.sender, amount);
        return true;
    }

    /**
     * @notice Approves another address (`spender`) to burn `amount` tokens
     * from the caller's balance.
     * @param spender The address that gains the right to burn tokens.
     * @param amount The number of tokens that `spender` may burn.
     * @return Returns true upon successful approval.
     */
    function approveBurn(address spender, uint256 amount) external returns (bool) {
        _burnAllowances[msg.sender][spender] = amount;
        emit BurnApproval(msg.sender, spender, amount);
        return true;
    }

    /**
     * @notice Burns tokens from a given address (`from`), assuming the caller
     * has sufficient burn allowance set by `from`.
     * @param from The address whose tokens will be burned.
     * @param amount The number of tokens to burn.
     * @return Returns true upon successful burn.
     */
    function burnFrom(address from, uint256 amount) external returns (bool) {
        require(_burnAllowances[from][msg.sender] >= amount, "HyperSolverToken: burn amount exceeds allowance");
        require(_balances[from] >= amount, "HyperSolverToken: burn amount exceeds balance");

        _burnAllowances[from][msg.sender] -= amount;
        _burn(from, amount);
        emit BurnedFrom(msg.sender, from, amount);
        return true;
    }

    //----------------------------------------------------------------------------------
    // ApproveAndCall / TransferAndCall
    //----------------------------------------------------------------------------------

    /**
     * @notice Allows the caller to set an allowance for `spender` and then calls `onApprove`
     * on the spender's contract address, enabling an atomic operation of approval + callback.
     * @param spender The contract address that will receive the allowance.
     * @param amount The allowance amount to be set.
     * @param data Additional data sent to the contract's `onApprove` function.
     * @return True if the entire sequence of actions succeeds.
     */
    function approveAndCall(
        address spender,
        uint256 amount,
        bytes calldata data
    )
        external
        nonReentrant
        returns (bool)
    {
        _approve(msg.sender, spender, amount);

        bool ok = IApproveAndCallReceiver(spender).onApprove(msg.sender, amount, data);
        require(ok, "HyperSolverToken: onApprove callback failed");

        emit ApproveAndCallExecuted(msg.sender, spender, amount, data);
        return true;
    }

    /**
     * @notice Transfers `amount` tokens to `to`, then calls `onTransfer` on `to`,
     * allowing an atomic transfer + callback in a single transaction.
     * @param to The contract address receiving the tokens.
     * @param amount The number of tokens to transfer.
     * @param data Additional data sent to the contract's `onTransfer` function.
     * @return True if the entire sequence of actions succeeds.
     */
    function transferAndCall(
        address to,
        uint256 amount,
        bytes calldata data
    )
        external
        nonReentrant
        returns (bool)
    {
        require(transfer(to, amount), "HyperSolverToken: transfer failed");

        bool ok = ITransferAndCallReceiver(to).onTransfer(msg.sender, amount, data);
        require(ok, "HyperSolverToken: onTransfer callback failed");

        emit TransferAndCallExecuted(msg.sender, to, amount, data);
        return true;
    }

    //----------------------------------------------------------------------------------
    // Custom Permit (EIP-2612-like)
    //----------------------------------------------------------------------------------

    /**
     * @notice Allows a user to set an allowance for a `spender` by presenting a signed message
     * instead of calling `approve` directly (often called "gasless approval").
     * @param owner The address that owns the tokens and signed the permit message.
     * @param spender The address approved to spend tokens on behalf of `owner`.
     * @param value The amount of tokens that `spender` is allowed to spend.
     * @param deadline A timestamp by which the permit must be used.
     * @param v Recovery byte of the signature.
     * @param r Half of the ECDSA signature pair.
     * @param s The other half of the ECDSA signature pair.
     */
    function permit(
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external nonReentrant {
        require(block.timestamp <= deadline, "HyperSolverToken: permit expired");

        address recoveredAddress = _recoverPermitSigner(owner, spender, value, deadline, v, r, s);
        require(recoveredAddress == owner && recoveredAddress != address(0), "HyperSolverToken: invalid permit signature");

        _approve(owner, spender, value);
    }

    /**
     * @dev Recovers the permit signer’s address using the EIP-712 signature standard.
     *
     * Steps performed:
     * 1. Nonce Handling: Retrieves and increments the owner’s nonce to prevent replay attacks.
     * 2. Hashing: Constructs a struct hash by encoding the token's name, symbol, contract address, chain ID, owner, spender, value, nonce, and deadline.
     * 3. Digest Creation: Forms an EIP-712 digest by combining the domain separator with the struct hash.
     * 4. Signature Recovery: Uses `ecrecover` with the digest and signature parameters (v, r, s) to recover the signer's address.
     *
     * Returns the address that signed the permit, which should match the owner if the signature is valid.
     */
    function _recoverPermitSigner(
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) internal returns (address) {
        uint256 currentNonce = nonces[owner]++;

        // Construct the struct hash with the customized fields (name, symbol, verifyingContract, chainId).
        bytes32 structHash = keccak256(
            abi.encode(
                PERMIT_TYPEHASH,
                keccak256(bytes(name)),
                keccak256(bytes(symbol)),
                address(this),
                block.chainid,
                owner,
                spender,
                value,
                currentNonce,
                deadline
            )
        );

        // EIP-712 digest
        bytes32 digest = keccak256(
            abi.encodePacked("\x19\x01", DOMAIN_SEPARATOR, structHash)
        );

        // Recover the address from the signature
        return ecrecover(digest, v, r, s);
    }

    //----------------------------------------------------------------------------------
    // Internal Functions
    //----------------------------------------------------------------------------------

    /**
     * @dev Creates `amount` new tokens and assigns them to `account`, increasing
     * the total supply. Emits a `Transfer` event from `address(0)` to `account`.
     */
    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "HyperSolverToken: mint to zero address");
        totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    /**
     * @dev Destroys `amount` tokens from `account`, reducing the total supply.
     * Emits a `Transfer` event from `account` to `address(0)`.
     */
    function _burn(address account, uint256 amount) internal {
        _balances[account] -= amount;
        totalSupply -= amount;
        emit Transfer(account, address(0), amount);
    }

    /**
     * @dev Updates `owner`'s allowance for `spender` to `amount`. Emits an `Approval` event.
     */
    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0), "HyperSolverToken: approve from zero address");
        require(spender != address(0), "HyperSolverToken: approve to zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    /**
     * @dev Grants the specified `role` to `account` if it does not already have it.
     * Also marks that `account` has had the role in `_roleHistory`.
     */
    function _grantRole(bytes32 role, address account) internal {
        if (!_hasRole[role][account]) {
            _hasRole[role][account] = true;
            _roleHistory[role][account] = true;
            emit RoleGranted(role, account);
        }
    }

    /**
     * @dev Revokes the specified `role` from `account` if it currently has it.
     */
    function _revokeRole(bytes32 role, address account) internal {
        if (_hasRole[role][account]) {
            _hasRole[role][account] = false;
            emit RoleRevoked(role, account);
        }
    }
}