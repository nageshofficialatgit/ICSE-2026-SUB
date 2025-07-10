// SPDX-License-Identifier: MIT
// File: @openzeppelin/contracts@5.0.0/utils/Context.sol


// OpenZeppelin Contracts (last updated v5.0.0) (utils/Context.sol)

pragma solidity ^0.8.20;

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

// File: @openzeppelin/contracts@5.0.0/access/Ownable.sol


// OpenZeppelin Contracts (last updated v5.0.0) (access/Ownable.sol)

pragma solidity ^0.8.20;


/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * The initial owner is set to the address provided by the deployer. This can
 * later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
    address private _owner;

    /**
     * @dev The caller account is not authorized to perform an operation.
     */
    error OwnableUnauthorizedAccount(address account);

    /**
     * @dev The owner is not a valid owner account. (eg. `address(0)`)
     */
    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Initializes the contract setting the address provided by the deployer as the initial owner.
     */
    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if the sender is not the owner.
     */
    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby disabling any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

// File: @openzeppelin/contracts@5.0.0/token/ERC20/IERC20.sol


// OpenZeppelin Contracts (last updated v5.0.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC20 standard as defined in the EIP.
 */
interface IERC20 {
    /**
     * @dev Emitted when `value` tokens are moved from one account (`from`) to
     * another (`to`).
     *
     * Note that `value` may be zero.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Emitted when the allowance of a `spender` for an `owner` is set by
     * a call to {approve}. `value` is the new allowance.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);

    /**
     * @dev Returns the value of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the value of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves a `value` amount of tokens from the caller's account to `to`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address to, uint256 value) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets a `value` amount of tokens as the allowance of `spender` over the
     * caller's tokens.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * IMPORTANT: Beware that changing an allowance with this method brings the risk
     * that someone may use both the old and the new allowance by unfortunate
     * transaction ordering. One possible solution to mitigate this race
     * condition is to first reduce the spender's allowance to 0 and set the
     * desired value afterwards:
     * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
     *
     * Emits an {Approval} event.
     */
    function approve(address spender, uint256 value) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to` using the
     * allowance mechanism. `value` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

// File: @openzeppelin/contracts@5.0.0/token/ERC20/extensions/IERC20Metadata.sol


// OpenZeppelin Contracts (last updated v5.0.0) (token/ERC20/extensions/IERC20Metadata.sol)

pragma solidity ^0.8.20;


/**
 * @dev Interface for the optional metadata functions from the ERC20 standard.
 */
interface IERC20Metadata is IERC20 {
    /**
     * @dev Returns the name of the token.
     */
    function name() external view returns (string memory);

    /**
     * @dev Returns the symbol of the token.
     */
    function symbol() external view returns (string memory);

    /**
     * @dev Returns the decimals places of the token.
     */
    function decimals() external view returns (uint8);
}

// File: @openzeppelin/contracts@5.0.0/interfaces/draft-IERC6093.sol


// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/draft-IERC6093.sol)
pragma solidity ^0.8.20;

/**
 * @dev Standard ERC20 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC20 tokens.
 */
interface IERC20Errors {
    /**
     * @dev Indicates an error related to the current `balance` of a `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     * @param balance Current balance for the interacting account.
     * @param needed Minimum amount required to perform a transfer.
     */
    error ERC20InsufficientBalance(address sender, uint256 balance, uint256 needed);

    /**
     * @dev Indicates a failure with the token `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     */
    error ERC20InvalidSender(address sender);

    /**
     * @dev Indicates a failure with the token `receiver`. Used in transfers.
     * @param receiver Address to which tokens are being transferred.
     */
    error ERC20InvalidReceiver(address receiver);

    /**
     * @dev Indicates a failure with the `spender`’s `allowance`. Used in transfers.
     * @param spender Address that may be allowed to operate on tokens without being their owner.
     * @param allowance Amount of tokens a `spender` is allowed to operate with.
     * @param needed Minimum amount required to perform a transfer.
     */
    error ERC20InsufficientAllowance(address spender, uint256 allowance, uint256 needed);

    /**
     * @dev Indicates a failure with the `approver` of a token to be approved. Used in approvals.
     * @param approver Address initiating an approval operation.
     */
    error ERC20InvalidApprover(address approver);

    /**
     * @dev Indicates a failure with the `spender` to be approved. Used in approvals.
     * @param spender Address that may be allowed to operate on tokens without being their owner.
     */
    error ERC20InvalidSpender(address spender);
}

/**
 * @dev Standard ERC721 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC721 tokens.
 */
interface IERC721Errors {
    /**
     * @dev Indicates that an address can't be an owner. For example, `address(0)` is a forbidden owner in EIP-20.
     * Used in balance queries.
     * @param owner Address of the current owner of a token.
     */
    error ERC721InvalidOwner(address owner);

    /**
     * @dev Indicates a `tokenId` whose `owner` is the zero address.
     * @param tokenId Identifier number of a token.
     */
    error ERC721NonexistentToken(uint256 tokenId);

    /**
     * @dev Indicates an error related to the ownership over a particular token. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     * @param tokenId Identifier number of a token.
     * @param owner Address of the current owner of a token.
     */
    error ERC721IncorrectOwner(address sender, uint256 tokenId, address owner);

    /**
     * @dev Indicates a failure with the token `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     */
    error ERC721InvalidSender(address sender);

    /**
     * @dev Indicates a failure with the token `receiver`. Used in transfers.
     * @param receiver Address to which tokens are being transferred.
     */
    error ERC721InvalidReceiver(address receiver);

    /**
     * @dev Indicates a failure with the `operator`’s approval. Used in transfers.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     * @param tokenId Identifier number of a token.
     */
    error ERC721InsufficientApproval(address operator, uint256 tokenId);

    /**
     * @dev Indicates a failure with the `approver` of a token to be approved. Used in approvals.
     * @param approver Address initiating an approval operation.
     */
    error ERC721InvalidApprover(address approver);

    /**
     * @dev Indicates a failure with the `operator` to be approved. Used in approvals.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     */
    error ERC721InvalidOperator(address operator);
}

/**
 * @dev Standard ERC1155 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC1155 tokens.
 */
interface IERC1155Errors {
    /**
     * @dev Indicates an error related to the current `balance` of a `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     * @param balance Current balance for the interacting account.
     * @param needed Minimum amount required to perform a transfer.
     * @param tokenId Identifier number of a token.
     */
    error ERC1155InsufficientBalance(address sender, uint256 balance, uint256 needed, uint256 tokenId);

    /**
     * @dev Indicates a failure with the token `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     */
    error ERC1155InvalidSender(address sender);

    /**
     * @dev Indicates a failure with the token `receiver`. Used in transfers.
     * @param receiver Address to which tokens are being transferred.
     */
    error ERC1155InvalidReceiver(address receiver);

    /**
     * @dev Indicates a failure with the `operator`’s approval. Used in transfers.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     * @param owner Address of the current owner of a token.
     */
    error ERC1155MissingApprovalForAll(address operator, address owner);

    /**
     * @dev Indicates a failure with the `approver` of a token to be approved. Used in approvals.
     * @param approver Address initiating an approval operation.
     */
    error ERC1155InvalidApprover(address approver);

    /**
     * @dev Indicates a failure with the `operator` to be approved. Used in approvals.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     */
    error ERC1155InvalidOperator(address operator);

    /**
     * @dev Indicates an array length mismatch between ids and values in a safeBatchTransferFrom operation.
     * Used in batch transfers.
     * @param idsLength Length of the array of token identifiers
     * @param valuesLength Length of the array of token amounts
     */
    error ERC1155InvalidArrayLength(uint256 idsLength, uint256 valuesLength);
}

// File: @openzeppelin/contracts@5.0.0/token/ERC20/ERC20.sol


// OpenZeppelin Contracts (last updated v5.0.0) (token/ERC20/ERC20.sol)

pragma solidity ^0.8.20;





/**
 * @dev Implementation of the {IERC20} interface.
 *
 * This implementation is agnostic to the way tokens are created. This means
 * that a supply mechanism has to be added in a derived contract using {_mint}.
 *
 * TIP: For a detailed writeup see our guide
 * https://forum.openzeppelin.com/t/how-to-implement-erc20-supply-mechanisms/226[How
 * to implement supply mechanisms].
 *
 * The default value of {decimals} is 18. To change this, you should override
 * this function so it returns a different value.
 *
 * We have followed general OpenZeppelin Contracts guidelines: functions revert
 * instead returning `false` on failure. This behavior is nonetheless
 * conventional and does not conflict with the expectations of ERC20
 * applications.
 *
 * Additionally, an {Approval} event is emitted on calls to {transferFrom}.
 * This allows applications to reconstruct the allowance for all accounts just
 * by listening to said events. Other implementations of the EIP may not emit
 * these events, as it isn't required by the specification.
 */
abstract contract ERC20 is Context, IERC20, IERC20Metadata, IERC20Errors {
    mapping(address account => uint256) private _balances;

    mapping(address account => mapping(address spender => uint256)) private _allowances;

    uint256 private _totalSupply;

    string private _name;
    string private _symbol;

    /**
     * @dev Sets the values for {name} and {symbol}.
     *
     * All two of these values are immutable: they can only be set once during
     * construction.
     */
    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
    }

    /**
     * @dev Returns the name of the token.
     */
    function name() public view virtual returns (string memory) {
        return _name;
    }

    /**
     * @dev Returns the symbol of the token, usually a shorter version of the
     * name.
     */
    function symbol() public view virtual returns (string memory) {
        return _symbol;
    }

    /**
     * @dev Returns the number of decimals used to get its user representation.
     * For example, if `decimals` equals `2`, a balance of `505` tokens should
     * be displayed to a user as `5.05` (`505 / 10 ** 2`).
     *
     * Tokens usually opt for a value of 18, imitating the relationship between
     * Ether and Wei. This is the default value returned by this function, unless
     * it's overridden.
     *
     * NOTE: This information is only used for _display_ purposes: it in
     * no way affects any of the arithmetic of the contract, including
     * {IERC20-balanceOf} and {IERC20-transfer}.
     */
    function decimals() public view virtual returns (uint8) {
        return 18;
    }

    /**
     * @dev See {IERC20-totalSupply}.
     */
    function totalSupply() public view virtual returns (uint256) {
        return _totalSupply;
    }

    /**
     * @dev See {IERC20-balanceOf}.
     */
    function balanceOf(address account) public view virtual returns (uint256) {
        return _balances[account];
    }

    /**
     * @dev See {IERC20-transfer}.
     *
     * Requirements:
     *
     * - `to` cannot be the zero address.
     * - the caller must have a balance of at least `value`.
     */
    function transfer(address to, uint256 value) public virtual returns (bool) {
        address owner = _msgSender();
        _transfer(owner, to, value);
        return true;
    }

    /**
     * @dev See {IERC20-allowance}.
     */
    function allowance(address owner, address spender) public view virtual returns (uint256) {
        return _allowances[owner][spender];
    }

    /**
     * @dev See {IERC20-approve}.
     *
     * NOTE: If `value` is the maximum `uint256`, the allowance is not updated on
     * `transferFrom`. This is semantically equivalent to an infinite approval.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     */
    function approve(address spender, uint256 value) public virtual returns (bool) {
        address owner = _msgSender();
        _approve(owner, spender, value);
        return true;
    }

    /**
     * @dev See {IERC20-transferFrom}.
     *
     * Emits an {Approval} event indicating the updated allowance. This is not
     * required by the EIP. See the note at the beginning of {ERC20}.
     *
     * NOTE: Does not update the allowance if the current allowance
     * is the maximum `uint256`.
     *
     * Requirements:
     *
     * - `from` and `to` cannot be the zero address.
     * - `from` must have a balance of at least `value`.
     * - the caller must have allowance for ``from``'s tokens of at least
     * `value`.
     */
    function transferFrom(address from, address to, uint256 value) public virtual returns (bool) {
        address spender = _msgSender();
        _spendAllowance(from, spender, value);
        _transfer(from, to, value);
        return true;
    }

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to`.
     *
     * This internal function is equivalent to {transfer}, and can be used to
     * e.g. implement automatic token fees, slashing mechanisms, etc.
     *
     * Emits a {Transfer} event.
     *
     * NOTE: This function is not virtual, {_update} should be overridden instead.
     */
    function _transfer(address from, address to, uint256 value) internal {
        if (from == address(0)) {
            revert ERC20InvalidSender(address(0));
        }
        if (to == address(0)) {
            revert ERC20InvalidReceiver(address(0));
        }
        _update(from, to, value);
    }

    /**
     * @dev Transfers a `value` amount of tokens from `from` to `to`, or alternatively mints (or burns) if `from`
     * (or `to`) is the zero address. All customizations to transfers, mints, and burns should be done by overriding
     * this function.
     *
     * Emits a {Transfer} event.
     */
    function _update(address from, address to, uint256 value) internal virtual {
        if (from == address(0)) {
            // Overflow check required: The rest of the code assumes that totalSupply never overflows
            _totalSupply += value;
        } else {
            uint256 fromBalance = _balances[from];
            if (fromBalance < value) {
                revert ERC20InsufficientBalance(from, fromBalance, value);
            }
            unchecked {
                // Overflow not possible: value <= fromBalance <= totalSupply.
                _balances[from] = fromBalance - value;
            }
        }

        if (to == address(0)) {
            unchecked {
                // Overflow not possible: value <= totalSupply or value <= fromBalance <= totalSupply.
                _totalSupply -= value;
            }
        } else {
            unchecked {
                // Overflow not possible: balance + value is at most totalSupply, which we know fits into a uint256.
                _balances[to] += value;
            }
        }

        emit Transfer(from, to, value);
    }

    /**
     * @dev Creates a `value` amount of tokens and assigns them to `account`, by transferring it from address(0).
     * Relies on the `_update` mechanism
     *
     * Emits a {Transfer} event with `from` set to the zero address.
     *
     * NOTE: This function is not virtual, {_update} should be overridden instead.
     */
    function _mint(address account, uint256 value) internal {
        if (account == address(0)) {
            revert ERC20InvalidReceiver(address(0));
        }
        _update(address(0), account, value);
    }

    /**
     * @dev Destroys a `value` amount of tokens from `account`, lowering the total supply.
     * Relies on the `_update` mechanism.
     *
     * Emits a {Transfer} event with `to` set to the zero address.
     *
     * NOTE: This function is not virtual, {_update} should be overridden instead
     */
    function _burn(address account, uint256 value) internal {
        if (account == address(0)) {
            revert ERC20InvalidSender(address(0));
        }
        _update(account, address(0), value);
    }

    /**
     * @dev Sets `value` as the allowance of `spender` over the `owner` s tokens.
     *
     * This internal function is equivalent to `approve`, and can be used to
     * e.g. set automatic allowances for certain subsystems, etc.
     *
     * Emits an {Approval} event.
     *
     * Requirements:
     *
     * - `owner` cannot be the zero address.
     * - `spender` cannot be the zero address.
     *
     * Overrides to this logic should be done to the variant with an additional `bool emitEvent` argument.
     */
    function _approve(address owner, address spender, uint256 value) internal {
        _approve(owner, spender, value, true);
    }

    /**
     * @dev Variant of {_approve} with an optional flag to enable or disable the {Approval} event.
     *
     * By default (when calling {_approve}) the flag is set to true. On the other hand, approval changes made by
     * `_spendAllowance` during the `transferFrom` operation set the flag to false. This saves gas by not emitting any
     * `Approval` event during `transferFrom` operations.
     *
     * Anyone who wishes to continue emitting `Approval` events on the`transferFrom` operation can force the flag to
     * true using the following override:
     * ```
     * function _approve(address owner, address spender, uint256 value, bool) internal virtual override {
     *     super._approve(owner, spender, value, true);
     * }
     * ```
     *
     * Requirements are the same as {_approve}.
     */
    function _approve(address owner, address spender, uint256 value, bool emitEvent) internal virtual {
        if (owner == address(0)) {
            revert ERC20InvalidApprover(address(0));
        }
        if (spender == address(0)) {
            revert ERC20InvalidSpender(address(0));
        }
        _allowances[owner][spender] = value;
        if (emitEvent) {
            emit Approval(owner, spender, value);
        }
    }

    /**
     * @dev Updates `owner` s allowance for `spender` based on spent `value`.
     *
     * Does not update the allowance value in case of infinite allowance.
     * Revert if not enough allowance is available.
     *
     * Does not emit an {Approval} event.
     */
    function _spendAllowance(address owner, address spender, uint256 value) internal virtual {
        uint256 currentAllowance = allowance(owner, spender);
        if (currentAllowance != type(uint256).max) {
            if (currentAllowance < value) {
                revert ERC20InsufficientAllowance(spender, currentAllowance, value);
            }
            unchecked {
                _approve(owner, spender, currentAllowance - value, false);
            }
        }
    }
}

// File: @openzeppelin/contracts@5.0.0/utils/Pausable.sol


// OpenZeppelin Contracts (last updated v5.0.0) (utils/Pausable.sol)

pragma solidity ^0.8.20;


/**
 * @dev Contract module which allows children to implement an emergency stop
 * mechanism that can be triggered by an authorized account.
 *
 * This module is used through inheritance. It will make available the
 * modifiers `whenNotPaused` and `whenPaused`, which can be applied to
 * the functions of your contract. Note that they will not be pausable by
 * simply including this module, only once the modifiers are put in place.
 */
abstract contract Pausable is Context {
    bool private _paused;

    /**
     * @dev Emitted when the pause is triggered by `account`.
     */
    event Paused(address account);

    /**
     * @dev Emitted when the pause is lifted by `account`.
     */
    event Unpaused(address account);

    /**
     * @dev The operation failed because the contract is paused.
     */
    error EnforcedPause();

    /**
     * @dev The operation failed because the contract is not paused.
     */
    error ExpectedPause();

    /**
     * @dev Initializes the contract in unpaused state.
     */
    constructor() {
        _paused = false;
    }

    /**
     * @dev Modifier to make a function callable only when the contract is not paused.
     *
     * Requirements:
     *
     * - The contract must not be paused.
     */
    modifier whenNotPaused() {
        _requireNotPaused();
        _;
    }

    /**
     * @dev Modifier to make a function callable only when the contract is paused.
     *
     * Requirements:
     *
     * - The contract must be paused.
     */
    modifier whenPaused() {
        _requirePaused();
        _;
    }

    /**
     * @dev Returns true if the contract is paused, and false otherwise.
     */
    function paused() public view virtual returns (bool) {
        return _paused;
    }

    /**
     * @dev Throws if the contract is paused.
     */
    function _requireNotPaused() internal view virtual {
        if (paused()) {
            revert EnforcedPause();
        }
    }

    /**
     * @dev Throws if the contract is not paused.
     */
    function _requirePaused() internal view virtual {
        if (!paused()) {
            revert ExpectedPause();
        }
    }

    /**
     * @dev Triggers stopped state.
     *
     * Requirements:
     *
     * - The contract must not be paused.
     */
    function _pause() internal virtual whenNotPaused {
        _paused = true;
        emit Paused(_msgSender());
    }

    /**
     * @dev Returns to normal state.
     *
     * Requirements:
     *
     * - The contract must be paused.
     */
    function _unpause() internal virtual whenPaused {
        _paused = false;
        emit Unpaused(_msgSender());
    }
}

// File: @openzeppelin/contracts@5.0.0/utils/ReentrancyGuard.sol


// OpenZeppelin Contracts (last updated v5.0.0) (utils/ReentrancyGuard.sol)

pragma solidity ^0.8.20;

/**
 * @dev Contract module that helps prevent reentrant calls to a function.
 *
 * Inheriting from `ReentrancyGuard` will make the {nonReentrant} modifier
 * available, which can be applied to functions to make sure there are no nested
 * (reentrant) calls to them.
 *
 * Note that because there is a single `nonReentrant` guard, functions marked as
 * `nonReentrant` may not call one another. This can be worked around by making
 * those functions `private`, and then adding `external` `nonReentrant` entry
 * points to them.
 *
 * TIP: If you would like to learn more about reentrancy and alternative ways
 * to protect against it, check out our blog post
 * https://blog.openzeppelin.com/reentrancy-after-istanbul/[Reentrancy After Istanbul].
 */
abstract contract ReentrancyGuard {
    // Booleans are more expensive than uint256 or any type that takes up a full
    // word because each write operation emits an extra SLOAD to first read the
    // slot's contents, replace the bits taken up by the boolean, and then write
    // back. This is the compiler's defense against contract upgrades and
    // pointer aliasing, and it cannot be disabled.

    // The values being non-zero value makes deployment a bit more expensive,
    // but in exchange the refund on every call to nonReentrant will be lower in
    // amount. Since refunds are capped to a percentage of the total
    // transaction's gas, it is best to keep them low in cases like this one, to
    // increase the likelihood of the full refund coming into effect.
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;

    uint256 private _status;

    /**
     * @dev Unauthorized reentrant call.
     */
    error ReentrancyGuardReentrantCall();

    constructor() {
        _status = NOT_ENTERED;
    }

    /**
     * @dev Prevents a contract from calling itself, directly or indirectly.
     * Calling a `nonReentrant` function from another `nonReentrant`
     * function is not supported. It is possible to prevent this from happening
     * by making the `nonReentrant` function external, and making it call a
     * `private` function that does the actual work.
     */
    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {
        // On the first call to nonReentrant, _status will be NOT_ENTERED
        if (_status == ENTERED) {
            revert ReentrancyGuardReentrantCall();
        }

        // Any calls to nonReentrant after this point will fail
        _status = ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == ENTERED;
    }
}

// File: YoutHNesT/YoutHNesT Token Contract Mar25 .sol


pragma solidity ^0.8.26;






/**
* @title YoutHNesT Token Contract 
* @dev This contract manages the main token contract including minting for the TGE, cliff, and vesting.
*
* ---- *** YoutHNesT *** More than a meme, a vision! ----
*
* --- The World's First Meme-Social Cryptocurrency. ---
* 
* -- Official launch price on TGE: $0.06 per token. –
*
* - Contact: info€youthnest.info -
* - Website: https://youthnest.info -
*/

contract YoutHNesTToken is ERC20, Ownable, Pausable, ReentrancyGuard {

    uint8 public constant DECIMALS = 4;
    uint256 public constant TOTAL_SUPPLY_TOKENS = 40000000000 * 10 ** uint256(DECIMALS);

    mapping(address => VestingSchedule) public vestingSchedules;
    mapping(address => bool) public approvedPreSaleContracts;
    mapping(address => uint256) public vestingBalances;

    address[] private privateInvestorList;
    address[] public publicInvestorList;

    uint256 private contractBalance;
    uint256 public lastProcessedIndex = 0;
    bool private mintedForTGE;
    uint256[9] public mintAmounts;
    address[9] public mintAddresses;

    // Events
    event PreSaleContractApproved(address indexed contractAddress);
    event PreSaleContractRemoved(address indexed contractAddress);
    event TokensMinted(address indexed beneficiary, uint256 amount);
    event TokensReleased(address indexed beneficiary, uint256 amount);
    event TokensVested(address indexed beneficiary, uint256 amount);
    event TransferFailed(address indexed beneficiary, uint256 amount);
    event ContractPaused(address account);
    event ContractUnpaused(address account);
    event AutomaticVestingCheckPerformed(uint256 processedCount, bool completed);
    event VestingScheduled(address indexed beneficiary, uint256 amount);
    event UnsoldTokensTransferred(address indexed recipient, uint256 amount);

    // Custom Errors
    error InvalidTokenAddress(); 
    error NotAuthorized();
    error CliffPeriodNotReached();
    error AllTokensAlreadyReleased();
    error NoTokensToRelease();
    error TokenTransferFailed();
    error InvalidBeneficiaryAddress();
    error InvalidAmount();
    error ExceedsMaxSupply();
    error InvalidContractAddress();
    error ContractNotApproved();
    error InsufficientBalance(uint256 requested, uint256 available);
    error ZeroAddress();
    error UnauthorizedContractCall(address caller);

    // Dates of TGE (Token generation event)
    uint256 public immutable TGE_START_TIMESTAMP = 1756677600; // Start date TGE
    uint256 public immutable TGE_END_TIMESTAMP = 1767221999;  // End date TGE

    // Vesting Structure
    struct VestingSchedule {
        uint256 totalAmount;
        uint256 releasedAmount;
        uint256 cliffDuration;
        uint256 vestingDuration;
        uint256 startTime;
        uint256 interval;
    }

    constructor() ERC20("YoutHNesT", "YHT") Ownable(msg.sender) {
    }

    // Set up vesting for tokens sold to investors
function _setVesting(
    address beneficiary,
    uint256 amount,
    uint256 cliffDuration,
    uint256 vestingDuration,
    uint256 startTime,
    uint256 interval
    ) internal {
    require(beneficiary != address(0), "Invalid beneficiary");
    require(amount > 0, "Amount must be greater than 0");

    vestingSchedules[beneficiary] = VestingSchedule({
        totalAmount: amount,
        releasedAmount: 0,
        cliffDuration: cliffDuration,
        vestingDuration: vestingDuration,
        startTime: startTime,
        interval: interval
    });

    emit VestingScheduled(beneficiary, amount);
}

    // Function to audit the vesting state of a beneficiary
    function auditVesting(address _beneficiary) external view returns (
        uint256 totalAmount,
        uint256 releasedAmount,
        uint256 remainingAmount,
        uint256 cliffDuration,
        uint256 vestingDuration,
        uint256 startTime,
        uint256 interval
    ) {
        VestingSchedule storage schedule = vestingSchedules[_beneficiary];
        
        totalAmount = schedule.totalAmount;
        releasedAmount = schedule.releasedAmount;
        remainingAmount = totalAmount - releasedAmount;
        cliffDuration = schedule.cliffDuration;
        vestingDuration = schedule.vestingDuration;
        startTime = schedule.startTime;
        interval = schedule.interval;
    }

    // Function for minting tokens during the TGE
    function mintForTGE() external onlyOwner whenNotPaused {
        require(!mintedForTGE, "TGE already executed");
 require(block.timestamp >= TGE_START_TIMESTAMP && block.timestamp <=     TGE_END_TIMESTAMP, "Outside TGE period");
        mintedForTGE = true;

        uint256 totalMintAmount = 
            1000000000 * 10 ** uint256(DECIMALS) + // Public Presale
            1000000000 * 10 ** uint256(DECIMALS) + // Private Round Sale
            9600000000 * 10 ** uint256(DECIMALS) + // Public Sale
            3200000000 * 10 ** uint256(DECIMALS) + // Liquidity
            8000000000 * 10 ** uint256(DECIMALS) + // Rewards & Holders
            3600000000 * 10 ** uint256(DECIMALS) + // Team & Advisors 1
            3600000000 * 10 ** uint256(DECIMALS) + // Team & Advisors 2 
            4400000000 * 10 ** uint256(DECIMALS) + // Marketing & Alliances
            5600000000 * 10 ** uint256(DECIMALS);   // Reserve

        require(totalSupply() + totalMintAmount <= TOTAL_SUPPLY_TOKENS, "Exceeds max supply");
           
        mintAddresses[0] = 0xDEB6B467E5Ee88778B628d71BAe1FA0B27cd9177; 
        mintAddresses[1] = 0xef5266F47F260d0DB67d3fEE2312C8eE35fA0e3a;
        mintAddresses[2] = 0xA18D5f1B8986e60642757dCe06370B8ad5350234;
        mintAddresses[3] = 0x247605dA2600d144c6326B63EA61f169c7F7ad08;
        mintAddresses[4] = 0xDfBB25ce5DAB5D3e6F89095E935A6A7eEc4e5C6D; 
        mintAddresses[5] = 0xa4810cc7684E2b74cE31784C6b48b1d6190D0a4b; 
        mintAddresses[6] = 0x9c5a95D1544645293654889a284f571B7283dA9B; 
        mintAddresses[7] = 0xCc5B5f3607342b8085591ffe9D77583a090BD3cf;
        mintAddresses[8] = 0xD5c43f5f83EC849243a1667d2C72968f0137B3f0;

        mintAmounts[0] = 1000000000 * 10 ** uint256(DECIMALS);
        mintAmounts[1] = 1000000000 * 10 ** uint256(DECIMALS);
        mintAmounts[2] = 9600000000 * 10 ** uint256(DECIMALS);
        mintAmounts[3] = 3200000000 * 10 ** uint256(DECIMALS);
        mintAmounts[4] = 8000000000 * 10 ** uint256(DECIMALS);
        mintAmounts[5] = 3600000000 * 10 ** uint256(DECIMALS);
        mintAmounts[6] = 3600000000 * 10 ** uint256(DECIMALS);
        mintAmounts[7] = 4400000000 * 10 ** uint256(DECIMALS);
        mintAmounts[8] = 5600000000 * 10 ** uint256(DECIMALS); 

_mint(address(this), totalMintAmount);  

emit TokensMinted(address(this), totalMintAmount); 

       contractBalance = totalMintAmount;

        // Setting up vesting
        _setVesting(mintAddresses[2], mintAmounts[2] * 50 / 100, 180 days, 24 * 30 days, TGE_START_TIMESTAMP, 30 days);
        _setVesting(mintAddresses[4], mintAmounts[4], 365 days, 10 * 365 days, TGE_START_TIMESTAMP, 30 days);
        _setVesting(mintAddresses[5], mintAmounts[5], 180 days, 24 * 30 days, TGE_START_TIMESTAMP, 30 days);
        _setVesting(mintAddresses[6], mintAmounts[6], 180 days, 24 * 30 days, TGE_START_TIMESTAMP, 30 days);
        _setVesting(mintAddresses[7], mintAmounts[7], 180 days, 24 * 30 days, TGE_START_TIMESTAMP, 30 days);
        _setVesting(mintAddresses[8], mintAmounts[8], 365 days, 24 * 30 days, TGE_START_TIMESTAMP, 30 days);
    }

function handleUnsoldTokens(
    uint256 privateSoldAmount,
    uint256 publicSoldAmount
) external onlyOwner {
    require(mintAmounts[0] >= publicSoldAmount, "Public sale: Sold amount exceeds allocated");
    require(mintAmounts[1] >= privateSoldAmount, "Private sale: Sold amount exceeds allocated");

    uint256 privateUnsoldAmount = mintAmounts[1] - privateSoldAmount;
    uint256 publicUnsoldAmount = mintAmounts[0] - publicSoldAmount;

    uint256 totalUnsold = privateUnsoldAmount + publicUnsoldAmount;
    require(contractBalance >= totalUnsold, "Insufficient contract balance");

    // We transfer ONLY the UNSold tokens from the public presale
    if (publicUnsoldAmount > 0) {
        _transfer(address(this), mintAddresses[0], publicUnsoldAmount);
        emit UnsoldTokensTransferred(mintAddresses[0], publicUnsoldAmount);
    }

    // We transfer ONLY the UNSold tokens from the private pre-sale
    if (privateUnsoldAmount > 0) {
        _transfer(address(this), mintAddresses[1], privateUnsoldAmount);
        emit UnsoldTokensTransferred(mintAddresses[1], privateUnsoldAmount);
    }

    contractBalance -= totalUnsold;
}
    
// Functions for distributing tokens to pre-sale contracts
  function distributeInitialTokensPrivateSale(address privateSaleContract) external onlyOwner whenNotPaused { require(approvedPreSaleContracts[privateSaleContract], "Contract not approved");
        
        IYoutHNesTPrivateSale sale = IYoutHNesTPrivateSale(privateSaleContract);
        uint256 investorCount = sale.getPrivateInvestorCount();

for (uint256 i = 0; i < investorCount; i++) {
    address investor = sale.getPrivateInvestor(i);
    uint256 amount = sale.getAllocation(investor);
    require(investor != address(0) && amount > 0, "Invalid investor");

    vestingSchedules[investor] = VestingSchedule({
        totalAmount: amount,
        releasedAmount: 0,
        cliffDuration: 90 days,
        vestingDuration: 180 days,
        startTime: block.timestamp,
        interval: 30 days
    });

    uint256 releaseAmount = (amount * 30) / 100;
    vestingSchedules[investor].releasedAmount += releaseAmount;

    _safeTokenTransfer(investor, releaseAmount);
    emit TokensReleased(investor, releaseAmount);
       }
  }

function distributeInitialTokensPublicSale(address publicSaleContract) external onlyOwner whenNotPaused {
    require(approvedPreSaleContracts[publicSaleContract], "Contract not approved");

    IYoutHNesTPublicPreSale sale = IYoutHNesTPublicPreSale(publicSaleContract);
    uint256 investorCount = sale.getPublicInvestorCount();

    for (uint256 i = 0; i < investorCount; i++) {
        address investor = sale.getPublicInvestor(i);
        uint256 amount = sale.getAllocation(investor);
        require(investor != address(0) && amount > 0, "Invalid investor");

        uint256 releaseAmount = (amount * 30) / 100;
        vestingSchedules[investor].releasedAmount += releaseAmount;
        _setVesting(investor, amount - releaseAmount, 90 days, 180 days, block.timestamp, 30 days);

        _safeTokenTransfer(investor, releaseAmount);
        emit TokensReleased(investor, releaseAmount);
    }
}

    // Function for secure token transfer
    function _safeTokenTransfer(address _beneficiary, uint256 amountToRelease) private {
if (contractBalance < amountToRelease) revert InsufficientBalance(amountToRelease, contractBalance);    _transfer(address(this), _beneficiary, amountToRelease); contractBalance -= amountToRelease; }

function releaseVestedTokens(address _beneficiary) external nonReentrant whenNotPaused {
    require(msg.sender == _beneficiary || msg.sender == owner(), "Not authorized to release tokens");
    _checkAndReleaseVestedTokens(_beneficiary);
}

    // Function to check and release vested tokens automatically
    function _checkAndReleaseVestedTokens(address _beneficiary) private {
        VestingSchedule storage schedule = vestingSchedules[_beneficiary];

        if (schedule.releasedAmount >= schedule.totalAmount) {
            return;
        }

        uint256 currentTime = block.timestamp;

        if (currentTime < schedule.startTime + schedule.cliffDuration) {
            return;
        }

        uint256 elapsedTime = currentTime - schedule.startTime;
        uint256 vestedAmount;

        if (elapsedTime >= schedule.vestingDuration) {
            vestedAmount = schedule.totalAmount;
        } else {
            vestedAmount = (schedule.totalAmount * elapsedTime) / schedule.vestingDuration;
        }

        uint256 amountToRelease = vestedAmount - schedule.releasedAmount;

        if (amountToRelease > 0) {
            schedule.releasedAmount += amountToRelease;
            _safeTokenTransfer(_beneficiary, amountToRelease);

            emit TokensReleased(_beneficiary, amountToRelease);
        }
    }

    // Emergency function to release all tokens manually
    function emergencyReleaseAllTokens(address _beneficiary) external onlyOwner {
        VestingSchedule storage schedule = vestingSchedules[_beneficiary];

        uint256 remainingAmount = schedule.totalAmount - schedule.releasedAmount;

        if (remainingAmount > 0) {
            schedule.releasedAmount = schedule.totalAmount;
            _safeTokenTransfer(_beneficiary, remainingAmount);

            emit TokensReleased(_beneficiary, remainingAmount);
        }
    }

       // Function to trigger automatic vesting check across all beneficiaries
    function releaseVestedTokensAutomatically() external nonReentrant whenNotPaused {
        uint256 maxIterations = 10; // Limiting the number of iterations to avoid excessive gas consumption
        uint256 count = 0;
        uint256 totalInvestors = publicInvestorList.length + privateInvestorList.length;

        for (uint256 i = lastProcessedIndex; i < totalInvestors && count < maxIterations; i++) {
        address investor = (i < publicInvestorList.length) 
            ? publicInvestorList[i] 
            : privateInvestorList[i - publicInvestorList.length];

        _checkAndReleaseVestedTokens(investor);
        count++;
        lastProcessedIndex = i + 1;
    }

    if (lastProcessedIndex >= totalInvestors) {
        lastProcessedIndex = 0;
    }

    emit AutomaticVestingCheckPerformed(count, lastProcessedIndex == 0);
}

    // Emergency function to withdraw funds
    function emergencyWithdraw() external onlyOwner {
        uint256 contractEthBalance = address(this).balance;
        if (contractEthBalance == 0) revert InsufficientBalance(0, contractEthBalance);
        (bool sent, ) = owner().call{value: contractEthBalance}("");
        if (!sent) revert TokenTransferFailed();
    }

    // Functions to pause and unpause the contract
    function pause() external onlyOwner {
        _pause();
        emit ContractPaused(msg.sender);
    }

    function unpause() external onlyOwner {
        _unpause();
        emit ContractUnpaused(msg.sender);
    }

    //Approve/Revoke Pre-sale Contracts Function
    function approvePreSaleContract(address contractAddress) external onlyOwner {
        if (contractAddress == address(0)) revert ZeroAddress();
        approvedPreSaleContracts[contractAddress] = true;
        emit PreSaleContractApproved(contractAddress);
    }

    function removePreSaleContract(address contractAddress) external onlyOwner {
        if (contractAddress == address(0)) revert ZeroAddress();
        approvedPreSaleContracts[contractAddress] = false;
    emit PreSaleContractRemoved(contractAddress);
    }
}   

// Interfaces for pre-sale contracts
interface IYoutHNesTPrivateSale {
    function getPrivateInvestorCount() external view returns (uint256);
    function getPrivateInvestor(uint256 index) external view returns (address);
    function getAllocation(address investor) external view returns (uint256);
}

interface IYoutHNesTPublicPreSale {
    function getPublicInvestorCount() external view returns (uint256);
    function getPublicInvestor(uint256 index) external view returns (address);
    function getAllocation(address investor) external view returns (uint256);
}