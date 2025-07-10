// File: @openzeppelin/contracts/utils/Context.sol


// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

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

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}

// File: @openzeppelin/contracts/access/Ownable.sol


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

// File: @openzeppelin/contracts/token/ERC20/IERC20.sol


// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC-20 standard as defined in the ERC.
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

// File: @openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol


// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC20/extensions/IERC20Metadata.sol)

pragma solidity ^0.8.20;


/**
 * @dev Interface for the optional metadata functions from the ERC-20 standard.
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

// File: @openzeppelin/contracts/interfaces/draft-IERC6093.sol


// OpenZeppelin Contracts (last updated v5.1.0) (interfaces/draft-IERC6093.sol)
pragma solidity ^0.8.20;

/**
 * @dev Standard ERC-20 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC-20 tokens.
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
 * @dev Standard ERC-721 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC-721 tokens.
 */
interface IERC721Errors {
    /**
     * @dev Indicates that an address can't be an owner. For example, `address(0)` is a forbidden owner in ERC-20.
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
 * @dev Standard ERC-1155 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC-1155 tokens.
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

// File: @openzeppelin/contracts/token/ERC20/ERC20.sol


// OpenZeppelin Contracts (last updated v5.2.0) (token/ERC20/ERC20.sol)

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
 * conventional and does not conflict with the expectations of ERC-20
 * applications.
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
     * Skips emitting an {Approval} event indicating an allowance update. This is not
     * required by the ERC. See {xref-ERC20-_approve-address-address-uint256-bool-}[_approve].
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
     *
     * ```solidity
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
        if (currentAllowance < type(uint256).max) {
            if (currentAllowance < value) {
                revert ERC20InsufficientAllowance(spender, currentAllowance, value);
            }
            unchecked {
                _approve(owner, spender, currentAllowance - value, false);
            }
        }
    }
}

// File: @openzeppelin/contracts/utils/ReentrancyGuard.sol


// OpenZeppelin Contracts (last updated v5.1.0) (utils/ReentrancyGuard.sol)

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
 * TIP: If EIP-1153 (transient storage) is available on the chain you're deploying at,
 * consider using {ReentrancyGuardTransient} instead.
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

// File: Amara.sol


pragma solidity 0.8.28;




// ===== Uniswap V2 Interfaces =====
interface IUniswapV2Router02 {
    function factory() external pure returns (address);
    function WETH() external pure returns (address);
    // Used for calculating minimum amounts (slippage protection)
    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;
    function swapExactETHForTokensSupportingFeeOnTransferTokens(
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable;
    function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);
}

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IUniswapV2Pair {
    function getReserves() external view returns (uint112, uint112, uint32);
}

contract Amara is ERC20, Ownable, ReentrancyGuard {
    address public constant BURN_ADDRESS = 0x000000000000000000000000000000000000dEaD;
    address private constant UNISWAP_V2_ROUTER = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;

    // ========= Basic Token Data =========
    uint256 public constant INITIAL_SUPPLY = 250000000000 * 10**18;

    // ========= Fee Settings (in basis points) =========
    uint256 internal constant FEE_DENOMINATOR = 10000;
    uint256 public constant INITIAL_BURN_FEE = 500;    // 5%
    uint256 public constant BASE_LIQUIDITY_FEE = 300;    // 3%

    // ========= Anti-Whale Settings =========
    uint256 public constant SELL_COOLDOWN = 60; // seconds between sells
    uint256 public constant SELL_THRESHOLD = INITIAL_SUPPLY / 200; // 0.5% of supply
    uint256 public constant WHALE_HOLDING_THRESHOLD = INITIAL_SUPPLY / 100; // 1% of supply
    uint256 public constant WHALE_EXTRA_SELL_FEE = 500; // extra 5%

    // ========= Time-Based Dynamics =========
    uint256 public _deploymentTime;  // used to calculate dynamic burn rate
    uint256 public constant BURN_FEE_INCREASE_INTERVAL = 90 days;
    uint256 public constant BURN_FEE_INCREASE_AMOUNT = 25;

    // ========= Staking Constants =========
    uint256 public constant STAKING_REWARD_RATE = 100;
    uint256 public constant STAKING_PERIOD = 30 days;

    // ========= Swap & Liquidity Settings =========
    IUniswapV2Router02 public uniswapRouter;
    address public uniswapPair;
    bool private _inSwap;
    uint256 public swapThreshold;        // token threshold for auto-liquidity injection
    uint256 public buybackSwapThreshold; // token threshold for buyback swap

    // ========= Fee Collection =========
    uint256 public liquidityTokensCollected;
    uint256 public lotteryPool;
    uint256 public buybackTokensCollected;

    // ========= Burn Tracking =========
    uint256 public totalBurned;

    // ========= Trade Analytics =========
    uint256 public totalTradeVolume;

    // ========= Advanced Burn Mechanisms =========
    // Milestone Burns
    uint256[] public milestoneThresholds;
    mapping(uint256 => bool) public milestoneBurnTriggered;
    uint256 public constant MILESTONE_BURN_PERCENTAGE = 500; // 5%
    // Super Burn
    bool public superBurnTriggered;
    uint256 public constant SUPER_BURN_THRESHOLD = INITIAL_SUPPLY * 20 / 100;
    uint256 public constant SUPER_BURN_PERCENTAGE = 1000; // 10%
    // Auto-Supply Reduction
    uint256 public constant AUTO_SUPPLY_REDUCTION_PERCENTAGE = 200; // 2%
    uint256 public constant AUTO_SUPPLY_REDUCTION_ETH_THRESHOLD = 100 ether;
    // Adaptive Liquidity
    uint256 public lastPrice; // scaled by 1e18
    uint256 public constant VOLATILITY_THRESHOLD_PERCENT = 10;
    uint256 public constant ADAPTIVE_LIQUIDITY_FEE_INCREASE = 200;

    // ========= Slippage Tolerance =========
    uint256 public constant SLIPPAGE_TOLERANCE_BP = 50; // 50 basis points
    mapping(address => bool) public isExcludedFromFee;

    // ========= Buyback Fallback =========
    uint256 public lastBuybackTime;
    uint256 public constant BUYBACK_FALLBACK_INTERVAL = 1 hours;

    // ========= Reserve Caching =========
    uint256 private cachedTokenReserve;
    uint256 private cachedETHReserve;
    uint256 private lastReservesUpdate;
    uint256 public constant RESERVES_UPDATE_INTERVAL = 5 minutes;

    // ========= Events =========
    event Airdrop(address indexed beneficiary, uint256 bonusAmount);
    event SwapAndLiquify(uint256 tokensSwapped, uint256 ethReceived, uint256 tokensIntoLiquidity);
    event Burn(address indexed burner, uint256 amount);
    event FeeDiscountGranted(address indexed user, uint256 discountUntil);
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount, uint256 reward);
    event LotteryAward(address indexed winner, uint256 amount);
    event ReceivedETH(address indexed sender, uint256 amount);

    // ========= Anti-Whale Purchase Restrictions =========
    uint256 public constant MAX_PURCHASE_AMOUNT_FIRST_30_DAYS = 10_000_000 * 10**18;
    uint256 public constant MAX_PURCHASES_PER_MINUTE = 1;

    struct PurchaseWindow {
        uint256 windowStart;
        uint256 count;
    }
    mapping(address => PurchaseWindow) public purchaseWindows;

    // ========= User Data Struct =========
    struct UserData {
        uint256 lastSell;
        uint256 feeDiscountUntil;
        uint256 firstReceived;
        uint256 stakedBalance;
        uint256 stakeTimestamp;
    }
    mapping(address => UserData) public userData;

    // ========= Modifiers =========
    modifier lockTheSwap() {
        _inSwap = true;
        _;
        _inSwap = false;
    }

    /// @dev External functions that modify state are protected by onlyRenounced and nonReentrant.
    modifier onlyRenounced() {
        require(owner() == address(0), "Access control: owner not renounced");
        _;
    }

    // ========= Constructor =========
    constructor() payable ERC20("Amara", "AMARA") Ownable(msg.sender) {
        _deploymentTime = block.timestamp;
        _mint(msg.sender, INITIAL_SUPPLY);
        userData[msg.sender].firstReceived = block.timestamp;

        uniswapRouter = IUniswapV2Router02(UNISWAP_V2_ROUTER);
        uniswapPair = IUniswapV2Factory(uniswapRouter.factory())
            .createPair(address(this), uniswapRouter.WETH());

        // Set thresholds, etc.
        swapThreshold = INITIAL_SUPPLY / 10000;
        buybackSwapThreshold = INITIAL_SUPPLY / 10000;
        milestoneThresholds.push(INITIAL_SUPPLY * 5 / 100);
        milestoneThresholds.push(INITIAL_SUPPLY * 10 / 100);
        milestoneThresholds.push(INITIAL_SUPPLY * 15 / 100);

        _updateCachedReserves();
        lastBuybackTime = block.timestamp;

        isExcludedFromFee[msg.sender] = true;              // Exclude the deployer
        isExcludedFromFee[address(this)] = true;             // Exclude the contract itself
        isExcludedFromFee[address(uniswapRouter)] = true;    // Exclude the Uniswap router
        isExcludedFromFee[uniswapPair] = true;               // Exclude the liquidity pair

        // Immediately renounce ownership.
        renounceOwnership();
    }

    // ========= External Functions =========
    function transfer(address recipient, uint256 amount)
    public
    override
    onlyRenounced
    nonReentrant
    returns (bool)
    {
        _feeTransfer(_msgSender(), recipient, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount)
    public
    override
    onlyRenounced
    nonReentrant
    returns (bool)
    {
        uint256 currentAllowance = allowance(sender, _msgSender());
        require(currentAllowance >= amount, "Allowance low");
        _approve(sender, _msgSender(), currentAllowance - amount);
        _feeTransfer(sender, recipient, amount);
        return true;
    }

    function stake(uint256 amount) external onlyRenounced nonReentrant {
        require(balanceOf(msg.sender) != 0, "Low balance");
        _feeTransfer(msg.sender, address(this), amount);
        userData[msg.sender].stakedBalance += amount;
        userData[msg.sender].stakeTimestamp = block.timestamp;
        emit Staked(msg.sender, amount);
    }

    function unstake(uint256 amount) external onlyRenounced nonReentrant {
        require(userData[msg.sender].stakedBalance >= amount, "Insufficient stake");
        uint256 stakedTime = block.timestamp - userData[msg.sender].stakeTimestamp;
        uint256 reward = (amount * STAKING_REWARD_RATE * stakedTime) / (STAKING_PERIOD * FEE_DENOMINATOR);
        userData[msg.sender].stakedBalance -= amount;
        _feeTransfer(address(this), msg.sender, amount + reward);
        emit Unstaked(msg.sender, amount, reward);
    }

    // Lottery: splits the lottery pool among multiple winners.
    function triggerLotteryAward(address[] calldata winners)
    external
    onlyRenounced
    nonReentrant
    {
        require(winners.length > 0, "No winners provided");
        uint256 totalPrize = lotteryPool;
        require(totalPrize > 0, "No lottery funds");
        uint256 share = totalPrize / winners.length;
        uint256 remainder = totalPrize - (share * winners.length);
        for (uint256 i = 0; i < winners.length; i++) {
            uint256 awardAmount = share;
            if (i == 0) {
                awardAmount += remainder; // assign remainder to first winner
            }
            // Direct transfer from contract to winner; no fee processing.
            super._transfer(address(this), winners[i], awardAmount);
            emit LotteryAward(winners[i], awardAmount);
        }
        lotteryPool = 0;
    }

    // ========= Internal Fee Transfer Function =========
    function _feeTransfer(address sender, address recipient, uint256 amount) internal {
        // If sender or recipient is fee-exempt, do a simple transfer.
        if (isExcludedFromFee[sender] || isExcludedFromFee[recipient]) {
            super._transfer(sender, recipient, amount);
            return;
        }

        // --- NEW: Anti-Whale Checks for Buys during the first 30 days ---
        // A buy occurs when tokens come from the uniswapPair.
        if (sender == uniswapPair && block.timestamp < _deploymentTime + 30 days && !isExcludedFromFee[recipient]) {
            // Prevent purchases of 50 million tokens or over.
            require(amount < MAX_PURCHASE_AMOUNT_FIRST_30_DAYS, "Anti-Whale: purchase amount too high");

            // Limit the number of purchases per 60 seconds.
            PurchaseWindow storage pw = purchaseWindows[recipient];
            if (block.timestamp < pw.windowStart + 60) {
                require(pw.count < MAX_PURCHASES_PER_MINUTE, "Anti-Whale: too many purchases in 60 seconds");
                pw.count++;
            } else {
                pw.windowStart = block.timestamp;
                pw.count = 1;
            }
        }
        // --- End of Anti-Whale Checks ---

        uint256 currentTime = block.timestamp;
        // Check-effects: update internal state before external calls.
        if (recipient == uniswapPair) {
            require(currentTime - userData[sender].lastSell >= SELL_COOLDOWN, "Cooldown");
            userData[sender].lastSell = currentTime;
        }
        if (sender == uniswapPair || recipient == uniswapPair) {
            totalTradeVolume += amount;
        }
        uint256 burnFee = currentBurnFee();
        uint256 liquidityFee = getAdaptiveLiquidityFee();
        // Apply discount if the sender held tokens for 30 days.
        if (userData[sender].firstReceived != 0 && currentTime - userData[sender].firstReceived >= 30 days) {
            burnFee = (burnFee * 80) / 100;
            liquidityFee = (liquidityFee * 80) / 100;
        }
        // Apply burn-to-boost discount.
        if (userData[sender].feeDiscountUntil >= currentTime) {
            burnFee /= 2;
            liquidityFee /= 2;
        }
        // Extra fee for large sells.
        if (recipient == uniswapPair && amount >= SELL_THRESHOLD) {
            liquidityFee += WHALE_EXTRA_SELL_FEE;
        }
        {
            uint256 senderBalance = balanceOf(sender);
            if (senderBalance >= WHALE_HOLDING_THRESHOLD && amount >= (senderBalance * 5) / 100) {
                liquidityFee += WHALE_EXTRA_SELL_FEE;
            }
        }
        uint256 burnAmount = (amount * burnFee) / FEE_DENOMINATOR;
        uint256 totalLiquidityFee = (amount * liquidityFee) / FEE_DENOMINATOR;
        uint256 lotteryFee = (totalLiquidityFee * 10) / 100;
        uint256 buybackFeeAmount = (totalLiquidityFee * 20) / 100;
        uint256 liquidityForLiquidity = totalLiquidityFee - lotteryFee - buybackFeeAmount;
        uint256 totalFees = burnAmount + totalLiquidityFee;
        require(amount > totalFees, "Fee too high");
        uint256 transferAmount = amount - totalFees;

        if (burnAmount != 0) {
            super._transfer(sender, address(0xdead), burnAmount);
            totalBurned += burnAmount;
            emit Burn(sender, burnAmount);
            if (recipient == address(0xdead)) {
                userData[sender].feeDiscountUntil = currentTime + 1 days;
                emit FeeDiscountGranted(sender, currentTime + 1 days);
            }
        }
        if (liquidityForLiquidity != 0) {
            super._transfer(sender, address(this), liquidityForLiquidity);
            liquidityTokensCollected += liquidityForLiquidity;
        }
        if (lotteryFee != 0) {
            super._transfer(sender, address(this), lotteryFee);
            lotteryPool += lotteryFee;
        }
        if (buybackFeeAmount != 0) {
            super._transfer(sender, address(this), buybackFeeAmount);
            buybackTokensCollected += buybackFeeAmount;
        }
        // Final transfer to recipient
        super._transfer(sender, recipient, transferAmount);
        if (userData[recipient].firstReceived == 0) {
            userData[recipient].firstReceived = currentTime;
        }
        // Trigger liquidity injection if conditions are met.
        if (liquidityTokensCollected >= swapThreshold && !_inSwap && sender != uniswapPair) {
            swapAndLiquify(liquidityTokensCollected);
            liquidityTokensCollected = 0;
        }
        if (!_inSwap && sender != uniswapPair &&
        (buybackTokensCollected >= buybackSwapThreshold ||
            (block.timestamp - lastBuybackTime >= BUYBACK_FALLBACK_INTERVAL && buybackTokensCollected > 0))) {
            swapAndBuybackAndBurn(buybackTokensCollected);
            buybackTokensCollected = 0;
            lastBuybackTime = block.timestamp;
        }
        checkMilestoneBurns();
        checkSuperBurn();
        checkAutoSupplyReduction();
        tryAirdropBonus(recipient);
    }

    // ========= Dynamic Burn Fee Calculation =========
    function currentBurnFee() public view returns (uint256) {
        // Note: integer division truncation is expected.
        uint256 periods = (block.timestamp - _deploymentTime) / BURN_FEE_INCREASE_INTERVAL;
        return INITIAL_BURN_FEE + (periods * BURN_FEE_INCREASE_AMOUNT);
    }

    // ========= Adaptive Liquidity Fee Calculation =========
    function getAdaptiveLiquidityFee() public returns (uint256) {
        uint256 fee = BASE_LIQUIDITY_FEE;
        ( uint256 tokenReserve, uint256 ethReserve ) = _getPairReserves();
        if (tokenReserve == 0) return fee;
        uint256 currentPrice = (ethReserve * 1e18) / tokenReserve;
        if (lastPrice == 0) {
            lastPrice = currentPrice;
            return fee;
        }
        uint256 priceChange = currentPrice > lastPrice ? currentPrice - lastPrice : lastPrice - currentPrice;
        uint256 changePercent = (priceChange * 100) / lastPrice;
        if (changePercent >= VOLATILITY_THRESHOLD_PERCENT) {
            fee += ADAPTIVE_LIQUIDITY_FEE_INCREASE;
        }
        lastPrice = currentPrice;
        return fee;
    }

    // ========= Reserve Caching =========
    function _updateCachedReserves() internal {
        // External call to getReserves; then update cached values.
        (uint112 reserve0, uint112 reserve1, ) = IUniswapV2Pair(uniswapPair).getReserves();

        address _weth = uniswapRouter.WETH();

        if (address(this) < _weth) {
            cachedTokenReserve = uint256(reserve0);
            cachedETHReserve = uint256(reserve1);
        } else {
            cachedTokenReserve = uint256(reserve1);
            cachedETHReserve = uint256(reserve0);
        }

        lastReservesUpdate = block.timestamp;
    }

    function _getPairReserves() internal returns (uint256 tokenReserve, uint256 ethReserve) {
        if (block.timestamp - lastReservesUpdate > RESERVES_UPDATE_INTERVAL && cachedTokenReserve > 0) {
            _updateCachedReserves();
        }
        return (cachedTokenReserve, cachedETHReserve);
    }

    // ========= Swap and Liquify =========
    function swapAndLiquify(uint256 tokenAmount) private lockTheSwap {
        address _this = address(this);
        uint256 half = tokenAmount >> 1; // using bit-shift for division by 2
        uint256 otherHalf = tokenAmount - half;
        uint256 initialBalance = _this.balance;
        swapTokensForETH(half);
        uint256 newBalance = _this.balance - initialBalance;
        addLiquidity(otherHalf, newBalance);
        emit SwapAndLiquify(half, newBalance, otherHalf);
    }

    // ========= Add Liquidity =========
    function addLiquidity(uint256 tokenAmount, uint256 ethAmount) private {
        address _this = address(this);

        // Approve token transfer to cover all possible scenarios
        _approve(_this, address(uniswapRouter), tokenAmount);

        // Add liquidity
        uniswapRouter.addLiquidityETH{value: ethAmount}(
            address(this),
            tokenAmount,
            0, // Accept any amount of tokens
            0, // Accept any amount of ETH
            address(BURN_ADDRESS), // Sends LP tokens to burn address for permanent liquidity lock
            block.timestamp
        );
    }

    // ========= Swap Tokens for ETH =========
    function swapTokensForETH(uint256 tokenAmount) private {
        address _this = address(this);
        address[] memory path = new address[](2);
        path[0] = _this;
        path[1] = uniswapRouter.WETH();
        uint[] memory amounts = uniswapRouter.getAmountsOut(tokenAmount, path);
        uint256 amountOutMin = (amounts[1] * (FEE_DENOMINATOR - SLIPPAGE_TOLERANCE_BP)) / FEE_DENOMINATOR;
        _approve(_this, address(uniswapRouter), tokenAmount);
        uniswapRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            amountOutMin,
            path,
            _this,
            block.timestamp
        );
    }

    // ========= Swap and Buyback & Burn =========
    function swapAndBuybackAndBurn(uint256 tokenAmount) private lockTheSwap {
        address _this = address(this);
        uint256 initialEth = _this.balance;
        _approve(_this, address(uniswapRouter), tokenAmount);
        address[] memory path = new address[](2);
        path[0] = _this;
        path[1] = uniswapRouter.WETH();
        uniswapRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            _this,
            block.timestamp
        );
        uint256 ethReceived = _this.balance - initialEth;
        if (ethReceived != 0) {
            swapETHForTokensAndBurn(ethReceived);
        }
    }

    // ========= Swap ETH for Tokens and Burn Them =========
    function swapETHForTokensAndBurn(uint256 ethAmount) internal lockTheSwap {
        require(ethAmount > 0, "ETH amount must be greater than zero");
        address[] memory path = new address[](2);
        path[0] = uniswapRouter.WETH();
        path[1] = address(this);

        uint[] memory amounts = uniswapRouter.getAmountsOut(ethAmount, path);
        uint256 amountOutMin = (amounts[1] * (FEE_DENOMINATOR - SLIPPAGE_TOLERANCE_BP)) / FEE_DENOMINATOR;

        uniswapRouter.swapExactETHForTokensSupportingFeeOnTransferTokens{value: ethAmount}(
            amountOutMin,
            path,
            BURN_ADDRESS,
            block.timestamp
        );

        emit Burn(BURN_ADDRESS, amounts[1]);
    }

    // ========= Airdrop Bonus =========
    function tryAirdropBonus(address beneficiary) internal {
        // Note: The on-chain pseudo-random mechanism is not secure.
        if (userData[beneficiary].firstReceived != 0 && block.timestamp - userData[beneficiary].firstReceived >= 30 days) {
            if (uint256(keccak256(abi.encodePacked(block.timestamp, block.prevrandao, beneficiary, totalTradeVolume))) % 1000 == 0) {
                uint256 bonus = INITIAL_SUPPLY / 10000;
                if (balanceOf(address(this)) >= bonus) {
                    super._transfer(address(this), beneficiary, bonus);
                    emit Airdrop(beneficiary, bonus);
                }
            }
        }
    }

    // ========= Milestone Burns =========
    function checkMilestoneBurns() internal {
        uint256 contractBalance = balanceOf(address(this));
        uint256 len = milestoneThresholds.length;
        for (uint256 i = 0; i < len; ++i) {
            uint256 threshold = milestoneThresholds[i];
            if (!milestoneBurnTriggered[threshold] && totalBurned >= threshold) {
                uint256 burnAmount = (contractBalance * MILESTONE_BURN_PERCENTAGE) / FEE_DENOMINATOR;
                if (burnAmount != 0) {
                    _burn(address(this), burnAmount);
                    totalBurned += burnAmount;
                    milestoneBurnTriggered[threshold] = true;
                    emit Burn(address(this), burnAmount);
                }
            }
        }
    }

    // ========= Super Burn =========
    function checkSuperBurn() internal {
        if (!superBurnTriggered && totalBurned >= SUPER_BURN_THRESHOLD) {
            uint256 contractBalance = balanceOf(address(this));
            uint256 burnAmount = (contractBalance * SUPER_BURN_PERCENTAGE) / FEE_DENOMINATOR;
            if (burnAmount != 0) {
                _burn(address(this), burnAmount);
                totalBurned += burnAmount;
                superBurnTriggered = true;
                emit Burn(address(this), burnAmount);
            }
        }
    }

    // ========= Auto-Supply Reduction =========
    function checkAutoSupplyReduction() internal {
        (, uint256 ethReserve) = _getPairReserves();
        if (ethReserve >= AUTO_SUPPLY_REDUCTION_ETH_THRESHOLD) {
            uint256 contractBalance = balanceOf(address(this));
            uint256 burnAmount = (contractBalance * AUTO_SUPPLY_REDUCTION_PERCENTAGE) / FEE_DENOMINATOR;
            if (burnAmount != 0) {
                _burn(address(this), burnAmount);
                totalBurned += burnAmount;
                emit Burn(address(this), burnAmount);
            }
        }
    }

    // ========= On-Chain Analytics =========
    function getCurrentBurnRate() external view returns (uint256 currentBurnRate) {
        currentBurnRate = currentBurnFee();
    }

    function getTotalBurned() external view returns (uint256 totalBurn) {
        totalBurn = totalBurned;
    }

    function getCirculatingSupply() external view returns (uint256 circulatingSupply) {
        circulatingSupply = totalSupply() - balanceOf(address(0xdead));
    }

    function getLastPrice() external view returns (uint256 lastPriceValue) {
        lastPriceValue = lastPrice;
    }

    function getTotalTradeVolume() external view returns (uint256 tradeVolume) {
        tradeVolume = totalTradeVolume;
    }

    function _getPairReservesView() internal view returns (uint256 tokenReserve, uint256 ethReserve) {
        (uint112 reserve0, uint112 reserve1, ) = IUniswapV2Pair(uniswapPair).getReserves();

        address _weth = uniswapRouter.WETH();

        if (address(this) < _weth) {
            tokenReserve = uint256(reserve0);
            ethReserve = uint256(reserve1);
        } else {
            tokenReserve = uint256(reserve1);
            ethReserve = uint256(reserve0);
        }
    }


    function computeAdaptiveLiquidityFee() external view returns (uint256 computedFee) {
        (uint256 tokenReserve, uint256 ethReserve) = _getPairReservesView(); // ✅ FIXED

        if (tokenReserve == 0) return BASE_LIQUIDITY_FEE;

        uint256 currentPrice = (ethReserve * 1e18) / tokenReserve;
        uint256 _lastPrice = lastPrice;

        if (_lastPrice == 0) {
            computedFee = BASE_LIQUIDITY_FEE;
        } else {
            uint256 priceChange = currentPrice > _lastPrice ? currentPrice - _lastPrice : _lastPrice - currentPrice;
            uint256 changePercent = (priceChange * 100) / _lastPrice;

            if (changePercent >= VOLATILITY_THRESHOLD_PERCENT) {
                computedFee = BASE_LIQUIDITY_FEE + ADAPTIVE_LIQUIDITY_FEE_INCREASE;
            } else {
                computedFee = BASE_LIQUIDITY_FEE;
            }
        }
    }


    function getFeeStructure() external view returns (uint256 currentBurnFeeValue, uint256 baseLiquidityFeeValue) {
        currentBurnFeeValue = currentBurnFee();
        baseLiquidityFeeValue = BASE_LIQUIDITY_FEE;
    }

    // ========= Fallback Function =========
    receive() external payable {
        emit ReceivedETH(msg.sender, msg.value);
    }
}