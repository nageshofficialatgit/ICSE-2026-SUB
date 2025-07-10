//SPDX-License-Identifier: MIT
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

pragma solidity ^0.8.4;





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

// File: contracts/contracts/contracts/interfaces/IStabilityModule.sol

pragma solidity 0.8.20;

interface IStabilityModule {
    struct Stake {
        uint256 amount;
        uint256 startTime;
        uint256 endTime;
        uint256 withdrawAmount;
        uint256 lastGovContractCall;
    }

    function addTokens(
        address _collateralType,
        uint256 _amount
    ) external payable;

    function stake(uint256 _amount) external;

    function getGovernanceStake(
        address _staker
    ) external view returns (Stake memory);

    function getTotalPoolAmount() external view returns (uint256);

    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) external returns (bool);

    function updateLastGovContractCall(address _voter) external;
}

// File: contracts/contracts/contracts/libraries/Math.sol


// Copied from https://github.com/dapphub/ds-math
// Added the div function from openzepeling safeMath

pragma solidity 0.8.20;

library DSMath {
    function add(uint256 x, uint256 y) internal pure returns (uint256 z) {
        require((z = x + y) >= x, "ds-math-add-overflow");
    }

    function sub(uint256 x, uint256 y) internal pure returns (uint256 z) {
        require((z = x - y) <= x, "ds-math-sub-underflow");
    }

    // slither-disable-next-line incorrect-equality
    function mul(uint256 x, uint256 y) internal pure returns (uint256 z) {
        require(y == 0 || (z = x * y) / y == x, "ds-math-mul-overflow");
    }

    /**
     * @dev Returns the integer division of two unsigned integers. Reverts on
     * division by zero. The result is rounded towards zero.
     *
     * Counterpart to Solidity's `/` operator. Note: this function uses a
     * `revert` opcode (which leaves remaining gas untouched) while Solidity
     * uses an invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return div(a, b, "SafeMath: division by zero");
    }

    /**
     * @dev Returns the integer division of two unsigned integers. Reverts with custom message on
     * division by zero. The result is rounded towards zero.
     *
     * Counterpart to Solidity's `/` operator. Note: this function uses a
     * `revert` opcode (which leaves remaining gas untouched) while Solidity
     * uses an invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function div(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        require(b > 0, errorMessage);
        uint256 c = a / b;
        // assert(a == b * c + a % b); // There is no case in which this doesn't hold

        return c;
    }

    uint256 public constant WAD = 10 ** 18;
    uint256 public constant RAY = 10 ** 28;

    //rounds to zero if x*y < WAD / 2
    function wmul(uint256 x, uint256 y) internal pure returns (uint256 z) {
        z = add(mul(x, y), WAD / 2) / WAD;
    }

    //rounds to zero if x*y < WAD / 2
    function rmul(uint256 x, uint256 y) internal pure returns (uint256 z) {
        z = add(mul(x, y), RAY / 2) / RAY;
    }

    //rounds to zero if x*y < WAD / 2
    function wdiv(uint256 x, uint256 y) internal pure returns (uint256 z) {
        z = add(mul(x, WAD), y / 2) / y;
    }

    //rounds to zero if x*y < RAY / 2
    function rdiv(uint256 x, uint256 y) internal pure returns (uint256 z) {
        z = mul((x / mul(y, RAY)), WAD);
    }

    // This famous algorithm is called "exponentiation by squaring"
    // and calculates x^n with x as fixed-point and n as regular unsigned.
    //
    // It's O(log n), instead of O(n) for naive repeated multiplication.
    //
    // These facts are why it works:
    //
    //  If n is even, then x^n = (x^2)^(n/2).
    //  If n is odd,  then x^n = x * x^(n-1),
    //   and applying the equation for even x gives
    //    x^n = x * (x^2)^((n-1) / 2).
    //
    //  Also, EVM division is flooring and
    //    floor[(n-1) / 2] = floor[n / 2].
    //
    // Not sure if this is a false positive - https://github.com/dapphub/ds-math/issues/18
    // slither-disable-next-line weak-prng
    function rpow(uint256 x, uint256 n) internal pure returns (uint256 z) {
        z = n % 2 != 0 ? x : RAY;

        for (n /= 2; n != 0; n /= 2) {
            x = rmul(x, x);

            if (n % 2 != 0) {
                z = rmul(z, x);
            }
        }
    }

    // babylonian method (https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Babylonian_method)
    function sqrt(uint y) internal pure returns (uint z) {
        if (y > 3) {
            z = y;
            uint x = y / 2 + 1;
            while (x < z) {
                z = x;
                x = (y / x + x) / 2;
            }
        } else if (y != 0) {
            z = 1;
        }
    }

    function min(uint x, uint y) internal pure returns (uint z) {
        return x <= y ? x : y;
    }
}

// File: contracts/contracts/contracts/Governance.sol

pragma solidity 0.8.20;




contract Governance is ERC20 {
    using DSMath for uint;

    bool initialized;

    IStabilityModule stabilityModule;
    address public rewardDistributor;
    address public immutable team;
    address public vestingContract;

    uint256 public lastMintTimestamp = 0;
    uint256 public totalDailyMinted;
    uint256 public constant DAILY_MINT_CAP = 72e24;
    uint256 public voteCount;

    struct Vote {
        uint256 startTime;
        uint256 tallyTime;
        uint256 amountSupporting;
        uint256 amountAgainst;
        uint256 amountAbstained;
        bool executed;
        bool result;
        bytes4 voteFunction;
        address voteAddress;
        address initiator;
        bytes data;
        mapping(address => bool) voted;
    }

    struct VestingSchedule {
        uint256 totalAmount;
        uint256 cliffEnd;
        uint256 vestingEnd;
        uint256 lastClaimTime;
        uint256 monthlyAmount;
        uint256 amountClaimed;
        uint8 schedule;
    }

    mapping(uint256 => Vote) public voteInfo;
    mapping(address => uint256) public lastVoteTimestamp;
    mapping(address => mapping(uint8 => VestingSchedule))
        public vestingSchedules;
    mapping(address => uint8[]) public userSchedules;

    event VoteProposed(
        address indexed user,
        uint256 voteId,
        uint256 timestamp,
        address indexed voteAddress
    );
    event Voted(address indexed user, uint256 voteCount);
    event VoteExecuted(address executor, bool result);

    /// @notice Thrown when address zero is provided
    error ZeroAddress();

    modifier onlyVoter() {
        require(
            (stabilityModule.getGovernanceStake(msg.sender).startTime >
                block.timestamp - 90 days) ||
                (stabilityModule
                    .getGovernanceStake(msg.sender)
                    .lastGovContractCall > block.timestamp - 90 days) ||
                (lastVoteTimestamp[msg.sender] > block.timestamp - 90 days),
            "stake is inactive, hasn't been used in 3 months"
        );

        require(
            stabilityModule.getGovernanceStake(msg.sender).startTime <
                block.timestamp - 30 days,
            "stake must be at least 30 days old!"
        );
        _;
    }

    modifier mustInit() {
        require(initialized, "contract is not initialized");
        _;
    }

    modifier onlyTeam() {
        require(msg.sender == team, "can only be call by team");
        _;
    }

    modifier voteExists(uint256 numberOfVote) {
        require(numberOfVote <= voteCount, "Vote does not exist");
        _;
    }

    constructor(address _team) ERC20("GOV", "GOV") {
        require(_team != address(0), "invalid address");
        _mint(_team, 14e25);

        team = _team;
    }

    function calculateAvailableTokens(
        address account,
        uint8 _schedule
    ) public view returns (uint256) {
        VestingSchedule storage schedule = vestingSchedules[account][_schedule];

        if (block.timestamp < schedule.cliffEnd) return 0;
        if (block.timestamp >= schedule.vestingEnd) return schedule.totalAmount;

        uint256 monthsSinceLastClaim = (block.timestamp -
            schedule.lastClaimTime) / 30 days;
        uint256 newVestedAmount = monthsSinceLastClaim * schedule.monthlyAmount;
        uint256 totalVested = schedule.amountClaimed + newVestedAmount;

        return
            totalVested > schedule.totalAmount
                ? schedule.totalAmount
                : totalVested;
    }

    function init(
        address rewardDistributorAddress,
        address stabilityModuleAddress,
        address vestingAddress
    ) external onlyTeam {
        require(!initialized, "contract is initialized");
        if (rewardDistributorAddress == address(0)) revert ZeroAddress();
        if (stabilityModuleAddress == address(0)) revert ZeroAddress();
        if (vestingAddress == address(0)) revert ZeroAddress();

        rewardDistributor = rewardDistributorAddress;
        stabilityModule = IStabilityModule(stabilityModuleAddress);
        vestingContract = vestingAddress;
        initialized = true;

        stabilityModule.updateLastGovContractCall(msg.sender);
    }
    //slither-disable-next-line divide-before-multiply
    function mintDaily() external mustInit onlyTeam {
        uint256 dailyMintAmount = 1e23;
        uint256 numDays = lastMintTimestamp == 0
            ? 1
            : (block.timestamp - lastMintTimestamp) / 1 days;

        require(numDays != 0, "number of days cannot be 0");

        uint256 totalMintAmount = dailyMintAmount * numDays;
        require(
            totalDailyMinted + totalMintAmount <= DAILY_MINT_CAP,
            "Daily mint cap of 72M reached"
        );

        //90,000 minted to reward distributor to handle incentives distribution
        _mint(rewardDistributor, 9e22 * numDays);

        //10,000 minted to a reserve
        _mint(address(team), 1e22 * numDays);

        totalDailyMinted += totalMintAmount;
        lastMintTimestamp = block.timestamp;

        stabilityModule.updateLastGovContractCall(msg.sender);
    }
    //slither-disable-next-line uninitialized-state
    function proposeVote(
        address voteAddress,
        bytes4 voteFunction,
        bytes memory voteData
    ) external onlyVoter mustInit {
        require(
            stabilityModule.getGovernanceStake(msg.sender).amount >
                stabilityModule.getTotalPoolAmount() / 10,
            "user needs to stake more tokens in pool to start vote!"
        );

        voteCount++;
        uint256 voteId = voteCount;
        Vote storage _thisVote = voteInfo[voteId];
        _thisVote.initiator = msg.sender;
        _thisVote.startTime = block.timestamp;
        _thisVote.tallyTime = 0;
        _thisVote.voteAddress = voteAddress;
        _thisVote.voteFunction = voteFunction;
        _thisVote.data = voteData;

        stabilityModule.updateLastGovContractCall(msg.sender);

        emit VoteProposed(
            msg.sender,
            voteId,
            _thisVote.startTime,
            _thisVote.voteAddress
        );
    }

    function executeVote(
        uint256 numberOfVote
    ) external onlyVoter mustInit voteExists(numberOfVote) {
        //75 percent of pool needs to vote

        Vote storage v = voteInfo[numberOfVote];

        require(
            v.amountSupporting + v.amountAgainst + v.amountAbstained >
                (stabilityModule.getTotalPoolAmount() * 3) / 4,
            "75% of pool has not voted yet!"
        );
        require(v.tallyTime == 0, "Vote has already been tallied");
        uint256 _duration = 2 days;
        require(
            block.timestamp - v.startTime > _duration,
            "Time for voting has not elapsed"
        );

        if (
            v.amountSupporting >
            (stabilityModule.getTotalPoolAmount() * 51) / 100
        ) {
            v.result = true;
            address _destination = v.voteAddress;
            bool _succ;
            bytes memory _res;
            (_succ, _res) = _destination.call(
                abi.encodePacked(v.voteFunction, v.data)
            );
            //When testing _destination.call can require higher gas than the standard. Be sure to increase the gas if it fails.
            require(_succ, "error running _destination.call");
        } else {
            v.result = false;
        }

        v.executed = true;
        v.tallyTime = block.timestamp;
        stabilityModule.updateLastGovContractCall(msg.sender);

        emit VoteExecuted(msg.sender, v.result);
    }

    function vote(
        uint256 numberOfVote,
        bool isSupports,
        bool isAbstains
    ) external onlyVoter mustInit voteExists(numberOfVote) {
        Vote storage v = voteInfo[numberOfVote];
        require(v.tallyTime == 0, "Vote has already been tallied");
        require(!v.voted[msg.sender], "Sender has already voted");

        v.voted[msg.sender] = true;
        if (isAbstains) {
            v.amountAbstained += balanceOf(msg.sender);
        } else if (isSupports) {
            v.amountSupporting += balanceOf(msg.sender);
        } else {
            v.amountAgainst += balanceOf(msg.sender);
        }

        stabilityModule.updateLastGovContractCall(msg.sender);
        lastVoteTimestamp[msg.sender] = block.timestamp;

        emit Voted(msg.sender, numberOfVote);
    }

    function setVestingSchedule(
        address investor,
        uint256 amount,
        uint8 schedule,
        uint256 vestingDuration,
        uint256 cliffDuration
    ) external {
        require(msg.sender == vestingContract, "Only vesting contract");
        require(investor != address(0), "Invalid investor address");

        uint256 monthlyAmount = (amount * 30 days) / vestingDuration;

        vestingSchedules[investor][schedule] = VestingSchedule({
            totalAmount: amount,
            cliffEnd: block.timestamp + cliffDuration,
            vestingEnd: block.timestamp + cliffDuration + vestingDuration,
            lastClaimTime: block.timestamp + cliffDuration,
            monthlyAmount: monthlyAmount,
            amountClaimed: 0,
            schedule: schedule
        });

        userSchedules[investor].push(schedule);
    }

    function getUserVestingSchedules(
        address user
    ) external view returns (uint8[] memory) {
        return userSchedules[user];
    }

    function _update(
        address from,
        address to,
        uint256 value
    ) internal virtual override {
        if (from != address(0) && from != vestingContract) {
            uint256 totalVested = 0;
            uint256 vestedBalance = 0;

            uint8[] storage schedules = userSchedules[from];
            if (schedules.length > 0) {
                for (uint i = 0; i < schedules.length; i++) {
                    VestingSchedule storage schedule = vestingSchedules[from][
                        schedules[i]
                    ];
                    if (block.timestamp >= schedule.cliffEnd) {
                        totalVested += calculateAvailableTokens(
                            from,
                            schedules[i]
                        );
                    }
                    vestedBalance += schedule.totalAmount;
                }

                uint256 freeBalance = balanceOf(from) > vestedBalance
                    ? balanceOf(from) - vestedBalance
                    : 0;
                require(
                    value <= freeBalance + totalVested,
                    "Amount exceeds available tokens"
                );
            }
        }
        super._update(from, to, value);
    }
}