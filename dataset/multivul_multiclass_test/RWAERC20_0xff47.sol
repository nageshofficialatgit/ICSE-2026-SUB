// SPDX-License-Identifier: MIT



// File @openzeppelin/contracts/access/IAccessControl.sol@v5.2.0

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (access/IAccessControl.sol)

pragma solidity ^0.8.20;

/**
 * @dev External interface of AccessControl declared to support ERC-165 detection.
 */
interface IAccessControl {
    /**
     * @dev The `account` is missing a role.
     */
    error AccessControlUnauthorizedAccount(address account, bytes32 neededRole);

    /**
     * @dev The caller of a function is not the expected one.
     *
     * NOTE: Don't confuse with {AccessControlUnauthorizedAccount}.
     */
    error AccessControlBadConfirmation();

    /**
     * @dev Emitted when `newAdminRole` is set as ``role``'s admin role, replacing `previousAdminRole`
     *
     * `DEFAULT_ADMIN_ROLE` is the starting admin for all roles, despite
     * {RoleAdminChanged} not being emitted signaling this.
     */
    event RoleAdminChanged(bytes32 indexed role, bytes32 indexed previousAdminRole, bytes32 indexed newAdminRole);

    /**
     * @dev Emitted when `account` is granted `role`.
     *
     * `sender` is the account that originated the contract call. This account bears the admin role (for the granted role).
     * Expected in cases where the role was granted using the internal {AccessControl-_grantRole}.
     */
    event RoleGranted(bytes32 indexed role, address indexed account, address indexed sender);

    /**
     * @dev Emitted when `account` is revoked `role`.
     *
     * `sender` is the account that originated the contract call:
     *   - if using `revokeRole`, it is the admin role bearer
     *   - if using `renounceRole`, it is the role bearer (i.e. `account`)
     */
    event RoleRevoked(bytes32 indexed role, address indexed account, address indexed sender);

    /**
     * @dev Returns `true` if `account` has been granted `role`.
     */
    function hasRole(bytes32 role, address account) external view returns (bool);

    /**
     * @dev Returns the admin role that controls `role`. See {grantRole} and
     * {revokeRole}.
     *
     * To change a role's admin, use {AccessControl-_setRoleAdmin}.
     */
    function getRoleAdmin(bytes32 role) external view returns (bytes32);

    /**
     * @dev Grants `role` to `account`.
     *
     * If `account` had not been already granted `role`, emits a {RoleGranted}
     * event.
     *
     * Requirements:
     *
     * - the caller must have ``role``'s admin role.
     */
    function grantRole(bytes32 role, address account) external;

    /**
     * @dev Revokes `role` from `account`.
     *
     * If `account` had been granted `role`, emits a {RoleRevoked} event.
     *
     * Requirements:
     *
     * - the caller must have ``role``'s admin role.
     */
    function revokeRole(bytes32 role, address account) external;

    /**
     * @dev Revokes `role` from the calling account.
     *
     * Roles are often managed via {grantRole} and {revokeRole}: this function's
     * purpose is to provide a mechanism for accounts to lose their privileges
     * if they are compromised (such as when a trusted device is misplaced).
     *
     * If the calling account had been granted `role`, emits a {RoleRevoked}
     * event.
     *
     * Requirements:
     *
     * - the caller must be `callerConfirmation`.
     */
    function renounceRole(bytes32 role, address callerConfirmation) external;
}


// File @openzeppelin/contracts/utils/Context.sol@v5.2.0

// Original license: SPDX_License_Identifier: MIT
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


// File @openzeppelin/contracts/utils/introspection/IERC165.sol@v5.2.0

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/introspection/IERC165.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC-165 standard, as defined in the
 * https://eips.ethereum.org/EIPS/eip-165[ERC].
 *
 * Implementers can declare support of contract interfaces, which can then be
 * queried by others ({ERC165Checker}).
 *
 * For an implementation, see {ERC165}.
 */
interface IERC165 {
    /**
     * @dev Returns true if this contract implements the interface defined by
     * `interfaceId`. See the corresponding
     * https://eips.ethereum.org/EIPS/eip-165#how-interfaces-are-identified[ERC section]
     * to learn more about how these ids are created.
     *
     * This function call must use less than 30 000 gas.
     */
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}


// File @openzeppelin/contracts/utils/introspection/ERC165.sol@v5.2.0

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/introspection/ERC165.sol)

pragma solidity ^0.8.20;

/**
 * @dev Implementation of the {IERC165} interface.
 *
 * Contracts that want to implement ERC-165 should inherit from this contract and override {supportsInterface} to check
 * for the additional interface id that will be supported. For example:
 *
 * ```solidity
 * function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
 *     return interfaceId == type(MyInterface).interfaceId || super.supportsInterface(interfaceId);
 * }
 * ```
 */
abstract contract ERC165 is IERC165 {
    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual returns (bool) {
        return interfaceId == type(IERC165).interfaceId;
    }
}


// File @openzeppelin/contracts/access/AccessControl.sol@v5.2.0

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (access/AccessControl.sol)

pragma solidity ^0.8.20;



/**
 * @dev Contract module that allows children to implement role-based access
 * control mechanisms. This is a lightweight version that doesn't allow enumerating role
 * members except through off-chain means by accessing the contract event logs. Some
 * applications may benefit from on-chain enumerability, for those cases see
 * {AccessControlEnumerable}.
 *
 * Roles are referred to by their `bytes32` identifier. These should be exposed
 * in the external API and be unique. The best way to achieve this is by
 * using `public constant` hash digests:
 *
 * ```solidity
 * bytes32 public constant MY_ROLE = keccak256("MY_ROLE");
 * ```
 *
 * Roles can be used to represent a set of permissions. To restrict access to a
 * function call, use {hasRole}:
 *
 * ```solidity
 * function foo() public {
 *     require(hasRole(MY_ROLE, msg.sender));
 *     ...
 * }
 * ```
 *
 * Roles can be granted and revoked dynamically via the {grantRole} and
 * {revokeRole} functions. Each role has an associated admin role, and only
 * accounts that have a role's admin role can call {grantRole} and {revokeRole}.
 *
 * By default, the admin role for all roles is `DEFAULT_ADMIN_ROLE`, which means
 * that only accounts with this role will be able to grant or revoke other
 * roles. More complex role relationships can be created by using
 * {_setRoleAdmin}.
 *
 * WARNING: The `DEFAULT_ADMIN_ROLE` is also its own admin: it has permission to
 * grant and revoke this role. Extra precautions should be taken to secure
 * accounts that have been granted it. We recommend using {AccessControlDefaultAdminRules}
 * to enforce additional security measures for this role.
 */
abstract contract AccessControl is Context, IAccessControl, ERC165 {
    struct RoleData {
        mapping(address account => bool) hasRole;
        bytes32 adminRole;
    }

    mapping(bytes32 role => RoleData) private _roles;

    bytes32 public constant DEFAULT_ADMIN_ROLE = 0x00;

    /**
     * @dev Modifier that checks that an account has a specific role. Reverts
     * with an {AccessControlUnauthorizedAccount} error including the required role.
     */
    modifier onlyRole(bytes32 role) {
        _checkRole(role);
        _;
    }

    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
        return interfaceId == type(IAccessControl).interfaceId || super.supportsInterface(interfaceId);
    }

    /**
     * @dev Returns `true` if `account` has been granted `role`.
     */
    function hasRole(bytes32 role, address account) public view virtual returns (bool) {
        return _roles[role].hasRole[account];
    }

    /**
     * @dev Reverts with an {AccessControlUnauthorizedAccount} error if `_msgSender()`
     * is missing `role`. Overriding this function changes the behavior of the {onlyRole} modifier.
     */
    function _checkRole(bytes32 role) internal view virtual {
        _checkRole(role, _msgSender());
    }

    /**
     * @dev Reverts with an {AccessControlUnauthorizedAccount} error if `account`
     * is missing `role`.
     */
    function _checkRole(bytes32 role, address account) internal view virtual {
        if (!hasRole(role, account)) {
            revert AccessControlUnauthorizedAccount(account, role);
        }
    }

    /**
     * @dev Returns the admin role that controls `role`. See {grantRole} and
     * {revokeRole}.
     *
     * To change a role's admin, use {_setRoleAdmin}.
     */
    function getRoleAdmin(bytes32 role) public view virtual returns (bytes32) {
        return _roles[role].adminRole;
    }

    /**
     * @dev Grants `role` to `account`.
     *
     * If `account` had not been already granted `role`, emits a {RoleGranted}
     * event.
     *
     * Requirements:
     *
     * - the caller must have ``role``'s admin role.
     *
     * May emit a {RoleGranted} event.
     */
    function grantRole(bytes32 role, address account) public virtual onlyRole(getRoleAdmin(role)) {
        _grantRole(role, account);
    }

    /**
     * @dev Revokes `role` from `account`.
     *
     * If `account` had been granted `role`, emits a {RoleRevoked} event.
     *
     * Requirements:
     *
     * - the caller must have ``role``'s admin role.
     *
     * May emit a {RoleRevoked} event.
     */
    function revokeRole(bytes32 role, address account) public virtual onlyRole(getRoleAdmin(role)) {
        _revokeRole(role, account);
    }

    /**
     * @dev Revokes `role` from the calling account.
     *
     * Roles are often managed via {grantRole} and {revokeRole}: this function's
     * purpose is to provide a mechanism for accounts to lose their privileges
     * if they are compromised (such as when a trusted device is misplaced).
     *
     * If the calling account had been revoked `role`, emits a {RoleRevoked}
     * event.
     *
     * Requirements:
     *
     * - the caller must be `callerConfirmation`.
     *
     * May emit a {RoleRevoked} event.
     */
    function renounceRole(bytes32 role, address callerConfirmation) public virtual {
        if (callerConfirmation != _msgSender()) {
            revert AccessControlBadConfirmation();
        }

        _revokeRole(role, callerConfirmation);
    }

    /**
     * @dev Sets `adminRole` as ``role``'s admin role.
     *
     * Emits a {RoleAdminChanged} event.
     */
    function _setRoleAdmin(bytes32 role, bytes32 adminRole) internal virtual {
        bytes32 previousAdminRole = getRoleAdmin(role);
        _roles[role].adminRole = adminRole;
        emit RoleAdminChanged(role, previousAdminRole, adminRole);
    }

    /**
     * @dev Attempts to grant `role` to `account` and returns a boolean indicating if `role` was granted.
     *
     * Internal function without access restriction.
     *
     * May emit a {RoleGranted} event.
     */
    function _grantRole(bytes32 role, address account) internal virtual returns (bool) {
        if (!hasRole(role, account)) {
            _roles[role].hasRole[account] = true;
            emit RoleGranted(role, account, _msgSender());
            return true;
        } else {
            return false;
        }
    }

    /**
     * @dev Attempts to revoke `role` to `account` and returns a boolean indicating if `role` was revoked.
     *
     * Internal function without access restriction.
     *
     * May emit a {RoleRevoked} event.
     */
    function _revokeRole(bytes32 role, address account) internal virtual returns (bool) {
        if (hasRole(role, account)) {
            _roles[role].hasRole[account] = false;
            emit RoleRevoked(role, account, _msgSender());
            return true;
        } else {
            return false;
        }
    }
}


// File @openzeppelin/contracts/interfaces/draft-IERC6093.sol@v5.2.0

// Original license: SPDX_License_Identifier: MIT
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


// File @openzeppelin/contracts/token/ERC20/IERC20.sol@v5.2.0

// Original license: SPDX_License_Identifier: MIT
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


// File @openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol@v5.2.0

// Original license: SPDX_License_Identifier: MIT
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


// File @openzeppelin/contracts/token/ERC20/ERC20.sol@v5.2.0

// Original license: SPDX_License_Identifier: MIT
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


// File @openzeppelin/contracts/utils/Pausable.sol@v5.2.0

// Original license: SPDX_License_Identifier: MIT
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


// File contracts/access/CustodianRole.sol

// Original license: SPDX_License_Identifier: MIT

pragma solidity ^0.8.20;

/**
 * @title CustodianRole
 * @dev A contract to add role based access control for RWA.
 */
abstract contract CustodianRole is AccessControl {
    bytes32 public constant ISSUER_ROLE = keccak256("ISSUER");
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER");
    bytes32 public constant AGENT_ROLE = keccak256("AGENT");
    bytes32 public constant MANAGER_ROLE = keccak256("MANAGER");

    /**
     * @dev Initializes the contract setting roles for the deployer address.
     */
    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, _msgSender());
        _grantRole(ISSUER_ROLE, _msgSender());
        _grantRole(VERIFIER_ROLE, _msgSender());
        _grantRole(AGENT_ROLE, _msgSender());
        _grantRole(MANAGER_ROLE, _msgSender());
    }

    /**
     * @dev Modifier that checks if an account has the ISSUER role.
     */
    modifier onlyIssuer() {
        _checkRole(ISSUER_ROLE);
        _;
    }

    /**
     * @dev Modifier that checks if an account has the VERIFIER role.
     */
    modifier onlyVerifier() {
        _checkRole(VERIFIER_ROLE);
        _;
    }

    /**
     * @dev Modifier that checks if an account has the AGENT role.
     */
    modifier onlyAgent() {
        _checkRole(AGENT_ROLE);
        _;
    }

    /**
     * @dev Modifier that checks if an account has the MANAGER role.
     */
    modifier onlyManager() {
        _checkRole(MANAGER_ROLE);
        _;
    }
}


// File @openzeppelin/contracts/utils/structs/EnumerableSet.sol@v5.2.0

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/structs/EnumerableSet.sol)
// This file was procedurally generated from scripts/generate/templates/EnumerableSet.js.

pragma solidity ^0.8.20;

/**
 * @dev Library for managing
 * https://en.wikipedia.org/wiki/Set_(abstract_data_type)[sets] of primitive
 * types.
 *
 * Sets have the following properties:
 *
 * - Elements are added, removed, and checked for existence in constant time
 * (O(1)).
 * - Elements are enumerated in O(n). No guarantees are made on the ordering.
 *
 * ```solidity
 * contract Example {
 *     // Add the library methods
 *     using EnumerableSet for EnumerableSet.AddressSet;
 *
 *     // Declare a set state variable
 *     EnumerableSet.AddressSet private mySet;
 * }
 * ```
 *
 * As of v3.3.0, sets of type `bytes32` (`Bytes32Set`), `address` (`AddressSet`)
 * and `uint256` (`UintSet`) are supported.
 *
 * [WARNING]
 * ====
 * Trying to delete such a structure from storage will likely result in data corruption, rendering the structure
 * unusable.
 * See https://github.com/ethereum/solidity/pull/11843[ethereum/solidity#11843] for more info.
 *
 * In order to clean an EnumerableSet, you can either remove all elements one by one or create a fresh instance using an
 * array of EnumerableSet.
 * ====
 */
library EnumerableSet {
    // To implement this library for multiple types with as little code
    // repetition as possible, we write it in terms of a generic Set type with
    // bytes32 values.
    // The Set implementation uses private functions, and user-facing
    // implementations (such as AddressSet) are just wrappers around the
    // underlying Set.
    // This means that we can only create new EnumerableSets for types that fit
    // in bytes32.

    struct Set {
        // Storage of set values
        bytes32[] _values;
        // Position is the index of the value in the `values` array plus 1.
        // Position 0 is used to mean a value is not in the set.
        mapping(bytes32 value => uint256) _positions;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function _add(Set storage set, bytes32 value) private returns (bool) {
        if (!_contains(set, value)) {
            set._values.push(value);
            // The value is stored at length-1, but we add 1 to all indexes
            // and use 0 as a sentinel value
            set._positions[value] = set._values.length;
            return true;
        } else {
            return false;
        }
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function _remove(Set storage set, bytes32 value) private returns (bool) {
        // We cache the value's position to prevent multiple reads from the same storage slot
        uint256 position = set._positions[value];

        if (position != 0) {
            // Equivalent to contains(set, value)
            // To delete an element from the _values array in O(1), we swap the element to delete with the last one in
            // the array, and then remove the last element (sometimes called as 'swap and pop').
            // This modifies the order of the array, as noted in {at}.

            uint256 valueIndex = position - 1;
            uint256 lastIndex = set._values.length - 1;

            if (valueIndex != lastIndex) {
                bytes32 lastValue = set._values[lastIndex];

                // Move the lastValue to the index where the value to delete is
                set._values[valueIndex] = lastValue;
                // Update the tracked position of the lastValue (that was just moved)
                set._positions[lastValue] = position;
            }

            // Delete the slot where the moved value was stored
            set._values.pop();

            // Delete the tracked position for the deleted slot
            delete set._positions[value];

            return true;
        } else {
            return false;
        }
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function _contains(Set storage set, bytes32 value) private view returns (bool) {
        return set._positions[value] != 0;
    }

    /**
     * @dev Returns the number of values on the set. O(1).
     */
    function _length(Set storage set) private view returns (uint256) {
        return set._values.length;
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function _at(Set storage set, uint256 index) private view returns (bytes32) {
        return set._values[index];
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function _values(Set storage set) private view returns (bytes32[] memory) {
        return set._values;
    }

    // Bytes32Set

    struct Bytes32Set {
        Set _inner;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function add(Bytes32Set storage set, bytes32 value) internal returns (bool) {
        return _add(set._inner, value);
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function remove(Bytes32Set storage set, bytes32 value) internal returns (bool) {
        return _remove(set._inner, value);
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(Bytes32Set storage set, bytes32 value) internal view returns (bool) {
        return _contains(set._inner, value);
    }

    /**
     * @dev Returns the number of values in the set. O(1).
     */
    function length(Bytes32Set storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function at(Bytes32Set storage set, uint256 index) internal view returns (bytes32) {
        return _at(set._inner, index);
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function values(Bytes32Set storage set) internal view returns (bytes32[] memory) {
        bytes32[] memory store = _values(set._inner);
        bytes32[] memory result;

        assembly ("memory-safe") {
            result := store
        }

        return result;
    }

    // AddressSet

    struct AddressSet {
        Set _inner;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function add(AddressSet storage set, address value) internal returns (bool) {
        return _add(set._inner, bytes32(uint256(uint160(value))));
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function remove(AddressSet storage set, address value) internal returns (bool) {
        return _remove(set._inner, bytes32(uint256(uint160(value))));
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(AddressSet storage set, address value) internal view returns (bool) {
        return _contains(set._inner, bytes32(uint256(uint160(value))));
    }

    /**
     * @dev Returns the number of values in the set. O(1).
     */
    function length(AddressSet storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function at(AddressSet storage set, uint256 index) internal view returns (address) {
        return address(uint160(uint256(_at(set._inner, index))));
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function values(AddressSet storage set) internal view returns (address[] memory) {
        bytes32[] memory store = _values(set._inner);
        address[] memory result;

        assembly ("memory-safe") {
            result := store
        }

        return result;
    }

    // UintSet

    struct UintSet {
        Set _inner;
    }

    /**
     * @dev Add a value to a set. O(1).
     *
     * Returns true if the value was added to the set, that is if it was not
     * already present.
     */
    function add(UintSet storage set, uint256 value) internal returns (bool) {
        return _add(set._inner, bytes32(value));
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function remove(UintSet storage set, uint256 value) internal returns (bool) {
        return _remove(set._inner, bytes32(value));
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(UintSet storage set, uint256 value) internal view returns (bool) {
        return _contains(set._inner, bytes32(value));
    }

    /**
     * @dev Returns the number of values in the set. O(1).
     */
    function length(UintSet storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

    /**
     * @dev Returns the value stored at position `index` in the set. O(1).
     *
     * Note that there are no guarantees on the ordering of values inside the
     * array, and it may change when more values are added or removed.
     *
     * Requirements:
     *
     * - `index` must be strictly less than {length}.
     */
    function at(UintSet storage set, uint256 index) internal view returns (uint256) {
        return uint256(_at(set._inner, index));
    }

    /**
     * @dev Return the entire set in an array
     *
     * WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed
     * to mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that
     * this function has an unbounded cost, and using it as part of a state-changing function may render the function
     * uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
     */
    function values(UintSet storage set) internal view returns (uint256[] memory) {
        bytes32[] memory store = _values(set._inner);
        uint256[] memory result;

        assembly ("memory-safe") {
            result := store
        }

        return result;
    }
}


// File contracts/token/ERC20/extensions/ERC20Mintable.sol

// Original license: SPDX_License_Identifier: MIT

pragma solidity ^0.8.20;

/**
 * @title ERC20Mintable
 * @dev Extension of ERC20 that adds a minting behavior.
 */
abstract contract ERC20Mintable is ERC20 {
    // indicates if minting is finished
    bool private _mintingFinished = false;

    /**
     * @dev Emitted during finish minting.
     */
    event MintFinished();

    /**
     * @dev Indicates a failure in minting as it has been finished.
     */
    error ERC20MintingFinished();

    /**
     * @dev Returns if minting is finished or not.
     */
    function mintingFinished() external view returns (bool) {
        return _mintingFinished;
    }

    /**
     * @dev Function to stop minting new tokens.
     *
     * WARNING: it allows everyone to finish minting. Access controls MUST be defined in derived contracts.
     */
    function _finishMinting() internal {
        if (_mintingFinished) {
            revert ERC20MintingFinished();
        }

        _mintingFinished = true;

        emit MintFinished();
    }

    /**
     * @dev Requires that minting is not finished.
     * Otherwise revert a mint.
     */
    function _update(address from, address to, uint256 value) internal virtual override {
        if (from == address(0) && _mintingFinished) {
            revert ERC20MintingFinished();
        }
        super._update(from, to, value);
    }
}


// File contracts/utils/ArrayUtils.sol

// Original license: SPDX_License_Identifier: MIT

pragma solidity ^0.8.20;

library ArrayUtils {
    /**
     * @dev The operation failed because the lengths of the provided arrays do not match.
     */
    error MismatchedArrayLengths();
}


// File contracts/token/ERC20/extensions/ERC20Authorized.sol

// Original license: SPDX_License_Identifier: MIT

pragma solidity ^0.8.20;

/**
 * @title ERC20Authorized
 * @dev Extension of ERC20 that adds:
 *
 * - an authorization status for addresses to receive and transfer tokens, including minting and burning.
 * - a freeze status for addresses to receive and transfer tokens, including minting and burning, also if authorized.
 */
abstract contract ERC20Authorized is ERC20Mintable {
    using EnumerableSet for EnumerableSet.AddressSet;

    // set of authorized addresses
    EnumerableSet.AddressSet private _authorizedAccounts;

    // mapping of frozen addresses
    mapping(address => bool) private _isFrozen;

    /**
     * @dev Emitted when an `account` is authorized to transfer tokens. Not emitted if already authorized.
     */
    event AccountAuthorized(address indexed account);

    /**
     * @dev Emitted when an `account` is unauthorized to transfer tokens. Not emitted if already unauthorized.
     */
    event AccountUnauthorized(address indexed account);

    /**
     * @dev Emitted when an `account` is frozen to transfer tokens. Not emitted if already frozen.
     */
    event AccountFrozen(address indexed account);

    /**
     * @dev Emitted when an `account` is unfrozen to transfer tokens. Not emitted if already unfrozen.
     */
    event AccountUnfrozen(address indexed account);

    /**
     * @dev The operation failed because the `account` is not authorized to transfer or frozen.
     * @param account The address from which the tokens are being transferred.
     */
    error ERC20UnauthorizedTransfer(address account);

    /**
     * @dev Authorize the deployer to transfer tokens.
     */
    constructor() {
        _setAuthorization(_msgSender(), true);
    }

    /**
     * @dev Returns the status of authorization for a given account.
     */
    function isAuthorized(address account) external view returns (bool) {
        return _authorizedAccounts.contains(account);
    }

    /**
     * @dev Returns the status of freeze for a given account.
     */
    function isFrozen(address account) external view returns (bool) {
        return _isFrozen[account];
    }

    /**
     * @dev Returns the number of authorized addresses.
     */
    function authorizedAccountsCount() external view returns (uint256) {
        return _authorizedAccounts.length();
    }

    /**
     * @dev Returns the authorized address at a specific index.
     * @param index The index in the authorized accounts list.
     */
    function authorizedAccountAt(uint256 index) external view returns (address) {
        return _authorizedAccounts.at(index);
    }

    /**
     * @dev Returns an array of all authorized addresses.
     */
    function authorizedAccounts() external view returns (address[] memory) {
        return _authorizedAccounts.values();
    }

    /**
     * @dev Sets the authorization status to receive and transfer tokens, including minting and burning.
     *
     * WARNING: it allows everyone to set the status. Access controls MUST be defined in derived contracts.
     *
     * @param account The address that will be authorized or not.
     * @param status The status of authorization.
     */
    function _setAuthorization(address account, bool status) internal {
        if (status) {
            if (_authorizedAccounts.add(account)) {
                emit AccountAuthorized(account);
            }
        } else {
            if (_authorizedAccounts.remove(account)) {
                emit AccountUnauthorized(account);
            }
        }
    }

    /**
     * @dev Sets the authorization status to receive and transfer tokens, including minting and burning.
     *
     * WARNING: it allows everyone to set the status. Access controls MUST be defined in derived contracts.
     *
     * @param accountList The addresses that will be authorized or not.
     * @param status The status of authorization.
     */
    function _setAuthorizationBatch(address[] calldata accountList, bool status) internal {
        uint256 length = accountList.length;
        for (uint256 i = 0; i < length; ++i) {
            _setAuthorization(accountList[i], status);
        }
    }

    /**
     * @dev Sets the freeze status for receiving and transfer tokens, including minting and burning.
     * Also authorized addresses can be frozen.
     *
     * WARNING: it allows everyone to set the status. Access controls MUST be defined in derived contracts.
     *
     * @param account The address that will be frozen or not.
     * @param status The status of authorization.
     */
    function _setFreeze(address account, bool status) internal {
        if (_isFrozen[account] != status) {
            _isFrozen[account] = status;
            if (status) {
                emit AccountFrozen(account);
            } else {
                emit AccountUnfrozen(account);
            }
        }
    }

    /**
     * @dev Sets the freeze status for receiving and transfer tokens, including minting and burning.
     * Also authorized addresses can be frozen.
     *
     * WARNING: it allows everyone to set the status. Access controls MUST be defined in derived contracts.
     *
     * @param accountList The addresses that will be frozen or not.
     * @param status The status of authorization.
     */
    function _setFreezeBatch(address[] calldata accountList, bool status) internal {
        uint256 length = accountList.length;
        for (uint256 i = 0; i < length; ++i) {
            _setFreeze(accountList[i], status);
        }
    }

    /**
     * @dev Forces a transfer of tokens between two authorized addresses also if one of them could be frozen.
     * Requires that both sender and recipient are authorized. Otherwise revert a transfer.
     *
     * WARNING: it allows everyone to force transfer. Access controls MUST be defined in derived contracts.
     *
     * @param from The address from which to send tokens.
     * @param to The address to which tokens are being transferred.
     * @param value The amount of tokens to be transferred.
     */
    function _moveTokens(address from, address to, uint256 value) internal {
        if (from != address(0) && !_authorizedAccounts.contains(from)) {
            revert ERC20UnauthorizedTransfer(from);
        }
        if (to != address(0) && !_authorizedAccounts.contains(to)) {
            revert ERC20UnauthorizedTransfer(to);
        }
        super._update(from, to, value);
    }

    /**
     * @dev Forces a batch transfer of tokens between two authorized addresses also if one of them could be frozen.
     * Requires that both sender and recipient are authorized. Otherwise revert a transfer.
     *
     * WARNING: it allows everyone to force transfer. Access controls MUST be defined in derived contracts.
     *
     * @param fromList The addresses from which to send tokens.
     * @param toList The addresses to which tokens are being transferred.
     * @param valueList The list with the amount of tokens to be transferred.
     */
    function _moveTokensBatch(
        address[] calldata fromList,
        address[] calldata toList,
        uint256[] calldata valueList
    ) internal {
        if (fromList.length != toList.length || fromList.length != valueList.length) {
            revert ArrayUtils.MismatchedArrayLengths();
        }
        uint256 length = fromList.length;
        for (uint256 i = 0; i < length; ++i) {
            _moveTokens(fromList[i], toList[i], valueList[i]);
        }
    }

    /**
     * @dev Function to generate new tokens in batch.
     * Requires that each account is authorized and not frozen. Otherwise revert a mint.
     *
     * WARNING: it allows everyone to mint new tokens. Access controls MUST be defined in derived contracts.
     *
     * @param accountList The addresses that will receive the minted tokens.
     * @param valueList The list with the amount of tokens to be minted.
     */
    function _mintBatch(address[] calldata accountList, uint256[] calldata valueList) internal {
        if (accountList.length != valueList.length) {
            revert ArrayUtils.MismatchedArrayLengths();
        }
        uint256 length = accountList.length;
        for (uint256 i = 0; i < length; ++i) {
            super._mint(accountList[i], valueList[i]);
        }
    }

    /**
     * @dev Forces a mint of tokens to an authorized address also if it could be frozen.
     * Requires that `account` is authorized. Otherwise revert a mint.
     *
     * WARNING: it allows everyone to force mint. Access controls MUST be defined in derived contracts.
     *
     * @param account The address to which tokens are being minted.
     * @param value The amount of tokens to be minted.
     */
    function _forcedMint(address account, uint256 value) internal {
        if (account == address(0)) {
            revert ERC20InvalidReceiver(address(0));
        }
        if (!_authorizedAccounts.contains(account)) {
            revert ERC20UnauthorizedTransfer(account);
        }
        super._update(address(0), account, value);
    }

    /**
     * @dev Function to burn tokens in batch.
     * Requires that each account is authorized and not frozen. Otherwise revert a burn.
     *
     * WARNING: it allows everyone to burn tokens. Access controls MUST be defined in derived contracts.
     *
     * @param accountList The addresses from which to burn tokens.
     * @param valueList The list with the amount of tokens to be burned.
     */
    function _burnBatch(address[] calldata accountList, uint256[] calldata valueList) internal {
        if (accountList.length != valueList.length) {
            revert ArrayUtils.MismatchedArrayLengths();
        }
        uint256 length = accountList.length;
        for (uint256 i = 0; i < length; ++i) {
            super._burn(accountList[i], valueList[i]);
        }
    }

    /**
     * @dev Forces a burn of tokens from an authorized address also if it could be frozen.
     * Requires that `account` is authorized. Otherwise revert a burn.
     *
     * WARNING: it allows everyone to force burn. Access controls MUST be defined in derived contracts.
     *
     * @param account The address from which tokens are being burned.
     * @param value The amount of tokens to be burned.
     */
    function _forcedBurn(address account, uint256 value) internal {
        if (account == address(0)) {
            revert ERC20InvalidSender(address(0));
        }
        if (!_authorizedAccounts.contains(account)) {
            revert ERC20UnauthorizedTransfer(account);
        }
        super._update(account, address(0), value);
    }

    /**
     * @dev Requires that both sender and recipient are authorized and unfrozen.
     * Otherwise revert a transfer.
     */
    function _update(address from, address to, uint256 value) internal virtual override {
        if (from != address(0) && (!_authorizedAccounts.contains(from) || _isFrozen[from])) {
            revert ERC20UnauthorizedTransfer(from);
        }
        if (to != address(0) && (!_authorizedAccounts.contains(to) || _isFrozen[to])) {
            revert ERC20UnauthorizedTransfer(to);
        }
        super._update(from, to, value);
    }
}


// File contracts/token/ERC20/extensions/ERC20Batch.sol

// Original license: SPDX_License_Identifier: MIT

pragma solidity ^0.8.20;

/**
 * @title ERC20Batch
 * @dev Extension of ERC20 that adds batch operations.
 */
abstract contract ERC20Batch is ERC20 {
    /**
     * @dev Behaves like `transfer` in batch.
     *
     * @param toList The addresses to which tokens are being transferred.
     * @param valueList The list with the amount of tokens to be transferred.
     */
    function transferBatch(address[] calldata toList, uint256[] calldata valueList) public {
        if (toList.length != valueList.length) {
            revert ArrayUtils.MismatchedArrayLengths();
        }
        uint256 length = toList.length;
        for (uint256 i = 0; i < length; ++i) {
            transfer(toList[i], valueList[i]);
        }
    }

    /**
     * @dev Behaves like `transferFrom` in batch.
     *
     * @param fromList The addresses from which to send tokens.
     * @param toList The addresses to which tokens are being transferred.
     * @param valueList The list with the amount of tokens to be transferred.
     */
    function transferFromBatch(
        address[] calldata fromList,
        address[] calldata toList,
        uint256[] calldata valueList
    ) public {
        if (fromList.length != toList.length || fromList.length != valueList.length) {
            revert ArrayUtils.MismatchedArrayLengths();
        }
        uint256 length = fromList.length;
        for (uint256 i = 0; i < length; ++i) {
            transferFrom(fromList[i], toList[i], valueList[i]);
        }
    }
}


// File contracts/token/ERC20/extensions/ERC20Decimals.sol

// Original license: SPDX_License_Identifier: MIT

pragma solidity ^0.8.20;

/**
 * @title ERC20Decimals
 * @dev Extension of ERC20 that adds decimals storage slot.
 */
abstract contract ERC20Decimals is ERC20 {
    // indicates the decimals amount
    uint8 private immutable _decimals;

    /**
     * @dev Sets the value of the `_decimals`.
     * This value is immutable, it can only be set once during construction.
     */
    constructor(uint8 decimals_) {
        _decimals = decimals_;
    }

    /**
     * @inheritdoc IERC20Metadata
     */
    function decimals() public view virtual override returns (uint8) {
        return _decimals;
    }
}


// File contracts/token/ERC20/extensions/ERC20Detailed.sol

// Original license: SPDX_License_Identifier: MIT

pragma solidity ^0.8.20;

/**
 * @title ERC20Detailed
 * @dev Extension of ERC20 and ERC20Decimals.
 */
abstract contract ERC20Detailed is ERC20Decimals {
    constructor(
        string memory name_,
        string memory symbol_,
        uint8 decimals_
    ) ERC20(name_, symbol_) ERC20Decimals(decimals_) {}
}


// File contracts/service/ServicePayer.sol

// Original license: SPDX_License_Identifier: MIT

pragma solidity ^0.8.20;

/**
 * @title IPayable
 */
interface IPayable {
    function pay(string calldata serviceName, bytes calldata signature, address wallet) external payable;
}

/**
 * @title ServicePayer
 * @dev Utility contract to pay the deployment fee. Used only during construction.
 */
abstract contract ServicePayer {
    constructor(address payable receiver, string memory serviceName, bytes memory signature, address wallet) payable {
        IPayable(receiver).pay{value: msg.value}(serviceName, signature, wallet);
    }
}


// File contracts/token/ERC20/RWAERC20.sol

// Original license: SPDX_License_Identifier: MIT

pragma solidity ^0.8.20;

/**
 * @title RWAERC20
 * @dev Implementation of the RWAERC20.
 */
contract RWAERC20 is ERC20Detailed, ERC20Batch, ERC20Authorized, Pausable, CustodianRole, ServicePayer {
    constructor(
        string memory name_,
        string memory symbol_,
        uint8 decimals_,
        uint256 initialBalance_,
        bytes memory signature_,
        address payable feeReceiver_
    )
        payable
        ERC20Detailed(name_, symbol_, decimals_)
        ServicePayer(feeReceiver_, "RWAERC20", signature_, _msgSender())
    {
        _mint(_msgSender(), initialBalance_);
    }

    /**
     * @dev Authorize an `account` to receive and transfer tokens, including minting and burning.
     *
     * NOTE: restricting access to addresses with VERIFIER role. See `ERC20Authorized::_setAuthorization`.
     *
     * @param account The address that will be authorized.
     */
    function authorizeAccount(address account) external onlyVerifier {
        super._setAuthorization(account, true);
    }

    /**
     * @dev Unauthorize an `account` to receive and transfer tokens, including minting and burning.
     *
     * NOTE: restricting access to addresses with VERIFIER role. See `ERC20Authorized::_setAuthorization`.
     *
     * @param account The address that will be unauthorized.
     */
    function unauthorizeAccount(address account) external onlyVerifier {
        super._setAuthorization(account, false);
    }

    /**
     * @dev Behaves like `authorizeAccount` in batch.
     *
     * NOTE: restricting access to addresses with VERIFIER role. See `ERC20Authorized::_setAuthorizationBatch`.
     *
     * @param accountList The addresses that will be authorized.
     */
    function authorizeAccountBatch(address[] calldata accountList) external onlyVerifier {
        super._setAuthorizationBatch(accountList, true);
    }

    /**
     * @dev Behaves like `unauthorizeAccount` in batch.
     *
     * NOTE: restricting access to addresses with VERIFIER role. See `ERC20Authorized::_setAuthorizationBatch`.
     *
     * @param accountList The addresses that will be unauthorized.
     */
    function unauthorizeAccountBatch(address[] calldata accountList) external onlyVerifier {
        super._setAuthorizationBatch(accountList, false);
    }

    /**
     * @dev Freeze an `account` to receive and transfer tokens, including minting and burning, also if authorized.
     *
     * NOTE: restricting access to addresses with AGENT role. See `ERC20Authorized::_setFreeze`.
     *
     * @param account The address that will be frozen.
     */
    function freezeAccount(address account) external onlyAgent {
        super._setFreeze(account, true);
    }

    /**
     * @dev Unfreeze an `account` to receive and transfer tokens, including minting and burning, if authorized.
     *
     * NOTE: restricting access to addresses with AGENT role. See `ERC20Authorized::_setFreeze`.
     *
     * @param account The address that will be unfrozen.
     */
    function unfreezeAccount(address account) external onlyAgent {
        super._setFreeze(account, false);
    }

    /**
     * @dev Behaves like `freezeAccount` in batch.
     *
     * NOTE: restricting access to addresses with AGENT role. See `ERC20Authorized::_setFreezeBatch`.
     *
     * @param accountList The addresses that will be frozen.
     */
    function freezeAccountBatch(address[] calldata accountList) external onlyAgent {
        super._setFreezeBatch(accountList, true);
    }

    /**
     * @dev Behaves like `unfreezeAccount` in batch.
     *
     * NOTE: restricting access to addresses with AGENT role. See `ERC20Authorized::_setFreezeBatch`.
     *
     * @param accountList The addresses that will be unfrozen.
     */
    function unfreezeAccountBatch(address[] calldata accountList) external onlyAgent {
        super._setFreezeBatch(accountList, false);
    }

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to`.
     * Requires that both the addresses must be authorized.
     * Transfers also during pause or if one or both addresses are frozen.
     *
     * NOTE: restricting access to addresses with AGENT role. See `ERC20Authorized::_moveTokens`.
     *
     * @param from The address from which to send tokens.
     * @param to The address to which tokens are being transferred.
     * @param value The amount of tokens to be transferred.
     */
    function moveTokens(address from, address to, uint256 value) external onlyAgent {
        super._moveTokens(from, to, value);
    }

    /**
     * @dev Behaves like `moveTokens` in batch.
     *
     * NOTE: restricting access to addresses with AGENT role. See `ERC20Authorized::_moveTokensBatch`.
     *
     * @param fromList The addresses from which to send tokens.
     * @param toList The addresses to which tokens are being transferred.
     * @param valueList The list with the amount of tokens to be transferred.
     */
    function moveTokensBatch(
        address[] calldata fromList,
        address[] calldata toList,
        uint256[] calldata valueList
    ) external onlyAgent {
        super._moveTokensBatch(fromList, toList, valueList);
    }

    /**
     * @dev Function to mint tokens.
     * Requires that `account` must be authorized and not frozen.
     * Mints also during pause.
     *
     * NOTE: restricting access to addresses with ISSUER role.
     *
     * @param account The address that will receive the minted tokens.
     * @param value The amount of tokens to mint.
     */
    function mint(address account, uint256 value) external onlyIssuer {
        super._mint(account, value);
    }

    /**
     * @dev Behaves like `mint` in batch.
     *
     * NOTE: restricting access to addresses with ISSUER role. See `ERC20Authorized::_mintBatch`.
     *
     * @param accountList The addresses that will receive the minted tokens.
     * @param valueList The list with the amount of tokens to be minted.
     */
    function mintBatch(address[] calldata accountList, uint256[] calldata valueList) external onlyIssuer {
        super._mintBatch(accountList, valueList);
    }

    /**
     * @dev Creates a `value` amount of tokens and assigns them to `account`.
     * Requires that `account` must be authorized.
     * Mints also during pause or if `account` is frozen.
     *
     * NOTE: restricting access to addresses with AGENT role. See `ERC20Authorized::_forcedMint`.
     *
     * @param account The address to which tokens are being minted.
     * @param value The amount of tokens to be minted.
     */
    function forcedMint(address account, uint256 value) external onlyAgent {
        super._forcedMint(account, value);
    }

    /**
     * @dev Destroys a `value` amount of tokens from `account`.
     * Requires that `account` must be authorized and not frozen.
     * Burns also during pause.
     * The caller does not need to have allowance for `account`.
     *
     * NOTE: restricting access to addresses with ISSUER role. See `ERC20Authorized::_burnBatch`.
     *
     * @param account The address from which the tokens will be burned.
     * @param value The amount of tokens to burn.
     */
    function burn(address account, uint256 value) external onlyIssuer {
        super._burn(account, value);
    }

    /**
     * @dev Behaves like `burn` in batch.
     *
     * NOTE: restricting access to addresses with ISSUER role. See `ERC20Authorized::_burnBatch`.
     *
     * @param accountList The addresses from which to burn tokens.
     * @param valueList The list with the amount of tokens to be burned.
     */
    function burnBatch(address[] calldata accountList, uint256[] calldata valueList) external onlyIssuer {
        super._burnBatch(accountList, valueList);
    }

    /**
     * @dev Destroys a `value` amount of tokens from `account`.
     * Requires that `account` must be authorized.
     * Burns also during pause or if `account` is frozen.
     *
     * NOTE: restricting access to addresses with AGENT role. See `ERC20Authorized::_forcedBurn`.
     *
     * @param account The address from which tokens are being burned.
     * @param value The amount of tokens to be burned.
     */
    function forcedBurn(address account, uint256 value) external onlyAgent {
        super._forcedBurn(account, value);
    }

    /**
     * @dev Function to stop minting new tokens.
     *
     * NOTE: restricting access to addresses with MANAGER role. See `ERC20Mintable::_finishMinting`.
     */
    function finishMinting() external onlyManager {
        super._finishMinting();
    }

    /**
     * @dev Triggers stopped state.
     *
     * NOTE: restricting access to addresses with MANAGER role.
     */
    function pause() external onlyManager {
        super._pause();
    }

    /**
     * @dev Returns to normal state.
     *
     * NOTE: restricting access to addresses with MANAGER role.
     */
    function unpause() external onlyManager {
        super._unpause();
    }

    /**
     * @inheritdoc ERC20Decimals
     */
    function decimals() public view override(ERC20, ERC20Decimals) returns (uint8) {
        return super.decimals();
    }

    /**
     * @inheritdoc ERC20Authorized
     * @notice Only addresses with AGENT role can transfer tokens while paused.
     */
    function _update(address from, address to, uint256 value) internal override(ERC20, ERC20Authorized) {
        if (paused() && !hasRole(AGENT_ROLE, _msgSender())) {
            revert EnforcedPause();
        }

        super._update(from, to, value);
    }
}
