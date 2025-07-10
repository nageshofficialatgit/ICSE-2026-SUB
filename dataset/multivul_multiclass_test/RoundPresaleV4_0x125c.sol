// File: @openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol


// OpenZeppelin Contracts (last updated v5.0.0) (proxy/utils/Initializable.sol)

pragma solidity ^0.8.20;

/**
 * @dev This is a base contract to aid in writing upgradeable contracts, or any kind of contract that will be deployed
 * behind a proxy. Since proxied contracts do not make use of a constructor, it's common to move constructor logic to an
 * external initializer function, usually called `initialize`. It then becomes necessary to protect this initializer
 * function so it can only be called once. The {initializer} modifier provided by this contract will have this effect.
 *
 * The initialization functions use a version number. Once a version number is used, it is consumed and cannot be
 * reused. This mechanism prevents re-execution of each "step" but allows the creation of new initialization steps in
 * case an upgrade adds a module that needs to be initialized.
 *
 * For example:
 *
 * [.hljs-theme-light.nopadding]
 * ```solidity
 * contract MyToken is ERC20Upgradeable {
 *     function initialize() initializer public {
 *         __ERC20_init("MyToken", "MTK");
 *     }
 * }
 *
 * contract MyTokenV2 is MyToken, ERC20PermitUpgradeable {
 *     function initializeV2() reinitializer(2) public {
 *         __ERC20Permit_init("MyToken");
 *     }
 * }
 * ```
 *
 * TIP: To avoid leaving the proxy in an uninitialized state, the initializer function should be called as early as
 * possible by providing the encoded function call as the `_data` argument to {ERC1967Proxy-constructor}.
 *
 * CAUTION: When used with inheritance, manual care must be taken to not invoke a parent initializer twice, or to ensure
 * that all initializers are idempotent. This is not verified automatically as constructors are by Solidity.
 *
 * [CAUTION]
 * ====
 * Avoid leaving a contract uninitialized.
 *
 * An uninitialized contract can be taken over by an attacker. This applies to both a proxy and its implementation
 * contract, which may impact the proxy. To prevent the implementation contract from being used, you should invoke
 * the {_disableInitializers} function in the constructor to automatically lock it when it is deployed:
 *
 * [.hljs-theme-light.nopadding]
 * ```
 * /// @custom:oz-upgrades-unsafe-allow constructor
 * constructor() {
 *     _disableInitializers();
 * }
 * ```
 * ====
 */
abstract contract Initializable {
    /**
     * @dev Storage of the initializable contract.
     *
     * It's implemented on a custom ERC-7201 namespace to reduce the risk of storage collisions
     * when using with upgradeable contracts.
     *
     * @custom:storage-location erc7201:openzeppelin.storage.Initializable
     */
    struct InitializableStorage {
        /**
         * @dev Indicates that the contract has been initialized.
         */
        uint64 _initialized;
        /**
         * @dev Indicates that the contract is in the process of being initialized.
         */
        bool _initializing;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.Initializable")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant INITIALIZABLE_STORAGE = 0xf0c57e16840df040f15088dc2f81fe391c3923bec73e23a9662efc9c229c6a00;

    /**
     * @dev The contract is already initialized.
     */
    error InvalidInitialization();

    /**
     * @dev The contract is not initializing.
     */
    error NotInitializing();

    /**
     * @dev Triggered when the contract has been initialized or reinitialized.
     */
    event Initialized(uint64 version);

    /**
     * @dev A modifier that defines a protected initializer function that can be invoked at most once. In its scope,
     * `onlyInitializing` functions can be used to initialize parent contracts.
     *
     * Similar to `reinitializer(1)`, except that in the context of a constructor an `initializer` may be invoked any
     * number of times. This behavior in the constructor can be useful during testing and is not expected to be used in
     * production.
     *
     * Emits an {Initialized} event.
     */
    modifier initializer() {
        // solhint-disable-next-line var-name-mixedcase
        InitializableStorage storage $ = _getInitializableStorage();

        // Cache values to avoid duplicated sloads
        bool isTopLevelCall = !$._initializing;
        uint64 initialized = $._initialized;

        // Allowed calls:
        // - initialSetup: the contract is not in the initializing state and no previous version was
        //                 initialized
        // - construction: the contract is initialized at version 1 (no reininitialization) and the
        //                 current contract is just being deployed
        bool initialSetup = initialized == 0 && isTopLevelCall;
        bool construction = initialized == 1 && address(this).code.length == 0;

        if (!initialSetup && !construction) {
            revert InvalidInitialization();
        }
        $._initialized = 1;
        if (isTopLevelCall) {
            $._initializing = true;
        }
        _;
        if (isTopLevelCall) {
            $._initializing = false;
            emit Initialized(1);
        }
    }

    /**
     * @dev A modifier that defines a protected reinitializer function that can be invoked at most once, and only if the
     * contract hasn't been initialized to a greater version before. In its scope, `onlyInitializing` functions can be
     * used to initialize parent contracts.
     *
     * A reinitializer may be used after the original initialization step. This is essential to configure modules that
     * are added through upgrades and that require initialization.
     *
     * When `version` is 1, this modifier is similar to `initializer`, except that functions marked with `reinitializer`
     * cannot be nested. If one is invoked in the context of another, execution will revert.
     *
     * Note that versions can jump in increments greater than 1; this implies that if multiple reinitializers coexist in
     * a contract, executing them in the right order is up to the developer or operator.
     *
     * WARNING: Setting the version to 2**64 - 1 will prevent any future reinitialization.
     *
     * Emits an {Initialized} event.
     */
    modifier reinitializer(uint64 version) {
        // solhint-disable-next-line var-name-mixedcase
        InitializableStorage storage $ = _getInitializableStorage();

        if ($._initializing || $._initialized >= version) {
            revert InvalidInitialization();
        }
        $._initialized = version;
        $._initializing = true;
        _;
        $._initializing = false;
        emit Initialized(version);
    }

    /**
     * @dev Modifier to protect an initialization function so that it can only be invoked by functions with the
     * {initializer} and {reinitializer} modifiers, directly or indirectly.
     */
    modifier onlyInitializing() {
        _checkInitializing();
        _;
    }

    /**
     * @dev Reverts if the contract is not in an initializing state. See {onlyInitializing}.
     */
    function _checkInitializing() internal view virtual {
        if (!_isInitializing()) {
            revert NotInitializing();
        }
    }

    /**
     * @dev Locks the contract, preventing any future reinitialization. This cannot be part of an initializer call.
     * Calling this in the constructor of a contract will prevent that contract from being initialized or reinitialized
     * to any version. It is recommended to use this to lock implementation contracts that are designed to be called
     * through proxies.
     *
     * Emits an {Initialized} event the first time it is successfully executed.
     */
    function _disableInitializers() internal virtual {
        // solhint-disable-next-line var-name-mixedcase
        InitializableStorage storage $ = _getInitializableStorage();

        if ($._initializing) {
            revert InvalidInitialization();
        }
        if ($._initialized != type(uint64).max) {
            $._initialized = type(uint64).max;
            emit Initialized(type(uint64).max);
        }
    }

    /**
     * @dev Returns the highest version that has been initialized. See {reinitializer}.
     */
    function _getInitializedVersion() internal view returns (uint64) {
        return _getInitializableStorage()._initialized;
    }

    /**
     * @dev Returns `true` if the contract is currently initializing. See {onlyInitializing}.
     */
    function _isInitializing() internal view returns (bool) {
        return _getInitializableStorage()._initializing;
    }

    /**
     * @dev Returns a pointer to the storage namespace.
     */
    // solhint-disable-next-line var-name-mixedcase
    function _getInitializableStorage() private pure returns (InitializableStorage storage $) {
        assembly {
            $.slot := INITIALIZABLE_STORAGE
        }
    }
}

// File: @openzeppelin/contracts/access/IAccessControl.sol


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

// File: @openzeppelin/contracts-upgradeable/utils/ContextUpgradeable.sol


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
abstract contract ContextUpgradeable is Initializable {
    function __Context_init() internal onlyInitializing {
    }

    function __Context_init_unchained() internal onlyInitializing {
    }
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

// File: @openzeppelin/contracts/utils/introspection/IERC165.sol


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

// File: @openzeppelin/contracts-upgradeable/utils/introspection/ERC165Upgradeable.sol


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
abstract contract ERC165Upgradeable is Initializable, IERC165 {
    function __ERC165_init() internal onlyInitializing {
    }

    function __ERC165_init_unchained() internal onlyInitializing {
    }
    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual returns (bool) {
        return interfaceId == type(IERC165).interfaceId;
    }
}

// File: @openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol


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
abstract contract AccessControlUpgradeable is Initializable, ContextUpgradeable, IAccessControl, ERC165Upgradeable {
    struct RoleData {
        mapping(address account => bool) hasRole;
        bytes32 adminRole;
    }

    bytes32 public constant DEFAULT_ADMIN_ROLE = 0x00;


    /// @custom:storage-location erc7201:openzeppelin.storage.AccessControl
    struct AccessControlStorage {
        mapping(bytes32 role => RoleData) _roles;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.AccessControl")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant AccessControlStorageLocation = 0x02dd7bc7dec4dceedda775e58dd541e08a116c6c53815c0bd028192f7b626800;

    function _getAccessControlStorage() private pure returns (AccessControlStorage storage $) {
        assembly {
            $.slot := AccessControlStorageLocation
        }
    }

    /**
     * @dev Modifier that checks that an account has a specific role. Reverts
     * with an {AccessControlUnauthorizedAccount} error including the required role.
     */
    modifier onlyRole(bytes32 role) {
        _checkRole(role);
        _;
    }

    function __AccessControl_init() internal onlyInitializing {
    }

    function __AccessControl_init_unchained() internal onlyInitializing {
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
        AccessControlStorage storage $ = _getAccessControlStorage();
        return $._roles[role].hasRole[account];
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
        AccessControlStorage storage $ = _getAccessControlStorage();
        return $._roles[role].adminRole;
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
        AccessControlStorage storage $ = _getAccessControlStorage();
        bytes32 previousAdminRole = getRoleAdmin(role);
        $._roles[role].adminRole = adminRole;
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
        AccessControlStorage storage $ = _getAccessControlStorage();
        if (!hasRole(role, account)) {
            $._roles[role].hasRole[account] = true;
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
        AccessControlStorage storage $ = _getAccessControlStorage();
        if (hasRole(role, account)) {
            $._roles[role].hasRole[account] = false;
            emit RoleRevoked(role, account, _msgSender());
            return true;
        } else {
            return false;
        }
    }
}

// File: @openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol


// OpenZeppelin Contracts (last updated v4.9.0) (security/ReentrancyGuard.sol)

pragma solidity ^0.8.0;


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
abstract contract ReentrancyGuardUpgradeable is Initializable {
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
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    function __ReentrancyGuard_init() internal onlyInitializing {
        __ReentrancyGuard_init_unchained();
    }

    function __ReentrancyGuard_init_unchained() internal onlyInitializing {
        _status = _NOT_ENTERED;
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
        // On the first call to nonReentrant, _status will be _NOT_ENTERED
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");

        // Any calls to nonReentrant after this point will fail
        _status = _ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = _NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == _ENTERED;
    }

    /**
     * @dev This empty reserved space is put in place to allow future versions to add new
     * variables without shifting down storage in the inheritance chain.
     * See https://docs.openzeppelin.com/contracts/4.x/upgradeable#storage_gaps
     */
    uint256[49] private __gap;
}

// File: @openzeppelin/contracts-upgradeable/security/PausableUpgradeable.sol


// OpenZeppelin Contracts (last updated v4.7.0) (security/Pausable.sol)

pragma solidity ^0.8.0;



/**
 * @dev Contract module which allows children to implement an emergency stop
 * mechanism that can be triggered by an authorized account.
 *
 * This module is used through inheritance. It will make available the
 * modifiers `whenNotPaused` and `whenPaused`, which can be applied to
 * the functions of your contract. Note that they will not be pausable by
 * simply including this module, only once the modifiers are put in place.
 */
abstract contract PausableUpgradeable is Initializable, ContextUpgradeable {
    /**
     * @dev Emitted when the pause is triggered by `account`.
     */
    event Paused(address account);

    /**
     * @dev Emitted when the pause is lifted by `account`.
     */
    event Unpaused(address account);

    bool private _paused;

    /**
     * @dev Initializes the contract in unpaused state.
     */
    function __Pausable_init() internal onlyInitializing {
        __Pausable_init_unchained();
    }

    function __Pausable_init_unchained() internal onlyInitializing {
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
        require(!paused(), "Pausable: paused");
    }

    /**
     * @dev Throws if the contract is not paused.
     */
    function _requirePaused() internal view virtual {
        require(paused(), "Pausable: not paused");
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

    /**
     * @dev This empty reserved space is put in place to allow future versions to add new
     * variables without shifting down storage in the inheritance chain.
     * See https://docs.openzeppelin.com/contracts/4.x/upgradeable#storage_gaps
     */
    uint256[49] private __gap;
}

// File: @openzeppelin/contracts/interfaces/draft-IERC1822.sol


// OpenZeppelin Contracts (last updated v5.1.0) (interfaces/draft-IERC1822.sol)

pragma solidity ^0.8.20;

/**
 * @dev ERC-1822: Universal Upgradeable Proxy Standard (UUPS) documents a method for upgradeability through a simplified
 * proxy whose upgrades are fully controlled by the current implementation.
 */
interface IERC1822Proxiable {
    /**
     * @dev Returns the storage slot that the proxiable contract assumes is being used to store the implementation
     * address.
     *
     * IMPORTANT: A proxy pointing at a proxiable contract should not be considered proxiable itself, because this risks
     * bricking a proxy that upgrades to it, by delegating to itself until out of gas. Thus it is critical that this
     * function revert if invoked through a proxy.
     */
    function proxiableUUID() external view returns (bytes32);
}

// File: @openzeppelin/contracts/proxy/beacon/IBeacon.sol


// OpenZeppelin Contracts (last updated v5.0.0) (proxy/beacon/IBeacon.sol)

pragma solidity ^0.8.20;

/**
 * @dev This is the interface that {BeaconProxy} expects of its beacon.
 */
interface IBeacon {
    /**
     * @dev Must return an address that can be used as a delegate call target.
     *
     * {UpgradeableBeacon} will check that this address is a contract.
     */
    function implementation() external view returns (address);
}

// File: @openzeppelin/contracts/interfaces/IERC1967.sol


// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC1967.sol)

pragma solidity ^0.8.20;

/**
 * @dev ERC-1967: Proxy Storage Slots. This interface contains the events defined in the ERC.
 */
interface IERC1967 {
    /**
     * @dev Emitted when the implementation is upgraded.
     */
    event Upgraded(address indexed implementation);

    /**
     * @dev Emitted when the admin account has changed.
     */
    event AdminChanged(address previousAdmin, address newAdmin);

    /**
     * @dev Emitted when the beacon is changed.
     */
    event BeaconUpgraded(address indexed beacon);
}

// File: @openzeppelin/contracts/utils/Errors.sol


// OpenZeppelin Contracts (last updated v5.1.0) (utils/Errors.sol)

pragma solidity ^0.8.20;

/**
 * @dev Collection of common custom errors used in multiple contracts
 *
 * IMPORTANT: Backwards compatibility is not guaranteed in future versions of the library.
 * It is recommended to avoid relying on the error API for critical functionality.
 *
 * _Available since v5.1._
 */
library Errors {
    /**
     * @dev The ETH balance of the account is not enough to perform the operation.
     */
    error InsufficientBalance(uint256 balance, uint256 needed);

    /**
     * @dev A call to an address target failed. The target may have reverted.
     */
    error FailedCall();

    /**
     * @dev The deployment failed.
     */
    error FailedDeployment();

    /**
     * @dev A necessary precompile is missing.
     */
    error MissingPrecompile(address);
}

// File: @openzeppelin/contracts/utils/Address.sol


// OpenZeppelin Contracts (last updated v5.2.0) (utils/Address.sol)

pragma solidity ^0.8.20;


/**
 * @dev Collection of functions related to the address type
 */
library Address {
    /**
     * @dev There's no code at `target` (it is not a contract).
     */
    error AddressEmptyCode(address target);

    /**
     * @dev Replacement for Solidity's `transfer`: sends `amount` wei to
     * `recipient`, forwarding all available gas and reverting on errors.
     *
     * https://eips.ethereum.org/EIPS/eip-1884[EIP1884] increases the gas cost
     * of certain opcodes, possibly making contracts go over the 2300 gas limit
     * imposed by `transfer`, making them unable to receive funds via
     * `transfer`. {sendValue} removes this limitation.
     *
     * https://consensys.net/diligence/blog/2019/09/stop-using-soliditys-transfer-now/[Learn more].
     *
     * IMPORTANT: because control is transferred to `recipient`, care must be
     * taken to not create reentrancy vulnerabilities. Consider using
     * {ReentrancyGuard} or the
     * https://solidity.readthedocs.io/en/v0.8.20/security-considerations.html#use-the-checks-effects-interactions-pattern[checks-effects-interactions pattern].
     */
    function sendValue(address payable recipient, uint256 amount) internal {
        if (address(this).balance < amount) {
            revert Errors.InsufficientBalance(address(this).balance, amount);
        }

        (bool success, bytes memory returndata) = recipient.call{value: amount}("");
        if (!success) {
            _revert(returndata);
        }
    }

    /**
     * @dev Performs a Solidity function call using a low level `call`. A
     * plain `call` is an unsafe replacement for a function call: use this
     * function instead.
     *
     * If `target` reverts with a revert reason or custom error, it is bubbled
     * up by this function (like regular Solidity function calls). However, if
     * the call reverted with no returned reason, this function reverts with a
     * {Errors.FailedCall} error.
     *
     * Returns the raw returned data. To convert to the expected return value,
     * use https://solidity.readthedocs.io/en/latest/units-and-global-variables.html?highlight=abi.decode#abi-encoding-and-decoding-functions[`abi.decode`].
     *
     * Requirements:
     *
     * - `target` must be a contract.
     * - calling `target` with `data` must not revert.
     */
    function functionCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but also transferring `value` wei to `target`.
     *
     * Requirements:
     *
     * - the calling contract must have an ETH balance of at least `value`.
     * - the called Solidity function must be `payable`.
     */
    function functionCallWithValue(address target, bytes memory data, uint256 value) internal returns (bytes memory) {
        if (address(this).balance < value) {
            revert Errors.InsufficientBalance(address(this).balance, value);
        }
        (bool success, bytes memory returndata) = target.call{value: value}(data);
        return verifyCallResultFromTarget(target, success, returndata);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a static call.
     */
    function functionStaticCall(address target, bytes memory data) internal view returns (bytes memory) {
        (bool success, bytes memory returndata) = target.staticcall(data);
        return verifyCallResultFromTarget(target, success, returndata);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a delegate call.
     */
    function functionDelegateCall(address target, bytes memory data) internal returns (bytes memory) {
        (bool success, bytes memory returndata) = target.delegatecall(data);
        return verifyCallResultFromTarget(target, success, returndata);
    }

    /**
     * @dev Tool to verify that a low level call to smart-contract was successful, and reverts if the target
     * was not a contract or bubbling up the revert reason (falling back to {Errors.FailedCall}) in case
     * of an unsuccessful call.
     */
    function verifyCallResultFromTarget(
        address target,
        bool success,
        bytes memory returndata
    ) internal view returns (bytes memory) {
        if (!success) {
            _revert(returndata);
        } else {
            // only check if target is a contract if the call was successful and the return data is empty
            // otherwise we already know that it was a contract
            if (returndata.length == 0 && target.code.length == 0) {
                revert AddressEmptyCode(target);
            }
            return returndata;
        }
    }

    /**
     * @dev Tool to verify that a low level call was successful, and reverts if it wasn't, either by bubbling the
     * revert reason or with a default {Errors.FailedCall} error.
     */
    function verifyCallResult(bool success, bytes memory returndata) internal pure returns (bytes memory) {
        if (!success) {
            _revert(returndata);
        } else {
            return returndata;
        }
    }

    /**
     * @dev Reverts with returndata if present. Otherwise reverts with {Errors.FailedCall}.
     */
    function _revert(bytes memory returndata) private pure {
        // Look for revert reason and bubble it up if present
        if (returndata.length > 0) {
            // The easiest way to bubble the revert reason is using memory via assembly
            assembly ("memory-safe") {
                let returndata_size := mload(returndata)
                revert(add(32, returndata), returndata_size)
            }
        } else {
            revert Errors.FailedCall();
        }
    }
}

// File: @openzeppelin/contracts/utils/StorageSlot.sol


// OpenZeppelin Contracts (last updated v5.1.0) (utils/StorageSlot.sol)
// This file was procedurally generated from scripts/generate/templates/StorageSlot.js.

pragma solidity ^0.8.20;

/**
 * @dev Library for reading and writing primitive types to specific storage slots.
 *
 * Storage slots are often used to avoid storage conflict when dealing with upgradeable contracts.
 * This library helps with reading and writing to such slots without the need for inline assembly.
 *
 * The functions in this library return Slot structs that contain a `value` member that can be used to read or write.
 *
 * Example usage to set ERC-1967 implementation slot:
 * ```solidity
 * contract ERC1967 {
 *     // Define the slot. Alternatively, use the SlotDerivation library to derive the slot.
 *     bytes32 internal constant _IMPLEMENTATION_SLOT = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;
 *
 *     function _getImplementation() internal view returns (address) {
 *         return StorageSlot.getAddressSlot(_IMPLEMENTATION_SLOT).value;
 *     }
 *
 *     function _setImplementation(address newImplementation) internal {
 *         require(newImplementation.code.length > 0);
 *         StorageSlot.getAddressSlot(_IMPLEMENTATION_SLOT).value = newImplementation;
 *     }
 * }
 * ```
 *
 * TIP: Consider using this library along with {SlotDerivation}.
 */
library StorageSlot {
    struct AddressSlot {
        address value;
    }

    struct BooleanSlot {
        bool value;
    }

    struct Bytes32Slot {
        bytes32 value;
    }

    struct Uint256Slot {
        uint256 value;
    }

    struct Int256Slot {
        int256 value;
    }

    struct StringSlot {
        string value;
    }

    struct BytesSlot {
        bytes value;
    }

    /**
     * @dev Returns an `AddressSlot` with member `value` located at `slot`.
     */
    function getAddressSlot(bytes32 slot) internal pure returns (AddressSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `BooleanSlot` with member `value` located at `slot`.
     */
    function getBooleanSlot(bytes32 slot) internal pure returns (BooleanSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `Bytes32Slot` with member `value` located at `slot`.
     */
    function getBytes32Slot(bytes32 slot) internal pure returns (Bytes32Slot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `Uint256Slot` with member `value` located at `slot`.
     */
    function getUint256Slot(bytes32 slot) internal pure returns (Uint256Slot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `Int256Slot` with member `value` located at `slot`.
     */
    function getInt256Slot(bytes32 slot) internal pure returns (Int256Slot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns a `StringSlot` with member `value` located at `slot`.
     */
    function getStringSlot(bytes32 slot) internal pure returns (StringSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `StringSlot` representation of the string storage pointer `store`.
     */
    function getStringSlot(string storage store) internal pure returns (StringSlot storage r) {
        assembly ("memory-safe") {
            r.slot := store.slot
        }
    }

    /**
     * @dev Returns a `BytesSlot` with member `value` located at `slot`.
     */
    function getBytesSlot(bytes32 slot) internal pure returns (BytesSlot storage r) {
        assembly ("memory-safe") {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `BytesSlot` representation of the bytes storage pointer `store`.
     */
    function getBytesSlot(bytes storage store) internal pure returns (BytesSlot storage r) {
        assembly ("memory-safe") {
            r.slot := store.slot
        }
    }
}

// File: @openzeppelin/contracts/proxy/ERC1967/ERC1967Utils.sol


// OpenZeppelin Contracts (last updated v5.2.0) (proxy/ERC1967/ERC1967Utils.sol)

pragma solidity ^0.8.22;





/**
 * @dev This library provides getters and event emitting update functions for
 * https://eips.ethereum.org/EIPS/eip-1967[ERC-1967] slots.
 */
library ERC1967Utils {
    /**
     * @dev Storage slot with the address of the current implementation.
     * This is the keccak-256 hash of "eip1967.proxy.implementation" subtracted by 1.
     */
    // solhint-disable-next-line private-vars-leading-underscore
    bytes32 internal constant IMPLEMENTATION_SLOT = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;

    /**
     * @dev The `implementation` of the proxy is invalid.
     */
    error ERC1967InvalidImplementation(address implementation);

    /**
     * @dev The `admin` of the proxy is invalid.
     */
    error ERC1967InvalidAdmin(address admin);

    /**
     * @dev The `beacon` of the proxy is invalid.
     */
    error ERC1967InvalidBeacon(address beacon);

    /**
     * @dev An upgrade function sees `msg.value > 0` that may be lost.
     */
    error ERC1967NonPayable();

    /**
     * @dev Returns the current implementation address.
     */
    function getImplementation() internal view returns (address) {
        return StorageSlot.getAddressSlot(IMPLEMENTATION_SLOT).value;
    }

    /**
     * @dev Stores a new address in the ERC-1967 implementation slot.
     */
    function _setImplementation(address newImplementation) private {
        if (newImplementation.code.length == 0) {
            revert ERC1967InvalidImplementation(newImplementation);
        }
        StorageSlot.getAddressSlot(IMPLEMENTATION_SLOT).value = newImplementation;
    }

    /**
     * @dev Performs implementation upgrade with additional setup call if data is nonempty.
     * This function is payable only if the setup call is performed, otherwise `msg.value` is rejected
     * to avoid stuck value in the contract.
     *
     * Emits an {IERC1967-Upgraded} event.
     */
    function upgradeToAndCall(address newImplementation, bytes memory data) internal {
        _setImplementation(newImplementation);
        emit IERC1967.Upgraded(newImplementation);

        if (data.length > 0) {
            Address.functionDelegateCall(newImplementation, data);
        } else {
            _checkNonPayable();
        }
    }

    /**
     * @dev Storage slot with the admin of the contract.
     * This is the keccak-256 hash of "eip1967.proxy.admin" subtracted by 1.
     */
    // solhint-disable-next-line private-vars-leading-underscore
    bytes32 internal constant ADMIN_SLOT = 0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103;

    /**
     * @dev Returns the current admin.
     *
     * TIP: To get this value clients can read directly from the storage slot shown below (specified by ERC-1967) using
     * the https://eth.wiki/json-rpc/API#eth_getstorageat[`eth_getStorageAt`] RPC call.
     * `0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103`
     */
    function getAdmin() internal view returns (address) {
        return StorageSlot.getAddressSlot(ADMIN_SLOT).value;
    }

    /**
     * @dev Stores a new address in the ERC-1967 admin slot.
     */
    function _setAdmin(address newAdmin) private {
        if (newAdmin == address(0)) {
            revert ERC1967InvalidAdmin(address(0));
        }
        StorageSlot.getAddressSlot(ADMIN_SLOT).value = newAdmin;
    }

    /**
     * @dev Changes the admin of the proxy.
     *
     * Emits an {IERC1967-AdminChanged} event.
     */
    function changeAdmin(address newAdmin) internal {
        emit IERC1967.AdminChanged(getAdmin(), newAdmin);
        _setAdmin(newAdmin);
    }

    /**
     * @dev The storage slot of the UpgradeableBeacon contract which defines the implementation for this proxy.
     * This is the keccak-256 hash of "eip1967.proxy.beacon" subtracted by 1.
     */
    // solhint-disable-next-line private-vars-leading-underscore
    bytes32 internal constant BEACON_SLOT = 0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50;

    /**
     * @dev Returns the current beacon.
     */
    function getBeacon() internal view returns (address) {
        return StorageSlot.getAddressSlot(BEACON_SLOT).value;
    }

    /**
     * @dev Stores a new beacon in the ERC-1967 beacon slot.
     */
    function _setBeacon(address newBeacon) private {
        if (newBeacon.code.length == 0) {
            revert ERC1967InvalidBeacon(newBeacon);
        }

        StorageSlot.getAddressSlot(BEACON_SLOT).value = newBeacon;

        address beaconImplementation = IBeacon(newBeacon).implementation();
        if (beaconImplementation.code.length == 0) {
            revert ERC1967InvalidImplementation(beaconImplementation);
        }
    }

    /**
     * @dev Change the beacon and trigger a setup call if data is nonempty.
     * This function is payable only if the setup call is performed, otherwise `msg.value` is rejected
     * to avoid stuck value in the contract.
     *
     * Emits an {IERC1967-BeaconUpgraded} event.
     *
     * CAUTION: Invoking this function has no effect on an instance of {BeaconProxy} since v5, since
     * it uses an immutable beacon without looking at the value of the ERC-1967 beacon slot for
     * efficiency.
     */
    function upgradeBeaconToAndCall(address newBeacon, bytes memory data) internal {
        _setBeacon(newBeacon);
        emit IERC1967.BeaconUpgraded(newBeacon);

        if (data.length > 0) {
            Address.functionDelegateCall(IBeacon(newBeacon).implementation(), data);
        } else {
            _checkNonPayable();
        }
    }

    /**
     * @dev Reverts if `msg.value` is not zero. It can be used to avoid `msg.value` stuck in the contract
     * if an upgrade doesn't perform an initialization call.
     */
    function _checkNonPayable() private {
        if (msg.value > 0) {
            revert ERC1967NonPayable();
        }
    }
}

// File: @openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol


// OpenZeppelin Contracts (last updated v5.2.0) (proxy/utils/UUPSUpgradeable.sol)

pragma solidity ^0.8.22;




/**
 * @dev An upgradeability mechanism designed for UUPS proxies. The functions included here can perform an upgrade of an
 * {ERC1967Proxy}, when this contract is set as the implementation behind such a proxy.
 *
 * A security mechanism ensures that an upgrade does not turn off upgradeability accidentally, although this risk is
 * reinstated if the upgrade retains upgradeability but removes the security mechanism, e.g. by replacing
 * `UUPSUpgradeable` with a custom implementation of upgrades.
 *
 * The {_authorizeUpgrade} function must be overridden to include access restriction to the upgrade mechanism.
 */
abstract contract UUPSUpgradeable is Initializable, IERC1822Proxiable {
    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    address private immutable __self = address(this);

    /**
     * @dev The version of the upgrade interface of the contract. If this getter is missing, both `upgradeTo(address)`
     * and `upgradeToAndCall(address,bytes)` are present, and `upgradeTo` must be used if no function should be called,
     * while `upgradeToAndCall` will invoke the `receive` function if the second argument is the empty byte string.
     * If the getter returns `"5.0.0"`, only `upgradeToAndCall(address,bytes)` is present, and the second argument must
     * be the empty byte string if no function should be called, making it impossible to invoke the `receive` function
     * during an upgrade.
     */
    string public constant UPGRADE_INTERFACE_VERSION = "5.0.0";

    /**
     * @dev The call is from an unauthorized context.
     */
    error UUPSUnauthorizedCallContext();

    /**
     * @dev The storage `slot` is unsupported as a UUID.
     */
    error UUPSUnsupportedProxiableUUID(bytes32 slot);

    /**
     * @dev Check that the execution is being performed through a delegatecall call and that the execution context is
     * a proxy contract with an implementation (as defined in ERC-1967) pointing to self. This should only be the case
     * for UUPS and transparent proxies that are using the current contract as their implementation. Execution of a
     * function through ERC-1167 minimal proxies (clones) would not normally pass this test, but is not guaranteed to
     * fail.
     */
    modifier onlyProxy() {
        _checkProxy();
        _;
    }

    /**
     * @dev Check that the execution is not being performed through a delegate call. This allows a function to be
     * callable on the implementing contract but not through proxies.
     */
    modifier notDelegated() {
        _checkNotDelegated();
        _;
    }

    function __UUPSUpgradeable_init() internal onlyInitializing {
    }

    function __UUPSUpgradeable_init_unchained() internal onlyInitializing {
    }
    /**
     * @dev Implementation of the ERC-1822 {proxiableUUID} function. This returns the storage slot used by the
     * implementation. It is used to validate the implementation's compatibility when performing an upgrade.
     *
     * IMPORTANT: A proxy pointing at a proxiable contract should not be considered proxiable itself, because this risks
     * bricking a proxy that upgrades to it, by delegating to itself until out of gas. Thus it is critical that this
     * function revert if invoked through a proxy. This is guaranteed by the `notDelegated` modifier.
     */
    function proxiableUUID() external view virtual notDelegated returns (bytes32) {
        return ERC1967Utils.IMPLEMENTATION_SLOT;
    }

    /**
     * @dev Upgrade the implementation of the proxy to `newImplementation`, and subsequently execute the function call
     * encoded in `data`.
     *
     * Calls {_authorizeUpgrade}.
     *
     * Emits an {Upgraded} event.
     *
     * @custom:oz-upgrades-unsafe-allow-reachable delegatecall
     */
    function upgradeToAndCall(address newImplementation, bytes memory data) public payable virtual onlyProxy {
        _authorizeUpgrade(newImplementation);
        _upgradeToAndCallUUPS(newImplementation, data);
    }

    /**
     * @dev Reverts if the execution is not performed via delegatecall or the execution
     * context is not of a proxy with an ERC-1967 compliant implementation pointing to self.
     * See {_onlyProxy}.
     */
    function _checkProxy() internal view virtual {
        if (
            address(this) == __self || // Must be called through delegatecall
            ERC1967Utils.getImplementation() != __self // Must be called through an active proxy
        ) {
            revert UUPSUnauthorizedCallContext();
        }
    }

    /**
     * @dev Reverts if the execution is performed via delegatecall.
     * See {notDelegated}.
     */
    function _checkNotDelegated() internal view virtual {
        if (address(this) != __self) {
            // Must not be called through delegatecall
            revert UUPSUnauthorizedCallContext();
        }
    }

    /**
     * @dev Function that should revert when `msg.sender` is not authorized to upgrade the contract. Called by
     * {upgradeToAndCall}.
     *
     * Normally, this function will use an xref:access.adoc[access control] modifier such as {Ownable-onlyOwner}.
     *
     * ```solidity
     * function _authorizeUpgrade(address) internal onlyOwner {}
     * ```
     */
    function _authorizeUpgrade(address newImplementation) internal virtual;

    /**
     * @dev Performs an implementation upgrade with a security check for UUPS proxies, and additional setup call.
     *
     * As a security check, {proxiableUUID} is invoked in the new implementation, and the return value
     * is expected to be the implementation slot in ERC-1967.
     *
     * Emits an {IERC1967-Upgraded} event.
     */
    function _upgradeToAndCallUUPS(address newImplementation, bytes memory data) private {
        try IERC1822Proxiable(newImplementation).proxiableUUID() returns (bytes32 slot) {
            if (slot != ERC1967Utils.IMPLEMENTATION_SLOT) {
                revert UUPSUnsupportedProxiableUUID(slot);
            }
            ERC1967Utils.upgradeToAndCall(newImplementation, data);
        } catch {
            // The implementation is not UUPS
            revert ERC1967Utils.ERC1967InvalidImplementation(newImplementation);
        }
    }
}

// File: presale/interfaces/ITokenMintable.sol


pragma solidity ^0.8.26;

/**
* @title ITokenMintable
* @dev Interface for token with mint and lock functionality
*/
interface ITokenMintable {
   function mint(address to, uint256 amount) external;
   function mintAndLock(address to, uint256 amount) external;
   function unlockTokens(address account, uint256 amount) external;
}


// File: presale/libraries/PresaleUtils.sol


pragma solidity ^0.8.26;

interface AggregatorV3Interface {
   function latestRoundData() external view returns (
       uint80 roundId,
       int256 answer,
       uint256 startedAt,
       uint256 updatedAt,
       uint80 answeredInRound
   );
}

/**
* @title PresaleUtils
* @dev Library providing utility functions for presale
*/
library PresaleUtils {
   // Maximum valid age for price feed
   uint256 public constant PRICE_FEED_MAX_AGE = 1 hours;

   /**
    * @dev Safe division function - prevents division by zero
    */
   function safeDiv(uint256 a, uint256 b) internal pure returns (uint256) {
       require(b > 0, "Division by zero");
       return a / b;
   }

   /**
    * @dev Try to get price from price feed
    */
   function tryGetPrice(AggregatorV3Interface feed) internal view returns (bool success, uint256 price) {
       try feed.latestRoundData() returns (uint80, int256 answer, uint256, uint256 updatedAt, uint80) {
           // Check if price is positive and recent
           if (answer > 0 && block.timestamp - updatedAt < PRICE_FEED_MAX_AGE) {
               return (true, uint256(answer));
           }
       } catch {}
       return (false, 0);
   }

   /**
    * @dev Get price using main feed, backup feed, and backup price
    */
   function getPrice(
       AggregatorV3Interface mainFeed,
       AggregatorV3Interface backupFeed,
       uint256 backupPrice
   ) internal view returns (uint256 price) {
       (bool success, uint256 fetchedPrice) = tryGetPrice(mainFeed);
       if (success) return fetchedPrice;
       
       if (address(backupFeed) != address(0)) {
           (success, fetchedPrice) = tryGetPrice(backupFeed);
           if (success) return fetchedPrice;
       }
       
       return backupPrice;
   }

   /**
    * @dev Convert ETH amount to USD value
    */
   function ethToUSD(uint256 amount, uint256 ethPrice) internal pure returns (uint256) {
       uint256 numerator = amount * ethPrice * 1e18;
       uint256 denominator = 1e18 * 1e8;
       return safeDiv(numerator, denominator);
   }

   /**
    * @dev Convert token amount to USD value
    */
   function tokenToUSD(
       uint256 amount, 
       uint256 tokenPrice, 
       uint8 tokenDecimals
   ) internal pure returns (uint256) {
       uint256 numerator = amount * tokenPrice * 1e18;
       uint256 denominator = 10**uint256(tokenDecimals) * 1e8;
       return safeDiv(numerator, denominator);
   }
}


// File: presale/RoundPresaleV4.sol


pragma solidity ^0.8.26;








/**
* @title RoundPresaleV4
* @dev Round-based token presale contract with UUPS upgradeable pattern
*/
contract RoundPresaleV4 is 
 Initializable, 
 AccessControlUpgradeable, 
 ReentrancyGuardUpgradeable, 
 PausableUpgradeable, 
 UUPSUpgradeable
{
  // Role definitions
  bytes32 public constant MANAGER_ROLE = keccak256("MANAGER_ROLE");
  bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");

  // Events
  event RoundCreated(uint256 indexed roundId, uint256 price);
  event RoundUpdated(uint256 indexed roundId);
  event RoundStarted(uint256 indexed roundId);
  event RoundEnded(uint256 indexed roundId, uint256 sold);
  event TokenPurchased(address indexed buyer, uint256 indexed orderId, uint256 amount);
  event TokenClaimed(address indexed buyer, uint256 indexed orderId);
  event SnapshotTaken(uint256 indexed snapshotId);
  event TokenAdded(address indexed token);
  event TokenRemoved(address indexed token);
  event FeedUpdated(address indexed token, bool isMain);
  event ReceiverChanged(address addr, uint256 share, bool isAdd);
  event FundsTransferred(address indexed token);
  event ConfigUpdated(string name, uint256 value);
  event CrossRoundPurchase(address indexed buyer, uint256 nextRoundId);
  event TokenMinted(address indexed buyer, uint256 amount);
  event TokenUnlocked(address indexed buyer, uint256 amount);
  event EmergencyUnlock(address indexed buyer, uint256 amount);

  // Round information
  struct Round {
      uint256 price;             // Token price (Wei)
      uint256 priceUSD;          // Token price (USD, 18 decimals)
      uint256 allocation;        // Round allocation
      uint256 sold;              // Number of tokens sold
      uint256 startTime;         // Round start time
      uint256 endTime;           // Round end time
      bool active;               // Round active status
      bool useUSDPrice;          // Use USD price flag
  }

  // Order information
  struct Order {
      address buyer;             // Buyer address
      uint256 amount;            // Purchase token amount
      uint256 price;             // Token price at purchase time
      uint256 timestamp;         // Order timestamp
      bool claimed;              // Claim status
  }

  // External view order detail information
  struct OrderDetail {
      uint256 orderId;
      uint256 amount;
      uint256 price;
      uint256 timestamp;
      bool claimed;
  }

  // Token information
  struct Token {
      uint8 decimals;            // Token decimals
      bool enabled;              // Enabled status
      address priceFeed;         // Price feed address
      address backupFeed;        // Backup price feed address
      uint256 backupPrice;       // Backup price (manual setting)
      uint256 minInvestment;     // Minimum investment amount
      uint256 threshold;         // Automatic transfer threshold
  }

  // Snapshot information
  struct Snapshot {
      uint256 timestamp;         // Snapshot timestamp
      uint256 balance;           // Snapshot balance
  }

  // Receiver information
  struct Receiver {
      address addr;              // Receiver address
      uint256 share;             // Share ratio
  }

  // Constants
  uint256 internal constant MIN_TOKENS = 1; // Minimum purchase token amount

  // State variables
  IERC20 public saleToken;                // Sale token
  AggregatorV3Interface public ethPriceFeed;         // ETH price feed
  AggregatorV3Interface public ethBackupFeed;        // ETH backup price feed
  uint256 public ethBackupPrice;                     // ETH backup price
  uint256 public ethThreshold;                       // ETH threshold
  uint256 public gasLimit;                           // Gas limit
  uint256 public minInvestment;                      // Minimum investment amount
  uint256 public batchLimit;                         // Batch limit
  uint256 public lastOrderId;                        // Last order ID
  uint256 public votePeriod;                         // Vote period
  uint256 public claimPeriod;                        // Claim period
  uint256 public requiredApprovals;                  // Required approvals
  uint256 public opDelay;                            // Operation delay
  uint256 public snapshotInterval;                   // Snapshot interval
  uint256 public lastSnapshotTime;                   // Last snapshot time
  uint256 public snapshotCount;                      // Snapshot count
  uint256 public priceUpdateInterval;                // Price update interval
  uint256 public lastPriceUpdateTime;                // Last price update time
  uint256 public maxPriceDeviation;                  // Maximum price deviation
  uint256 public roundDuration;                      // Round duration
  uint256 public currentRound;                       // Current round
  uint256 public totalRounds;                        // Total rounds
  uint256 public maxRound;                           // Maximum round
  uint256 public totalShares;                        // Total shares
  uint256 public pendingETH;                         // Pending ETH (for backward compatibility)
  uint256 public totalDistributed;                   // Total distributed
  uint256 public totalRaisedETH;                     // Total raised ETH
  uint256 public totalRaisedUSD;                     // Total raised USD
  bool public claimEnabled;                          // Claim enabled flag
  bool public autoIncreaseEnabled;                   // Auto increase enabled flag
  uint256 public autoIncreaseRate;                   // Auto increase rate
  uint256 public lastRoundCheckTime;                 // Last round check time
  bool public autoCreateEnabled;                     // Auto create enabled flag
  uint256 public defaultAllocation;                  // Default allocation
  uint256 public defaultPriceUSD;                    // Default USD price
  bool public crossRoundPurchaseEnabled;             // Cross round purchase enabled flag
  bool public directMintEnabled;                     // Direct mint enabled flag

  // Mappings
  mapping(uint256 => Round) public rounds;           // Round ID => Round information
  mapping(uint256 => Order) public orders;           // Order ID => Order information
  mapping(address => Token) public tokens;           // Token address => Token information
  mapping(uint256 => Snapshot) public snapshots;     // Snapshot ID => Snapshot information
  mapping(address => uint256[]) public userOrders;   // User address => Order ID array

  // Arrays
  address[] public tokenList;                        // Supported token list
  Receiver[] public receivers;                       // Fund receivers list

  /// @custom:oz-upgrades-unsafe-allow constructor
  constructor() {
      _disableInitializers();
  }

  /**
   * @dev Initialize contract (used instead of constructor)
   * @param _saleToken Address of the token being sold
   * @param _ethPriceFeed Address of the ETH/USD price feed
   */
  function initialize(
      address _saleToken,
      address _ethPriceFeed
  ) public initializer {
      __AccessControl_init();
      __ReentrancyGuard_init();
      __Pausable_init();
      __UUPSUpgradeable_init();

      _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
      _grantRole(MANAGER_ROLE, msg.sender);
      _grantRole(UPGRADER_ROLE, msg.sender);
      
      saleToken = IERC20(_saleToken);
      ethPriceFeed = AggregatorV3Interface(_ethPriceFeed);
      
      ethBackupPrice = 2000 * 10**8;
      ethThreshold = 1 ether;
      gasLimit = 300000;
      batchLimit = 100;
      minInvestment = 50 * 10**18;
      votePeriod = 3 days;
      claimPeriod = 30 days;
      requiredApprovals = 1;
      opDelay = 1 days;
      snapshotInterval = 1 days;
      priceUpdateInterval = 1 hours;
      maxPriceDeviation = 10;
      roundDuration = 7 days;
      autoIncreaseRate = 10;
      lastRoundCheckTime = block.timestamp;
      directMintEnabled = false;
  }

  /**
   * @dev Initialize V2 variables
   * @param _defaultAllocation Default allocation for auto-created rounds
   * @param _defaultPriceUSD Default USD price for auto-created rounds
   * @param autoIncreaseRateValue Rate at which price increases between rounds
   */
  function initV2(
      uint256 _defaultAllocation,
      uint256 _defaultPriceUSD,
      uint256 autoIncreaseRateValue
  ) external onlyRole(DEFAULT_ADMIN_ROLE) {
      lastRoundCheckTime = block.timestamp;
      maxRound = 0;
      autoIncreaseRate = autoIncreaseRateValue;
      autoIncreaseEnabled = true;
      defaultAllocation = _defaultAllocation;
      defaultPriceUSD = _defaultPriceUSD;
      autoCreateEnabled = true;
      crossRoundPurchaseEnabled = true;
  }

  /**
   * @dev Get ETH/USD price
   */
  function getETHPrice() public view returns (uint256) {
      return PresaleUtils.getPrice(ethPriceFeed, ethBackupFeed, ethBackupPrice);
  }

  /**
   * @dev Get token price
   */
  function getTokenPrice(address token) public view returns (uint256) {
      Token storage tokenInfo = tokens[token];
      require(tokenInfo.enabled, "Token not enabled");
      
      return PresaleUtils.getPrice(
          AggregatorV3Interface(tokenInfo.priceFeed),
          AggregatorV3Interface(tokenInfo.backupFeed),
          tokenInfo.backupPrice
      );
  }

  /**
   * @dev Convert ETH amount to USD value
   */
  function ethToUSD(uint256 amount) internal view returns (uint256) {
      return PresaleUtils.ethToUSD(amount, getETHPrice());
  }

  /**
   * @dev Convert token amount to USD value
   */
  function tokenToUSD(address token, uint256 amount) internal view returns (uint256) {
      uint256 tokenPrice = getTokenPrice(token);
      uint8 tokenDecimals = tokens[token].decimals;
      
      return PresaleUtils.tokenToUSD(amount, tokenPrice, tokenDecimals);
  }

  /**
   * @dev Check price updates
   */
  function _checkPrices() internal {
      if (block.timestamp < lastPriceUpdateTime + priceUpdateInterval) return;
      lastPriceUpdateTime = block.timestamp;
      getETHPrice();
  }

  /**
   * @dev Check round status and automatic transition
   */
  function _checkRound() internal {
      if (block.timestamp < lastRoundCheckTime + 1 hours) return;
      
      lastRoundCheckTime = block.timestamp;
      
      if (currentRound > 0 && rounds[currentRound].active && block.timestamp >= rounds[currentRound].endTime) {
          _endRound();
          
          uint256 nextRoundId = currentRound + 1;
          
          if (maxRound != 0 && nextRoundId > maxRound) return;
          
          if (nextRoundId <= totalRounds) {
              _startRound(nextRoundId);
          } else if (autoCreateEnabled && defaultAllocation > 0 && defaultPriceUSD > 0) {
              _createAndStartNextRound();
          }
      }
  }

  /**
   * @dev Automatically create and start next round
   */
  function _createAndStartNextRound() internal {
      totalRounds++;
      uint256 roundId = totalRounds;
      
      uint256 newPriceUSD = defaultPriceUSD;
      if (autoIncreaseEnabled && currentRound > 0) {
          newPriceUSD = defaultPriceUSD * (100 + autoIncreaseRate) / 100;
      }
      
      rounds[roundId] = Round({
          price: 0,
          priceUSD: newPriceUSD,
          allocation: defaultAllocation,
          sold: 0,
          startTime: block.timestamp,
          endTime: block.timestamp + roundDuration,
          active: true,
          useUSDPrice: true
      });
      
      currentRound = roundId;
      
      emit RoundCreated(roundId, 0);
      emit RoundStarted(roundId);
  }

  /**
   * @dev Process current round in cross round purchase
   */
  function _processCurrentRound(
      address buyer,
      uint256 remainingAllocation,
      Round storage round
  ) internal returns (bool) {
      if (remainingAllocation > 0) {
          _createOrder(buyer, remainingAllocation, round.useUSDPrice ? round.priceUSD : round.price);
          
          if (directMintEnabled) {
              ITokenMintable(address(saleToken)).mintAndLock(buyer, remainingAllocation);
              totalDistributed = totalDistributed + remainingAllocation;
              emit TokenMinted(buyer, remainingAllocation);
          }
          
          emit TokenPurchased(buyer, lastOrderId, remainingAllocation);
          round.sold = round.allocation;
      }
      
      _endRound();
      emit RoundEnded(currentRound, round.sold);
      
      return true;
  }

  /**
   * @dev Calculate tokens for next round
   */
  function _calculateNextRoundTokens(
      uint256 remainingTokens,
      uint256 investmentUSD,
      uint256 tokensAmount,
      address payToken,
      Round storage nextRound
  ) internal view returns (uint256) {
      uint256 remainingUSD;
      if (nextRound.useUSDPrice) {
          remainingUSD = (remainingTokens * nextRound.priceUSD) / 1e18;
      } else {
          if (payToken == address(0)) {
              uint256 ethAmount = (remainingTokens * nextRound.price) / 1e18;
              remainingUSD = ethAmount * getETHPrice() / 1e8;
          } else {
              if (tokensAmount > 0) {
                  remainingUSD = PresaleUtils.safeDiv(investmentUSD * remainingTokens, tokensAmount);
              } else {
                  remainingUSD = 0;
              }
          }
      }
      
      uint256 nextRoundTokens;
      if (nextRound.useUSDPrice) {
          nextRoundTokens = (remainingUSD * 1e18) / nextRound.priceUSD;
      } else {
          if (payToken == address(0)) {
              uint256 ethValue = remainingUSD * 1e18 / getETHPrice();
              nextRoundTokens = ethValue * 1e18 / nextRound.price;
          } else {
              nextRoundTokens = remainingTokens;
          }
      }
      
      return nextRoundTokens;
  }

  /**
   * @dev Process next round purchase
   */
  function _processNextRound(
      address buyer,
      uint256 nextRoundTokens,
      Round storage nextRound
  ) internal returns (uint256) {
      uint256 nextRoundAllocation = nextRound.allocation - nextRound.sold;
      uint256 actualTokens = nextRoundTokens <= nextRoundAllocation ? nextRoundTokens : nextRoundAllocation;
      
      if (actualTokens > 0) {
          _createOrder(buyer, actualTokens, nextRound.useUSDPrice ? nextRound.priceUSD : nextRound.price);
          nextRound.sold += actualTokens;
          
          if (directMintEnabled) {
              ITokenMintable(address(saleToken)).mintAndLock(buyer, actualTokens);
              totalDistributed += actualTokens;
              emit TokenMinted(buyer, actualTokens);
          }
          
          emit TokenPurchased(buyer, lastOrderId, actualTokens);
      }
      
      return actualTokens;
  }

  /**
   * @dev Handle cross round purchase
   */
  function _handleCrossRoundPurchase(
      address buyer, 
      uint256 tokensAmount, 
      uint256 remainingAllocation, 
      uint256 investmentUSD,
      address payToken
  ) internal {
      // Process current round
      Round storage round = rounds[currentRound];
      _processCurrentRound(buyer, remainingAllocation, round);
      
      // Calculate remaining tokens
      uint256 remainingTokens = tokensAmount - remainingAllocation;
      
      // If no tokens remain, exit early
      if (remainingTokens == 0) {
          return;
      }
      
      // Infinite loop prevention: limit max rounds to process
      uint256 maxRoundsToProcess = 10; // Max 10 rounds
      uint256 processedRounds = 1; // Already processed 1 round
      
      // Process remaining tokens across rounds
      while (remainingTokens > 0 && processedRounds < maxRoundsToProcess) {
          // Setup next round
          uint256 nextRoundId = currentRound + 1;
          bool nextRoundStarted = false;
          
          if (nextRoundId <= totalRounds) {
              _startRound(nextRoundId);
              nextRoundStarted = true;
          } else if (autoCreateEnabled && defaultAllocation > 0 && defaultPriceUSD > 0) {
              _createAndStartNextRound();
              nextRoundStarted = true;
          } else {
              // No next round available
              emit CrossRoundPurchase(buyer, 0);
              break;
          }
          
          if (!nextRoundStarted) break;
          
          emit CrossRoundPurchase(buyer, currentRound);
          
          // Process next round purchase
          Round storage nextRound = rounds[currentRound]; // Current round has been updated
          
          // Calculate tokens for next round
          uint256 nextRoundTokens = _calculateNextRoundTokens(
              remainingTokens,
              investmentUSD,
              tokensAmount,
              payToken,
              nextRound
          );
          
          // Process purchase in next round
          uint256 actualTokens = _processNextRound(buyer, nextRoundTokens, nextRound);
          
          // Update remaining tokens
          if (actualTokens > 0) {
              remainingTokens -= actualTokens;
          } else {
              // No tokens purchased, exit loop
              break;
          }
          
          // Check if round is complete
          if (nextRound.sold >= nextRound.allocation) {
              _endRound();
              emit RoundEnded(currentRound, nextRound.sold);
          } else {
              // Round not complete, stop processing
              break;
          }
          
          processedRounds++;
      }
  }

  /**
   * @dev Create a round
   * @param price Token price in ETH (if useUSD is false)
   * @param priceUSD Token price in USD (if useUSD is true)
   * @param allocation Total allocation for the round
   * @param useUSD Whether to use USD price
   */
  function createRound(
      uint256 price,
      uint256 priceUSD,
      uint256 allocation,
      bool useUSD
  ) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(allocation > 0, "Allocation must be greater than 0");
      require(useUSD ? priceUSD > 0 : price > 0, "Price must be greater than 0");
      
      totalRounds++;
      uint256 roundId = totalRounds;
      
      rounds[roundId] = Round({
          price: price,
          priceUSD: priceUSD,
          allocation: allocation,
          sold: 0,
          startTime: 0,
          endTime: 0,
          active: false,
          useUSDPrice: useUSD
      });
      
      emit RoundCreated(roundId, useUSD ? priceUSD : price);
  }

  /**
   * @dev Create USD round
   * @param priceUSD Token price in USD
   * @param allocation Total allocation for the round
   */
  function createUSDRound(
      uint256 priceUSD,
      uint256 allocation
  ) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(allocation > 0, "Allocation must be greater than 0");
      require(priceUSD > 0, "Price must be greater than 0");
      
      totalRounds++;
      uint256 roundId = totalRounds;
      
      rounds[roundId] = Round({
          price: 0,
          priceUSD: priceUSD,
          allocation: allocation,
          sold: 0,
          startTime: 0,
          endTime: 0,
          active: false,
          useUSDPrice: true
      });
      
      emit RoundCreated(roundId, priceUSD);
  }

  /**
   * @dev Start a round
   * @param roundId ID of the round to start
   */
  function startRound(uint256 roundId) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(roundId > 0 && roundId <= totalRounds, "Invalid round ID");
      require(!rounds[roundId].active, "Round already active");
      
      if (currentRound > 0 && rounds[currentRound].active) {
          _endRound();
      }
      
      _startRound(roundId);
  }

  /**
   * @dev Internal function to start round
   * @param roundId ID of the round to start
   */
  function _startRound(uint256 roundId) internal {
      Round storage round = rounds[roundId];
      round.active = true;
      round.startTime = block.timestamp;
      round.endTime = block.timestamp + roundDuration;
      
      currentRound = roundId;
      
      emit RoundStarted(roundId);
  }

  /**
   * @dev End current round
   */
  function endRound() external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(currentRound > 0, "No active round");
      require(rounds[currentRound].active, "Round not active");
      
      _endRound();
  }

  /**
   * @dev Internal function to end round
   */
  function _endRound() internal {
      Round storage round = rounds[currentRound];
      round.active = false;
      round.endTime = block.timestamp;
      
      emit RoundEnded(currentRound, round.sold);
      
      if (autoIncreaseEnabled && currentRound < totalRounds) {
          Round storage nextRound = rounds[currentRound + 1];
          
          if (nextRound.useUSDPrice) {
              nextRound.priceUSD = nextRound.priceUSD * (100 + autoIncreaseRate) / 100;
              emit ConfigUpdated("price", nextRound.priceUSD);
          } else {
              nextRound.price = nextRound.price * (100 + autoIncreaseRate) / 100;
              emit ConfigUpdated("price", nextRound.price);
          }
      }
  }

  /**
   * @dev Create order
   */
  function _createOrder(address buyer, uint256 amount, uint256 price) internal {
      lastOrderId++;
      
      orders[lastOrderId] = Order({
          buyer: buyer,
          amount: amount,
          price: price,
          timestamp: block.timestamp,
          claimed: directMintEnabled
      });
      
      userOrders[buyer].push(lastOrderId);
  }

  /**
   * @dev Internal buy function
   */
  function _buy(address buyer, uint256 amount, address payToken) internal {
      require(buyer != address(0), "Invalid buyer address");
      require(amount > 0, "Amount must be greater than 0");
      require(currentRound > 0, "No active round");
      
      _checkRound();
      
      require(rounds[currentRound].active, "Round not active");
      
      _checkPrices();
      
      // Calculate investment amount in USD
      uint256 investmentUSD;
      if (payToken == address(0)) {
          investmentUSD = ethToUSD(amount);
          totalRaisedETH = totalRaisedETH + amount;
      } else {
          investmentUSD = tokenToUSD(payToken, amount);
      }
      
      require(investmentUSD >= minInvestment, "Investment too small");
      totalRaisedUSD = totalRaisedUSD + investmentUSD;
      
      _processPurchase(buyer, investmentUSD, payToken, amount);
  }

  /**
   * @dev Process purchase for a round
   */
  function _processPurchase(
      address buyer, 
      uint256 investmentUSD, 
      address payToken, 
      uint256 payAmount
  ) internal {
      Round storage round = rounds[currentRound];
      
      // Calculate number of tokens to purchase
      uint256 tokensAmount;
      if (round.useUSDPrice) {
          // Calculate tokens based on USD price
          uint256 calculatedTokens = (investmentUSD * 1e18) / round.priceUSD;
          // Ensure minimum token amount
          tokensAmount = calculatedTokens < MIN_TOKENS ? MIN_TOKENS : calculatedTokens;
      } else {
          if (payToken == address(0)) {
              // Calculate tokens based on ETH price
              uint256 calculatedTokens = payAmount * 1e18 / round.price;
              // Ensure minimum token amount
              tokensAmount = calculatedTokens < MIN_TOKENS ? MIN_TOKENS : calculatedTokens;
          } else {
              // Calculate tokens based on ETH equivalent
              uint256 ethValue = investmentUSD * 1e18 / getETHPrice();
              uint256 calculatedTokens = ethValue * 1e18 / round.price;
              // Ensure minimum token amount
              tokensAmount = calculatedTokens < MIN_TOKENS ? MIN_TOKENS : calculatedTokens;
          }
      }
      
      uint256 remainingAllocation = round.allocation - round.sold;
      
      if (tokensAmount <= remainingAllocation) {
          // Standard case: purchase fits in current round
          _createOrder(buyer, tokensAmount, round.useUSDPrice ? round.priceUSD : round.price);
          round.sold = round.sold + tokensAmount;
          
          // If direct mint is enabled, mint and lock tokens immediately
          if (directMintEnabled) {
              ITokenMintable(address(saleToken)).mintAndLock(buyer, tokensAmount);
              totalDistributed = totalDistributed + tokensAmount;
              emit TokenMinted(buyer, tokensAmount);
          }
          
          emit TokenPurchased(buyer, lastOrderId, tokensAmount);
          
          // Check if round is now sold out
          if (round.sold >= round.allocation) {
              _endRound();
              emit RoundEnded(currentRound, round.sold);
              
              if (autoCreateEnabled && (maxRound == 0 || currentRound < maxRound)) {
                  lastRoundCheckTime = block.timestamp;
                  _createAndStartNextRound();
              }
          }
      } else if (crossRoundPurchaseEnabled) {
          // Cross round purchase processing
          _handleCrossRoundPurchase(buyer, tokensAmount, remainingAllocation, investmentUSD, payToken);
      } else {
          // Cross round purchase not enabled, only buy what fits
          _createOrder(buyer, remainingAllocation, round.useUSDPrice ? round.priceUSD : round.price);
          round.sold = round.allocation;
          
          // If direct mint is enabled, mint and lock tokens immediately
          if (directMintEnabled) {
              ITokenMintable(address(saleToken)).mintAndLock(buyer, remainingAllocation);
              totalDistributed = totalDistributed + remainingAllocation;
              emit TokenMinted(buyer, remainingAllocation);
          }
          
          emit TokenPurchased(buyer, lastOrderId, remainingAllocation);
          
          _endRound();
          emit RoundEnded(currentRound, round.sold);
          
          if (autoCreateEnabled && (maxRound == 0 || currentRound < maxRound)) {
              lastRoundCheckTime = block.timestamp;
              
              uint256 nextRoundId = currentRound + 1;
              if (nextRoundId <= totalRounds) {
                  _startRound(nextRoundId);
              } else if (defaultAllocation > 0 && defaultPriceUSD > 0) {
                  _createAndStartNextRound();
              }
          }
      }
      
      // Check if ETH balance exceeds threshold and transfer if needed
      if (address(this).balance >= ethThreshold) {
          _transferETH();
      }
      
      // Check token and transfer if needed
      if (payToken != address(0)) {
          _checkAndTransferToken(payToken);
      }
  }

  /**
   * @dev Transfer asset
   */
  function _transferAsset(address token, uint256 amount) internal {
      uint256 totalTransferred = 0;
      
      for (uint256 i = 0; i < receivers.length; i++) {
          uint256 share = amount * receivers[i].share / totalShares;
          
          if (token == address(0)) {
              (bool success, ) = receivers[i].addr.call{value: share, gas: gasLimit}("");
              require(success, "ETH transfer failed");
          } else {
              // Using direct transfer instead of SafeERC20
              require(IERC20(token).transfer(receivers[i].addr, share), "Token transfer failed");
          }
          
          totalTransferred += share;
      }
      
      if (totalTransferred < amount && receivers.length > 0) {
          uint256 remaining = amount - totalTransferred;
          
          if (token == address(0)) {
              (bool success, ) = receivers[0].addr.call{value: remaining, gas: gasLimit}("");
              require(success, "ETH transfer failed");
          } else {
              // Using direct transfer instead of SafeERC20
              require(IERC20(token).transfer(receivers[0].addr, remaining), "Token transfer failed");
          }
      }
  }

  /**
   * @dev Internal ETH transfer function
   */
  function _transferETH() internal {
      uint256 amount = address(this).balance;
      require(amount > 0, "No ETH balance");
      
      _transferAsset(address(0), amount);
      
      emit FundsTransferred(address(0));
  }

  /**
   * @dev Check and transfer token if threshold is reached
   */
  function _checkAndTransferToken(address token) internal {
      if (token == address(0) || token == address(saleToken)) return;
      
      IERC20 tokenContract = IERC20(token);
      uint256 balance = tokenContract.balanceOf(address(this));
      
      // Get token threshold or use minInvestment as default
      uint256 tokenThreshold = tokens[token].threshold;
      if (tokenThreshold == 0) {
          tokenThreshold = tokens[token].minInvestment;
      }
      
      if (balance >= tokenThreshold) {
          _transferAsset(token, balance);
          emit FundsTransferred(token);
      }
  }

  /**
   * @dev Claim tokens
   * @param orderId ID of the order to claim
   */
  function claim(uint256 orderId) external nonReentrant {
      require(claimEnabled, "Claiming not enabled");
      require(orderId > 0 && orderId <= lastOrderId, "Invalid order ID");
      
      Order storage order = orders[orderId];
      require(order.buyer == msg.sender, "Not the buyer");
      require(!order.claimed, "Already claimed");
      require(block.timestamp <= order.timestamp + claimPeriod, "Claim period expired");
      
      order.claimed = true;
      uint256 amount = order.amount;
      
      // In direct mint mode, don't increase totalDistributed (already increased at purchase time)
      if (!directMintEnabled) {
          totalDistributed = totalDistributed + amount;
      }
      
      emit TokenClaimed(order.buyer, orderId);
      
      if (directMintEnabled) {
          // In direct mint mode, unlock tokens
          ITokenMintable(address(saleToken)).unlockTokens(order.buyer, amount);
          emit TokenUnlocked(order.buyer, amount);
      } else {
          // Original method: transfer tokens
          require(saleToken.transfer(order.buyer, amount), "Token transfer failed");
      }
  }

  /**
   * @dev Toggle claim
   * @param enabled Whether claiming is enabled
   */
  function toggleClaim(bool enabled) public onlyRole(DEFAULT_ADMIN_ROLE) {
      claimEnabled = enabled;
      emit ConfigUpdated("claiming", enabled ? 1 : 0);
  }

  /**
   * @dev Claim all orders for a buyer
   * @param buyer Address of the buyer
   */
  function claimAll(address buyer) external nonReentrant {
      require(claimEnabled, "Claiming not enabled");
      require(buyer != address(0), "Invalid buyer address");
      
      uint256[] memory buyerOrders = userOrders[buyer];
      require(buyerOrders.length > 0, "No orders found");
      
      uint256 limit = batchLimit < buyerOrders.length ? batchLimit : buyerOrders.length;
      uint256 totalAmount = 0;
      
      for (uint256 i = 0; i < limit; i++) {
          uint256 orderId = buyerOrders[i];
          Order storage order = orders[orderId];
          
          if (!order.claimed && block.timestamp <= order.timestamp + claimPeriod) {
              order.claimed = true;
              totalAmount += order.amount;
              
              emit TokenClaimed(buyer, orderId);
          }
      }
      
      if (totalAmount > 0) {
          if (!directMintEnabled) {
              totalDistributed += totalAmount;
          }
          
          if (directMintEnabled) {
              ITokenMintable(address(saleToken)).unlockTokens(buyer, totalAmount);
              emit TokenUnlocked(buyer, totalAmount);
          } else {
              require(saleToken.transfer(buyer, totalAmount), "Token transfer failed");
          }
      }
  }

  /**
   * @dev Emergency unlock mechanism
   * @param buyer Address of the buyer
   * @param orderIds Array of order IDs to unlock
   */
  function emergencyUnlock(address buyer, uint256[] calldata orderIds) external onlyRole(DEFAULT_ADMIN_ROLE) nonReentrant {
      require(directMintEnabled, "Only available in direct mint mode");
      
      uint256 totalAmount = 0;
      for (uint256 i = 0; i < orderIds.length; i++) {
          uint256 orderId = orderIds[i];
          require(orderId > 0 && orderId <= lastOrderId, "Invalid order ID");
          
          Order storage order = orders[orderId];
          
          if (order.buyer == buyer && !order.claimed) {
              order.claimed = true;
              totalAmount += order.amount;
              emit TokenClaimed(buyer, orderId);
          }
      }
      
      if (totalAmount > 0) {
          ITokenMintable(address(saleToken)).unlockTokens(buyer, totalAmount);
          emit EmergencyUnlock(buyer, totalAmount);
      }
  }

  /**
   * @dev Toggle direct mint
   * @param enabled Whether direct mint is enabled
   */
  function toggleDirectMint(bool enabled) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(!rounds[currentRound].active || currentRound == 0, "Cannot change mint mode during active round");
      directMintEnabled = enabled;
      emit ConfigUpdated("directMint", enabled ? 1 : 0);
  }

  /**
   * @dev Buy tokens with ETH
   */
  function buy() external payable whenNotPaused {
      require(msg.value > 0, "No ETH sent");
      _buy(msg.sender, msg.value, address(0));
  }

  /**
   * @dev Buy tokens with ERC20
   * @param token Address of the token to use for purchase
   * @param amount Amount of tokens to use
   */
  function buyWithToken(address token, uint256 amount) external whenNotPaused nonReentrant {
      require(token != address(0), "Invalid token address");
      require(tokens[token].enabled, "Token not enabled");
      require(amount > 0, "Amount must be greater than 0");
      
      require(IERC20(token).transferFrom(msg.sender, address(this), amount), "Token transfer failed");
      
      _buy(msg.sender, amount, token);
      
      if (address(this).balance >= ethThreshold) {
          _transferETH();
      }
  }

  /**
   * @dev Add receiver
   * @param addr Address of the receiver
   * @param share Share of the receiver
   */
  function addRcvr(address addr, uint256 share) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(addr != address(0), "Invalid address");
      require(share > 0, "Share must be greater than 0");
      
      for (uint256 i = 0; i < receivers.length; i++) {
          require(receivers[i].addr != addr, "Receiver already exists");
      }
      
      receivers.push(Receiver({
          addr: addr,
          share: share
      }));
      
      totalShares += share;
      
      emit ReceiverChanged(addr, share, true);
  }

  /**
   * @dev Add token
   * @param token Address of the token
   * @param priceFeed Address of the price feed
   * @param backupFeed Address of the backup feed
   * @param backupPrice Backup price
   */
  function addToken(
      address token,
      address priceFeed,
      address backupFeed,
      uint256 backupPrice
  ) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(token != address(0), "Invalid token address");
      require(priceFeed != address(0), "Invalid price feed address");
      require(backupPrice > 0, "Invalid price");
      require(!tokens[token].enabled, "Token already enabled");
      require(backupPrice < 10**20, "Price too large");
      
      IERC20MetadataUpgradeable tokenContract = IERC20MetadataUpgradeable(token);
      uint8 decimals = tokenContract.decimals();
      
      tokens[token] = Token({
          decimals: decimals,
          enabled: true,
          priceFeed: priceFeed,
          backupFeed: backupFeed,
          backupPrice: backupPrice,
          minInvestment: minInvestment,
          threshold: 0
      });
      
      tokenList.push(token);
      
      emit TokenAdded(token);
  }

  /**
   * @dev Set min investment
   */
  function setMin(uint256 amount) external onlyRole(DEFAULT_ADMIN_ROLE) {
      minInvestment = amount;
      emit ConfigUpdated("min", amount);
  }

  /**
   * @dev Set ETH threshold
   */
  function setETHThreshold(uint256 threshold) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(threshold > 0, "Threshold must be greater than 0");
      ethThreshold = threshold;
      emit ConfigUpdated("threshold", threshold);
  }

  /**
   * @dev Set max round
   */
  function setMax(uint256 max) external onlyRole(DEFAULT_ADMIN_ROLE) {
      maxRound = max;
      emit ConfigUpdated("max", max);
  }

  /**
   * @dev Update round check time
   */
  function updCheckTime() external onlyRole(DEFAULT_ADMIN_ROLE) {
      lastRoundCheckTime = block.timestamp;
  }

  /**
   * @dev Set default allocation
   */
  function setAlloc(uint256 allocation) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(allocation > 0, "Allocation must be greater than 0");
      defaultAllocation = allocation;
      emit ConfigUpdated("allocation", allocation);
  }

  /**
   * @dev Set default USD price
   */
  function setPrice(uint256 priceUSD) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(priceUSD > 0, "Price must be greater than 0");
      defaultPriceUSD = priceUSD;
      emit ConfigUpdated("price", priceUSD);
  }

  /**
   * @dev Pause contract
   */
  function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
      _pause();
  }

  /**
   * @dev Unpause contract
   */
  function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
      _unpause();
  }

  /**
   * @dev Change sale token
   */
  function changeToken(address newToken) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(newToken != address(0), "Invalid token address");
      saleToken = IERC20(newToken);
  }

  /**
   * @dev Get orders by buyer
   */
  function getOrders(address buyer) external view returns (uint256[] memory) {
      return userOrders[buyer];
  }

  /**
   * @dev Get order count
   */
  function getOrderCount(address buyer) external view returns (uint256) {
      return userOrders[buyer].length;
  }

  /**
   * @dev Toggle auto settings
   */
  function toggleAuto(bool autoIncrease, bool autoCreate) external onlyRole(DEFAULT_ADMIN_ROLE) {
      autoIncreaseEnabled = autoIncrease;
      autoCreateEnabled = autoCreate;
      emit ConfigUpdated("auto", autoIncrease ? 1 : 0);
  }

  /**
   * @dev Set auto increase rate
   */
  function setRate(uint256 rate) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(rate > 0 && rate <= 50, "Rate must be between 1 and 50");
      autoIncreaseRate = rate;
      emit ConfigUpdated("rate", rate);
  }

  /**
   * @dev Set round duration
   */
  function setDuration(uint256 duration) external onlyRole(DEFAULT_ADMIN_ROLE) {
      require(duration >= 600, "Duration must be at least 600 seconds");
      roundDuration = duration;
      emit ConfigUpdated("duration", duration);
  }

  /**
   * @dev Toggle cross-round purchase
   */
  function toggleCross(bool enabled) external onlyRole(DEFAULT_ADMIN_ROLE) {
      crossRoundPurchaseEnabled = enabled;
      emit ConfigUpdated("crossRound", enabled ? 1 : 0);
  }

  /**
   * @dev Manual ETH transfer
   */
  function transferETH() external onlyRole(DEFAULT_ADMIN_ROLE) nonReentrant {
      _transferETH();
  }

  /**
   * @dev Emergency ETH transfer function
   */
  function emergencyETH() external onlyRole(DEFAULT_ADMIN_ROLE) nonReentrant {
      uint256 balance = address(this).balance;
      require(balance > 0, "No ETH balance");
      
      _transferAsset(address(0), balance);
      emit FundsTransferred(address(0));
  }

  /**
   * @dev Transfer ERC20 token
   */
  function transferToken(address token) external onlyRole(DEFAULT_ADMIN_ROLE) nonReentrant {
      require(token != address(0), "Invalid token address");
      require(token != address(saleToken), "Cannot transfer sale token");
      
      IERC20 tokenContract = IERC20(token);
      uint256 amount = tokenContract.balanceOf(address(this));
      require(amount > 0, "No token balance");
      
      _transferAsset(token, amount);
      
      emit FundsTransferred(token);
  }

  /**
   * @dev Reset contract state
   */
  function reset(
      address _saleToken,
      address _ethPriceFeed
  ) external onlyRole(DEFAULT_ADMIN_ROLE) {
      saleToken = IERC20(_saleToken);
      ethPriceFeed = AggregatorV3Interface(_ethPriceFeed);
      ethBackupFeed = AggregatorV3Interface(address(0));
      
      ethBackupPrice = 2000 * 10**8;
      ethThreshold = 0.1 ether;
      gasLimit = 300000;
      batchLimit = 100;
      minInvestment = 1 * 10**18;
      claimPeriod = 365 days;
      snapshotInterval = 1 days;
      priceUpdateInterval = 1 hours;
      roundDuration = 500;
      autoIncreaseRate = 10;
      
      lastOrderId = 0;
      snapshotCount = 0;
      totalRounds = 0;
      currentRound = 0;
      maxRound = 0;
      totalShares = 0;
      
      pendingETH = 0;
      totalDistributed = 0;
      totalRaisedETH = 50;
      totalRaisedUSD = 0;
      
      claimEnabled = false;
      autoIncreaseEnabled = true;
      autoCreateEnabled = true;
      crossRoundPurchaseEnabled = true;
      
      lastSnapshotTime = block.timestamp;
      lastPriceUpdateTime = block.timestamp;
      lastRoundCheckTime = block.timestamp;
      
      delete tokenList;
      delete receivers;
      
      defaultAllocation = 0;
      defaultPriceUSD = 0;
      
      emit ConfigUpdated("reset", block.timestamp);
  }

  /**
   * @dev Check upgrade authorization (UUPS pattern)
   * @param newImplementation Address of the new implementation
   */
  function _authorizeUpgrade(address newImplementation) internal override onlyRole(UPGRADER_ROLE) {
      // Authorization check is done by the role check in the modifier
  }

  /**
   * @dev Receive function to buy tokens when receiving ETH
   */
  receive() external payable {
      require(msg.value > 0, "No ETH sent");
      _buy(msg.sender, msg.value, address(0));
  }
}

// IERC20 interface definition
interface IERC20 {
 function totalSupply() external view returns (uint256);
 function balanceOf(address account) external view returns (uint256);
 function transfer(address recipient, uint256 amount) external returns (bool);
 function allowance(address owner, address spender) external view returns (uint256);
 function approve(address spender, uint256 amount) external returns (bool);
 function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
 function decimals() external view returns (uint8);
}

// IERC20MetadataUpgradeable interface definition
interface IERC20MetadataUpgradeable is IERC20 {
 function name() external view returns (string memory);
 function symbol() external view returns (string memory);
}