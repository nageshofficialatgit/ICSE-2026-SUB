// Sources flattened with hardhat v2.22.10 https://hardhat.org

// SPDX-License-Identifier: MIT AND MPL-2.0

// File @openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
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


// File @openzeppelin/contracts-upgradeable/utils/ContextUpgradeable.sol@v5.0.2

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


// File @openzeppelin/contracts/utils/introspection/IERC165.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (utils/introspection/IERC165.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC165 standard, as defined in the
 * https://eips.ethereum.org/EIPS/eip-165[EIP].
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
     * https://eips.ethereum.org/EIPS/eip-165#how-interfaces-are-identified[EIP section]
     * to learn more about how these ids are created.
     *
     * This function call must use less than 30 000 gas.
     */
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}


// File @openzeppelin/contracts-upgradeable/utils/introspection/ERC165Upgradeable.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (utils/introspection/ERC165.sol)

pragma solidity ^0.8.20;


/**
 * @dev Implementation of the {IERC165} interface.
 *
 * Contracts that want to implement ERC165 should inherit from this contract and override {supportsInterface} to check
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


// File @openzeppelin/contracts/access/IAccessControl.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (access/IAccessControl.sol)

pragma solidity ^0.8.20;

/**
 * @dev External interface of AccessControl declared to support ERC165 detection.
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
     * `sender` is the account that originated the contract call, an admin role
     * bearer except when using {AccessControl-_setupRole}.
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


// File @openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol@v5.0.2

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


// File @openzeppelin/contracts-upgradeable/metatx/ERC2771ContextUpgradeable.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.1) (metatx/ERC2771Context.sol)

pragma solidity ^0.8.20;


/**
 * @dev Context variant with ERC2771 support.
 *
 * WARNING: Avoid using this pattern in contracts that rely in a specific calldata length as they'll
 * be affected by any forwarder whose `msg.data` is suffixed with the `from` address according to the ERC2771
 * specification adding the address size in bytes (20) to the calldata size. An example of an unexpected
 * behavior could be an unintended fallback (or another function) invocation while trying to invoke the `receive`
 * function only accessible if `msg.data.length == 0`.
 *
 * WARNING: The usage of `delegatecall` in this contract is dangerous and may result in context corruption.
 * Any forwarded request to this contract triggering a `delegatecall` to itself will result in an invalid {_msgSender}
 * recovery.
 */
abstract contract ERC2771ContextUpgradeable is Initializable, ContextUpgradeable {
    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    address private immutable _trustedForwarder;

    /**
     * @dev Initializes the contract with a trusted forwarder, which will be able to
     * invoke functions on this contract on behalf of other accounts.
     *
     * NOTE: The trusted forwarder can be replaced by overriding {trustedForwarder}.
     */
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor(address trustedForwarder_) {
        _trustedForwarder = trustedForwarder_;
    }

    /**
     * @dev Returns the address of the trusted forwarder.
     */
    function trustedForwarder() public view virtual returns (address) {
        return _trustedForwarder;
    }

    /**
     * @dev Indicates whether any particular address is the trusted forwarder.
     */
    function isTrustedForwarder(address forwarder) public view virtual returns (bool) {
        return forwarder == trustedForwarder();
    }

    /**
     * @dev Override for `msg.sender`. Defaults to the original `msg.sender` whenever
     * a call is not performed by the trusted forwarder or the calldata length is less than
     * 20 bytes (an address length).
     */
    function _msgSender() internal view virtual override returns (address) {
        uint256 calldataLength = msg.data.length;
        uint256 contextSuffixLength = _contextSuffixLength();
        if (isTrustedForwarder(msg.sender) && calldataLength >= contextSuffixLength) {
            return address(bytes20(msg.data[calldataLength - contextSuffixLength:]));
        } else {
            return super._msgSender();
        }
    }

    /**
     * @dev Override for `msg.data`. Defaults to the original `msg.data` whenever
     * a call is not performed by the trusted forwarder or the calldata length is less than
     * 20 bytes (an address length).
     */
    function _msgData() internal view virtual override returns (bytes calldata) {
        uint256 calldataLength = msg.data.length;
        uint256 contextSuffixLength = _contextSuffixLength();
        if (isTrustedForwarder(msg.sender) && calldataLength >= contextSuffixLength) {
            return msg.data[:calldataLength - contextSuffixLength];
        } else {
            return super._msgData();
        }
    }

    /**
     * @dev ERC-2771 specifies the context as being a single address (20 bytes).
     */
    function _contextSuffixLength() internal view virtual override returns (uint256) {
        return 20;
    }
}


// File @openzeppelin/contracts/interfaces/draft-IERC6093.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
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


// File @openzeppelin/contracts/token/ERC20/IERC20.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
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


// File @openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
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


// File @openzeppelin/contracts-upgradeable/token/ERC20/ERC20Upgradeable.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
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
abstract contract ERC20Upgradeable is Initializable, ContextUpgradeable, IERC20, IERC20Metadata, IERC20Errors {
    /// @custom:storage-location erc7201:openzeppelin.storage.ERC20
    struct ERC20Storage {
        mapping(address account => uint256) _balances;

        mapping(address account => mapping(address spender => uint256)) _allowances;

        uint256 _totalSupply;

        string _name;
        string _symbol;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.ERC20")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant ERC20StorageLocation = 0x52c63247e1f47db19d5ce0460030c497f067ca4cebf71ba98eeadabe20bace00;

    function _getERC20Storage() private pure returns (ERC20Storage storage $) {
        assembly {
            $.slot := ERC20StorageLocation
        }
    }

    /**
     * @dev Sets the values for {name} and {symbol}.
     *
     * All two of these values are immutable: they can only be set once during
     * construction.
     */
    function __ERC20_init(string memory name_, string memory symbol_) internal onlyInitializing {
        __ERC20_init_unchained(name_, symbol_);
    }

    function __ERC20_init_unchained(string memory name_, string memory symbol_) internal onlyInitializing {
        ERC20Storage storage $ = _getERC20Storage();
        $._name = name_;
        $._symbol = symbol_;
    }

    /**
     * @dev Returns the name of the token.
     */
    function name() public view virtual returns (string memory) {
        ERC20Storage storage $ = _getERC20Storage();
        return $._name;
    }

    /**
     * @dev Returns the symbol of the token, usually a shorter version of the
     * name.
     */
    function symbol() public view virtual returns (string memory) {
        ERC20Storage storage $ = _getERC20Storage();
        return $._symbol;
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
        ERC20Storage storage $ = _getERC20Storage();
        return $._totalSupply;
    }

    /**
     * @dev See {IERC20-balanceOf}.
     */
    function balanceOf(address account) public view virtual returns (uint256) {
        ERC20Storage storage $ = _getERC20Storage();
        return $._balances[account];
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
        ERC20Storage storage $ = _getERC20Storage();
        return $._allowances[owner][spender];
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
        ERC20Storage storage $ = _getERC20Storage();
        if (from == address(0)) {
            // Overflow check required: The rest of the code assumes that totalSupply never overflows
            $._totalSupply += value;
        } else {
            uint256 fromBalance = $._balances[from];
            if (fromBalance < value) {
                revert ERC20InsufficientBalance(from, fromBalance, value);
            }
            unchecked {
                // Overflow not possible: value <= fromBalance <= totalSupply.
                $._balances[from] = fromBalance - value;
            }
        }

        if (to == address(0)) {
            unchecked {
                // Overflow not possible: value <= totalSupply or value <= fromBalance <= totalSupply.
                $._totalSupply -= value;
            }
        } else {
            unchecked {
                // Overflow not possible: balance + value is at most totalSupply, which we know fits into a uint256.
                $._balances[to] += value;
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
        ERC20Storage storage $ = _getERC20Storage();
        if (owner == address(0)) {
            revert ERC20InvalidApprover(address(0));
        }
        if (spender == address(0)) {
            revert ERC20InvalidSpender(address(0));
        }
        $._allowances[owner][spender] = value;
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


// File @openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol@v5.0.2

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
abstract contract PausableUpgradeable is Initializable, ContextUpgradeable {
    /// @custom:storage-location erc7201:openzeppelin.storage.Pausable
    struct PausableStorage {
        bool _paused;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.Pausable")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant PausableStorageLocation = 0xcd5ed15c6e187e77e9aee88184c21f4f2182ab5827cb3b7e07fbedcd63f03300;

    function _getPausableStorage() private pure returns (PausableStorage storage $) {
        assembly {
            $.slot := PausableStorageLocation
        }
    }

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
    function __Pausable_init() internal onlyInitializing {
        __Pausable_init_unchained();
    }

    function __Pausable_init_unchained() internal onlyInitializing {
        PausableStorage storage $ = _getPausableStorage();
        $._paused = false;
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
        PausableStorage storage $ = _getPausableStorage();
        return $._paused;
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
        PausableStorage storage $ = _getPausableStorage();
        $._paused = true;
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
        PausableStorage storage $ = _getPausableStorage();
        $._paused = false;
        emit Unpaused(_msgSender());
    }
}


// File @openzeppelin/contracts/utils/math/Math.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (utils/math/Math.sol)

pragma solidity ^0.8.20;

/**
 * @dev Standard math utilities missing in the Solidity language.
 */
library Math {
    /**
     * @dev Muldiv operation overflow.
     */
    error MathOverflowedMulDiv();

    enum Rounding {
        Floor, // Toward negative infinity
        Ceil, // Toward positive infinity
        Trunc, // Toward zero
        Expand // Away from zero
    }

    /**
     * @dev Returns the addition of two unsigned integers, with an overflow flag.
     */
    function tryAdd(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            uint256 c = a + b;
            if (c < a) return (false, 0);
            return (true, c);
        }
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, with an overflow flag.
     */
    function trySub(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b > a) return (false, 0);
            return (true, a - b);
        }
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, with an overflow flag.
     */
    function tryMul(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            // Gas optimization: this is cheaper than requiring 'a' not being zero, but the
            // benefit is lost if 'b' is also tested.
            // See: https://github.com/OpenZeppelin/openzeppelin-contracts/pull/522
            if (a == 0) return (true, 0);
            uint256 c = a * b;
            if (c / a != b) return (false, 0);
            return (true, c);
        }
    }

    /**
     * @dev Returns the division of two unsigned integers, with a division by zero flag.
     */
    function tryDiv(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a / b);
        }
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers, with a division by zero flag.
     */
    function tryMod(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a % b);
        }
    }

    /**
     * @dev Returns the largest of two numbers.
     */
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a > b ? a : b;
    }

    /**
     * @dev Returns the smallest of two numbers.
     */
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }

    /**
     * @dev Returns the average of two numbers. The result is rounded towards
     * zero.
     */
    function average(uint256 a, uint256 b) internal pure returns (uint256) {
        // (a + b) / 2 can overflow.
        return (a & b) + (a ^ b) / 2;
    }

    /**
     * @dev Returns the ceiling of the division of two numbers.
     *
     * This differs from standard division with `/` in that it rounds towards infinity instead
     * of rounding towards zero.
     */
    function ceilDiv(uint256 a, uint256 b) internal pure returns (uint256) {
        if (b == 0) {
            // Guarantee the same behavior as in a regular Solidity division.
            return a / b;
        }

        // (a + b - 1) / b can overflow on addition, so we distribute.
        return a == 0 ? 0 : (a - 1) / b + 1;
    }

    /**
     * @notice Calculates floor(x * y / denominator) with full precision. Throws if result overflows a uint256 or
     * denominator == 0.
     * @dev Original credit to Remco Bloemen under MIT license (https://xn--2-umb.com/21/muldiv) with further edits by
     * Uniswap Labs also under MIT license.
     */
    function mulDiv(uint256 x, uint256 y, uint256 denominator) internal pure returns (uint256 result) {
        unchecked {
            // 512-bit multiply [prod1 prod0] = x * y. Compute the product mod 2^256 and mod 2^256 - 1, then use
            // use the Chinese Remainder Theorem to reconstruct the 512 bit result. The result is stored in two 256
            // variables such that product = prod1 * 2^256 + prod0.
            uint256 prod0 = x * y; // Least significant 256 bits of the product
            uint256 prod1; // Most significant 256 bits of the product
            assembly {
                let mm := mulmod(x, y, not(0))
                prod1 := sub(sub(mm, prod0), lt(mm, prod0))
            }

            // Handle non-overflow cases, 256 by 256 division.
            if (prod1 == 0) {
                // Solidity will revert if denominator == 0, unlike the div opcode on its own.
                // The surrounding unchecked block does not change this fact.
                // See https://docs.soliditylang.org/en/latest/control-structures.html#checked-or-unchecked-arithmetic.
                return prod0 / denominator;
            }

            // Make sure the result is less than 2^256. Also prevents denominator == 0.
            if (denominator <= prod1) {
                revert MathOverflowedMulDiv();
            }

            ///////////////////////////////////////////////
            // 512 by 256 division.
            ///////////////////////////////////////////////

            // Make division exact by subtracting the remainder from [prod1 prod0].
            uint256 remainder;
            assembly {
                // Compute remainder using mulmod.
                remainder := mulmod(x, y, denominator)

                // Subtract 256 bit number from 512 bit number.
                prod1 := sub(prod1, gt(remainder, prod0))
                prod0 := sub(prod0, remainder)
            }

            // Factor powers of two out of denominator and compute largest power of two divisor of denominator.
            // Always >= 1. See https://cs.stackexchange.com/q/138556/92363.

            uint256 twos = denominator & (0 - denominator);
            assembly {
                // Divide denominator by twos.
                denominator := div(denominator, twos)

                // Divide [prod1 prod0] by twos.
                prod0 := div(prod0, twos)

                // Flip twos such that it is 2^256 / twos. If twos is zero, then it becomes one.
                twos := add(div(sub(0, twos), twos), 1)
            }

            // Shift in bits from prod1 into prod0.
            prod0 |= prod1 * twos;

            // Invert denominator mod 2^256. Now that denominator is an odd number, it has an inverse modulo 2^256 such
            // that denominator * inv = 1 mod 2^256. Compute the inverse by starting with a seed that is correct for
            // four bits. That is, denominator * inv = 1 mod 2^4.
            uint256 inverse = (3 * denominator) ^ 2;

            // Use the Newton-Raphson iteration to improve the precision. Thanks to Hensel's lifting lemma, this also
            // works in modular arithmetic, doubling the correct bits in each step.
            inverse *= 2 - denominator * inverse; // inverse mod 2^8
            inverse *= 2 - denominator * inverse; // inverse mod 2^16
            inverse *= 2 - denominator * inverse; // inverse mod 2^32
            inverse *= 2 - denominator * inverse; // inverse mod 2^64
            inverse *= 2 - denominator * inverse; // inverse mod 2^128
            inverse *= 2 - denominator * inverse; // inverse mod 2^256

            // Because the division is now exact we can divide by multiplying with the modular inverse of denominator.
            // This will give us the correct result modulo 2^256. Since the preconditions guarantee that the outcome is
            // less than 2^256, this is the final result. We don't need to compute the high bits of the result and prod1
            // is no longer required.
            result = prod0 * inverse;
            return result;
        }
    }

    /**
     * @notice Calculates x * y / denominator with full precision, following the selected rounding direction.
     */
    function mulDiv(uint256 x, uint256 y, uint256 denominator, Rounding rounding) internal pure returns (uint256) {
        uint256 result = mulDiv(x, y, denominator);
        if (unsignedRoundsUp(rounding) && mulmod(x, y, denominator) > 0) {
            result += 1;
        }
        return result;
    }

    /**
     * @dev Returns the square root of a number. If the number is not a perfect square, the value is rounded
     * towards zero.
     *
     * Inspired by Henry S. Warren, Jr.'s "Hacker's Delight" (Chapter 11).
     */
    function sqrt(uint256 a) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }

        // For our first guess, we get the biggest power of 2 which is smaller than the square root of the target.
        //
        // We know that the "msb" (most significant bit) of our target number `a` is a power of 2 such that we have
        // `msb(a) <= a < 2*msb(a)`. This value can be written `msb(a)=2**k` with `k=log2(a)`.
        //
        // This can be rewritten `2**log2(a) <= a < 2**(log2(a) + 1)`
        // → `sqrt(2**k) <= sqrt(a) < sqrt(2**(k+1))`
        // → `2**(k/2) <= sqrt(a) < 2**((k+1)/2) <= 2**(k/2 + 1)`
        //
        // Consequently, `2**(log2(a) / 2)` is a good first approximation of `sqrt(a)` with at least 1 correct bit.
        uint256 result = 1 << (log2(a) >> 1);

        // At this point `result` is an estimation with one bit of precision. We know the true value is a uint128,
        // since it is the square root of a uint256. Newton's method converges quadratically (precision doubles at
        // every iteration). We thus need at most 7 iteration to turn our partial result with one bit of precision
        // into the expected uint128 result.
        unchecked {
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            return min(result, a / result);
        }
    }

    /**
     * @notice Calculates sqrt(a), following the selected rounding direction.
     */
    function sqrt(uint256 a, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = sqrt(a);
            return result + (unsignedRoundsUp(rounding) && result * result < a ? 1 : 0);
        }
    }

    /**
     * @dev Return the log in base 2 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     */
    function log2(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >> 128 > 0) {
                value >>= 128;
                result += 128;
            }
            if (value >> 64 > 0) {
                value >>= 64;
                result += 64;
            }
            if (value >> 32 > 0) {
                value >>= 32;
                result += 32;
            }
            if (value >> 16 > 0) {
                value >>= 16;
                result += 16;
            }
            if (value >> 8 > 0) {
                value >>= 8;
                result += 8;
            }
            if (value >> 4 > 0) {
                value >>= 4;
                result += 4;
            }
            if (value >> 2 > 0) {
                value >>= 2;
                result += 2;
            }
            if (value >> 1 > 0) {
                result += 1;
            }
        }
        return result;
    }

    /**
     * @dev Return the log in base 2, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log2(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log2(value);
            return result + (unsignedRoundsUp(rounding) && 1 << result < value ? 1 : 0);
        }
    }

    /**
     * @dev Return the log in base 10 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     */
    function log10(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >= 10 ** 64) {
                value /= 10 ** 64;
                result += 64;
            }
            if (value >= 10 ** 32) {
                value /= 10 ** 32;
                result += 32;
            }
            if (value >= 10 ** 16) {
                value /= 10 ** 16;
                result += 16;
            }
            if (value >= 10 ** 8) {
                value /= 10 ** 8;
                result += 8;
            }
            if (value >= 10 ** 4) {
                value /= 10 ** 4;
                result += 4;
            }
            if (value >= 10 ** 2) {
                value /= 10 ** 2;
                result += 2;
            }
            if (value >= 10 ** 1) {
                result += 1;
            }
        }
        return result;
    }

    /**
     * @dev Return the log in base 10, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log10(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log10(value);
            return result + (unsignedRoundsUp(rounding) && 10 ** result < value ? 1 : 0);
        }
    }

    /**
     * @dev Return the log in base 256 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     *
     * Adding one to the result gives the number of pairs of hex symbols needed to represent `value` as a hex string.
     */
    function log256(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >> 128 > 0) {
                value >>= 128;
                result += 16;
            }
            if (value >> 64 > 0) {
                value >>= 64;
                result += 8;
            }
            if (value >> 32 > 0) {
                value >>= 32;
                result += 4;
            }
            if (value >> 16 > 0) {
                value >>= 16;
                result += 2;
            }
            if (value >> 8 > 0) {
                result += 1;
            }
        }
        return result;
    }

    /**
     * @dev Return the log in base 256, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log256(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log256(value);
            return result + (unsignedRoundsUp(rounding) && 1 << (result << 3) < value ? 1 : 0);
        }
    }

    /**
     * @dev Returns whether a provided rounding mode is considered rounding up for unsigned integers.
     */
    function unsignedRoundsUp(Rounding rounding) internal pure returns (bool) {
        return uint8(rounding) % 2 == 1;
    }
}


// File @openzeppelin/contracts/utils/StorageSlot.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (utils/StorageSlot.sol)
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
 * Example usage to set ERC1967 implementation slot:
 * ```solidity
 * contract ERC1967 {
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
        /// @solidity memory-safe-assembly
        assembly {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `BooleanSlot` with member `value` located at `slot`.
     */
    function getBooleanSlot(bytes32 slot) internal pure returns (BooleanSlot storage r) {
        /// @solidity memory-safe-assembly
        assembly {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `Bytes32Slot` with member `value` located at `slot`.
     */
    function getBytes32Slot(bytes32 slot) internal pure returns (Bytes32Slot storage r) {
        /// @solidity memory-safe-assembly
        assembly {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `Uint256Slot` with member `value` located at `slot`.
     */
    function getUint256Slot(bytes32 slot) internal pure returns (Uint256Slot storage r) {
        /// @solidity memory-safe-assembly
        assembly {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `StringSlot` with member `value` located at `slot`.
     */
    function getStringSlot(bytes32 slot) internal pure returns (StringSlot storage r) {
        /// @solidity memory-safe-assembly
        assembly {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `StringSlot` representation of the string storage pointer `store`.
     */
    function getStringSlot(string storage store) internal pure returns (StringSlot storage r) {
        /// @solidity memory-safe-assembly
        assembly {
            r.slot := store.slot
        }
    }

    /**
     * @dev Returns an `BytesSlot` with member `value` located at `slot`.
     */
    function getBytesSlot(bytes32 slot) internal pure returns (BytesSlot storage r) {
        /// @solidity memory-safe-assembly
        assembly {
            r.slot := slot
        }
    }

    /**
     * @dev Returns an `BytesSlot` representation of the bytes storage pointer `store`.
     */
    function getBytesSlot(bytes storage store) internal pure returns (BytesSlot storage r) {
        /// @solidity memory-safe-assembly
        assembly {
            r.slot := store.slot
        }
    }
}


// File @openzeppelin/contracts/utils/Arrays.sol@v5.0.2

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (utils/Arrays.sol)

pragma solidity ^0.8.20;


/**
 * @dev Collection of functions related to array types.
 */
library Arrays {
    using StorageSlot for bytes32;

    /**
     * @dev Searches a sorted `array` and returns the first index that contains
     * a value greater or equal to `element`. If no such index exists (i.e. all
     * values in the array are strictly less than `element`), the array length is
     * returned. Time complexity O(log n).
     *
     * `array` is expected to be sorted in ascending order, and to contain no
     * repeated elements.
     */
    function findUpperBound(uint256[] storage array, uint256 element) internal view returns (uint256) {
        uint256 low = 0;
        uint256 high = array.length;

        if (high == 0) {
            return 0;
        }

        while (low < high) {
            uint256 mid = Math.average(low, high);

            // Note that mid will always be strictly less than high (i.e. it will be a valid array index)
            // because Math.average rounds towards zero (it does integer division with truncation).
            if (unsafeAccess(array, mid).value > element) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        // At this point `low` is the exclusive upper bound. We will return the inclusive upper bound.
        if (low > 0 && unsafeAccess(array, low - 1).value == element) {
            return low - 1;
        } else {
            return low;
        }
    }

    /**
     * @dev Access an array in an "unsafe" way. Skips solidity "index-out-of-range" check.
     *
     * WARNING: Only use if you are certain `pos` is lower than the array length.
     */
    function unsafeAccess(address[] storage arr, uint256 pos) internal pure returns (StorageSlot.AddressSlot storage) {
        bytes32 slot;
        // We use assembly to calculate the storage slot of the element at index `pos` of the dynamic array `arr`
        // following https://docs.soliditylang.org/en/v0.8.20/internals/layout_in_storage.html#mappings-and-dynamic-arrays.

        /// @solidity memory-safe-assembly
        assembly {
            mstore(0, arr.slot)
            slot := add(keccak256(0, 0x20), pos)
        }
        return slot.getAddressSlot();
    }

    /**
     * @dev Access an array in an "unsafe" way. Skips solidity "index-out-of-range" check.
     *
     * WARNING: Only use if you are certain `pos` is lower than the array length.
     */
    function unsafeAccess(bytes32[] storage arr, uint256 pos) internal pure returns (StorageSlot.Bytes32Slot storage) {
        bytes32 slot;
        // We use assembly to calculate the storage slot of the element at index `pos` of the dynamic array `arr`
        // following https://docs.soliditylang.org/en/v0.8.20/internals/layout_in_storage.html#mappings-and-dynamic-arrays.

        /// @solidity memory-safe-assembly
        assembly {
            mstore(0, arr.slot)
            slot := add(keccak256(0, 0x20), pos)
        }
        return slot.getBytes32Slot();
    }

    /**
     * @dev Access an array in an "unsafe" way. Skips solidity "index-out-of-range" check.
     *
     * WARNING: Only use if you are certain `pos` is lower than the array length.
     */
    function unsafeAccess(uint256[] storage arr, uint256 pos) internal pure returns (StorageSlot.Uint256Slot storage) {
        bytes32 slot;
        // We use assembly to calculate the storage slot of the element at index `pos` of the dynamic array `arr`
        // following https://docs.soliditylang.org/en/v0.8.20/internals/layout_in_storage.html#mappings-and-dynamic-arrays.

        /// @solidity memory-safe-assembly
        assembly {
            mstore(0, arr.slot)
            slot := add(keccak256(0, 0x20), pos)
        }
        return slot.getUint256Slot();
    }

    /**
     * @dev Access an array in an "unsafe" way. Skips solidity "index-out-of-range" check.
     *
     * WARNING: Only use if you are certain `pos` is lower than the array length.
     */
    function unsafeMemoryAccess(uint256[] memory arr, uint256 pos) internal pure returns (uint256 res) {
        assembly {
            res := mload(add(add(arr, 0x20), mul(pos, 0x20)))
        }
    }

    /**
     * @dev Access an array in an "unsafe" way. Skips solidity "index-out-of-range" check.
     *
     * WARNING: Only use if you are certain `pos` is lower than the array length.
     */
    function unsafeMemoryAccess(address[] memory arr, uint256 pos) internal pure returns (address res) {
        assembly {
            res := mload(add(add(arr, 0x20), mul(pos, 0x20)))
        }
    }
}


// File contracts/interfaces/engine/draft-IERC1643.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

/// @title IERC1643 Document Management 
/// (part of the ERC1400 Security Token Standards)
interface IERC1643 {
    // Document Management
    function getDocument(bytes32 _name) external view returns (string memory , bytes32, uint256);
    function getAllDocuments() external view returns (bytes32[] memory);
}


// File contracts/interfaces/engine/IAuthorizationEngine.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

/*
* @dev minimum interface to define an AuthorizationEngine
*/
interface IAuthorizationEngine {
    /**
     * @dev Returns true if the operation is authorized, and false otherwise.
     */
    function operateOnGrantRole(
        bytes32 role, address account
    ) external returns (bool isValid);

    /**
     * @dev Returns true if the operation is authorized, and false otherwise.
     */
    function operateOnRevokeRole(
        bytes32 role, address account
    ) external returns (bool isValid);
   
}


// File contracts/interfaces/engine/IDebtGlobal.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

/**
* @notice interface to represent debt tokens
*/
interface IDebtGlobal {
    struct DebtBase {
        uint256 interestRate;
        uint256 parValue;
        string guarantor;
        string bondHolder;
        string maturityDate;
        string interestScheduleFormat;
        string interestPaymentDate;
        string dayCountConvention;
        string businessDayConvention;
        string publicHolidaysCalendar;
        string issuanceDate;
        string couponFrequency;
    }

    struct CreditEvents {
        bool flagDefault;
        bool flagRedeemed;
        string rating;
    }
}


// File contracts/interfaces/engine/IDebtEngine.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

/*
* @dev minimum interface to define a DebtEngine
*/
interface IDebtEngine is IDebtGlobal {
    /**
     * @dev Returns debt information
     */
    function debt() external view returns(IDebtGlobal.DebtBase memory);
    /**
     * @dev Returns credit events
     */
    function creditEvents() external view returns(IDebtGlobal.CreditEvents memory);
   
}


// File contracts/interfaces/draft-IERC1404/draft-IERC1404.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

/*
* @dev Contrary to the ERC-1404, this interface does not inherit from the ERC20 interface
*/
interface IERC1404 {
    /**
     * @dev See ERC-1404
     *
     */
    function detectTransferRestriction(
        address _from,
        address _to,
        uint256 _amount
    ) external view returns (uint8);

    /**
     * @dev See ERC-1404
     *
     */
    function messageForTransferRestriction(
        uint8 _restrictionCode
    ) external view returns (string memory);
}


// File contracts/interfaces/draft-IERC1404/draft-IERC1404EnumCode.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

interface IERC1404EnumCode {
    /* 
    * @dev leave the code 4-9 free/unused for further additions in your ruleEngine implementation
    */
    enum REJECTED_CODE_BASE {
        TRANSFER_OK,
        TRANSFER_REJECTED_PAUSED,
        TRANSFER_REJECTED_FROM_FROZEN,
        TRANSFER_REJECTED_TO_FROZEN
    }
}


// File contracts/interfaces/draft-IERC1404/draft-IERC1404Wrapper.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;


interface IERC1404Wrapper is IERC1404, IERC1404EnumCode  {

    /**
     * @dev Returns true if the transfer is valid, and false otherwise.
     */
    function validateTransfer(
        address _from,
        address _to,
        uint256 _amount
    ) external view returns (bool isValid);
}


// File contracts/interfaces/engine/IRuleEngine.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

/*
* @dev minimum interface to define a RuleEngine
*/
interface IRuleEngine is IERC1404Wrapper {
    /**
     * @dev Returns true if the operation is a success, and false otherwise.
     */
    function operateOnTransfer(
        address _from,
        address _to,
        uint256 _amount
    ) external returns (bool isValid);
   
}


// File contracts/interfaces/ICMTATConstructor.sol

// Original license: SPDX_License_Identifier: MPL-2.0
pragma solidity ^0.8.20;




/**
* @notice interface to represent arguments used for CMTAT constructor / initialize
*/
interface ICMTATConstructor {
    struct Engine {
        IRuleEngine ruleEngine;
        IDebtEngine debtEngine;
        IAuthorizationEngine authorizationEngine;
        IERC1643 documentEngine;
    }
    struct ERC20Attributes {
        // name of the token,
        string nameIrrevocable;
        // name of the symbol
        string symbolIrrevocable;
        // number of decimals of the token, must be 0 to be compliant with Swiss law as per CMTAT specifications (non-zero decimal number may be needed for other use cases)
        uint8 decimalsIrrevocable;
    }
    struct BaseModuleAttributes {
        // name of the tokenId
        string tokenId;
        // terms associated with the token
        string terms;
        // additional information to describe the token
        string information;
    }
}


// File contracts/libraries/Errors.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

/*
* @dev CMTAT custom errors
*/
library Errors {
    // CMTAT
    error CMTAT_InvalidTransfer(address from, address to, uint256 amount);

    // SnapshotModule
    error CMTAT_SnapshotModule_SnapshotScheduledInThePast(
        uint256 time,
        uint256 timestamp
    );
    error CMTAT_SnapshotModule_SnapshotTimestampBeforeLastSnapshot(
        uint256 time,
        uint256 lastSnapshotTimestamp
    );
    error CMTAT_SnapshotModule_SnapshotTimestampAfterNextSnapshot(
        uint256 time,
        uint256 nextSnapshotTimestamp
    );
    error CMTAT_SnapshotModule_SnapshotTimestampBeforePreviousSnapshot(
        uint256 time,
        uint256 previousSnapshotTimestamp
    );
    error CMTAT_SnapshotModule_SnapshotAlreadyExists();
    error CMTAT_SnapshotModule_SnapshotAlreadyDone();
    error CMTAT_SnapshotModule_NoSnapshotScheduled();
    error CMTAT_SnapshotModule_SnapshotNotFound();

    // ERC20BaseModule
    error CMTAT_ERC20BaseModule_WrongAllowance(
        address spender,
        uint256 currentAllowance,
        uint256 allowanceProvided
    );

    // BurnModule
    error CMTAT_BurnModule_EmptyAccounts();
    error CMTAT_BurnModule_AccountsValueslengthMismatch();

    // MintModule
    error CMTAT_MintModule_EmptyAccounts();
    error CMTAT_MintModule_AccountsValueslengthMismatch();

    // ERC20BaseModule
    error CMTAT_ERC20BaseModule_EmptyTos();
    error CMTAT_ERC20BaseModule_TosValueslengthMismatch();

    // DebtModule
    error CMTAT_DebtModule_SameValue();

    // ValidationModule
    error CMTAT_ValidationModule_SameValue();

    // AuthorizationModule
    error CMTAT_AuthorizationModule_AddressZeroNotAllowed();
    error CMTAT_AuthorizationModule_InvalidAuthorization();
    error CMTAT_AuthorizationModule_AuthorizationEngineAlreadySet(); 

    // DocumentModule
    error CMTAT_DocumentModule_SameValue();

    // PauseModule
    error CMTAT_PauseModule_ContractIsDeactivated();
}


// File contracts/modules/security/AuthorizationModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;



abstract contract AuthorizationModule is AccessControlUpgradeable {
    /* ============ Events ============ */
    /**
     * @dev Emitted when a rule engine is set.
     */
    event AuthorizationEngine(IAuthorizationEngine indexed newAuthorizationEngine);
    /* ============ ERC-7201 ============ */
    // keccak256(abi.encode(uint256(keccak256("CMTAT.storage.AuthorizationModule")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant AuthorizationModuleStorageLocation = 0x59b7f077fa4ad020f9053fd2197fef0113b19f0b11dcfe516e88cbc0e9226d00;
    /* ==== ERC-7201 State Variables === */
    struct AuthorizationModuleStorage {
        IAuthorizationEngine _authorizationEngine;
    }
    /* ============  Initializer Function ============ */
    /**
     * @dev
     *
     * - The grant to the admin role is done by AccessControlDefaultAdminRules
     * - The control of the zero address is done by AccessControlDefaultAdminRules
     *
     */
    function __AuthorizationModule_init_unchained(address admin, IAuthorizationEngine authorizationEngine_)
    internal onlyInitializing {
        if(admin == address(0)){
            revert Errors.CMTAT_AuthorizationModule_AddressZeroNotAllowed();
        }
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        if (address(authorizationEngine_) != address (0)) {
            AuthorizationModuleStorage storage $ = _getAuthorizationModuleStorage();
            $._authorizationEngine = authorizationEngine_;
            emit AuthorizationEngine(authorizationEngine_);
        }
    }


    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    function authorizationEngine() public view virtual returns (IAuthorizationEngine) {
        AuthorizationModuleStorage storage $ = _getAuthorizationModuleStorage();
        return $._authorizationEngine;
    }


    /*
    * @notice set an authorizationEngine if not already set
    * @dev once an AuthorizationEngine is set, it is not possible to unset it
    */
    function setAuthorizationEngine(
        IAuthorizationEngine authorizationEngine_
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        AuthorizationModuleStorage storage $ = _getAuthorizationModuleStorage();
        if (address($._authorizationEngine) != address (0)){
            revert Errors.CMTAT_AuthorizationModule_AuthorizationEngineAlreadySet();
        }
        $._authorizationEngine = authorizationEngine_;
        emit AuthorizationEngine(authorizationEngine_);
    }

    function grantRole(bytes32 role, address account) public override onlyRole(getRoleAdmin(role)) {
        AuthorizationModuleStorage storage $ = _getAuthorizationModuleStorage();
        if (address($._authorizationEngine) != address (0)) {
            bool result = $._authorizationEngine.operateOnGrantRole(role, account);
            if(!result) {
                // Operation rejected by the authorizationEngine
               revert Errors.CMTAT_AuthorizationModule_InvalidAuthorization();
            }
        }
        return AccessControlUpgradeable.grantRole(role, account);
    }

    function revokeRole(bytes32 role, address account) public override onlyRole(getRoleAdmin(role)) {
        AuthorizationModuleStorage storage $ = _getAuthorizationModuleStorage();
        if (address($._authorizationEngine) != address (0)) {
            bool result = $._authorizationEngine.operateOnRevokeRole(role, account);
            if(!result) {
                // Operation rejected by the authorizationEngine
               revert Errors.CMTAT_AuthorizationModule_InvalidAuthorization();
            }
        }
        return AccessControlUpgradeable.revokeRole(role, account);
    }

    /** 
     * @dev Returns `true` if `account` has been granted `role`.
     */
    function hasRole(
        bytes32 role,
        address account
    ) public view virtual override(AccessControlUpgradeable) returns (bool) {
        // The Default Admin has all roles
        if (AccessControlUpgradeable.hasRole(DEFAULT_ADMIN_ROLE, account)) {
            return true;
        }
        return AccessControlUpgradeable.hasRole(role, account);
    }


    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/


    /* ============ ERC-7201 ============ */
    function _getAuthorizationModuleStorage() private pure returns (AuthorizationModuleStorage storage $) {
        assembly {
            $.slot := AuthorizationModuleStorageLocation
        }
    }
}


// File contracts/modules/internal/ValidationModuleInternal.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;



/**
 * @dev Validation module.
 *
 * Useful for to restrict and validate transfers
 */
abstract contract ValidationModuleInternal is
    Initializable,
    ContextUpgradeable
{
    /* ============ Events ============ */
    /**
     * @dev Emitted when a rule engine is set.
     */
    event RuleEngine(IRuleEngine indexed newRuleEngine);
    /* ============ ERC-7201 ============ */
    // keccak256(abi.encode(uint256(keccak256("CMTAT.storage.ValidationModuleInternal")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant ValidationModuleInternalStorageLocation = 0xb3e8f29e401cfa802cad91001b5f9eb50decccdb111d80cb07177ab650b04700;
    /* ==== ERC-7201 State Variables === */
    struct ValidationModuleInternalStorage {
        IRuleEngine _ruleEngine;
    }
    /* ============  Initializer Function ============ */
    function __Validation_init_unchained(
        IRuleEngine ruleEngine_
    ) internal onlyInitializing {
        if (address(ruleEngine_) != address(0)) {
            ValidationModuleInternalStorage storage $ = _getValidationModuleInternalStorage();
            $._ruleEngine = ruleEngine_;
            emit RuleEngine(ruleEngine_);
        }
    }


    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    
    function ruleEngine() public view returns(IRuleEngine){
        ValidationModuleInternalStorage storage $ = _getValidationModuleInternalStorage();
        return $._ruleEngine;
    }


    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /**
    * @dev before making a call to this function, you have to check if a ruleEngine is set.
    */
    function _validateTransfer(
        address from,
        address to,
        uint256 amount
    ) internal view returns (bool) {
        ValidationModuleInternalStorage storage $ = _getValidationModuleInternalStorage();
        return $._ruleEngine.validateTransfer(from, to, amount);
    }

    /**
    * @dev before making a call to this function, you have to check if a ruleEngine is set.
    */
    function _messageForTransferRestriction(
        uint8 restrictionCode
    ) internal view returns (string memory) {
        ValidationModuleInternalStorage storage $ = _getValidationModuleInternalStorage();
        return $._ruleEngine.messageForTransferRestriction(restrictionCode);
    }

    /**
    * @dev before making a call to this function, you have to check if a ruleEngine is set.
    */
    function _detectTransferRestriction(
        address from,
        address to,
        uint256 amount
    ) internal view returns (uint8) {
        ValidationModuleInternalStorage storage $ = _getValidationModuleInternalStorage();
        return $._ruleEngine.detectTransferRestriction(from, to, amount);
    }

    function _operateOnTransfer(address from, address to, uint256 amount) virtual internal returns (bool) {
        ValidationModuleInternalStorage storage $ = _getValidationModuleInternalStorage();
        return $._ruleEngine.operateOnTransfer(from, to, amount);
    }


    /* ============ ERC-7201 ============ */
    function _getValidationModuleInternalStorage() internal pure returns (ValidationModuleInternalStorage storage $) {
        assembly {
            $.slot := ValidationModuleInternalStorageLocation
        }
    }
}


// File contracts/modules/internal/EnforcementModuleInternal.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;



/**
 * @dev Enforcement module.
 *
 * Allows the issuer to freeze transfers from a given address
 */
abstract contract EnforcementModuleInternal is
    Initializable,
    ContextUpgradeable
{
    /* ============ Events ============ */
    /**
     * @notice Emitted when an address is frozen.
     */
    event Freeze(
        address indexed enforcer,
        address indexed owner,
        string indexed reasonIndexed,
        string reason
    );

    /**
     * @notice Emitted when an address is unfrozen.
     */
    event Unfreeze(
        address indexed enforcer,
        address indexed owner,
        string indexed reasonIndexed,
        string reason
    );

     /* ============ ERC-7201 ============ */
    // keccak256(abi.encode(uint256(keccak256("CMTAT.storage.EnforcementModuleInternal")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant EnforcementModuleInternalStorageLocation = 0x0c7bc8a17be064111d299d7669f49519cb26c58611b72d9f6ccc40a1e1184e00;
    

    /* ==== ERC-7201 State Variables === */
    struct EnforcementModuleInternalStorage {
        mapping(address => bool) _frozen;
    }


    /*//////////////////////////////////////////////////////////////
                         INITIALIZER FUNCTION
    //////////////////////////////////////////////////////////////*/
    function __Enforcement_init_unchained() internal onlyInitializing {
        // no variable to initialize
    }

    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /**
     * @dev Returns true if the account is frozen, and false otherwise.
     */
    function frozen(address account) public view virtual returns (bool) {
        EnforcementModuleInternalStorage storage $ = _getEnforcementModuleInternalStorage();
        return $._frozen[account];
    }

    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /**
     * @dev Freezes an address.
     * @param account the account to freeze
     * @param reason indicate why the account was frozen.
     *
     */
    function _freeze(
        address account,
        string calldata reason
    ) internal virtual returns (bool) {
        EnforcementModuleInternalStorage storage $ = _getEnforcementModuleInternalStorage();
        if ($._frozen[account]) {
            return false;
        }
        $._frozen[account] = true;
        emit Freeze(_msgSender(), account, reason, reason);
        return true;
    }

    /**
     * @dev Unfreezes an address.
     * @param account the account to unfreeze
     * @param reason indicate why the account was unfrozen.
     */
    function _unfreeze(
        address account,
        string calldata reason
    ) internal virtual returns (bool) {
        EnforcementModuleInternalStorage storage $ = _getEnforcementModuleInternalStorage();
        if (!$._frozen[account]) {
            return false;
        }
        $._frozen[account] = false;
        emit Unfreeze(_msgSender(), account, reason, reason);

        return true;
    }

    /* ============ ERC-7201 ============ */
    function _getEnforcementModuleInternalStorage() private pure returns (EnforcementModuleInternalStorage storage $) {
        assembly {
            $.slot := EnforcementModuleInternalStorageLocation
        }
    }
}


// File contracts/modules/wrapper/core/EnforcementModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;


/**
 * @title Enforcement module.
 * @dev 
 *
 * Allows the issuer to freeze transfers from a given address
 */
abstract contract EnforcementModule is
    EnforcementModuleInternal,
    AuthorizationModule
{
    /* ============ State Variables ============ */
    bytes32 public constant ENFORCER_ROLE = keccak256("ENFORCER_ROLE");
    string internal constant TEXT_TRANSFER_REJECTED_FROM_FROZEN =
        "Address FROM is frozen";

    string internal constant TEXT_TRANSFER_REJECTED_TO_FROZEN =
        "Address TO is frozen";

    /* ============  Initializer Function ============ */
    function __EnforcementModule_init_unchained() internal onlyInitializing {
        // no variable to initialize
    }

    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    /**
     * @notice Freezes an address.
     * @param account the account to freeze
     * @param reason indicate why the account was frozen.
     */
    function freeze(
        address account,
        string calldata reason
    ) public onlyRole(ENFORCER_ROLE) returns (bool) {
        return _freeze(account, reason);
    }

    /**
     * @notice Unfreezes an address.
     * @param account the account to unfreeze
     * @param reason indicate why the account was unfrozen.
     *
     *
     */
    function unfreeze(
        address account,
        string calldata reason
    ) public onlyRole(ENFORCER_ROLE) returns (bool) {
        return _unfreeze(account, reason);
    }
}


// File contracts/modules/wrapper/core/PauseModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;


/**
 * @title Pause Module
 * @dev 
 * Put in pause or deactivate the contract
 * The issuer must be able to “pause” the smart contract, 
 * to prevent execution of transactions on the distributed ledger until the issuer puts an end to the pause. 
 *
 * Useful for scenarios such as preventing trades until the end of an evaluation
 * period, or having an emergency switch for freezing all token transfers in the
 * event of a large bug.
 */
abstract contract PauseModule is PausableUpgradeable, AuthorizationModule {
    /* ============ State Variables ============ */
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    string internal constant TEXT_TRANSFER_REJECTED_PAUSED =
        "All transfers paused";
    /* ============ Events ============ */
    event Deactivated(address account);
    /* ============ ERC-7201 ============ */
    // keccak256(abi.encode(uint256(keccak256("CMTAT.storage.ERC20BaseModule")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant PauseModuleStorageLocation = 0x9bd8d607565c0370ae5f91651ca67fd26d4438022bf72037316600e29e6a3a00;
    /* ==== ERC-7201 State Variables === */
    struct PauseModuleStorage {
        bool _isDeactivated;
    }
    /* ============  Initializer Function ============ */
    function __PauseModule_init_unchained() internal onlyInitializing {
        // no variable to initialize
    }
    
    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    /**
     * @notice Pauses all token transfers.
     * @dev See {ERC20Pausable} and {Pausable-_pause}.
     *
     * Requirements:
     *
     * - the caller must have the `PAUSER_ROLE`.
     *
     */
    function pause() public onlyRole(PAUSER_ROLE) {
        _pause();
    }

    /**
     * @notice Unpauses all token transfers.
     * @dev See {ERC20Pausable} and {Pausable-_unpause}.
     *
     * Requirements:
     *
     * - the caller must have the `PAUSER_ROLE`.
     */
    function unpause() public onlyRole(PAUSER_ROLE) {
        PauseModuleStorage storage $ = _getPauseModuleStorage();
        if($._isDeactivated){
            revert Errors.CMTAT_PauseModule_ContractIsDeactivated();
        }
        _unpause();
    }

    /**
    * @notice  deactivate the contract
    * Warning: the operation is irreversible, be careful
    * @dev
    * Emits a {Deactivated} event indicating that the contract has been deactivated.
    * Requirements:
    *
    * - the caller must have the `DEFAULT_ADMIN_ROLE`.
    */
    function deactivateContract()
        public
        onlyRole(DEFAULT_ADMIN_ROLE)
    {
        PauseModuleStorage storage $ = _getPauseModuleStorage();
        $._isDeactivated = true;
       _pause();
       emit Deactivated(_msgSender());
    }

    /**
    * @notice Returns true if the contract is deactivated, and false otherwise.
    */
    function deactivated() view public returns (bool){
        PauseModuleStorage storage $ = _getPauseModuleStorage();
        return $._isDeactivated;
    }


    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/


    /* ============ ERC-7201 ============ */
    function _getPauseModuleStorage() private pure returns (PauseModuleStorage storage $) {
        assembly {
            $.slot := PauseModuleStorageLocation
        }
    }
}


// File contracts/modules/wrapper/controllers/ValidationModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;




/**
 * @dev Validation module.
 *
 * Useful for to restrict and validate transfers
 */
abstract contract ValidationModule is
    ValidationModuleInternal,
    PauseModule,
    EnforcementModule,
    IERC1404Wrapper
{
    /* ============ State Variables ============ */
    string constant TEXT_TRANSFER_OK = "No restriction";
    string constant TEXT_UNKNOWN_CODE = "Unknown code";

    /* ============  Initializer Function ============ */
    function __ValidationModule_init_unchained() internal onlyInitializing {
        // no variable to initialize
    }


    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /*
    * @notice set a RuleEngine
    * @param ruleEngine_ the call will be reverted if the new value of ruleEngine is the same as the current one
    */
    function setRuleEngine(
        IRuleEngine ruleEngine_
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        ValidationModuleInternalStorage storage $ = _getValidationModuleInternalStorage();
        if ($._ruleEngine == ruleEngine_){
             revert Errors.CMTAT_ValidationModule_SameValue();
        }
        $._ruleEngine = ruleEngine_;
        emit RuleEngine(ruleEngine_);
    }

    /**
     * @dev ERC1404 returns the human readable explaination corresponding to the error code returned by detectTransferRestriction
     * @param restrictionCode The error code returned by detectTransferRestriction
     * @return message The human readable explaination corresponding to the error code returned by detectTransferRestriction
     */
    function messageForTransferRestriction(
        uint8 restrictionCode
    ) external view override returns (string memory message) {
          ValidationModuleInternalStorage storage $ = _getValidationModuleInternalStorage();
        if (restrictionCode == uint8(REJECTED_CODE_BASE.TRANSFER_OK)) {
            return TEXT_TRANSFER_OK;
        } else if (
            restrictionCode ==
            uint8(REJECTED_CODE_BASE.TRANSFER_REJECTED_PAUSED)
        ) {
            return TEXT_TRANSFER_REJECTED_PAUSED;
        } else if (
            restrictionCode ==
            uint8(REJECTED_CODE_BASE.TRANSFER_REJECTED_FROM_FROZEN)
        ) {
            return TEXT_TRANSFER_REJECTED_FROM_FROZEN;
        } else if (
            restrictionCode ==
            uint8(REJECTED_CODE_BASE.TRANSFER_REJECTED_TO_FROZEN)
        ) {
            return TEXT_TRANSFER_REJECTED_TO_FROZEN;
        } else if (address($._ruleEngine) != address(0)) {
            return _messageForTransferRestriction(restrictionCode);
        } else {
            return TEXT_UNKNOWN_CODE;
        }
    }
    
    /**
     * @dev ERC1404 check if _value token can be transferred from _from to _to
     * @param from address The address which you want to send tokens from
     * @param to address The address which you want to transfer to
     * @param amount uint256 the amount of tokens to be transferred
     * @return code of the rejection reason
     */
    function detectTransferRestriction(
        address from,
        address to,
        uint256 amount
    ) public view override returns (uint8 code) {
        ValidationModuleInternalStorage storage $ = _getValidationModuleInternalStorage();
        if (paused()) {
            return uint8(REJECTED_CODE_BASE.TRANSFER_REJECTED_PAUSED);
        } else if (frozen(from)) {
            return uint8(REJECTED_CODE_BASE.TRANSFER_REJECTED_FROM_FROZEN);
        } else if (frozen(to)) {
            return uint8(REJECTED_CODE_BASE.TRANSFER_REJECTED_TO_FROZEN);
        } else if (address($._ruleEngine) != address(0)) {
            return _detectTransferRestriction(from, to, amount);
        } else {
            return uint8(REJECTED_CODE_BASE.TRANSFER_OK);
        }
    }

    function validateTransfer(
        address from,
        address to,
        uint256 amount
    ) public view override returns (bool) {
        if (!_validateTransferByModule(from, to, amount)) {
            return false;
        }
        ValidationModuleInternalStorage storage $ = _getValidationModuleInternalStorage();
        if (address($._ruleEngine) != address(0)) {
            return _validateTransfer(from, to, amount);
        }
        return true;
    }


    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    function _validateTransferByModule(
        address from,
        address to,
        uint256 /*amount*/
    ) internal view returns (bool) {
        if (paused() || frozen(from) || frozen(to)) {
            return false;
        }
        return true;
    }

    function _operateOnTransfer(address from, address to, uint256 amount) override internal returns (bool){
        if (!_validateTransferByModule(from, to, amount)){
            return false;
        }
        ValidationModuleInternalStorage storage $ = _getValidationModuleInternalStorage();
        if (address($._ruleEngine) != address(0)) {
            return ValidationModuleInternal._operateOnTransfer(from, to, amount);
        }
        return true;
    }
}


// File contracts/modules/wrapper/core/BaseModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

abstract contract BaseModule is AuthorizationModule {
    /* ============ State Variables ============ */
    /** 
    * @notice 
    * Get the current version of the smart contract
    */
    string public constant VERSION = "2.5.0";
    
    /* ============ Events ============ */
    event Term(string indexed newTermIndexed, string newTerm);
    event TokenId(string indexed newTokenIdIndexed, string newTokenId);
    event Information(
        string indexed newInformationIndexed,
        string newInformation
    );
    event Flag(uint256 indexed newFlag);
    /* ============ ERC-7201 ============ */
    // keccak256(abi.encode(uint256(keccak256("CMTAT.storage.BaseModule")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant BaseModuleStorageLocation = 0xa98e72f7f70574363edb12c42a03ac1feb8cc898a6e0a30f6eefbab7093e0d00;

    /* ==== ERC-7201 State Variables === */
    struct BaseModuleStorage {
            string _tokenId;
            string _terms;
            string _information;
    }
    /* ============  Initializer Function ============ */
    /**
     * @dev Sets the values for {name} and {symbol}.
     *
     * All two of these values are immutable: they can only be set once during
     * construction.
     */
    function __Base_init_unchained(
        string memory tokenId_,
        string memory terms_,
        string memory information_
    ) internal onlyInitializing {
        BaseModuleStorage storage $ = _getBaseModuleStorage();
        $._tokenId = tokenId_;
        $._terms = terms_;
        $._information = information_;
    }

    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    function tokenId() public view virtual returns (string memory) {
        BaseModuleStorage storage $ = _getBaseModuleStorage();
        return $._tokenId;
    }

    function terms() public view virtual returns (string memory) {
        BaseModuleStorage storage $ = _getBaseModuleStorage();
        return $._terms;
    }
    function information() public view virtual returns (string memory) {
        BaseModuleStorage storage $ = _getBaseModuleStorage();
        return $._information;
    }

    /** 
    * @notice the tokenId will be changed even if the new value is the same as the current one
    */
    function setTokenId(
        string calldata tokenId_
    ) public onlyRole(DEFAULT_ADMIN_ROLE) {
        BaseModuleStorage storage $ = _getBaseModuleStorage();
        $._tokenId = tokenId_;
        emit TokenId(tokenId_, tokenId_);
    }

    /** 
    * @notice The terms will be changed even if the new value is the same as the current one
    */
    function setTerms(
        string calldata terms_
    ) public onlyRole(DEFAULT_ADMIN_ROLE) {
        BaseModuleStorage storage $ = _getBaseModuleStorage();
        $._terms  = terms_;
        emit Term(terms_, terms_);
    }

    /** 
    * @notice The information will be changed even if the new value is the same as the current one
    */
    function setInformation(
        string calldata information_
    ) public onlyRole(DEFAULT_ADMIN_ROLE) {
        BaseModuleStorage storage $ = _getBaseModuleStorage();
        $._information  = information_;
        emit Information(information_, information_);
    }


    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /* ============ ERC-7201 ============ */
    function _getBaseModuleStorage() private pure returns (BaseModuleStorage storage $) {
        assembly {
            $.slot := BaseModuleStorageLocation
        }
    }

}


// File contracts/modules/wrapper/core/ERC20BaseModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

// required OZ imports here


/**
 * @title ERC20Base module
 * @dev 
 *
 * Contains ERC-20 base functions and extension
 * Inherits from ERC-20
 * 
 */
abstract contract ERC20BaseModule is ERC20Upgradeable {
    /* ============ Events ============ */
    /**
    * @notice Emitted when the specified `spender` spends the specified `value` tokens owned by the specified `owner` reducing the corresponding allowance.
    * @dev The allowance can be also "spend" with the function BurnFrom, but in this case, the emitted event is BurnFrom.
    */
    event Spend(address indexed owner, address indexed spender, uint256 value);
    /* ============ ERC-7201 ============ */
    // keccak256(abi.encode(uint256(keccak256("CMTAT.storage.ERC20BaseModule")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant ERC20BaseModuleStorageLocation = 0x9bd8d607565c0370ae5f91651ca67fd26d4438022bf72037316600e29e6a3a00;
    /* ==== ERC-7201 State Variables === */
    struct ERC20BaseModuleStorage {
        uint8 _decimals;
    }

    /* ============  Initializer Function ============ */
    /**
     * @dev Initializers: Sets the values for decimals.
     *
     * this value is immutable: it can only be set once during
     * construction/initialization.
     */
    function __ERC20BaseModule_init_unchained(
        uint8 decimals_
    ) internal onlyInitializing {
        ERC20BaseModuleStorage storage $ = _getERC20BaseModuleStorage();
        $._decimals = decimals_;
    }
    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    /**
     *
     * @notice Returns the number of decimals used to get its user representation.
     * @inheritdoc ERC20Upgradeable
     */
    function decimals() public view virtual override returns (uint8) {
        ERC20BaseModuleStorage storage $ = _getERC20BaseModuleStorage();
        return $._decimals;
    }

    /**
     * @notice batch version of transfer
     * @param tos can not be empty, must have the same length as values
     * @param values can not be empty
     * @dev See {OpenZeppelin ERC20-transfer & ERC1155-safeBatchTransferFrom}.
     *
     *
     * Requirements:
     * - `tos` and `values` must have the same length
     * - `tos`cannot contain a zero address (check made by transfer)
     * - the caller must have a balance cooresponding to the total values
     */
    function transferBatch(
        address[] calldata tos,
        uint256[] calldata values
    ) public returns (bool) {
        if (tos.length == 0) {
            revert Errors.CMTAT_ERC20BaseModule_EmptyTos();
        }
        // We do not check that values is not empty since
        // this require will throw an error in this case.
        if (bool(tos.length != values.length)) {
            revert Errors.CMTAT_ERC20BaseModule_TosValueslengthMismatch();
        }
        // No need of unchecked block since Soliditiy 0.8.22
        for (uint256 i = 0; i < tos.length; ++i) {
            // We call directly the internal function transfer
            // The reason is that the public function adds only the owner address recovery
            ERC20Upgradeable._transfer(_msgSender(), tos[i], values[i]);
        }
        // not really useful
        // Here only to keep the same behaviour as transfer
        return true;
    }

    /**
     * @notice Transfers `value` amount of tokens from address `from` to address `to`
     * @custom:dev-cmtat
     * Emits a {Spend} event indicating the spended allowance.
     * @inheritdoc ERC20Upgradeable
     *
     */
    function transferFrom(
        address from,
        address to,
        uint256 value
    ) public virtual override returns (bool) {
        bool result = ERC20Upgradeable.transferFrom(from, to, value);
        // The result will be normally always true because OpenZeppelin will revert in case of an error
        if (result) {
            emit Spend(from, _msgSender(), value);
        }

        return result;
    }

    /**
    * @param addresses list of address to know their balance
    * @return balances ,totalSupply array with balance for each address, totalSupply
    * @dev useful for the snapshot rule
    */
    function balanceInfo(address[] calldata addresses) public view returns(uint256[] memory balances , uint256 totalSupply) {
        balances = new uint256[](addresses.length);
        for(uint256 i = 0; i < addresses.length; ++i){
            balances[i] = ERC20Upgradeable.balanceOf(addresses[i]);
        }
        totalSupply = ERC20Upgradeable.totalSupply();
    }


    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/


    /* ============ ERC-7201 ============ */
    function _getERC20BaseModuleStorage() private pure returns (ERC20BaseModuleStorage storage $) {
        assembly {
            $.slot := ERC20BaseModuleStorageLocation
        }
    }
}


// File contracts/interfaces/ICCIPToken.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

/**
* @notice CCIP Pool with mint
*/
interface ICCIPMintERC20 {
  /// @notice Mints new tokens for a given address.
  /// @param account The address to mint the new tokens to.
  /// @param value The number of tokens to be minted.
  /// @dev this function increases the total supply.
  function mint(address account, uint256 value) external;
}

/**
* @notice CCIP Pool with burnFrom
*/
interface ICCIPBurnFromERC20 {
  /// @notice Burns tokens from a given address..
  /// @param account The address to burn tokens from.
  /// @param value The number of tokens to be burned.
  /// @dev this function decreases the total supply.
  function burnFrom(address account, uint256 value) external;
}


// File contracts/modules/wrapper/core/ERC20BurnModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;



/**
 * @title ERC20Burn module.
 * @dev 
 *
 * Contains all burn functions, inherits from ERC-20
 */
abstract contract ERC20BurnModule is ERC20Upgradeable, ICCIPBurnFromERC20, AuthorizationModule {
    /* ============ State Variables ============ */
    bytes32 public constant BURNER_ROLE = keccak256("BURNER_ROLE");
    bytes32 public constant BURNER_FROM_ROLE = keccak256("BURNER_FROM_ROLE");
    
    /* ============ Events ============ */
    /**
    * @notice Emitted when the specified `value` amount of tokens owned by `owner`are destroyed with the given `reason`
    */
    event Burn(address indexed owner, uint256 value, string reason);
    /**
    * @notice Emitted when the specified `spender` burns the specified `value` tokens owned by the specified `owner` reducing the corresponding allowance.
    */
    event BurnFrom(address indexed owner, address indexed spender, uint256 value);


    /* ============  Initializer Function ============ */
    function __ERC20BurnModule_init_unchained() internal onlyInitializing {
        // no variable to initialize
    }

    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /**
     * @notice Destroys a `value` amount of tokens from `account`, by transferring it to address(0).
     * @dev
     * See {ERC20-_burn}
     * Emits a {Burn} event
     * Emits a {Transfer} event with `to` set to the zero address  (emits inside _burn).
     * Requirements:
     * - the caller must have the `BURNER_ROLE`.
     */
    function burn(
        address account,
        uint256 value,
        string calldata reason
    ) public onlyRole(BURNER_ROLE) {
        _burn(account, value);
        emit Burn(account, value, reason);
    }


    /**
     *
     * @notice batch version of {burn}.
     * @dev
     * See {ERC20-_burn} and {OpenZeppelin ERC1155_burnBatch}.
     *
     * For each burn action:
     * -Emits a {Burn} event
     * -Emits a {Transfer} event with `to` set to the zero address  (emits inside _burn).
     * The burn `reason`is the same for all `accounts` which tokens are burnt.
     * Requirements:
     * - `accounts` and `values` must have the same length
     * - the caller must have the `BURNER_ROLE`.
     */
    function burnBatch(
        address[] calldata accounts,
        uint256[] calldata values,
        string calldata reason
    ) public onlyRole(BURNER_ROLE) {
        if (accounts.length == 0) {
            revert Errors.CMTAT_BurnModule_EmptyAccounts();
        }
        // We do not check that values is not empty since
        // this require will throw an error in this case.
        if (bool(accounts.length != values.length)) {
            revert Errors.CMTAT_BurnModule_AccountsValueslengthMismatch();
        }
        // No need of unchecked block since Soliditiy 0.8.22
        for (uint256 i = 0; i < accounts.length; ++i ) {
            _burn(accounts[i], values[i]);
            emit Burn(accounts[i], values[i], reason);
        }
    }

    /**
     * @notice Destroys `amount` tokens from `account`, deducting from the caller's
     * allowance.
     * @dev 
     * Can be used to authorize a bridge (e.g. CCIP) to burn token owned by the bridge
     * No string parameter reason to be compatible with Bridge, e.g. CCIP
     * 
     * See {ERC20-_burn} and {ERC20-allowance}.
     *
     * Requirements:
     *
     * - the caller must have allowance for ``accounts``'s tokens of at least
     * `value`.
     */
    function burnFrom(address account, uint256 value)
        public
        onlyRole(BURNER_FROM_ROLE)
    {
        // Allowance check
        address sender =  _msgSender();
        uint256 currentAllowance = allowance(account, sender);
        if(currentAllowance < value){
            // ERC-6093
            revert ERC20InsufficientAllowance(sender, currentAllowance, value);
        }
        // Update allowance
        unchecked {
            _approve(account, sender, currentAllowance - value);
        }
        // burn
        _burn(account, value);
        // We also emit a burn event since its a burn operation
        emit Burn(account, value, "burnFrom");
        // Specific event for the operation
        emit BurnFrom(account, sender, value);
    }
}


// File contracts/modules/wrapper/core/ERC20MintModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;



/**
 * @title ERC20Mint module.
 * @dev 
 *
 * Contains all mint functions, inherits from ERC-20
 */
abstract contract ERC20MintModule is ERC20Upgradeable, ICCIPMintERC20, AuthorizationModule {
    /* ============ State Variables ============ */
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    /* ============ Events ============ */
    /**
     * @notice Emitted when the specified  `value` amount of new tokens are created and
     * allocated to the specified `account`.
     */
    event Mint(address indexed account, uint256 value);


    /* ============  Initializer Function ============ */
    function __ERC20MintModule_init_unchained() internal onlyInitializing {
        // no variable to initialize
    }

    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    /**
     * @notice  Creates a `value` amount of tokens and assigns them to `account`, by transferring it from address(0)
     * @param account token receiver
     * @param value amount of tokens
     * @dev
     * See {OpenZeppelin ERC20-_mint}.
     * Emits a {Mint} event.
     * Emits a {Transfer} event with `from` set to the zero address (emits inside _mint).
     *
     * Requirements:
     * - `account` cannot be the zero address (check made by _mint).
     * - The caller must have the `MINTER_ROLE`.
     */
    function mint(address account, uint256 value) public onlyRole(MINTER_ROLE) {
        _mint(account, value);
        emit Mint(account, value);
    }

    /**
     *
     * @notice batch version of {mint}
     * @dev
     * See {OpenZeppelin ERC20-_mint} and {OpenZeppelin ERC1155_mintBatch}.
     *
     * For each mint action:
     * - Emits a {Mint} event.
     * - Emits a {Transfer} event with `from` set to the zero address (emits inside _mint).
     *
     * Requirements:
     * - `accounts` and `values` must have the same length
     * - `accounts` cannot contain a zero address (check made by _mint).
     * - the caller must have the `MINTER_ROLE`.
     */
    function mintBatch(
        address[] calldata accounts,
        uint256[] calldata values
    ) public onlyRole(MINTER_ROLE) {
        if (accounts.length == 0) {
            revert Errors.CMTAT_MintModule_EmptyAccounts();
        }
        // We do not check that values is not empty since
        // this require will throw an error in this case.
        if (bool(accounts.length != values.length)) {
            revert Errors.CMTAT_MintModule_AccountsValueslengthMismatch();
        }
        // No need of unchecked block since Soliditiy 0.8.22
        for (uint256 i = 0; i < accounts.length; ++i ) {
            _mint(accounts[i], values[i]);
            emit Mint(accounts[i], values[i]);
        }
    }
}


// File contracts/modules/wrapper/extensions/DebtModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;



/**
 * @title Debt module
 * @dev 
 *
 * Retrieve debt and creditEvents information from a debtEngine
 */
abstract contract DebtModule is AuthorizationModule, IDebtEngine {
    /* ============ State Variables ============ */
    bytes32 public constant DEBT_ROLE = keccak256("DEBT_ROLE");
    /* ============ ERC-7201 ============ */
    // keccak256(abi.encode(uint256(keccak256("CMTAT.storage.DebtModule")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant DebtModuleStorageLocation = 0xf8a315cc5f2213f6481729acd86e55db7ccc930120ccf9fb78b53dcce75f7c00;
 
    /* ==== ERC-7201 State Variables === */
    struct DebtModuleStorage {
        IDebtEngine _debtEngine;
    }
    /* ============ Events ============ */
    /**
    * @dev Emitted when a rule engine is set.
    */
    event DebtEngine(IDebtEngine indexed newDebtEngine);


    /* ============  Initializer Function ============ */
    /**
     * @dev
     *
     * - The grant to the admin role is done by AccessControlDefaultAdminRules
     * - The control of the zero address is done by AccessControlDefaultAdminRules
     *
     */
    function __DebtModule_init_unchained(IDebtEngine debtEngine_)
    internal onlyInitializing {
        if (address(debtEngine_) != address (0)) {
            DebtModuleStorage storage $ = _getDebtModuleStorage();
            $._debtEngine = debtEngine_;
            emit DebtEngine(debtEngine_);
        }
        

    }
    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    function debtEngine() public view virtual returns (IDebtEngine) {
        DebtModuleStorage storage $ = _getDebtModuleStorage();
        return $._debtEngine;
    }

    /*
    * @notice set an authorizationEngine if not already set
    * 
    */
    function setDebtEngine(
        IDebtEngine debtEngine_
    ) external onlyRole(DEBT_ROLE) {
        DebtModuleStorage storage $ = _getDebtModuleStorage();
        if ($._debtEngine == debtEngine_){
            revert Errors.CMTAT_DebtModule_SameValue();
        }
        $._debtEngine = debtEngine_;
        emit DebtEngine(debtEngine_);
    }

    function debt() public view returns(DebtBase memory debtBaseResult){
        DebtModuleStorage storage $ = _getDebtModuleStorage();
        if(address($._debtEngine) != address(0)){
            debtBaseResult =  $._debtEngine.debt();
        }
    }

    function creditEvents() public view returns(CreditEvents memory creditEventsResult){
        DebtModuleStorage storage $ = _getDebtModuleStorage();
        if(address($._debtEngine) != address(0)){
            creditEventsResult =  $._debtEngine.creditEvents();
        }
    }


    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    
    /* ============ ERC-7201 ============ */
    function _getDebtModuleStorage() private pure returns (DebtModuleStorage storage $) {
        assembly {
            $.slot := DebtModuleStorageLocation
        }
    }

}


// File contracts/modules/wrapper/extensions/DocumentModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;



/**
 * @title Document module
 * @dev 
 *
 * Retrieve documents from a documentEngine
 */

abstract contract DocumentModule is AuthorizationModule, IERC1643 {
    /* ============ Events ============ */
    /**
     * @dev Emitted when a rule engine is set.
     */
    event DocumentEngine(IERC1643 indexed newDocumentEngine);
   
    /* ============ ERC-7201 ============ */
     bytes32 public constant DOCUMENT_ROLE = keccak256("DOCUMENT_ROLE");
    // keccak256(abi.encode(uint256(keccak256("CMTAT.storage.DocumentModule")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant DocumentModuleStorageLocation = 0x5edcb2767f407e647b6a4171ef53e8015a3eff0bb2b6e7765b1a26332bc43000;
    /* ==== ERC-7201 State Variables === */
    struct DocumentModuleStorage {
        IERC1643  _documentEngine;
    }

    /* ============  Initializer Function ============ */
    /**
     * @dev
     *
     * - The grant to the admin role is done by AccessControlDefaultAdminRules
     * - The control of the zero address is done by AccessControlDefaultAdminRules
     *
     */
    function __DocumentModule_init_unchained(IERC1643 documentEngine_)
    internal onlyInitializing {
        if (address(documentEngine_) != address (0)) {
            DocumentModuleStorage storage $ = _getDocumentModuleStorage();
            $._documentEngine = documentEngine_;
            emit DocumentEngine(documentEngine_);
        }
    }

    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    function documentEngine() public view virtual returns (IERC1643) {
        DocumentModuleStorage storage $ = _getDocumentModuleStorage();
        return $._documentEngine;
    }

    /*
    * @notice set an authorizationEngine if not already set
    * 
    */
    function setDocumentEngine(
        IERC1643 documentEngine_
    ) external onlyRole(DOCUMENT_ROLE) {
        DocumentModuleStorage storage $ = _getDocumentModuleStorage();
        if ($._documentEngine == documentEngine_){
             revert Errors.CMTAT_DocumentModule_SameValue();
        }
        $._documentEngine = documentEngine_;
        emit DocumentEngine(documentEngine_);
    }


    function getDocument(bytes32 _name) public view returns (string memory, bytes32, uint256){
        DocumentModuleStorage storage $ = _getDocumentModuleStorage();
        if(address($._documentEngine) != address(0)){
            return $._documentEngine.getDocument( _name);
        } else{
            return ("",0x0, 0);
        }
    }

    function getAllDocuments() public view returns (bytes32[] memory documents){
        DocumentModuleStorage storage $ = _getDocumentModuleStorage();
        if(address($._documentEngine) != address(0)){
            documents =  $._documentEngine.getAllDocuments();
        }
    }


    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /* ============ ERC-7201 ============ */
    function _getDocumentModuleStorage() private pure returns (DocumentModuleStorage storage $) {
        assembly {
            $.slot := DocumentModuleStorageLocation
        }
    } 
}


// File contracts/interfaces/ICMTATSnapshot.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

/**
* @notice minimum interface to represent a CMTAT with snapshot
*/
interface ICMTATSnapshot {
    /** 
    * @notice Return the number of tokens owned by the given owner at the time when the snapshot with the given time was created.
    * @return value stored in the snapshot, or the actual balance if no snapshot
    */
    function snapshotBalanceOf(uint256 time,address owner) external view returns (uint256);
    /**
    * @dev See {OpenZeppelin - ERC20Snapshot}
    * Retrieves the total supply at the specified time.
    * @return value stored in the snapshot, or the actual totalSupply if no snapshot
    */
    function snapshotTotalSupply(uint256 time) external view returns (uint256);
    /**
    * @notice Return snapshotBalanceOf and snapshotTotalSupply to avoid multiple calls
    * @return ownerBalance ,  totalSupply - see snapshotBalanceOf and snapshotTotalSupply
    */
    function snapshotInfo(uint256 time, address owner) external view returns (uint256 ownerBalance, uint256 totalSupply);
    /**
    * @notice Return snapshotBalanceOf for each address in the array and the total supply
    * @return ownerBalances array with the balance of each address, the total supply
    */
    function snapshotInfoBatch(uint256 time, address[] calldata addresses) external view returns (uint256[] memory ownerBalances, uint256 totalSupply);

    /**
    * @notice Return snapshotBalanceOf for each address in the array and the total supply
    * @return ownerBalances array with the balance of each address, the total supply
    */
    function snapshotInfoBatch(uint256[] calldata times, address[] calldata addresses) external view returns (uint256[][] memory ownerBalances, uint256[] memory totalSupply);


}


// File contracts/modules/internal/base/SnapshotModuleBase.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;


/**
 * @dev Base for the Snapshot module
 *
 * Useful to take a snapshot of token holder balance and total supply at a specific time
 * Inspired by Openzeppelin - ERC20Snapshot but use the time as Id instead of a counter.
 * Contrary to OpenZeppelin, the function _getCurrentSnapshotId is not available 
 *  because overriding this function can break the contract.
 */

abstract contract SnapshotModuleBase is Initializable {
    using Arrays for uint256[];
    /* ============ Structs ============ *
    /** 
    * @dev See {OpenZeppelin - ERC20Snapshot}
    * Snapshotted values have arrays of ids (time) and the value corresponding to that id.
    * ids is expected to be sorted in ascending order, and to contain no repeated elements 
    * because we use findUpperBound in the function _valueAt
    */
    struct Snapshots {
        uint256[] ids;
        uint256[] values;
    }
    /* ============ Events ============ */
    /**
    @notice Emitted when the snapshot with the specified oldTime was scheduled or rescheduled at the specified newTime.
    */
    event SnapshotSchedule(uint256 indexed oldTime, uint256 indexed newTime);

    /** 
    * @notice Emitted when the scheduled snapshot with the specified time was cancelled.
    */
    event SnapshotUnschedule(uint256 indexed time);

    /* ============ ERC-7201 ============ */
    // keccak256(abi.encode(uint256(keccak256("CMTAT.storage.SnapshotModuleBase")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant SnapshotModuleBaseStorageLocation = 0x649d9af4a0486294740af60c5e3bf61210e7b49108a80b1f369042ea9fd02000;
    /* ==== ERC-7201 State Variables === */
    struct SnapshotModuleBaseStorage {
        /**
        * @dev See {OpenZeppelin - ERC20Snapshot}
        */
        mapping(address => Snapshots) _accountBalanceSnapshots;
        Snapshots _totalSupplySnapshots;
        /**
        * @dev time instead of a counter for OpenZeppelin
        */
        // Initialized to zero
        uint256  _currentSnapshotTime;
        // Initialized to zero
        uint256  _currentSnapshotIndex;
        /** 
        * @dev 
        * list of scheduled snapshot (time)
        * This list is sorted in ascending order
        */
        uint256[] _scheduledSnapshots;
    }
    /*//////////////////////////////////////////////////////////////
                         INITIALIZER FUNCTION
    //////////////////////////////////////////////////////////////*/
    function __SnapshotModuleBase_init_unchained() internal onlyInitializing {
        // Nothing to do
        // _currentSnapshotTime & _currentSnapshotIndex are initialized to zero
    }

    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    /** 
    *  
    * @notice Get all snapshots
    */
    function getAllSnapshots() public view returns (uint256[] memory) {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        return $._scheduledSnapshots;
    }

    /** 
    * @dev 
    * Get the next scheduled snapshots
    */
    function getNextSnapshots() public view returns (uint256[] memory) {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        uint256[] memory nextScheduledSnapshot = new uint256[](0);
        // no snapshot were planned
        if ($._scheduledSnapshots.length > 0) {
            (
                uint256 timeLowerBound,
                uint256 indexLowerBound
            ) = _findScheduledMostRecentPastSnapshot();
            // All snapshots are situated in the futur
            if ((timeLowerBound == 0) && ($._currentSnapshotTime == 0)) {
                return $._scheduledSnapshots;
            } else {
                // There are snapshots situated in the futur
                if (indexLowerBound + 1 != $._scheduledSnapshots.length) {
                    // All next snapshots are located after the snapshot specified by indexLowerBound
                    uint256 arraySize = $._scheduledSnapshots.length -
                        indexLowerBound -
                        1;
                    nextScheduledSnapshot = new uint256[](arraySize);
                    // No need of unchecked block since Soliditiy 0.8.22
                    for (uint256 i; i < arraySize; ++i) {
                        nextScheduledSnapshot[i] = $._scheduledSnapshots[
                            indexLowerBound + 1 + i
                        ];
                    }
                }
            }
        }
        return nextScheduledSnapshot;
    }

    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /** 
    * @dev schedule a snapshot at the specified time
    * You can only add a snapshot after the last previous
    */
    function _scheduleSnapshot(uint256 time) internal {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        // Check the time firstly to avoid an useless read of storage
       _checkTimeInThePast(time);

        if ($._scheduledSnapshots.length > 0) {
            // We check the last snapshot on the list
            uint256 nextSnapshotTime = $._scheduledSnapshots[
                $._scheduledSnapshots.length - 1
            ];
            if (time < nextSnapshotTime) {
                revert Errors
                    .CMTAT_SnapshotModule_SnapshotTimestampBeforeLastSnapshot(
                        time,
                        nextSnapshotTime
                    );
            } else if (time == nextSnapshotTime) {
                revert Errors.CMTAT_SnapshotModule_SnapshotAlreadyExists();
            }
        }
        $._scheduledSnapshots.push(time);
        emit SnapshotSchedule(0, time);
    }

    /** 
    * @dev schedule a snapshot at the specified time
    */
    function _scheduleSnapshotNotOptimized(uint256 time) internal {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        _checkTimeInThePast(time);
        (bool isFound, uint256 index) = _findScheduledSnapshotIndex(time);
        // Perfect match
        if (isFound) {
            revert Errors.CMTAT_SnapshotModule_SnapshotAlreadyExists();
        }
        // if no upper bound match found, we push the snapshot at the end of the list
        if (index == $._scheduledSnapshots.length) {
            $._scheduledSnapshots.push(time);
        } else {
            $._scheduledSnapshots.push(
                $._scheduledSnapshots[$._scheduledSnapshots.length - 1]
            );
            for (uint256 i = $._scheduledSnapshots.length - 2; i > index; ) {
                $._scheduledSnapshots[i] = $._scheduledSnapshots[i - 1];
                unchecked {
                    --i;
                }
            }
            $._scheduledSnapshots[index] = time;
        }
        emit SnapshotSchedule(0, time);
    }

    /** 
    * @dev reschedule a scheduled snapshot at the specified newTime
    */
    function _rescheduleSnapshot(uint256 oldTime, uint256 newTime) internal {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        // Check the time firstly to avoid an useless read of storage
        _checkTimeSnapshotAlreadyDone(oldTime);
        _checkTimeInThePast(newTime);
        if ($._scheduledSnapshots.length == 0) {
            revert Errors.CMTAT_SnapshotModule_NoSnapshotScheduled();
        }
        uint256 index = _findAndRevertScheduledSnapshotIndex(oldTime);
        if (index + 1 < $._scheduledSnapshots.length) {
            uint256 nextSnapshotTime = $._scheduledSnapshots[index + 1];
            if (newTime > nextSnapshotTime) {
                revert Errors
                    .CMTAT_SnapshotModule_SnapshotTimestampAfterNextSnapshot(
                        newTime,
                        nextSnapshotTime
                    );
            } else if (newTime == nextSnapshotTime) {
                revert Errors.CMTAT_SnapshotModule_SnapshotAlreadyExists();
            }
        }
        if (index > 0) {
            if (newTime <= $._scheduledSnapshots[index - 1])
                revert Errors
                    .CMTAT_SnapshotModule_SnapshotTimestampBeforePreviousSnapshot(
                        newTime,
                        $._scheduledSnapshots[index - 1]
                    );
        }
        $._scheduledSnapshots[index] = newTime;

        emit SnapshotSchedule(oldTime, newTime);
    }

    /**
    * @dev unschedule the last scheduled snapshot
    */
    function _unscheduleLastSnapshot(uint256 time) internal {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        // Check the time firstly to avoid an useless read of storage
        _checkTimeSnapshotAlreadyDone(time);
        if ($._scheduledSnapshots.length == 0) {
            revert Errors.CMTAT_SnapshotModule_NoSnapshotScheduled();
        }
        // All snapshot time are unique, so we do not check the indice
        if (time !=$._scheduledSnapshots[$._scheduledSnapshots.length - 1]) {
            revert Errors.CMTAT_SnapshotModule_SnapshotNotFound();
        }
        $._scheduledSnapshots.pop();
        emit SnapshotUnschedule(time);
    }

    /** 
    * @dev unschedule (remove) a scheduled snapshot in three steps:
    * - search the snapshot in the list
    * - If found, move all next snapshots one position to the left
    * - Reduce the array size by deleting the last snapshot
    */
    function _unscheduleSnapshotNotOptimized(uint256 time) internal {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        _checkTimeSnapshotAlreadyDone(time);
        
        uint256 index = _findAndRevertScheduledSnapshotIndex(time);
        // No need of unchecked block since Soliditiy 0.8.22
        for (uint256 i = index; i + 1 < $._scheduledSnapshots.length; ++i ) {
            $._scheduledSnapshots[i] = $._scheduledSnapshots[i + 1];
        }
        $._scheduledSnapshots.pop();
    }

    /**
    * @dev See {OpenZeppelin - ERC20Snapshot}
    * @param time where we want a snapshot
    * @param snapshots the struct where are stored the snapshots
    * @return  snapshotExist true if a snapshot is found, false otherwise
    * value 0 if no snapshot, balance value if a snapshot exists
    */
    function _valueAt(
        uint256 time,
        Snapshots storage snapshots
    ) internal view returns (bool snapshotExist, uint256 value) {
        // When a valid snapshot is queried, there are three possibilities:
        //  a) The queried value was not modified after the snapshot was taken. Therefore, a snapshot entry was never
        //  created for this id, and all stored snapshot ids are smaller than the requested one. The value that corresponds
        //  to this id is the current one.
        //  b) The queried value was modified after the snapshot was taken. Therefore, there will be an entry with the
        //  requested id, and its value is the one to return.
        //  c) More snapshots were created after the requested one, and the queried value was later modified. There will be
        //  no entry for the requested id: the value that corresponds to it is that of the smallest snapshot id that is
        //  larger than the requested one.
        //
        // In summary, we need to find an element in an array, returning the index of the smallest value that is larger if
        // it is not found, unless said value doesn't exist (e.g. when all values are smaller). Arrays.findUpperBound does
        // exactly this.

        uint256 index = snapshots.ids.findUpperBound(time);

        if (index == snapshots.ids.length) {
            return (false, 0);
        } else {
            return (true, snapshots.values[index]);
        }
    }

    /** 
    * @dev 
    * Inside a struct Snapshots:
    * - Update the array ids to the current Snapshot time if this one is greater than the snapshot times stored in ids.
    * - Update the value to the corresponding value.
    */
    function _updateSnapshot(
        Snapshots storage snapshots,
        uint256 currentValue
    ) internal {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        uint256 current = $._currentSnapshotTime;
        if (_lastSnapshot(snapshots.ids) < current) {
            snapshots.ids.push(current);
            snapshots.values.push(currentValue);
        }
    }

    /** 
    * @dev
    * Set the currentSnapshotTime by retrieving the most recent snapshot
    * if a snapshot exists, clear all past scheduled snapshot
    */
    function _setCurrentSnapshot() internal {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        (
            uint256 scheduleSnapshotTime,
            uint256 scheduleSnapshotIndex
        ) = _findScheduledMostRecentPastSnapshot();
        if (scheduleSnapshotTime > 0) {
            $._currentSnapshotTime = scheduleSnapshotTime;
            $._currentSnapshotIndex = scheduleSnapshotIndex;
        }
    }

    /**
    * @return the last snapshot time inside a snapshot ids array
    */
    function _lastSnapshot(
        uint256[] storage ids
    ) private view returns (uint256) {
        if (ids.length == 0) {
            return 0;
        } else {
            return ids[ids.length - 1];
        }
    }

    /** 
    * @dev Find the snapshot index at the specified time
    * @return (true, index) if the snapshot exists, (false, 0) otherwise
    */
    function _findScheduledSnapshotIndex(
        uint256 time
    ) private view returns (bool, uint256) {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        uint256 indexFound = $._scheduledSnapshots.findUpperBound(time);
        uint256 _scheduledSnapshotsLength = $._scheduledSnapshots.length;
        // Exact match
        if (
            indexFound != _scheduledSnapshotsLength &&
            $._scheduledSnapshots[indexFound] == time
        ) {
            return (true, indexFound);
        }
        // Upper bound match
        else if (indexFound != _scheduledSnapshotsLength) {
            return (false, indexFound);
        }
        // no match
        else {
            return (false, _scheduledSnapshotsLength);
        }
    }

    /** 
    * @dev find the most recent past snapshot
    * The complexity of this function is O(N) because we go through the whole list
    */
    function _findScheduledMostRecentPastSnapshot()
        private
        view
        returns (uint256 time, uint256 index)
    {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        uint256 currentArraySize = $._scheduledSnapshots.length;
        // no snapshot or the current snapshot already points on the last snapshot
        if (
            currentArraySize == 0 ||
            (($._currentSnapshotIndex + 1 == currentArraySize) && (time != 0))
        ) {
            return (0, currentArraySize);
        }
        // mostRecent is initialized in the loop
        uint256 mostRecent;
        index = currentArraySize;
        // No need of unchecked block since Soliditiy 0.8.22
        for (uint256 i = $._currentSnapshotIndex; i < currentArraySize; ++i ) {
            if ($._scheduledSnapshots[i] <= block.timestamp) {
                mostRecent = $._scheduledSnapshots[i];
                index = i;
            } else {
                // All snapshot are planned in the futur
                break;
            }
        }
        return (mostRecent, index);
    }

    /* ============ Utility functions ============ */


    function _findAndRevertScheduledSnapshotIndex(
        uint256 time
    ) private view returns (uint256){
        (bool isFound, uint256 index) = _findScheduledSnapshotIndex(time);
        if (!isFound) {
            revert Errors.CMTAT_SnapshotModule_SnapshotNotFound();
        }
        return index;
    }
    function _checkTimeInThePast(uint256 time) internal view{
        if (time <= block.timestamp) {
                    revert Errors.CMTAT_SnapshotModule_SnapshotScheduledInThePast(
                        time,
                        block.timestamp
                    );
                }
    }
    function _checkTimeSnapshotAlreadyDone(uint256 time) internal view{
        if (time <= block.timestamp) {
            revert Errors.CMTAT_SnapshotModule_SnapshotAlreadyDone();
        }
    }

    /* ============ ERC-7201 ============ */
    function _getSnapshotModuleBaseStorage() internal pure returns (SnapshotModuleBaseStorage storage $) {
        assembly {
            $.slot := SnapshotModuleBaseStorageLocation
        }
    }
}


// File contracts/modules/internal/ERC20SnapshotModuleInternal.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;




/**
 * @dev Snapshot module internal.
 *
 * Useful to take a snapshot of token holder balance and total supply at a specific time
 * Inspired by Openzeppelin - ERC20Snapshot but use the time as Id instead of a counter.
 * Contrary to OpenZeppelin, the function _getCurrentSnapshotId is not available 
   because overriding this function can break the contract.
 */

abstract contract ERC20SnapshotModuleInternal is ICMTATSnapshot, SnapshotModuleBase, ERC20Upgradeable {
    using Arrays for uint256[];
    /* ============  Initializer Function ============ */
    function __ERC20Snapshot_init_unchained() internal onlyInitializing {
        // Nothing to do
        // _currentSnapshotTime & _currentSnapshotIndex are initialized to zero
    }


    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    /**
    * @notice Return snapshotBalanceOf and snapshotTotalSupply to avoid multiple calls
    * @return ownerBalance ,  totalSupply - see snapshotBalanceOf and snapshotTotalSupply
    */
    function snapshotInfo(uint256 time, address owner) public view returns (uint256 ownerBalance, uint256 totalSupply) {
        ownerBalance = snapshotBalanceOf(time, owner);
        totalSupply = snapshotTotalSupply(time);
    }

    /**
    * @notice Return snapshotBalanceOf for each address in the array and the total supply
    * @return ownerBalances array with the balance of each address, the total supply
    */
    function snapshotInfoBatch(uint256 time, address[] calldata addresses) public view returns (uint256[] memory ownerBalances, uint256 totalSupply) {
        ownerBalances = new uint256[](addresses.length);
        for(uint256 i = 0; i < addresses.length; ++i){
             ownerBalances[i]  = snapshotBalanceOf(time, addresses[i]);
        }
        totalSupply = snapshotTotalSupply(time);
    }

    /**
    * @notice Return snapshotBalanceOf for each address in the array and the total supply
    * @return ownerBalances array with the balance of each address, the total supply
    */
    function snapshotInfoBatch(uint256[] calldata times, address[] calldata addresses) public view returns (uint256[][] memory ownerBalances, uint256[] memory totalSupply) {
        ownerBalances = new uint256[][](times.length);
        totalSupply = new uint256[](times.length);
        for(uint256 iT = 0; iT < times.length; ++iT){
            (ownerBalances[iT], totalSupply[iT]) = snapshotInfoBatch(times[iT],addresses);
        }
    }

    /** 
    * @notice Return the number of tokens owned by the given owner at the time when the snapshot with the given time was created.
    * @return value stored in the snapshot, or the actual balance if no snapshot
    */
    function snapshotBalanceOf(
        uint256 time,
        address owner
    ) public view returns (uint256) {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        (bool snapshotted, uint256 value) = _valueAt(
            time,
            $._accountBalanceSnapshots[owner]
        );

        return snapshotted ? value : balanceOf(owner);
    }

    /**
    * @dev See {OpenZeppelin - ERC20Snapshot}
    * Retrieves the total supply at the specified time.
    * @return value stored in the snapshot, or the actual totalSupply if no snapshot
    */
    function snapshotTotalSupply(uint256 time) public view returns (uint256) {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        (bool snapshotted, uint256 value) = _valueAt(
            time,
            $._totalSupplySnapshots
        );
        return snapshotted ? value : totalSupply();
    }

    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /** 
    * @dev Update balance and/or total supply snapshots before the values are modified. This is implemented
    * in the _beforeTokenTransfer hook, which is executed for _mint, _burn, and _transfer operations.
    */
    function _snapshotUpdate(
        address from,
        address to
    ) internal virtual  {
        _setCurrentSnapshot();
        if (from != address(0)) {
            // for both burn and transfer
            _updateAccountSnapshot(from);
            if (to != address(0)) {
                // transfer
                _updateAccountSnapshot(to);
            } else {
                // burn
                _updateTotalSupplySnapshot();
            }
        } else {
            // mint
            _updateAccountSnapshot(to);
            _updateTotalSupplySnapshot();
        }
    }

    /**
    * @dev See {OpenZeppelin - ERC20Snapshot}
    */
    function _updateAccountSnapshot(address account) private {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        _updateSnapshot($._accountBalanceSnapshots[account], balanceOf(account));
    }

    /**
    * @dev See {OpenZeppelin - ERC20Snapshot}
    */
    function _updateTotalSupplySnapshot() private {
        SnapshotModuleBaseStorage storage $ = _getSnapshotModuleBaseStorage();
        _updateSnapshot($._totalSupplySnapshots, totalSupply());
    }
}


// File contracts/modules/wrapper/extensions/ERC20SnapshotModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;


/**
 * @title Snapshot module
 * @dev 
 *
 * Useful to take a snapshot of token holder balance and total supply at a specific time
 */

abstract contract ERC20SnapshotModule is
    ERC20SnapshotModuleInternal,
    AuthorizationModule
{
    /* ============ State Variables ============ */
    bytes32 public constant SNAPSHOOTER_ROLE = keccak256("SNAPSHOOTER_ROLE");
    /* ============  Initializer Function ============ */
    function __ERC20SnasphotModule_init_unchained() internal onlyInitializing {
        // no variable to initialize
    }
    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    /** 
    * @notice 
    * Schedule a snapshot at the given time specified as a number of seconds since epoch.
    * The time cannot be before the time of the latest scheduled, but not yet created snapshot.  
    */
    function scheduleSnapshot(uint256 time) public onlyRole(SNAPSHOOTER_ROLE) {
        _scheduleSnapshot(time);
    }

    /** 
    * @notice 
    * Schedule a snapshot at the given time specified as a number of seconds since epoch.
    * The time cannot be before the time of the latest scheduled, but not yet created snapshot.  
    */
    function scheduleSnapshotNotOptimized(
        uint256 time
    ) public onlyRole(SNAPSHOOTER_ROLE) {
        _scheduleSnapshotNotOptimized(time);
    }

    /** 
    * @notice
    * Reschedule the scheduled snapshot, but not yet created snapshot with the given oldTime to be created at the given newTime specified as a number of seconds since epoch. 
    * The newTime cannot be before the time of the previous scheduled, but not yet created snapshot, or after the time fo the next scheduled snapshot. 
    */
    function rescheduleSnapshot(
        uint256 oldTime,
        uint256 newTime
    ) public onlyRole(SNAPSHOOTER_ROLE) {
        _rescheduleSnapshot(oldTime, newTime);
    }

    /**
    * @notice 
    * Cancel creation of the scheduled snapshot, but not yet created snapshot with the given time. 
    * There should not be any other snapshots scheduled after this one. 
    */
    function unscheduleLastSnapshot(
        uint256 time
    ) public onlyRole(SNAPSHOOTER_ROLE) {
        _unscheduleLastSnapshot(time);
    }

    /** 
    * @notice 
    * Cancel creation of the scheduled snapshot, but not yet created snapshot with the given time. 
    */
    function unscheduleSnapshotNotOptimized(
        uint256 time
    ) public onlyRole(SNAPSHOOTER_ROLE) {
        _unscheduleSnapshotNotOptimized(time);
    }
}


// File contracts/modules/wrapper/extensions/MetaTxModule.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

/**
 * @title Meta transaction (gasless) module.
 * @dev 
 *
 * Useful for to provide UX where the user does not pay gas for token exchange
 * To follow OpenZeppelin, this contract does not implement the functions init & init_unchained.
 * ()
 */
abstract contract MetaTxModule is ERC2771ContextUpgradeable {
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor(
        address trustedForwarder
    ) ERC2771ContextUpgradeable(trustedForwarder) {
        // Nothing to do
    }
}


// File contracts/modules/CMTAT_BASE.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

// required OZ imports here







/*
* SnapshotModule:
* Add this import in case you add the SnapshotModule
*/







abstract contract CMTAT_BASE is
    Initializable,
    ContextUpgradeable,
    // Core
    BaseModule,
    PauseModule,
    ERC20MintModule,
    ERC20BurnModule,
    EnforcementModule,
    ValidationModule,
    ERC20BaseModule,
    // Extension
    MetaTxModule,
    ERC20SnapshotModule,
    DebtModule,
    DocumentModule
{   

    /*//////////////////////////////////////////////////////////////
                         INITIALIZER FUNCTION
    //////////////////////////////////////////////////////////////*/
    /**
     * @notice
     * initialize the proxy contract
     * The calls to this function will revert if the contract was deployed without a proxy
     * @param admin address of the admin of contract (Access Control)
     * @param ERC20Attributes_ ERC20 name, symbol and decimals
     * @param baseModuleAttributes_ tokenId, terms, information
     * @param engines_ external contract
     */
    function initialize(
        address admin,
        ICMTATConstructor.ERC20Attributes memory ERC20Attributes_,
        ICMTATConstructor.BaseModuleAttributes memory baseModuleAttributes_,
        ICMTATConstructor.Engine memory engines_ 
    ) public virtual initializer {
        __CMTAT_init(
            admin,
            ERC20Attributes_,
            baseModuleAttributes_,
            engines_
        );
    }


    /**
     * @dev calls the different initialize functions from the different modules
     */
    function __CMTAT_init(
        address admin,
        ICMTATConstructor.ERC20Attributes memory ERC20Attributes_,
        ICMTATConstructor.BaseModuleAttributes memory baseModuleAttributes_,
        ICMTATConstructor.Engine memory engines_ 
    ) internal onlyInitializing {
        /* OpenZeppelin library */
        // OZ init_unchained functions are called firstly due to inheritance
        __Context_init_unchained();
        __ERC20_init_unchained(ERC20Attributes_.nameIrrevocable, ERC20Attributes_.symbolIrrevocable);
        // AccessControlUpgradeable inherits from ERC165Upgradeable
        __ERC165_init_unchained();
        // AuthorizationModule inherits from AccessControlUpgradeable
        __AccessControl_init_unchained();
        __Pausable_init_unchained();

        /* Internal Modules */
        __Enforcement_init_unchained();
        /*
        SnapshotModule:
        Add these two calls in case you add the SnapshotModule
            */
        __SnapshotModuleBase_init_unchained();
        __ERC20Snapshot_init_unchained();
    
        __Validation_init_unchained(engines_ .ruleEngine);

        /* Wrapper */
        // AuthorizationModule_init_unchained is called firstly due to inheritance
        __AuthorizationModule_init_unchained(admin, engines_ .authorizationEngine);
        __ERC20BurnModule_init_unchained();
        __ERC20MintModule_init_unchained();
        // EnforcementModule_init_unchained is called before ValidationModule_init_unchained due to inheritance
        __EnforcementModule_init_unchained();
        __ERC20BaseModule_init_unchained(ERC20Attributes_.decimalsIrrevocable);
        // PauseModule_init_unchained is called before ValidationModule_init_unchained due to inheritance
        __PauseModule_init_unchained();
        __ValidationModule_init_unchained();

        /*
        SnapshotModule:
        Add this call in case you add the SnapshotModule
        */
        __ERC20SnasphotModule_init_unchained();
        __DocumentModule_init_unchained(engines_ .documentEngine);
        __DebtModule_init_unchained(engines_ .debtEngine);

        /* Other modules */
        __Base_init_unchained(baseModuleAttributes_.tokenId, baseModuleAttributes_.terms, baseModuleAttributes_.information);

        /* own function */
        __CMTAT_init_unchained();
    }

    function __CMTAT_init_unchained() internal onlyInitializing {
        // no variable to initialize
    }


    /*//////////////////////////////////////////////////////////////
                            PUBLIC/EXTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /**
     * @notice Returns the number of decimals used to get its user representation.
     */
    function decimals()
        public
        view
        virtual
        override(ERC20Upgradeable, ERC20BaseModule)
        returns (uint8)
    {
        return ERC20BaseModule.decimals();
    }

    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    )
        public
        virtual
        override(ERC20Upgradeable, ERC20BaseModule)
        returns (bool)
    {
        return ERC20BaseModule.transferFrom(sender, recipient, amount);
    }

    /**
    * @notice burn and mint atomically
    * @param from current token holder to burn tokens
    * @param to receiver to send the new minted tokens
    * @param amountToBurn number of tokens to burn
    * @param amountToMint number of tokens to mint
    * @dev 
    * - The access control is managed by the functions burn (ERC20BurnModule) and mint (ERC20MintModule)
    * - Input validation is also managed by the functions burn and mint
    * - You can mint more tokens than burnt
    */
    function burnAndMint(address from, address to, uint256 amountToBurn, uint256 amountToMint, string calldata reason) public  {
        burn(from, amountToBurn, reason);
        mint(to, amountToMint);
    }

    /*//////////////////////////////////////////////////////////////
                            INTERNAL/PRIVATE FUNCTIONS
    //////////////////////////////////////////////////////////////*/
    /**
     * @dev
     *
     */
    function _update(
        address from,
        address to,
        uint256 amount
    ) internal override(ERC20Upgradeable) {
        if (!ValidationModule._operateOnTransfer(from, to, amount)) {
            revert Errors.CMTAT_InvalidTransfer(from, to, amount);
        }
        /*
        SnapshotModule:
        Add this in case you add the SnapshotModule
        We call the SnapshotModule only if the transfer is valid
        */
        ERC20SnapshotModuleInternal._snapshotUpdate(from, to);
        ERC20Upgradeable._update(from, to, amount);
    }
    /*//////////////////////////////////////////////////////////////
                            METAXTX MODULE
    //////////////////////////////////////////////////////////////*/
    /**
     * @dev This surcharge is not necessary if you do not use the MetaTxModule
     */
    function _msgSender()
        internal
        view
        override(ERC2771ContextUpgradeable, ContextUpgradeable)
        returns (address sender)
    {
        return ERC2771ContextUpgradeable._msgSender();
    }

    /**
     * @dev This surcharge is not necessary if you do not use the MetaTxModule
     */
    function _contextSuffixLength() internal view 
    override(ERC2771ContextUpgradeable, ContextUpgradeable)
    returns (uint256) {
         return ERC2771ContextUpgradeable._contextSuffixLength();
    }

    /**
     * @dev This surcharge is not necessary if you do not use the MetaTxModule
     */
    function _msgData()
        internal
        view
        override(ERC2771ContextUpgradeable, ContextUpgradeable)
        returns (bytes calldata)
    {
        return ERC2771ContextUpgradeable._msgData();
    }
}


// File contracts/CMTAT_STANDALONE.sol

// Original license: SPDX_License_Identifier: MPL-2.0

pragma solidity ^0.8.20;

/**
* @title CMTAT version for a standalone deployment (without proxy)
*/
contract CMTAT_STANDALONE is CMTAT_BASE {
    /**
     * @notice Contract version for standalone deployment
     * @param forwarderIrrevocable address of the forwarder, required for the gasless support
     * @param admin address of the admin of contract (Access Control)
     * @param ERC20Attributes_ ERC20 name, symbol and decimals
     * @param baseModuleAttributes_ tokenId, terms, information
     * @param engines_ external contract
     */
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor(
        address forwarderIrrevocable,
        address admin,
        ICMTATConstructor.ERC20Attributes memory ERC20Attributes_,
        ICMTATConstructor.BaseModuleAttributes memory baseModuleAttributes_,
        ICMTATConstructor.Engine memory engines_ 
    ) MetaTxModule(forwarderIrrevocable) {
        // Initialize the contract to avoid front-running
        // Warning : do not initialize the proxy
        initialize(
            admin,
            ERC20Attributes_,
            baseModuleAttributes_,
            engines_
        );
    }
}