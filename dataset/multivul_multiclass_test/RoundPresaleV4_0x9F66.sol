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

// File: @openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol


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
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;

    /// @custom:storage-location erc7201:openzeppelin.storage.ReentrancyGuard
    struct ReentrancyGuardStorage {
        uint256 _status;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.ReentrancyGuard")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant ReentrancyGuardStorageLocation = 0x9b779b17422d0df92223018b32b4d1fa46e071723d6817e2486d003becc55f00;

    function _getReentrancyGuardStorage() private pure returns (ReentrancyGuardStorage storage $) {
        assembly {
            $.slot := ReentrancyGuardStorageLocation
        }
    }

    /**
     * @dev Unauthorized reentrant call.
     */
    error ReentrancyGuardReentrantCall();

    function __ReentrancyGuard_init() internal onlyInitializing {
        __ReentrancyGuard_init_unchained();
    }

    function __ReentrancyGuard_init_unchained() internal onlyInitializing {
        ReentrancyGuardStorage storage $ = _getReentrancyGuardStorage();
        $._status = NOT_ENTERED;
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
        ReentrancyGuardStorage storage $ = _getReentrancyGuardStorage();
        // On the first call to nonReentrant, _status will be NOT_ENTERED
        if ($._status == ENTERED) {
            revert ReentrancyGuardReentrantCall();
        }

        // Any calls to nonReentrant after this point will fail
        $._status = ENTERED;
    }

    function _nonReentrantAfter() private {
        ReentrancyGuardStorage storage $ = _getReentrancyGuardStorage();
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        $._status = NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        ReentrancyGuardStorage storage $ = _getReentrancyGuardStorage();
        return $._status == ENTERED;
    }
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

// File: @openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol


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

// File: @openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol


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
abstract contract OwnableUpgradeable is Initializable, ContextUpgradeable {
    /// @custom:storage-location erc7201:openzeppelin.storage.Ownable
    struct OwnableStorage {
        address _owner;
    }

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.Ownable")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant OwnableStorageLocation = 0x9016d09d72d40fdae2fd8ceac6b6234c7706214fd39c1cd1e609a0528c199300;

    function _getOwnableStorage() private pure returns (OwnableStorage storage $) {
        assembly {
            $.slot := OwnableStorageLocation
        }
    }

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
    function __Ownable_init(address initialOwner) internal onlyInitializing {
        __Ownable_init_unchained(initialOwner);
    }

    function __Ownable_init_unchained(address initialOwner) internal onlyInitializing {
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
        OwnableStorage storage $ = _getOwnableStorage();
        return $._owner;
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
        OwnableStorage storage $ = _getOwnableStorage();
        address oldOwner = $._owner;
        $._owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
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

// File: @openzeppelin/contracts-upgradeable/token/ERC20/IERC20Upgradeable.sol


// OpenZeppelin Contracts (last updated v4.9.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.0;

/**
 * @dev Interface of the ERC20 standard as defined in the EIP.
 */
interface IERC20Upgradeable {
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
     * @dev Returns the amount of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the amount of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves `amount` tokens from the caller's account to `to`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address to, uint256 amount) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets `amount` as the allowance of `spender` over the caller's tokens.
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
    function approve(address spender, uint256 amount) external returns (bool);

    /**
     * @dev Moves `amount` tokens from `from` to `to` using the
     * allowance mechanism. `amount` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

// File: @openzeppelin/contracts-upgradeable/token/ERC20/extensions/IERC20PermitUpgradeable.sol


// OpenZeppelin Contracts (last updated v4.9.4) (token/ERC20/extensions/IERC20Permit.sol)

pragma solidity ^0.8.0;

/**
 * @dev Interface of the ERC20 Permit extension allowing approvals to be made via signatures, as defined in
 * https://eips.ethereum.org/EIPS/eip-2612[EIP-2612].
 *
 * Adds the {permit} method, which can be used to change an account's ERC20 allowance (see {IERC20-allowance}) by
 * presenting a message signed by the account. By not relying on {IERC20-approve}, the token holder account doesn't
 * need to send a transaction, and thus is not required to hold Ether at all.
 *
 * ==== Security Considerations
 *
 * There are two important considerations concerning the use of `permit`. The first is that a valid permit signature
 * expresses an allowance, and it should not be assumed to convey additional meaning. In particular, it should not be
 * considered as an intention to spend the allowance in any specific way. The second is that because permits have
 * built-in replay protection and can be submitted by anyone, they can be frontrun. A protocol that uses permits should
 * take this into consideration and allow a `permit` call to fail. Combining these two aspects, a pattern that may be
 * generally recommended is:
 *
 * ```solidity
 * function doThingWithPermit(..., uint256 value, uint256 deadline, uint8 v, bytes32 r, bytes32 s) public {
 *     try token.permit(msg.sender, address(this), value, deadline, v, r, s) {} catch {}
 *     doThing(..., value);
 * }
 *
 * function doThing(..., uint256 value) public {
 *     token.safeTransferFrom(msg.sender, address(this), value);
 *     ...
 * }
 * ```
 *
 * Observe that: 1) `msg.sender` is used as the owner, leaving no ambiguity as to the signer intent, and 2) the use of
 * `try/catch` allows the permit to fail and makes the code tolerant to frontrunning. (See also
 * {SafeERC20-safeTransferFrom}).
 *
 * Additionally, note that smart contract wallets (such as Argent or Safe) are not able to produce permit signatures, so
 * contracts should have entry points that don't rely on permit.
 */
interface IERC20PermitUpgradeable {
    /**
     * @dev Sets `value` as the allowance of `spender` over ``owner``'s tokens,
     * given ``owner``'s signed approval.
     *
     * IMPORTANT: The same issues {IERC20-approve} has related to transaction
     * ordering also apply here.
     *
     * Emits an {Approval} event.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     * - `deadline` must be a timestamp in the future.
     * - `v`, `r` and `s` must be a valid `secp256k1` signature from `owner`
     * over the EIP712-formatted function arguments.
     * - the signature must use ``owner``'s current nonce (see {nonces}).
     *
     * For more information on the signature format, see the
     * https://eips.ethereum.org/EIPS/eip-2612#specification[relevant EIP
     * section].
     *
     * CAUTION: See Security Considerations above.
     */
    function permit(
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external;

    /**
     * @dev Returns the current nonce for `owner`. This value must be
     * included whenever a signature is generated for {permit}.
     *
     * Every successful call to {permit} increases ``owner``'s nonce by one. This
     * prevents a signature from being used multiple times.
     */
    function nonces(address owner) external view returns (uint256);

    /**
     * @dev Returns the domain separator used in the encoding of the signature for {permit}, as defined by {EIP712}.
     */
    // solhint-disable-next-line func-name-mixedcase
    function DOMAIN_SEPARATOR() external view returns (bytes32);
}

// File: @openzeppelin/contracts-upgradeable/utils/AddressUpgradeable.sol


// OpenZeppelin Contracts (last updated v4.9.0) (utils/Address.sol)

pragma solidity ^0.8.1;

/**
 * @dev Collection of functions related to the address type
 */
library AddressUpgradeable {
    /**
     * @dev Returns true if `account` is a contract.
     *
     * [IMPORTANT]
     * ====
     * It is unsafe to assume that an address for which this function returns
     * false is an externally-owned account (EOA) and not a contract.
     *
     * Among others, `isContract` will return false for the following
     * types of addresses:
     *
     *  - an externally-owned account
     *  - a contract in construction
     *  - an address where a contract will be created
     *  - an address where a contract lived, but was destroyed
     *
     * Furthermore, `isContract` will also return true if the target contract within
     * the same transaction is already scheduled for destruction by `SELFDESTRUCT`,
     * which only has an effect at the end of a transaction.
     * ====
     *
     * [IMPORTANT]
     * ====
     * You shouldn't rely on `isContract` to protect against flash loan attacks!
     *
     * Preventing calls from contracts is highly discouraged. It breaks composability, breaks support for smart wallets
     * like Gnosis Safe, and does not provide security since it can be circumvented by calling from a contract
     * constructor.
     * ====
     */
    function isContract(address account) internal view returns (bool) {
        // This method relies on extcodesize/address.code.length, which returns 0
        // for contracts in construction, since the code is only stored at the end
        // of the constructor execution.

        return account.code.length > 0;
    }

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
     * https://solidity.readthedocs.io/en/v0.8.0/security-considerations.html#use-the-checks-effects-interactions-pattern[checks-effects-interactions pattern].
     */
    function sendValue(address payable recipient, uint256 amount) internal {
        require(address(this).balance >= amount, "Address: insufficient balance");

        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Address: unable to send value, recipient may have reverted");
    }

    /**
     * @dev Performs a Solidity function call using a low level `call`. A
     * plain `call` is an unsafe replacement for a function call: use this
     * function instead.
     *
     * If `target` reverts with a revert reason, it is bubbled up by this
     * function (like regular Solidity function calls).
     *
     * Returns the raw returned data. To convert to the expected return value,
     * use https://solidity.readthedocs.io/en/latest/units-and-global-variables.html?highlight=abi.decode#abi-encoding-and-decoding-functions[`abi.decode`].
     *
     * Requirements:
     *
     * - `target` must be a contract.
     * - calling `target` with `data` must not revert.
     *
     * _Available since v3.1._
     */
    function functionCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0, "Address: low-level call failed");
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`], but with
     * `errorMessage` as a fallback revert reason when `target` reverts.
     *
     * _Available since v3.1._
     */
    function functionCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0, errorMessage);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but also transferring `value` wei to `target`.
     *
     * Requirements:
     *
     * - the calling contract must have an ETH balance of at least `value`.
     * - the called Solidity function must be `payable`.
     *
     * _Available since v3.1._
     */
    function functionCallWithValue(address target, bytes memory data, uint256 value) internal returns (bytes memory) {
        return functionCallWithValue(target, data, value, "Address: low-level call with value failed");
    }

    /**
     * @dev Same as {xref-Address-functionCallWithValue-address-bytes-uint256-}[`functionCallWithValue`], but
     * with `errorMessage` as a fallback revert reason when `target` reverts.
     *
     * _Available since v3.1._
     */
    function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value,
        string memory errorMessage
    ) internal returns (bytes memory) {
        require(address(this).balance >= value, "Address: insufficient balance for call");
        (bool success, bytes memory returndata) = target.call{value: value}(data);
        return verifyCallResultFromTarget(target, success, returndata, errorMessage);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a static call.
     *
     * _Available since v3.3._
     */
    function functionStaticCall(address target, bytes memory data) internal view returns (bytes memory) {
        return functionStaticCall(target, data, "Address: low-level static call failed");
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-string-}[`functionCall`],
     * but performing a static call.
     *
     * _Available since v3.3._
     */
    function functionStaticCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal view returns (bytes memory) {
        (bool success, bytes memory returndata) = target.staticcall(data);
        return verifyCallResultFromTarget(target, success, returndata, errorMessage);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a delegate call.
     *
     * _Available since v3.4._
     */
    function functionDelegateCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionDelegateCall(target, data, "Address: low-level delegate call failed");
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-string-}[`functionCall`],
     * but performing a delegate call.
     *
     * _Available since v3.4._
     */
    function functionDelegateCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal returns (bytes memory) {
        (bool success, bytes memory returndata) = target.delegatecall(data);
        return verifyCallResultFromTarget(target, success, returndata, errorMessage);
    }

    /**
     * @dev Tool to verify that a low level call to smart-contract was successful, and revert (either by bubbling
     * the revert reason or using the provided one) in case of unsuccessful call or if target was not a contract.
     *
     * _Available since v4.8._
     */
    function verifyCallResultFromTarget(
        address target,
        bool success,
        bytes memory returndata,
        string memory errorMessage
    ) internal view returns (bytes memory) {
        if (success) {
            if (returndata.length == 0) {
                // only check isContract if the call was successful and the return data is empty
                // otherwise we already know that it was a contract
                require(isContract(target), "Address: call to non-contract");
            }
            return returndata;
        } else {
            _revert(returndata, errorMessage);
        }
    }

    /**
     * @dev Tool to verify that a low level call was successful, and revert if it wasn't, either by bubbling the
     * revert reason or using the provided one.
     *
     * _Available since v4.3._
     */
    function verifyCallResult(
        bool success,
        bytes memory returndata,
        string memory errorMessage
    ) internal pure returns (bytes memory) {
        if (success) {
            return returndata;
        } else {
            _revert(returndata, errorMessage);
        }
    }

    function _revert(bytes memory returndata, string memory errorMessage) private pure {
        // Look for revert reason and bubble it up if present
        if (returndata.length > 0) {
            // The easiest way to bubble the revert reason is using memory via assembly
            /// @solidity memory-safe-assembly
            assembly {
                let returndata_size := mload(returndata)
                revert(add(32, returndata), returndata_size)
            }
        } else {
            revert(errorMessage);
        }
    }
}

// File: @openzeppelin/contracts-upgradeable/token/ERC20/utils/SafeERC20Upgradeable.sol


// OpenZeppelin Contracts (last updated v4.9.3) (token/ERC20/utils/SafeERC20.sol)

pragma solidity ^0.8.0;




/**
 * @title SafeERC20
 * @dev Wrappers around ERC20 operations that throw on failure (when the token
 * contract returns false). Tokens that return no value (and instead revert or
 * throw on failure) are also supported, non-reverting calls are assumed to be
 * successful.
 * To use this library you can add a `using SafeERC20 for IERC20;` statement to your contract,
 * which allows you to call the safe operations as `token.safeTransfer(...)`, etc.
 */
library SafeERC20Upgradeable {
    using AddressUpgradeable for address;

    /**
     * @dev Transfer `value` amount of `token` from the calling contract to `to`. If `token` returns no value,
     * non-reverting calls are assumed to be successful.
     */
    function safeTransfer(IERC20Upgradeable token, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transfer.selector, to, value));
    }

    /**
     * @dev Transfer `value` amount of `token` from `from` to `to`, spending the approval given by `from` to the
     * calling contract. If `token` returns no value, non-reverting calls are assumed to be successful.
     */
    function safeTransferFrom(IERC20Upgradeable token, address from, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transferFrom.selector, from, to, value));
    }

    /**
     * @dev Deprecated. This function has issues similar to the ones found in
     * {IERC20-approve}, and its usage is discouraged.
     *
     * Whenever possible, use {safeIncreaseAllowance} and
     * {safeDecreaseAllowance} instead.
     */
    function safeApprove(IERC20Upgradeable token, address spender, uint256 value) internal {
        // safeApprove should only be called when setting an initial allowance,
        // or when resetting it to zero. To increase and decrease it, use
        // 'safeIncreaseAllowance' and 'safeDecreaseAllowance'
        require(
            (value == 0) || (token.allowance(address(this), spender) == 0),
            "SafeERC20: approve from non-zero to non-zero allowance"
        );
        _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, value));
    }

    /**
     * @dev Increase the calling contract's allowance toward `spender` by `value`. If `token` returns no value,
     * non-reverting calls are assumed to be successful.
     */
    function safeIncreaseAllowance(IERC20Upgradeable token, address spender, uint256 value) internal {
        uint256 oldAllowance = token.allowance(address(this), spender);
        _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, oldAllowance + value));
    }

    /**
     * @dev Decrease the calling contract's allowance toward `spender` by `value`. If `token` returns no value,
     * non-reverting calls are assumed to be successful.
     */
    function safeDecreaseAllowance(IERC20Upgradeable token, address spender, uint256 value) internal {
        unchecked {
            uint256 oldAllowance = token.allowance(address(this), spender);
            require(oldAllowance >= value, "SafeERC20: decreased allowance below zero");
            _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, oldAllowance - value));
        }
    }

    /**
     * @dev Set the calling contract's allowance toward `spender` to `value`. If `token` returns no value,
     * non-reverting calls are assumed to be successful. Meant to be used with tokens that require the approval
     * to be set to zero before setting it to a non-zero value, such as USDT.
     */
    function forceApprove(IERC20Upgradeable token, address spender, uint256 value) internal {
        bytes memory approvalCall = abi.encodeWithSelector(token.approve.selector, spender, value);

        if (!_callOptionalReturnBool(token, approvalCall)) {
            _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, 0));
            _callOptionalReturn(token, approvalCall);
        }
    }

    /**
     * @dev Use a ERC-2612 signature to set the `owner` approval toward `spender` on `token`.
     * Revert on invalid signature.
     */
    function safePermit(
        IERC20PermitUpgradeable token,
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) internal {
        uint256 nonceBefore = token.nonces(owner);
        token.permit(owner, spender, value, deadline, v, r, s);
        uint256 nonceAfter = token.nonces(owner);
        require(nonceAfter == nonceBefore + 1, "SafeERC20: permit did not succeed");
    }

    /**
     * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
     * on the return value: the return value is optional (but if data is returned, it must not be false).
     * @param token The token targeted by the call.
     * @param data The call data (encoded using abi.encode or one of its variants).
     */
    function _callOptionalReturn(IERC20Upgradeable token, bytes memory data) private {
        // We need to perform a low level call here, to bypass Solidity's return data size checking mechanism, since
        // we're implementing it ourselves. We use {Address-functionCall} to perform this call, which verifies that
        // the target address contains contract code and also asserts for success in the low-level call.

        bytes memory returndata = address(token).functionCall(data, "SafeERC20: low-level call failed");
        require(returndata.length == 0 || abi.decode(returndata, (bool)), "SafeERC20: ERC20 operation did not succeed");
    }

    /**
     * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
     * on the return value: the return value is optional (but if data is returned, it must not be false).
     * @param token The token targeted by the call.
     * @param data The call data (encoded using abi.encode or one of its variants).
     *
     * This is a variant of {_callOptionalReturn} that silents catches all reverts and returns a bool instead.
     */
    function _callOptionalReturnBool(IERC20Upgradeable token, bytes memory data) private returns (bool) {
        // We need to perform a low level call here, to bypass Solidity's return data size checking mechanism, since
        // we're implementing it ourselves. We cannot use {Address-functionCall} here since this should return false
        // and not revert is the subcall reverts.

        (bool success, bytes memory returndata) = address(token).call(data);
        return
            success && (returndata.length == 0 || abi.decode(returndata, (bool))) && AddressUpgradeable.isContract(address(token));
    }
}

// File: @openzeppelin/contracts-upgradeable/token/ERC20/extensions/IERC20MetadataUpgradeable.sol


// OpenZeppelin Contracts v4.4.1 (token/ERC20/extensions/IERC20Metadata.sol)

pragma solidity ^0.8.0;


/**
 * @dev Interface for the optional metadata functions from the ERC20 standard.
 *
 * _Available since v4.1._
 */
interface IERC20MetadataUpgradeable is IERC20Upgradeable {
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

// File: @chainlink/contracts/src/v0.8/shared/interfaces/AggregatorV3Interface.sol


pragma solidity ^0.8.0;

// solhint-disable-next-line interface-starts-with-i
interface AggregatorV3Interface {
  function decimals() external view returns (uint8);

  function description() external view returns (string memory);

  function version() external view returns (uint256);

  function getRoundData(
    uint80 _roundId
  ) external view returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);

  function latestRoundData()
    external
    view
    returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);
}

// File: round-presale-v4 (11)/contracts/RoundPresaleBase.sol


pragma solidity ^0.8.26;











/**
* @title RoundPresaleBase
* @dev Base contract for round-based token presale
*/
abstract contract RoundPresaleBase is 
    Initializable, 
    ReentrancyGuardUpgradeable, 
    PausableUpgradeable, 
    OwnableUpgradeable,
    AccessControlUpgradeable,
    UUPSUpgradeable
{
    using SafeERC20Upgradeable for IERC20Upgradeable;

    // Role definitions
    bytes32 public constant MANAGER_ROLE = keccak256("MANAGER_ROLE");

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
    event VestingContractSet(address indexed vestingContract);
    event VestingEnabled(bool enabled);
    event PresaleStartTimeSet(uint256 startTime);

    // Round information
    struct Round {
        uint256 price;             // Token price in Wei
        uint256 priceUSD;          // Token price in USD (18 decimals)
        uint256 allocation;        // Token allocation for the round
        uint256 sold;              // Tokens sold in the round
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

    // Order detail for external view
    struct OrderDetail {
        uint256 orderId;
        uint256 amount;
        uint256 price;
        uint256 timestamp;
        bool claimed;
    }

    // Round calculation for cross-round purchases
    struct RoundCalculation {
        uint256 roundId;           // Round ID
        uint256 priceUSD;          // Price in USD
        uint256 allocation;        // Round allocation
        uint256 tokensToProcess;   // Tokens to process in this round
        bool exists;               // Whether the round exists or needs to be created
    }

    // Token information
    struct TokenInfo {
        bool enabled;              // Token enabled status
        address priceFeed;         // Price feed address
        address backupFeed;        // Backup price feed address
        uint256 backupPrice;       // Backup price (8 decimals)
        uint256 decimals;          // Token decimals
    }

    // Receiver information
    struct ReceiverInfo {
        address addr;              // Receiver address
        uint256 share;             // Share percentage
    }

    // Sale token
    IERC20Upgradeable public saleToken;
    
    // Price feeds
    AggregatorV3Interface public ethPriceFeed;
    AggregatorV3Interface public ethBackupFeed;
    
    // Configuration
    uint256 public ethBackupPrice;                     // ETH backup price (8 decimals)
    uint256 public ethThreshold;                       // ETH threshold for auto-transfer
    uint256 public gasLimit;                           // Gas limit for transfers
    uint256 public batchLimit;                         // Batch limit for operations
    uint256 public minInvestment;                      // Minimum investment amount
    uint256 public claimPeriod;                        // Claim period
    uint256 public snapshotInterval;                   // Snapshot interval
    uint256 public priceUpdateInterval;                // Price update interval
    uint256 public roundDuration;                      // Round duration
    uint256 public MIN_TOKENS;                         // Minimum token amount
    
    // State variables
    uint256 public lastOrderId;                        // Last order ID
    uint256 public snapshotCount;                      // Snapshot count
    uint256 public totalRounds;                        // Total rounds
    uint256 public currentRound;                       // Current round
    uint256 public maxRound;                           // Maximum round
    uint256 public totalShares;                        // Total shares
    uint256 public pendingETH;                         // Pending ETH
    uint256 public totalDistributed;                   // Total distributed
    uint256 public totalRaisedETH;                     // Total raised ETH
    uint256 public totalRaisedUSD;                     // Total raised USD
    bool public claimEnabled;                          // Claim enabled
    bool public autoIncreaseEnabled;                   // Auto increase enabled
    uint256 public autoIncreaseRate;                   // Auto increase rate
    uint256 public lastRoundCheckTime;                 // Last round check time
    bool public autoCreateEnabled;                     // Auto create enabled
    uint256 public defaultAllocation;                  // Default allocation
    uint256 public defaultPriceUSD;                    // Default price USD
    bool public crossRoundPurchaseEnabled;             // Cross round purchase enabled
    bool public directMintEnabled;                     // Direct mint enabled
    
    // Timestamps
    uint256 public lastSnapshotTime;                   // Last snapshot time
    uint256 public lastPriceUpdateTime;                // Last price update time
    
    // Mappings
    mapping(uint256 => Round) public rounds;           // Round ID => Round
    mapping(uint256 => Order) public orders;           // Order ID => Order
    mapping(address => uint256[]) public userOrders;   // User => Order IDs
    mapping(address => TokenInfo) public tokens;       // Token => Token info
    mapping(uint256 => uint256[]) public snapshots;    // Snapshot ID => Order IDs
    
    // Lists
    address[] public tokenList;                        // List of tokens
    ReceiverInfo[] public receivers;                   // List of receivers
    
    // Vesting
    address public vestingContract;                    // Vesting contract address
    bool public vestingEnabled;                        // Vesting enabled flag
    uint256 public presaleStartTime;                   // Presale start time for dynamic vesting
    
    /**
     * @dev Initialize base contract
     */
    function __RoundPresaleBase_init() internal onlyInitializing {
        __Ownable_init(msg.sender);
        __Pausable_init();
        __ReentrancyGuard_init();
        __AccessControl_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MANAGER_ROLE, msg.sender);
        
        MIN_TOKENS = 1;
        presaleStartTime = block.timestamp; 
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
            claimed: false
        });
        
        userOrders[buyer].push(lastOrderId);
    }
    
    /**
     * @dev Set vesting contract
     * @param _vestingContract Address of the vesting contract
     */
    function setVestingContract(address _vestingContract) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_vestingContract != address(0), "Invalid vesting contract address");
        vestingContract = _vestingContract;
        emit VestingContractSet(_vestingContract);
    }
    
    /**
     * @dev Enable or disable vesting
     * @param _enabled Whether vesting should be enabled
     */
    function enableVesting(bool _enabled) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (_enabled) {
            require(vestingContract != address(0), "Vesting contract not set");
        }
        vestingEnabled = _enabled;
        emit VestingEnabled(_enabled);
    }
    
    /**
     * @dev Set presale start time
     * @param _startTime Presale start time
     */
    function setPresaleStartTime(uint256 _startTime) external onlyRole(DEFAULT_ADMIN_ROLE) {
        presaleStartTime = _startTime;
        emit PresaleStartTimeSet(_startTime);
    }
    
    /**
     * @dev Get ETH price from Chainlink
     * @return ETH price in USD (8 decimals)
     */
    function getETHPrice() public view returns (uint256) {
        try ethPriceFeed.latestRoundData() returns (
            uint80 /* _roundId */,
            int256 answer,
            uint256 /* _startedAt */,
            uint256 /* _updatedAt */,
            uint80 /* _answeredInRound */
        ) {
            if (answer <= 0) {
                return ethBackupPrice;
            }
            return uint256(answer);
        } catch {
            try ethBackupFeed.latestRoundData() returns (
                uint80 /* _roundId */,
                int256 answer,
                uint256 /* _startedAt */,
                uint256 /* _updatedAt */,
                uint80 /* _answeredInRound */
            ) {
                if (answer <= 0) {
                    return ethBackupPrice;
                }
                return uint256(answer);
            } catch {
                return ethBackupPrice;
            }
        }
    }
    
    /**
     * @dev Convert ETH to USD
     * @param amount Amount of ETH
     * @return Amount in USD (18 decimals)
     */
    function ethToUSD(uint256 amount) public view returns (uint256) {
        uint256 ethPrice = getETHPrice();
        return amount * ethPrice / 1e8;
    }
    
    /**
     * @dev Convert token to USD
     * @param token Token address
     * @param amount Amount of tokens
     * @return Amount in USD (18 decimals)
     */
    function tokenToUSD(address token, uint256 amount) public view returns (uint256) {
        TokenInfo storage tokenInfo = tokens[token];
        require(tokenInfo.enabled, "Token not enabled");
        
        uint256 tokenPrice;
        try AggregatorV3Interface(tokenInfo.priceFeed).latestRoundData() returns (
            uint80 /* _roundId */,
            int256 answer,
            uint256 /* _startedAt */,
            uint256 /* _updatedAt */,
            uint80 /* _answeredInRound */
        ) {
            if (answer <= 0) {
                tokenPrice = tokenInfo.backupPrice;
            } else {
                tokenPrice = uint256(answer);
            }
        } catch {
            try AggregatorV3Interface(tokenInfo.backupFeed).latestRoundData() returns (
                uint80 /* _roundId */,
                int256 answer,
                uint256 /* _startedAt */,
                uint256 /* _updatedAt */,
                uint80 /* _answeredInRound */
            ) {
                if (answer <= 0) {
                    tokenPrice = tokenInfo.backupPrice;
                } else {
                    tokenPrice = uint256(answer);
                }
            } catch {
                tokenPrice = tokenInfo.backupPrice;
            }
        }
        
        uint256 decimals = tokenInfo.decimals;
        if (decimals == 0) {
            try IERC20MetadataUpgradeable(token).decimals() returns (uint8 dec) {
                decimals = dec;
            } catch {
                decimals = 18;
            }
        }
        
        return amount * tokenPrice / (10 ** decimals);
    }
}


// File: round-presale-v4 (11)/contracts/RoundPresaleRounds.sol


pragma solidity ^0.8.26;


/**
 * @title RoundPresaleRounds
 * @dev Contract that provides round management functionality
 */
abstract contract RoundPresaleRounds is RoundPresaleBase {
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
        
        uint256 newPriceUSD;
        if (autoIncreaseEnabled && currentRound > 0) {
            // Calculate new round price based on previous round price
            Round storage prevRound = rounds[currentRound];
            if (prevRound.useUSDPrice) {
                newPriceUSD = prevRound.priceUSD * (100 + autoIncreaseRate) / 100;
            } else {
                // Convert ETH price to USD for calculation
                uint256 ethPrice = getETHPrice();
                newPriceUSD = (prevRound.price * ethPrice / 1e8) * (100 + autoIncreaseRate) / 100;
            }
        } else {
            newPriceUSD = defaultPriceUSD;
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
     * @dev Internal function to start round
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
     * @dev Calculate the price for the next round
     */
    function _calculateNextRoundPrice(Round storage currentRoundData) internal view returns (uint256) {
        uint256 newPriceUSD;
        if (autoIncreaseEnabled) {
            if (currentRoundData.useUSDPrice) {
                newPriceUSD = currentRoundData.priceUSD * (100 + autoIncreaseRate) / 100;
            } else {
                uint256 ethPrice = getETHPrice();
                newPriceUSD = (currentRoundData.price * ethPrice / 1e8) * (100 + autoIncreaseRate) / 100;
            }
        } else {
            newPriceUSD = defaultPriceUSD;
        }
        return newPriceUSD;
    }
    
    /**
     * @dev Check prices and update if needed
     */
    function _checkPrices() internal {
        if (block.timestamp < lastPriceUpdateTime + priceUpdateInterval) return;
        lastPriceUpdateTime = block.timestamp;
    }
}


// File: round-presale-v4 (11)/contracts/RoundPresaleFunds.sol


pragma solidity ^0.8.26;


/**
 * @title RoundPresaleFunds
 * @dev Contract that provides fund management functionality
 */
abstract contract RoundPresaleFunds is RoundPresaleBase {
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
                IERC20Upgradeable(token).transfer(receivers[i].addr, share);
            }
            
            totalTransferred += share;
        }
        
        if (totalTransferred < amount && receivers.length > 0) {
            uint256 remaining = amount - totalTransferred;
            
            if (token == address(0)) {
                (bool success, ) = receivers[0].addr.call{value: remaining, gas: gasLimit}("");
                require(success, "ETH transfer failed");
            } else {
                IERC20Upgradeable(token).transfer(receivers[0].addr, remaining);
            }
        }
    }

    /**
     * @dev Get receiver addresses
     */
    function _getReceiverAddresses() internal view returns (address[] memory) {
        address[] memory addresses = new address[](receivers.length);
        for (uint256 i = 0; i < receivers.length; i++) {
            addresses[i] = receivers[i].addr;
        }
        return addresses;
    }

    /**
     * @dev Get receiver shares
     */
    function _getReceiverShares() internal view returns (uint256[] memory) {
        uint256[] memory shares = new uint256[](receivers.length);
        for (uint256 i = 0; i < receivers.length; i++) {
            shares[i] = receivers[i].share;
        }
        return shares;
    }

    /**
     * @dev Internal ETH transfer function
     */
    function _transferETH() internal {
        uint256 amount = address(this).balance;
        require(amount > 0, "No ETH to transfer");
        
        _transferAsset(address(0), amount);
        
        emit FundsTransferred(address(0));
    }

    /**
     * @dev Check and transfer token if threshold is reached
     */
    function _checkAndTransferToken(address token) internal {
        if (token == address(0) || token == address(saleToken)) return;
        
        IERC20Upgradeable tokenContract = IERC20Upgradeable(token);
        uint256 balance = tokenContract.balanceOf(address(this));
        
        // Get token threshold or use minInvestment as default
        uint256 tokenThreshold = tokens[token].backupPrice;
        if (tokenThreshold == 0) {
            tokenThreshold = minInvestment;
        }
        
        if (balance >= tokenThreshold) {
            _transferAsset(token, balance);
            emit FundsTransferred(token);
        }
    }
}


// File: round-presale-v4 (11)/contracts/interfaces/ITokenMintable.sol


pragma solidity ^0.8.26;

interface ITokenMintable {
    function mintAndLock(address to, uint256 amount) external;
    function mintWithVesting(address to, uint256 amount, uint256 timestamp) external;
    function unlockTokens(address account, uint256 amount) external;
    function setVestingContract(address vestingContract) external;
}


// File: round-presale-v4 (11)/contracts/interfaces/ITokenVestable.sol


pragma solidity ^0.8.26;

interface ITokenVestable {
    function lockWithVesting(address account, uint256 amount, uint256 purchaseTime, uint256 presaleStartTime) external;
    function releaseVestedTokens(address account) external returns (uint256);
    function getVestingInfo(address account) external view returns (
        uint256 totalAmount,
        uint256 releasedAmount,
        uint256 releasableAmount,
        uint256 purchaseTime,
        uint256 nextReleaseTime,
        uint256 vestingDuration
    );
    function startVesting() external;
    function addInvestment(address investor, uint256 investmentUSD) external returns (uint256);
}


// File: round-presale-v4 (11)/contracts/libraries/PresaleUtils.sol


pragma solidity ^0.8.26;


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


// File: round-presale-v4 (11)/contracts/libraries/PresalePurchaseLib.sol


pragma solidity ^0.8.26;


/**
* @title PresalePurchaseLib
* @dev Library providing purchase calculation functions
*/
library PresalePurchaseLib {
   // Round structure definition to match RoundPresaleBase.Round
   struct Round {
       uint256 price;             // Token price in Wei
       uint256 priceUSD;          // Token price in USD (18 decimals)
       uint256 allocation;        // Token allocation for the round
       uint256 sold;              // Tokens sold in the round
       uint256 startTime;         // Round start time
       uint256 endTime;           // Round end time
       bool active;               // Round active status
       bool useUSDPrice;          // Use USD price flag
   }

   /**
    * @dev Calculate tokens for next round
    */
   function calculateNextRoundTokens(
       uint256 remainingTokens,
       uint256 investmentUSD,
       uint256 tokensAmount,
       address payToken,
       Round memory nextRound,
       uint256 ethPrice
   ) internal pure returns (uint256) {
       uint256 remainingUSD;
       if (nextRound.useUSDPrice) {
           remainingUSD = (remainingTokens * nextRound.priceUSD) / 1e18;
       } else {
           if (payToken == address(0)) {
               uint256 ethAmount = (remainingTokens * nextRound.price) / 1e18;
               remainingUSD = ethAmount * ethPrice / 1e8;
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
               uint256 ethValue = remainingUSD * 1e18 / ethPrice;
               nextRoundTokens = ethValue * 1e18 / nextRound.price;
           } else {
               nextRoundTokens = remainingTokens;
           }
       }
       
       return nextRoundTokens;
   }

   /**
    * @dev Calculate token amount based on investment
    */
   function calculateTokenAmount(
       bool useUSDPrice,
       uint256 priceUSD,
       uint256 price,
       uint256 investmentUSD,
       uint256 payAmount,
       address payToken,
       uint256 ethPrice,
       uint256 minTokens
   ) internal pure returns (uint256) {
       uint256 tokensAmount;
       
       if (useUSDPrice) {
           // Calculate tokens based on USD price
           uint256 calculatedTokens = (investmentUSD * 1e18) / priceUSD;
           // Ensure minimum token amount
           tokensAmount = calculatedTokens < minTokens ? minTokens : calculatedTokens;
       } else {
           if (payToken == address(0)) {
               // Calculate tokens based on ETH price
               uint256 calculatedTokens = payAmount * 1e18 / price;
               // Ensure minimum token amount
               tokensAmount = calculatedTokens < minTokens ? minTokens : calculatedTokens;
           } else {
               // Calculate tokens based on ETH equivalent
               uint256 ethValue = investmentUSD * 1e18 / ethPrice;
               uint256 calculatedTokens = ethValue * 1e18 / price;
               // Ensure minimum token amount
               tokensAmount = calculatedTokens < minTokens ? minTokens : calculatedTokens;
           }
       }
       
       return tokensAmount;
   }
}


// File: round-presale-v4 (11)/contracts/RoundPresalePurchase.sol


pragma solidity ^0.8.26;






/**
* @title RoundPresalePurchase
* @dev Contract that provides purchase functionality
*/
abstract contract RoundPresalePurchase is RoundPresaleRounds, RoundPresaleFunds {
    /**
     * @dev Calculate tokens for next round
     */
    function _calculateNextRoundTokens(
        uint256 remainingTokens,
        uint256 investmentUSD,
        uint256 tokensAmount,
        address payToken,
        Round memory nextRound
    ) internal view returns (uint256) {
        // Manually convert to PresalePurchaseLib.Round struct
        PresalePurchaseLib.Round memory libRound = PresalePurchaseLib.Round({
            price: nextRound.price,
            priceUSD: nextRound.priceUSD,
            allocation: nextRound.allocation,
            sold: nextRound.sold,
            startTime: nextRound.startTime,
            endTime: nextRound.endTime,
            active: nextRound.active,
            useUSDPrice: nextRound.useUSDPrice
        });
        
        return PresalePurchaseLib.calculateNextRoundTokens(
            remainingTokens,
            investmentUSD,
            tokensAmount,
            payToken,
            libRound,
            getETHPrice()
        );
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
                if (vestingEnabled && vestingContract != address(0)) {
                    // Mint with vesting - pass presale start time
                    ITokenMintable(address(saleToken)).mintWithVesting(buyer, remainingAllocation, block.timestamp);
                    ITokenVestable(vestingContract).lockWithVesting(buyer, remainingAllocation, block.timestamp, presaleStartTime);
                } else {
                    // Standard mint and lock
                    ITokenMintable(address(saleToken)).mintAndLock(buyer, remainingAllocation);
                }
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
            
            emit TokenPurchased(buyer, lastOrderId, actualTokens);
        }
        
        return actualTokens;
    }

    /**
     * @dev Calculate the required rounds and tokens to process in each round
     */
    function _calculateRequiredRounds(
        uint256 remainingTokens,
        uint256 investmentUSD,
        uint256 tokensAmount,
        address payToken
    ) internal view returns (RoundCalculation[] memory) {
        // Calculate up to 30 rounds (gas limit consideration)
        RoundCalculation[] memory calculations = new RoundCalculation[](30);
        uint256 calculationCount = 0;
        
        uint256 tokensLeft = remainingTokens;
        uint256 nextRoundId = currentRound + 1;
        
        // Check existing rounds
        while (tokensLeft > 0 && calculationCount < 30 && nextRoundId <= totalRounds) {
            Round storage nextRound = rounds[nextRoundId];
            
            // Copy from storage to memory
            Round memory nextRoundMemory = Round({
                price: nextRound.price,
                priceUSD: nextRound.priceUSD,
                allocation: nextRound.allocation,
                sold: nextRound.sold,
                startTime: nextRound.startTime,
                endTime: nextRound.endTime,
                active: nextRound.active,
                useUSDPrice: nextRound.useUSDPrice
            });
            
            uint256 nextRoundTokens = _calculateNextRoundTokens(
                tokensLeft,
                investmentUSD,
                tokensAmount,
                payToken,
                nextRoundMemory
            );
            
            uint256 nextRoundAllocation = nextRound.allocation - nextRound.sold;
            uint256 actualTokens = nextRoundTokens <= nextRoundAllocation ? nextRoundTokens : nextRoundAllocation;
            
            if (actualTokens > 0) {
                calculations[calculationCount] = RoundCalculation({
                    roundId: nextRoundId,
                    priceUSD: nextRound.useUSDPrice ? nextRound.priceUSD : 0,
                    allocation: nextRoundAllocation,
                    tokensToProcess: actualTokens,
                    exists: true
                });
                
                tokensLeft -= actualTokens;
                calculationCount++;
            }
            
            if (actualTokens < nextRoundTokens) {
                // Cannot process all tokens in this round
                break;
            }
            
            nextRoundId++;
        }
        
        // Calculate rounds that need to be auto-created
        if (tokensLeft > 0 && calculationCount < 30 && autoCreateEnabled && defaultAllocation > 0 && defaultPriceUSD > 0) {
            uint256 currentPriceUSD;
            bool useCurrentRound = false;
            
            // Get price from current round or last calculated round
            if (calculationCount > 0) {
                currentPriceUSD = calculations[calculationCount - 1].priceUSD;
                useCurrentRound = true;
            } else if (currentRound > 0) {
                Round storage currentRoundData = rounds[currentRound];
                if (currentRoundData.useUSDPrice) {
                    currentPriceUSD = currentRoundData.priceUSD;
                } else {
                    uint256 ethPrice = getETHPrice();
                    currentPriceUSD = currentRoundData.price * ethPrice / 1e8;
                }
                useCurrentRound = true;
            } else {
                currentPriceUSD = defaultPriceUSD;
            }
            
            while (tokensLeft > 0 && calculationCount < 30) {
                uint256 newPriceUSD;
                if (useCurrentRound && autoIncreaseEnabled) {
                    newPriceUSD = currentPriceUSD * (100 + autoIncreaseRate) / 100;
                } else {
                    newPriceUSD = defaultPriceUSD;
                }
                
                // Create temporary round structure
                Round memory tempRound = Round({
                    price: 0,
                    priceUSD: newPriceUSD,
                    allocation: defaultAllocation,
                    sold: 0,
                    startTime: 0,
                    endTime: 0,
                    active: false,
                    useUSDPrice: true
                });
                
                // Calculate tokens to process in this round
                uint256 nextRoundTokens = _calculateNextRoundTokens(
                    tokensLeft,
                    investmentUSD,
                    tokensAmount,
                    payToken,
                    tempRound
                );
                
                uint256 actualTokens = nextRoundTokens <= defaultAllocation ? nextRoundTokens : defaultAllocation;
                
                if (actualTokens > 0) {
                    calculations[calculationCount] = RoundCalculation({
                        roundId: nextRoundId,
                        priceUSD: newPriceUSD,
                        allocation: defaultAllocation,
                        tokensToProcess: actualTokens,
                        exists: false
                    });
                    
                    tokensLeft -= actualTokens;
                    calculationCount++;
                    currentPriceUSD = newPriceUSD;
                    useCurrentRound = true;
                } else {
                    break;
                }
                
                nextRoundId++;
            }
        }
        
        // Resize array to actual calculation count
        RoundCalculation[] memory result = new RoundCalculation[](calculationCount);
        for (uint256 i = 0; i < calculationCount; i++) {
            result[i] = calculations[i];
        }
        
        return result;
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
        
        // Exit if no tokens remain
        if (remainingTokens == 0) {
            return;
        }
        
        // Calculate required rounds
        RoundCalculation[] memory requiredRounds = _calculateRequiredRounds(
            remainingTokens,
            investmentUSD,
            tokensAmount,
            payToken
        );
        
        // Exit if no rounds calculated
        if (requiredRounds.length == 0) {
            emit CrossRoundPurchase(buyer, 0);
            return;
        }
        
        // Total tokens to mint
        uint256 totalTokensToMint = 0;
        
        // Process each round
        for (uint256 i = 0; i < requiredRounds.length; i++) {
            RoundCalculation memory calc = requiredRounds[i];
            
            // Start or create round
            if (calc.exists) {
                _startRound(calc.roundId);
            } else {
                // Create new round
                totalRounds++;
                uint256 roundId = totalRounds;
                
                rounds[roundId] = Round({
                    price: 0,
                    priceUSD: calc.priceUSD,
                    allocation: calc.allocation,
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
            
            emit CrossRoundPurchase(buyer, currentRound);
            
            // Create order and update round state
            Round storage currentRoundData = rounds[currentRound];
            _createOrder(buyer, calc.tokensToProcess, currentRoundData.useUSDPrice ? currentRoundData.priceUSD : currentRoundData.price);
            currentRoundData.sold += calc.tokensToProcess;
            
            // Accumulate token amount
            totalTokensToMint += calc.tokensToProcess;
            
            emit TokenPurchased(buyer, lastOrderId, calc.tokensToProcess);
            
            // Check if round is complete
            if (currentRoundData.sold >= currentRoundData.allocation) {
                _endRound();
                emit RoundEnded(currentRound, currentRoundData.sold);
                
                // Continue to next round if not the last one
                if (i < requiredRounds.length - 1) {
                    continue;
                }
                
                // Create next round if this is the last round and auto-create is enabled
                if (autoCreateEnabled && (maxRound == 0 || currentRound < maxRound)) {
                    lastRoundCheckTime = block.timestamp;
                    _createAndStartNextRound();
                }
            }
        }
        
        // Mint all tokens at once
        if (directMintEnabled && totalTokensToMint > 0) {
            if (vestingEnabled && vestingContract != address(0)) {
                // Mint with vesting - pass presale start time
                ITokenMintable(address(saleToken)).mintWithVesting(buyer, totalTokensToMint, block.timestamp);
                ITokenVestable(vestingContract).lockWithVesting(buyer, totalTokensToMint, block.timestamp, presaleStartTime);
            } else {
                // Standard mint and lock
                ITokenMintable(address(saleToken)).mintAndLock(buyer, totalTokensToMint);
            }
            totalDistributed += totalTokensToMint;
            emit TokenMinted(buyer, totalTokensToMint);
        }
    }

    /**
     * @dev Internal buy function
     */
    function _buy(address buyer, uint256 amount, address payToken) internal {
        require(buyer != address(0), "Invalid address");
        require(amount > 0, "Invalid amount");
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
        
        require(investmentUSD >= minInvestment, "Too small");
        totalRaisedUSD = totalRaisedUSD + investmentUSD;
        
        // Track investment amount and update level (if vesting contract is set)
        if (vestingEnabled && vestingContract != address(0)) {
            (bool success,) = vestingContract.call(
                abi.encodeWithSignature("addInvestment(address,uint256)", buyer, investmentUSD)
            );
        }
        
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
        uint256 tokensAmount = PresalePurchaseLib.calculateTokenAmount(
            round.useUSDPrice,
            round.priceUSD,
            round.price,
            investmentUSD,
            payAmount,
            payToken,
            getETHPrice(),
            MIN_TOKENS
        );
        
        uint256 remainingAllocation = round.allocation - round.sold;
        
        if (tokensAmount <= remainingAllocation) {
            // Standard case: purchase fits in current round
            _createOrder(buyer, tokensAmount, round.useUSDPrice ? round.priceUSD : round.price);
            round.sold = round.sold + tokensAmount;
            
            // If direct mint is enabled, mint and lock tokens immediately
            if (directMintEnabled) {
                if (vestingEnabled && vestingContract != address(0)) {
                    // Mint with vesting - pass presale start time
                    ITokenMintable(address(saleToken)).mintWithVesting(buyer, tokensAmount, block.timestamp);
                    ITokenVestable(vestingContract).lockWithVesting(buyer, tokensAmount, block.timestamp, presaleStartTime);
                } else {
                    // Standard mint and lock
                    ITokenMintable(address(saleToken)).mintAndLock(buyer, tokensAmount);
                }
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
                if (vestingEnabled && vestingContract != address(0)) {
                    // Mint with vesting - pass presale start time
                    ITokenMintable(address(saleToken)).mintWithVesting(buyer, remainingAllocation, block.timestamp);
                    ITokenVestable(vestingContract).lockWithVesting(buyer, remainingAllocation, block.timestamp, presaleStartTime);
                } else {
                    // Standard mint and lock
                    ITokenMintable(address(saleToken)).mintAndLock(buyer, remainingAllocation);
                }
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
}


// File: round-presale-v4 (11)/contracts/RoundPresaleClaim.sol


pragma solidity ^0.8.26;




/**
* @title RoundPresaleClaim
* @dev Contract that provides claim functionality
*/
abstract contract RoundPresaleClaim is RoundPresaleBase {
    /**
     * @dev Initialize claim contract
     */
    function __RoundPresaleClaim_init() internal onlyInitializing {
        // Initialization logic (ReentrancyGuard initialization removed)
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
            if (vestingEnabled && vestingContract != address(0)) {
                // In vesting mode, release vested tokens
                try ITokenVestable(vestingContract).releaseVestedTokens(order.buyer) returns (uint256 releasable) {
                    if (releasable > 0) {
                        emit TokenUnlocked(order.buyer, releasable);
                    }
                } catch {
                    // If vesting hasn't started yet, just mark as claimed but don't release tokens
                }
            } else {
                // In direct mint mode, unlock tokens
                ITokenMintable(address(saleToken)).unlockTokens(order.buyer, amount);
                emit TokenUnlocked(order.buyer, amount);
            }
        } else {
            // Original method: transfer tokens
            require(saleToken.transfer(order.buyer, amount), "Token transfer failed");
        }
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
        uint256 claimedCount = 0;
        
        for (uint256 i = 0; i < limit; i++) {
            uint256 orderId = buyerOrders[i];
            Order storage order = orders[orderId];
            
            if (!order.claimed && block.timestamp <= order.timestamp + claimPeriod) {
                order.claimed = true;
                totalAmount += order.amount;
                claimedCount++;
                
                emit TokenClaimed(buyer, orderId);
            }
        }
        
        if (totalAmount > 0) {
            if (!directMintEnabled) {
                totalDistributed += totalAmount;
            }
            
            if (directMintEnabled) {
                if (vestingEnabled && vestingContract != address(0)) {
                    // In vesting mode, release vested tokens
                    try ITokenVestable(vestingContract).releaseVestedTokens(buyer) returns (uint256 releasable) {
                        if (releasable > 0) {
                            emit TokenUnlocked(buyer, releasable);
                        }
                    } catch {
                        // If vesting hasn't started yet, just mark as claimed but don't release tokens
                    }
                } else {
                    // In direct mint mode, unlock tokens
                    ITokenMintable(address(saleToken)).unlockTokens(buyer, totalAmount);
                    emit TokenUnlocked(buyer, totalAmount);
                }
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
            if (vestingEnabled && vestingContract != address(0)) {
                // Force release all tokens in vesting contract
                ITokenMintable(address(saleToken)).unlockTokens(buyer, totalAmount);
            } else {
                ITokenMintable(address(saleToken)).unlockTokens(buyer, totalAmount);
            }
            emit EmergencyUnlock(buyer, totalAmount);
        }
    }
    
    /**
     * @dev Get vesting information for a buyer
     * @param buyer Address of the buyer
     * @return totalAmount Total amount of tokens being vested
     * @return releasedAmount Amount of tokens already released
     * @return releasableAmount Amount of tokens that can be released now
     * @return purchaseTime Time when tokens were purchased
     * @return nextReleaseTime Time of next token release
     * @return vestingDuration Custom vesting duration for this account
     */
    function getVestingInfo(address buyer) external view returns (
        uint256 totalAmount,
        uint256 releasedAmount,
        uint256 releasableAmount,
        uint256 purchaseTime,
        uint256 nextReleaseTime,
        uint256 vestingDuration
    ) {
        require(vestingEnabled && vestingContract != address(0), "Vesting not enabled");
        return ITokenVestable(vestingContract).getVestingInfo(buyer);
    }
}


// File: round-presale-v4 (11)/contracts/RoundPresaleAdmin.sol


pragma solidity ^0.8.26;


/**
 * @title RoundPresaleAdmin
 * @dev Contract that provides admin functionality
 */
abstract contract RoundPresaleAdmin is RoundPresaleBase {
    /**
     * @dev Set min investment
     * @param amount Minimum investment amount
     */
    function setMin(uint256 amount) external onlyRole(DEFAULT_ADMIN_ROLE) {
        minInvestment = amount;
        emit ConfigUpdated("min", amount);
    }

    /**
     * @dev Set ETH threshold
     * @param threshold ETH threshold for auto-transfer
     */
    function setETHThreshold(uint256 threshold) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(threshold > 0, "Threshold must be greater than 0");
        ethThreshold = threshold;
        emit ConfigUpdated("threshold", threshold);
    }

    /**
     * @dev Set max round
     * @param max Maximum round number
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
     * @param allocation Default allocation for auto-created rounds
     */
    function setAlloc(uint256 allocation) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(allocation > 0, "Allocation must be greater than 0");
        defaultAllocation = allocation;
        emit ConfigUpdated("allocation", allocation);
    }

    /**
     * @dev Set default USD price
     * @param priceUSD Default USD price for auto-created rounds
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
     * @param newToken New sale token address
     */
    function changeToken(address newToken) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(newToken != address(0), "Invalid token address");
        saleToken = IERC20Upgradeable(newToken);
    }

    /**
     * @dev Enable or disable direct mint
     * @param enabled Whether direct mint should be enabled
     */
    function enableDirectMint(bool enabled) external onlyRole(DEFAULT_ADMIN_ROLE) {
        directMintEnabled = enabled;
        emit ConfigUpdated("directMint", enabled ? 1 : 0);
    }

    /**
     * @dev Enable or disable claim
     * @param enabled Whether claim should be enabled
     */
    function enableClaim(bool enabled) external onlyRole(DEFAULT_ADMIN_ROLE) {
        claimEnabled = enabled;
        emit ConfigUpdated("claim", enabled ? 1 : 0);
    }

    /**
     * @dev UUPS upgrade authorization
     * @param newImplementation Address of the new implementation
     */
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}


// File: round-presale-v4 (11)/contracts/RoundPresaleV4.sol


pragma solidity ^0.8.26;




/**
* @title RoundPresaleV4
* @dev Main contract for round-based token presale with vesting support
*/
contract RoundPresaleV4 is RoundPresalePurchase, RoundPresaleClaim, RoundPresaleAdmin {
   /**
    * @dev Initialize the contract
    */
   function init(
       address _saleToken,
       address _ethPriceFeed
   ) public initializer {
       __RoundPresaleBase_init();
       __RoundPresaleClaim_init();
       
       saleToken = IERC20Upgradeable(_saleToken);
       ethPriceFeed = AggregatorV3Interface(_ethPriceFeed);
       
       ethBackupPrice = 2000 * 10**8;
       ethThreshold = 1 ether;
       gasLimit = 300000;
       batchLimit = 100;
       minInvestment = 50 * 10**18;
       claimPeriod = 30 days;
       snapshotInterval = 1 days;
       priceUpdateInterval = 1 hours;
       roundDuration = 7 days;
       autoIncreaseRate = 10;
       lastRoundCheckTime = block.timestamp;
       presaleStartTime = block.timestamp; 
   }
   
   /**
    * @dev Initialize V4 variables
    * @param _vestingContract Address of the vesting contract
    */
   function initV4(
       address _vestingContract
   ) external onlyRole(DEFAULT_ADMIN_ROLE) {
       if (_vestingContract != address(0)) {
           vestingContract = _vestingContract;
           emit VestingContractSet(_vestingContract);
       }
   }
   
   /**
    * @dev Buy tokens with ETH
    */
   function buy() external payable nonReentrant {
       _buy(msg.sender, msg.value, address(0));
   }
   
   /**
    * @dev Buy tokens with ERC20 token
    * @param token Address of the token to pay with
    * @param amount Amount of tokens to pay
    */
   function buyWithToken(address token, uint256 amount) external nonReentrant {
       require(tokens[token].enabled, "Token not enabled");
       require(IERC20Upgradeable(token).transferFrom(msg.sender, address(this), amount), "Transfer failed");
       
       _buy(msg.sender, amount, token);
   }
}