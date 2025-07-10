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

// File: @openzeppelin/contracts/security/ReentrancyGuard.sol


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
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor() {
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

// File: https://github.com/limitbreakinc/creator-token-contracts/blob/main/contracts/interfaces/IEOARegistry.sol


pragma solidity ^0.8.4;


interface IEOARegistry is IERC165 {
    function isVerifiedEOA(address account) external view returns (bool);
}
// File: https://github.com/limitbreakinc/creator-token-contracts/blob/main/contracts/utils/TransferPolicy.sol


pragma solidity ^0.8.4;

enum AllowlistTypes {
    Operators,
    PermittedContractReceivers
}

enum ReceiverConstraints {
    None,
    NoCode,
    EOA
}

enum CallerConstraints {
    None,
    OperatorWhitelistEnableOTC,
    OperatorWhitelistDisableOTC
}

enum StakerConstraints {
    None,
    CallerIsTxOrigin,
    EOA
}

enum TransferSecurityLevels {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six
}

struct TransferSecurityPolicy {
    CallerConstraints callerConstraints;
    ReceiverConstraints receiverConstraints;
}

struct CollectionSecurityPolicy {
    TransferSecurityLevels transferSecurityLevel;
    uint120 operatorWhitelistId;
    uint120 permittedContractReceiversId;
}

// File: https://github.com/limitbreakinc/creator-token-contracts/blob/main/contracts/interfaces/ITransferSecurityRegistry.sol


pragma solidity ^0.8.4;


interface ITransferSecurityRegistry {
    event AddedToAllowlist(AllowlistTypes indexed kind, uint256 indexed id, address indexed account);
    event CreatedAllowlist(AllowlistTypes indexed kind, uint256 indexed id, string indexed name);
    event ReassignedAllowlistOwnership(AllowlistTypes indexed kind, uint256 indexed id, address indexed newOwner);
    event RemovedFromAllowlist(AllowlistTypes indexed kind, uint256 indexed id, address indexed account);
    event SetAllowlist(AllowlistTypes indexed kind, address indexed collection, uint120 indexed id);
    event SetTransferSecurityLevel(address indexed collection, TransferSecurityLevels level);

    function createOperatorWhitelist(string calldata name) external returns (uint120);
    function createPermittedContractReceiverAllowlist(string calldata name) external returns (uint120);
    function reassignOwnershipOfOperatorWhitelist(uint120 id, address newOwner) external;
    function reassignOwnershipOfPermittedContractReceiverAllowlist(uint120 id, address newOwner) external;
    function renounceOwnershipOfOperatorWhitelist(uint120 id) external;
    function renounceOwnershipOfPermittedContractReceiverAllowlist(uint120 id) external;
    function setTransferSecurityLevelOfCollection(address collection, TransferSecurityLevels level) external;
    function setOperatorWhitelistOfCollection(address collection, uint120 id) external;
    function setPermittedContractReceiverAllowlistOfCollection(address collection, uint120 id) external;
    function addOperatorToWhitelist(uint120 id, address operator) external;
    function addPermittedContractReceiverToAllowlist(uint120 id, address receiver) external;
    function removeOperatorFromWhitelist(uint120 id, address operator) external;
    function removePermittedContractReceiverFromAllowlist(uint120 id, address receiver) external;
    function getCollectionSecurityPolicy(address collection) external view returns (CollectionSecurityPolicy memory);
    function getWhitelistedOperators(uint120 id) external view returns (address[] memory);
    function getPermittedContractReceivers(uint120 id) external view returns (address[] memory);
    function isOperatorWhitelisted(uint120 id, address operator) external view returns (bool);
    function isContractReceiverPermitted(uint120 id, address receiver) external view returns (bool);
}
// File: https://github.com/limitbreakinc/creator-token-contracts/blob/main/contracts/interfaces/ITransferValidator.sol


pragma solidity ^0.8.4;


interface ITransferValidator {
    function applyCollectionTransferPolicy(address caller, address from, address to) external view;
}
// File: https://github.com/limitbreakinc/creator-token-contracts/blob/main/contracts/interfaces/ICreatorTokenTransferValidator.sol


pragma solidity ^0.8.4;




interface ICreatorTokenTransferValidator is ITransferSecurityRegistry, ITransferValidator, IEOARegistry {}
// File: airdropcontract.sol


pragma solidity ^0.8.25;





interface IERC721 {
    function safeTransferFrom(address from, address to, uint256 tokenId) external;
}

contract ElmonX_Bulk_AirDrop is Ownable, ReentrancyGuard, ICreatorTokenTransferValidator {
    bytes32 private constant MODULE_TYPE = bytes32("AirdropERC721");
    uint256 private constant VERSION = 1.0;

    // Storage for transfer security policies
    mapping(address => CollectionSecurityPolicy) private collectionPolicies;
    mapping(uint120 => address[]) private operatorWhitelists;
    mapping(uint120 => address[]) private permittedContractReceiverAllowlists;
    mapping(uint120 => address) private operatorWhitelistOwners;
    mapping(uint120 => address) private permittedContractReceiverAllowlistOwners;
    uint120 private nextOperatorWhitelistId = 1;
    uint120 private nextPermittedContractReceiverAllowlistId = 1;

    constructor() Ownable(msg.sender) {}

    // Metadata functions (assuming part of IEOARegistry via ICreatorTokenTransferValidator)
    function contractType() external pure returns (bytes32) {
        return MODULE_TYPE;
    }

    function contractVersion() external pure returns (uint8) {
        return uint8(VERSION);
    }

    // IEOARegistry function required by ICreatorTokenTransferValidator
    function isVerifiedEOA(address) external pure returns (bool) {
        return true; // Adjust as needed (e.g., add EOA verification logic)
    }

    // Airdrop function
    function airdrop(
        address _tokenAddress,
        address _tokenOwner,
        address[] memory _recipients,
        uint256[] memory _tokenIds
    ) external nonReentrant onlyOwner {
        uint256 len = _tokenIds.length;
        require(len == _recipients.length, "length mismatch");

        IERC721 token = IERC721(_tokenAddress);

        for (uint256 i = 0; i < len; i++) {
            token.safeTransferFrom(_tokenOwner, _recipients[i], _tokenIds[i]);
        }
    }

    // IERC165 implementation
    function supportsInterface(bytes4 interfaceId) external pure override returns (bool) {
        return interfaceId == type(IERC165).interfaceId ||
               interfaceId == type(ICreatorTokenTransferValidator).interfaceId;
    }

    // ICreatorTokenTransferValidator implementations
    function applyCollectionTransferPolicy(address, address, address) external pure override {
        return; // Allows all transfers by default
    }

    function setTransferSecurityLevelOfCollection(address collection, TransferSecurityLevels level) external override onlyOwner {
        collectionPolicies[collection].transferSecurityLevel = level;
    }

    function setOperatorWhitelistOfCollection(address collection, uint120 operatorWhitelistId) external override onlyOwner {
        collectionPolicies[collection].operatorWhitelistId = operatorWhitelistId;
    }

    function setPermittedContractReceiverAllowlistOfCollection(address collection, uint120 permittedContractReceiversId) external override onlyOwner {
        collectionPolicies[collection].permittedContractReceiversId = permittedContractReceiversId;
    }

    function getCollectionSecurityPolicy(address collection) external view override returns (CollectionSecurityPolicy memory) {
        CollectionSecurityPolicy memory policy = collectionPolicies[collection];
        if (policy.transferSecurityLevel == TransferSecurityLevels.Zero && policy.operatorWhitelistId == 0 && policy.permittedContractReceiversId == 0) {
            return CollectionSecurityPolicy({
                transferSecurityLevel: TransferSecurityLevels.Zero,
                operatorWhitelistId: 0,
                permittedContractReceiversId: 0
            });
        }
        return policy;
    }

    function getWhitelistedOperators(uint120 operatorWhitelistId) external view override returns (address[] memory) {
        return operatorWhitelists[operatorWhitelistId];
    }

    function getPermittedContractReceivers(uint120 permittedContractReceiversId) external view override returns (address[] memory) {
        return permittedContractReceiverAllowlists[permittedContractReceiversId];
    }

    function isOperatorWhitelisted(uint120 operatorWhitelistId, address operator) external view override returns (bool) {
        address[] memory whitelist = operatorWhitelists[operatorWhitelistId];
        for (uint256 i = 0; i < whitelist.length; i++) {
            if (whitelist[i] == operator) {
                return true;
            }
        }
        return false;
    }

    function isContractReceiverPermitted(uint120 permittedContractReceiversId, address receiver) external view override returns (bool) {
        address[] memory allowlist = permittedContractReceiverAllowlists[permittedContractReceiversId];
        for (uint256 i = 0; i < allowlist.length; i++) {
            if (allowlist[i] == receiver) {
                return true;
            }
        }
        return false;
    }

    // ITransferSecurityRegistry implementations
    function createOperatorWhitelist(string calldata name) external override onlyOwner returns (uint120) {
        uint120 id = nextOperatorWhitelistId++;
        operatorWhitelistOwners[id] = msg.sender;
        emit OperatorWhitelistCreated(id, name, msg.sender);
        return id;
    }

    function createPermittedContractReceiverAllowlist(string calldata name) external override onlyOwner returns (uint120) {
        uint120 id = nextPermittedContractReceiverAllowlistId++;
        permittedContractReceiverAllowlistOwners[id] = msg.sender;
        emit PermittedContractReceiverAllowlistCreated(id, name, msg.sender);
        return id;
    }

    function addPermittedContractReceiverToAllowlist(uint120 id, address receiver) external override {
        require(permittedContractReceiverAllowlistOwners[id] == msg.sender, "Not the owner");
        permittedContractReceiverAllowlists[id].push(receiver);
    }

    function removeOperatorFromWhitelist(uint120 id, address operator) external override {
        require(operatorWhitelistOwners[id] == msg.sender, "Not the owner");
        address[] storage whitelist = operatorWhitelists[id];
        for (uint256 i = 0; i < whitelist.length; i++) {
            if (whitelist[i] == operator) {
                whitelist[i] = whitelist[whitelist.length - 1];
                whitelist.pop();
                break;
            }
        }
    }

    function removePermittedContractReceiverFromAllowlist(uint120 id, address receiver) external override {
        require(permittedContractReceiverAllowlistOwners[id] == msg.sender, "Not the owner");
        address[] storage allowlist = permittedContractReceiverAllowlists[id];
        for (uint256 i = 0; i < allowlist.length; i++) {
            if (allowlist[i] == receiver) {
                allowlist[i] = allowlist[allowlist.length - 1];
                allowlist.pop();
                break;
            }
        }
    }

    function reassignOwnershipOfOperatorWhitelist(uint120 id, address newOwner) external override {
        require(operatorWhitelistOwners[id] == msg.sender, "Not the owner");
        operatorWhitelistOwners[id] = newOwner;
    }

    function reassignOwnershipOfPermittedContractReceiverAllowlist(uint120 id, address newOwner) external override {
        require(permittedContractReceiverAllowlistOwners[id] == msg.sender, "Not the owner");
        permittedContractReceiverAllowlistOwners[id] = newOwner;
    }

    function renounceOwnershipOfOperatorWhitelist(uint120 id) external override {
        require(operatorWhitelistOwners[id] == msg.sender, "Not the owner");
        operatorWhitelistOwners[id] = address(0);
    }

    function renounceOwnershipOfPermittedContractReceiverAllowlist(uint120 id) external override {
        require(permittedContractReceiverAllowlistOwners[id] == msg.sender, "Not the owner");
        permittedContractReceiverAllowlistOwners[id] = address(0);
    }

    // Optional: Helper functions to manage whitelists (restricted to owner)
    function addOperatorToWhitelist(uint120 operatorWhitelistId, address operator) external onlyOwner {
        operatorWhitelists[operatorWhitelistId].push(operator);
    }

    function addPermittedContractReceiver(uint120 permittedContractReceiversId, address receiver) external onlyOwner {
        permittedContractReceiverAllowlists[permittedContractReceiversId].push(receiver);
    }

    // Events
    event OperatorWhitelistCreated(uint120 indexed id, string name, address owner);
    event PermittedContractReceiverAllowlistCreated(uint120 indexed id, string name, address owner);
}