// SPDX-License-Identifier: MIT

//File： @openzeppelin/contracts/utils/Errors.sol

// OpenZeppelin Contracts (last updated v5.1.0) (utils/Errors.sol)
pragma solidity 0.8.28;

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

//File： @openzeppelin/contracts/utils/Address.sol

// OpenZeppelin Contracts (last updated v5.2.0) (utils/Address.sol)

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

        (bool success, bytes memory returndata) = recipient.call{value: amount}(
            ""
        );
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
    function functionCall(
        address target,
        bytes memory data
    ) internal returns (bytes memory) {
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
    function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value
    ) internal returns (bytes memory) {
        if (address(this).balance < value) {
            revert Errors.InsufficientBalance(address(this).balance, value);
        }
        (bool success, bytes memory returndata) = target.call{value: value}(
            data
        );
        return verifyCallResultFromTarget(target, success, returndata);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a static call.
     */
    function functionStaticCall(
        address target,
        bytes memory data
    ) internal view returns (bytes memory) {
        (bool success, bytes memory returndata) = target.staticcall(data);
        return verifyCallResultFromTarget(target, success, returndata);
    }

    /**
     * @dev Same as {xref-Address-functionCall-address-bytes-}[`functionCall`],
     * but performing a delegate call.
     */
    function functionDelegateCall(
        address target,
        bytes memory data
    ) internal returns (bytes memory) {
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
    function verifyCallResult(
        bool success,
        bytes memory returndata
    ) internal pure returns (bytes memory) {
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

//File： @openzeppelin/contracts/utils/ReentrancyGuard.sol

// OpenZeppelin Contracts (last updated v5.1.0) (utils/ReentrancyGuard.sol)

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

//File： @chainlink/contracts/src/v0.8/automation/interfaces/AutomationCompatibleInterface.sol

// solhint-disable-next-line interface-starts-with-i
interface AutomationCompatibleInterface {
    /**
     * @notice method that is simulated by the keepers to see if any work actually
     * needs to be performed. This method does does not actually need to be
     * executable, and since it is only ever simulated it can consume lots of gas.
     * @dev To ensure that it is never called, you may want to add the
     * cannotExecute modifier from KeeperBase to your implementation of this
     * method.
     * @param checkData specified in the upkeep registration so it is always the
     * same for a registered upkeep. This can easily be broken down into specific
     * arguments using `abi.decode`, so multiple upkeeps can be registered on the
     * same contract and easily differentiated by the contract.
     * @return upkeepNeeded boolean to indicate whether the keeper should call
     * performUpkeep or not.
     * @return performData bytes that the keeper should call performUpkeep with, if
     * upkeep is needed. If you would like to encode data to decode later, try
     * `abi.encode`.
     */
    function checkUpkeep(
        bytes calldata checkData
    ) external returns (bool upkeepNeeded, bytes memory performData);

    /**
     * @notice method that is actually executed by the keepers, via the registry.
     * The data returned by the checkUpkeep simulation will be passed into
     * this method to actually be executed.
     * @dev The input to this method should not be trusted, and the caller of the
     * method should not even be restricted to any single registry. Anyone should
     * be able call it, and the input should be validated, there is no guarantee
     * that the data passed in is the performData returned from checkUpkeep. This
     * could happen due to malicious keepers, racing keepers, or simply a state
     * change while the performUpkeep transaction is waiting for confirmation.
     * Always validate the data passed in.
     * @param performData is the data which was passed back from the checkData
     * simulation. If it is encoded, it can easily be decoded into other types by
     * calling `abi.decode`. This data should not be trusted, and should be
     * validated against the contract's current state.
     */
    function performUpkeep(bytes calldata performData) external;
}

//File： @chainlink/contracts/src/v0.8/automation/AutomationBase.sol

contract AutomationBase {
    error OnlySimulatedBackend();

    /**
     * @notice method that allows it to be simulated via eth_call by checking that
     * the sender is the zero address.
     */
    function _preventExecution() internal view {
        // solhint-disable-next-line avoid-tx-origin
        if (
            tx.origin != address(0) &&
            tx.origin != address(0x1111111111111111111111111111111111111111)
        ) {
            revert OnlySimulatedBackend();
        }
    }

    /**
     * @notice modifier that allows it to be simulated via eth_call by checking
     * that the sender is the zero address.
     */
    modifier cannotExecute() {
        _preventExecution();
        _;
    }
}

//File： @chainlink/contracts/src/v0.8/automation/KeeperCompatible.sol

/**
 * @notice This is a deprecated interface. Please use AutomationCompatible directly.
 */
// solhint-disable-next-line no-unused-import

// solhint-disable-next-line no-unused-import

// solhint-disable-next-line no-unused-import

//File： @chainlink/contracts/src/v0.8/shared/interfaces/AggregatorV3Interface.sol

// solhint-disable-next-line interface-starts-with-i
interface AggregatorV3Interface {
    function decimals() external view returns (uint8);

    function description() external view returns (string memory);

    function version() external view returns (uint256);

    function getRoundData(
        uint80 _roundId
    )
        external
        view
        returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );

    function latestRoundData()
        external
        view
        returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );
}

//File： @openzeppelin/contracts/utils/Context.sol

// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

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

//File： @openzeppelin/contracts/access/Ownable.sol

// OpenZeppelin Contracts (last updated v5.0.0) (access/Ownable.sol)

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

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

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

//File： @openzeppelin/contracts/proxy/utils/Initializable.sol

// OpenZeppelin Contracts (last updated v5.0.0) (proxy/utils/Initializable.sol)

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
    bytes32 private constant INITIALIZABLE_STORAGE =
        0xf0c57e16840df040f15088dc2f81fe391c3923bec73e23a9662efc9c229c6a00;

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
    function _getInitializableStorage()
        private
        pure
        returns (InitializableStorage storage $)
    {
        assembly {
            $.slot := INITIALIZABLE_STORAGE
        }
    }
}

//File： @chainlink/contracts/src/v0.8/automation/AutomationCompatible.sol

abstract contract KeeperCompatible is
    AutomationBase,
    AutomationCompatibleInterface
{

}

//File： fs://8264b29a6b9244718712c9d9cd3fd5b2/token.sol

interface IUniswapV2Router02 {
    function swapExactTokensForETH(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);

    function swapExactETHForTokens(
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external payable returns (uint256[] memory amounts);

    function getAmountsOut(
        uint256 amountIn,
        address[] calldata path
    ) external view returns (uint256[] memory amounts);
}

// Uniswap V2 interfaces
interface IUniswapV2Pair {
    function getReserves()
        external
        view
        returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);

    function token0() external view returns (address);

    function token1() external view returns (address);

    function sync() external;
}

interface IERC20 {
    function totalSupply() external view returns (uint256);

    function balanceOf(address who) external view returns (uint256);

    function allowance(
        address owner,
        address spender
    ) external view returns (uint256);

    function transfer(address to, uint256 value) external returns (bool);

    function approve(address spender, uint256 value) external returns (bool);

    function transferFrom(
        address from,
        address to,
        uint256 value
    ) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
}

abstract contract ERC20Detailed is IERC20 {
    string private _name;
    string private _symbol;
    uint8 private _decimals;

    function initialize(
        string memory name_,
        string memory symbol_,
        uint8 decimals_
    ) internal {
        _name = name_;
        _symbol = symbol_;
        _decimals = decimals_;
    }

    function name() public view returns (string memory) {
        return _name;
    }

    function symbol() public view returns (string memory) {
        return _symbol;
    }

    function decimals() public view returns (uint8) {
        return _decimals;
    }
}

interface IDogePriceOracle {
    function getPrice() external view returns (uint256);
}

contract CollarToken is
    ERC20Detailed,
    Ownable,
    KeeperCompatible,
    ReentrancyGuard
{
    IDogePriceOracle public priceFeedDOGE; // price feed for DOGE/USD
    AggregatorV3Interface public priceFeedETH; // Chainlink price feed for ETH/USD
    uint256 public rebaseInterval = 24 hours; // Rebase interval
    uint256 public lastRebaseTime; // Timestamp of last rebase
    uint256 public targetRatio = 500 ether; // Target: 1 Collar = 500 DOGE
    uint256 private MAX_TARGET_THRESHOLD = 501 ether;

    // Uniswap V2 variables
    address public uniswapPair;

    using Address for address;
    uint256 private constant DECIMALS = 18;
    uint256 private constant TOTAL_GONS =
        type(uint256).max - (type(uint256).max % 1e18);
    uint256 private _gonsPerFragment;
    uint256 private _totalSupply;
    bool public rebaseActive = false;
    mapping(address => bool) public keepers;
    uint256 private constant MIN_DEVIATION_THRESHOLD = 1e15; // 0.1% (1e18 = 100%)

    mapping(address => uint256) private _gonBalances;
    mapping(address => mapping(address => uint256)) private _allowedFragments;

    // Multisig state variables
    address[] public owners;
    uint256 public required;
    struct Proposal {
        string action; // Descriptive name of the action (e.g., "setUniswapPaireToken")
        address target; // Target address (typically this contract)
        uint256 value; // ETH value to send (usually 0 for these functions)
        bytes data; // Encoded function call data
        uint256 approvalCount; // Number of approvals received
        mapping(address => bool) approved; // Tracks which owners have approved
        bool executed; // Whether the proposal has been executed
    }
    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;

    event Rebase(uint256 oldSupply, uint256 newSupply);
    event KeeperAdded(address newKeeper);
    event UniswapPairUpdated(address newUniswapPair);
    event RebaseIntervalUpdated(uint256 newInterval);
    event PriceFeedEthUpdated(address newPriceFeedETH);
    event TargetRatioUpdated(uint256 newTargetRatio);
    event PriceFeedDogeUpdated(address newPriceFeedDoge);
    event RebaseStateChanged(bool isActive);
    event KeeperRemoved(address removedKeeper);
    event ProposalCreated(
        uint256 indexed proposalId,
        string action,
        address target,
        uint256 value,
        bytes data
    );
    event ProposalApproved(uint256 indexed proposalId, address indexed owner);
    event ProposalExecuted(uint256 indexed proposalId);

    constructor(
        uint256 _initialSupply,
        address _priceFeedDOGE,
        address _priceFeedETH,
        address[] memory _owners,
        uint256 _required,
        address _initialTokenHolder
    ) Ownable(address(this)) {
        require(_initialSupply > 0, "Initial supply must be greater than 0");
        require(_owners.length >= _required, "Owners less than required");
        require(_required > 0, "Required must be greater than 0");
        require(_initialTokenHolder != address(0), "Invalid token holder");

        // Initialize token details
        ERC20Detailed.initialize("DOGE FATHER", "COLLAR", uint8(DECIMALS));
        _totalSupply = _initialSupply * 10 ** DECIMALS;
        _gonsPerFragment = TOTAL_GONS / _totalSupply;
        _gonBalances[_initialTokenHolder] = TOTAL_GONS;

        // Initialize multisig parameters
        owners = _owners;
        required = _required;

        priceFeedETH = AggregatorV3Interface(_priceFeedETH); 
        priceFeedDOGE = IDogePriceOracle(_priceFeedDOGE);

        emit Transfer(address(0), _initialTokenHolder, _totalSupply);
    }

    modifier onlyKeeper() {
        require(keepers[msg.sender], "Not authorized keeper");
        _;
    }

    // Multisig helper function
    function isOwner(address account) public view returns (bool) {
        for (uint256 i = 0; i < owners.length; i++) {
            if (owners[i] == account) {
                return true;
            }
        }
        return false;
    }

    // Multisig proposal functions
    function createProposal(
        string memory action,
        address target,
        uint256 value,
        bytes memory data
    ) external {
        require(isOwner(msg.sender), "Only owners can create proposals");
        proposalCount++;
        Proposal storage proposal = proposals[proposalCount];
        proposal.action = action;
        proposal.target = target;
        proposal.value = value;
        proposal.data = data;
        proposal.approvalCount = 0;
        proposal.executed = false;
        emit ProposalCreated(proposalCount, action, target, value, data);
    }

    function approveProposal(uint256 proposalId) external {
        require(isOwner(msg.sender), "Only owners can approve");
        Proposal storage proposal = proposals[proposalId];
        require(proposal.data.length > 0, "Proposal does not exist");
        require(!proposal.approved[msg.sender], "Already approved");
        proposal.approved[msg.sender] = true;
        proposal.approvalCount++;
        emit ProposalApproved(proposalId, msg.sender);
    }

    function executeProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.approvalCount >= required, "Not enough approvals");
        require(!proposal.executed, "Already executed");
        proposal.executed = true;
        (bool success, ) = proposal.target.call{value: proposal.value}(
            proposal.data
        );
        require(success, "Execution failed");
        emit ProposalExecuted(proposalId);
    }

    // Chainlink Keeper-compatible checkUpkeep function
    function checkUpkeep(
        bytes calldata
    ) external view override returns (bool upkeepNeeded, bytes memory) {
        upkeepNeeded =
            block.timestamp >= lastRebaseTime + rebaseInterval &&
            (rebaseActive || _shouldActivateRebasing());
        return (upkeepNeeded, "");
    }

    // Chainlink Keeper-compatible performUpkeep function
    function performUpkeep(
        bytes calldata
    ) external override nonReentrant onlyKeeper {
        require(
            block.timestamp >= lastRebaseTime + rebaseInterval,
            "Rebase interval not met"
        );
        // Sync before checking if rebasing should be active
        if (uniswapPair != address(0)) {
            IUniswapV2Pair(uniswapPair).sync();
        }
        if (!rebaseActive) {
            if (_shouldActivateRebasing()) {
                rebaseActive = true; // Now daily rebasing can start
            } else {
                return; // Exit function if the ratio hasn't been met yet
            }
        }

        _rebase();
    }

    // Internal helper to determine if rebasing should activate
    function _shouldActivateRebasing() internal view returns (bool) {
        uint256 currentPrice = getCollarPriceInUSD();
        uint256 dogePrice = getDogePrice();
        uint256 targetPrice = (dogePrice * targetRatio) / 1e18;

        return currentPrice >= targetPrice;
    }

    function getDogePrice() public view returns (uint256) {
        uint256 price = priceFeedDOGE.getPrice();
        require(price > 0, "Invalid DOGE price");
        return price;
    }

    // Fetch ETH price using Chainlink Price Feed
    function getEthPrice() public view returns (uint256) {
        // Fetch the latest round data
        (, int256 price, , uint256 updatedAt, ) = priceFeedETH
            .latestRoundData();

        // Ensure the price is valid
        require(price > 0, "Invalid ETH price");

        // Ensure the price feed data is recent
        require(block.timestamp - updatedAt < 1 hours, "Stale ETH price");

        // Adjust for decimals (Chainlink prices typically have 8 decimals)
        uint256 feedDecimals = priceFeedETH.decimals();
        return uint256(price) * (10 ** (18 - feedDecimals));
    }

    // Get Collar price from Uniswap V2
    function getCollarPriceFromV2() public view returns (uint256) {
        if (uniswapPair == address(0)) {
            return 0;
        }

        IUniswapV2Pair pair = IUniswapV2Pair(uniswapPair);
        (uint112 reserve0, uint112 reserve1, ) = pair.getReserves();

        require(reserve0 > 0 && reserve1 > 0, "Insufficient liquidity");

        // Determine which reserve corresponds to COLLAR
        bool isCollarToken0 = pair.token0() == address(this);
        uint112 collarReserve = isCollarToken0 ? reserve0 : reserve1;
        uint112 tokenReserve = isCollarToken0 ? reserve1 : reserve0;

        // Calculate price of 1 COLLAR in terms of the other token
        // price = tokenReserve / collarReserve
        return (uint256(tokenReserve) * 1e18) / uint256(collarReserve);
    }

    function getCollarPriceInUSD() public view returns (uint256) {
        // Get the Collar token price in ETH from Uniswap
        uint256 collarPriceInETH = getCollarPriceFromV2();
        if (collarPriceInETH == 0) {
            return 0; // Return 0 if price can't be determined yet
        }

        // Fetch ETH price in USD using Chainlink's ETH/USD price feed
        uint256 ethPriceInUSD = getEthPrice();

        // Calculate the Collar price in USD (Collar price in ETH * ETH price in USD)
        uint256 collarPriceInUSD = (collarPriceInETH * ethPriceInUSD) / 1e18;

        return collarPriceInUSD; // Return in 18 decimals
    }

    function _rebase() internal {
        // First sync the Uniswap pair to get the latest reserves
        if (uniswapPair != address(0)) {
            IUniswapV2Pair(uniswapPair).sync();
        }

        uint256 currentPrice = getCollarPriceInUSD();
        uint256 dogePrice = getDogePrice();

        require(_totalSupply > 0, "Rebase failed: total supply is zero");

        // Calculate the target price (1 Collar = targetRatio DOGE)
        uint256 targetPrice = (dogePrice * targetRatio) / 1e18;
        require(targetPrice > 0, "Invalid target price");
        require(currentPrice > 0, "Invalid current price");

        uint256 oldSupply = _totalSupply;
        uint256 adjustment;

        // Calculate the percentage deviation from the target price
        uint256 deviation;
        if (currentPrice > targetPrice) {
            // If Collar is overvalued, calculate the percentage above the target
            deviation = ((currentPrice - targetPrice) * 1e18) / targetPrice;
        } else if (currentPrice < targetPrice) {
            // If Collar is undervalued, calculate the percentage below the target
            deviation = ((targetPrice - currentPrice) * 1e18) / targetPrice;
        } else {
            return; // No adjustment needed if price is exactly at target
        }

        if (deviation < MIN_DEVIATION_THRESHOLD) {
            return;
        }

        // Calculate the supply adjustment based on the deviation
        adjustment = (_totalSupply * deviation) / 1e18;

        // Apply the adjustment with overflow/underflow checks
        if (currentPrice > targetPrice) {
            // If Collar is overvalued, increase the supply
            require(
                _totalSupply + adjustment >= _totalSupply,
                "Overflow detected"
            );
            _totalSupply += adjustment;
        } else if (currentPrice < targetPrice) {
            // If Collar is undervalued, decrease the supply
            require(adjustment <= _totalSupply, "Underflow detected");
            _totalSupply -= adjustment;
        } else {
            return;
        }

        // Update _gonsPerFragment to reflect the new supply
        _gonsPerFragment = TOTAL_GONS / _totalSupply;
        lastRebaseTime = block.timestamp;

        // Call sync() on the Uniswap pair to update reserves after rebase
        if (uniswapPair != address(0)) {
            IUniswapV2Pair(uniswapPair).sync();
        }

        emit Rebase(oldSupply, _totalSupply);
    }

    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _gonBalances[account] / _gonsPerFragment;
    }

    function approve(
        address spender,
        uint256 value
    ) public override returns (bool) {
        _allowedFragments[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function allowance(
        address owner_,
        address spender
    ) public view override returns (uint256) {
        return _allowedFragments[owner_][spender];
    }

    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) public override returns (bool) {
        require(
            sender != address(0),
            "Transfer from zero address is not allowed"
        );
        require(
            recipient != address(0),
            "Transfer to zero address is not allowed"
        );
        require(amount > 0, "Transfer amount must be greater than zero");

        uint256 gonAmount = amount * _gonsPerFragment;
        require(_gonBalances[sender] >= gonAmount, "Insufficient balance");

        // Check allowance
        uint256 currentAllowance = _allowedFragments[sender][msg.sender];
        require(
            currentAllowance >= amount,
            "Transfer amount exceeds allowance"
        );

        _gonBalances[sender] -= gonAmount;
        _gonBalances[recipient] += gonAmount;

        // Update allowance
        _allowedFragments[sender][msg.sender] -= amount;

        emit Transfer(sender, recipient, amount);
        return true;
    }

    function transfer(
        address recipient,
        uint256 amount
    ) public override returns (bool) {
        require(
            recipient != address(0),
            "Transfer to zero address is not allowed"
        );

        uint256 gonAmount = amount * _gonsPerFragment;
        require(_gonBalances[msg.sender] >= gonAmount, "Insufficient balance");

        _gonBalances[msg.sender] -= gonAmount;
        _gonBalances[recipient] += gonAmount;

        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    // Function to set or update the paired token
    function setUniswapPaireToken(address _uniswapPair) external onlyOwner {
        require(_uniswapPair != address(0), "Invalid paired token");
        uniswapPair = _uniswapPair;
        emit UniswapPairUpdated(_uniswapPair);
    }

    function addKeeper(address _keeper) external onlyOwner {
        require(_keeper != address(0), "Invalid address");
        keepers[_keeper] = true;
        emit KeeperAdded(_keeper);
    }

    function setTargetRatio(uint256 _targetRatio) external onlyOwner {
        require(
            _targetRatio > 0 && _targetRatio < MAX_TARGET_THRESHOLD,
            "Invalid target ratio"
        );
        targetRatio = _targetRatio;
        emit TargetRatioUpdated(_targetRatio);
    }

    function removeKeeper(address _keeper) external onlyOwner {
        require(_keeper != address(0), "Invalid address");
        require(keepers[_keeper], "Not a keeper");
        keepers[_keeper] = false;
        emit KeeperRemoved(_keeper);
    }

    function setRebaseInterval(uint256 _interval) external onlyOwner {
        rebaseInterval = _interval;
        emit RebaseIntervalUpdated(_interval);
    }

    function toggleRebaseActive(bool _active) external onlyOwner {
        rebaseActive = _active;
        emit RebaseStateChanged(_active);
    }

    function setPriceFeedEth(address _newPriceFeedETH) external onlyOwner {
        require(_newPriceFeedETH != address(0), "Invalid address");
        require(isContract(_newPriceFeedETH), "Address is not a contract");
        priceFeedETH = AggregatorV3Interface(_newPriceFeedETH);
        emit PriceFeedEthUpdated(_newPriceFeedETH);
    }

    function setPriceFeedDoge(address _newPriceFeedDoge) external onlyOwner {
        require(_newPriceFeedDoge != address(0), "Invalid address");
        require(isContract(_newPriceFeedDoge), "Address is not a contract");
        priceFeedDOGE = IDogePriceOracle(_newPriceFeedDoge);
        emit PriceFeedDogeUpdated(_newPriceFeedDoge);
    }

    function isContract(address account) internal view returns (bool) {
        // According to EIP-1052, 0x0 is the value returned for not-yet created accounts
        // and 0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470 is returned
        // for accounts without code, i.e. `keccak256('')`
        bytes32 codehash;
        bytes32 accountHash = 0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470;
        // solhint-disable-next-line no-inline-assembly
        assembly {
            codehash := extcodehash(account)
        }
        return (codehash != accountHash && codehash != 0x0);
    }
}