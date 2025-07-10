// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

/// @notice This library contains various token pool functions to aid constructing the return data.
library Pool {
    // The tag used to signal support for the pool v1 standard
    // bytes4(keccak256("CCIP_POOL_V1"))
    bytes4 public constant CCIP_POOL_V1 = 0xaff2afbf;

    // The number of bytes in the return data for a pool v1 releaseOrMint call.
    // This should match the size of the ReleaseOrMintOutV1 struct.
    uint16 public constant CCIP_POOL_V1_RET_BYTES = 32;

    // The default max number of bytes in the return data for a pool v1 lockOrBurn call.
    // This data can be used to send information to the destination chain token pool. Can be overwritten
    // in the TokenTransferFeeConfig.destBytesOverhead if more data is required.
    uint32 public constant CCIP_LOCK_OR_BURN_V1_RET_BYTES = 32;

    struct LockOrBurnInV1 {
        bytes receiver; //  The recipient of the tokens on the destination chain, abi encoded
        uint64 remoteChainSelector; // ─╮ The chain ID of the destination chain
        address originalSender; // ─────╯ The original sender of the tx on the source chain
        uint256 amount; //  The amount of tokens to lock or burn, denominated in the source token's decimals
        address localToken; //  The address on this chain of the token to lock or burn
    }

    struct LockOrBurnOutV1 {
        // The address of the destination token, abi encoded in the case of EVM chains
        // This value is UNTRUSTED as any pool owner can return whatever value they want.
        bytes destTokenAddress;
        // Optional pool data to be transferred to the destination chain. Be default this is capped at
        // CCIP_LOCK_OR_BURN_V1_RET_BYTES bytes. If more data is required, the TokenTransferFeeConfig.destBytesOverhead
        // has to be set for the specific token.
        bytes destPoolData;
    }

    struct ReleaseOrMintInV1 {
        bytes originalSender; //          The original sender of the tx on the source chain
        uint64 remoteChainSelector; // ─╮ The chain ID of the source chain
        address receiver; // ───────────╯ The recipient of the tokens on the destination chain.
        uint256 amount; //                The amount of tokens to release or mint, denominated in the source token's decimals
        address localToken; //            The address on this chain of the token to release or mint
        /// @dev WARNING: sourcePoolAddress should be checked prior to any processing of funds. Make sure it matches the
        /// expected pool address for the given remoteChainSelector.
        bytes sourcePoolAddress; //       The address of the source pool, abi encoded in the case of EVM chains
        bytes sourcePoolData; //          The data received from the source pool to process the release or mint
        /// @dev WARNING: offchainTokenData is untrusted data.
        bytes offchainTokenData; //       The offchain data to process the release or mint
    }

    struct ReleaseOrMintOutV1 {
        // The number of tokens released or minted on the destination chain, denominated in the local token's decimals.
        // This value is expected to be equal to the ReleaseOrMintInV1.amount in the case where the source and destination
        // chain have the same number of decimals.
        uint256 destinationAmount;
    }
}

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

pragma solidity ^0.8.0;

/// @notice Shared public interface for multiple V1 pool types.
/// Each pool type handles a different child token model (lock/unlock, mint/burn.)
interface IPoolV1 is IERC165 {
    /// @notice Lock tokens into the pool or burn the tokens.
    /// @param lockOrBurnIn Encoded data fields for the processing of tokens on the source chain.
    /// @return lockOrBurnOut Encoded data fields for the processing of tokens on the destination chain.
    function lockOrBurn(
        Pool.LockOrBurnInV1 calldata lockOrBurnIn
    ) external returns (Pool.LockOrBurnOutV1 memory lockOrBurnOut);

    /// @notice Releases or mints tokens to the receiver address.
    /// @param releaseOrMintIn All data required to release or mint tokens.
    /// @return releaseOrMintOut The amount of tokens released or minted on the local chain, denominated
    /// in the local token's decimals.
    /// @dev The offramp asserts that the balanceOf of the receiver has been incremented by exactly the number
    /// of tokens that is returned in ReleaseOrMintOutV1.destinationAmount. If the amounts do not match, the tx reverts.
    function releaseOrMint(
        Pool.ReleaseOrMintInV1 calldata releaseOrMintIn
    ) external returns (Pool.ReleaseOrMintOutV1 memory);

    /// @notice Checks whether a remote chain is supported in the token pool.
    /// @param remoteChainSelector The selector of the remote chain.
    /// @return true if the given chain is a permissioned remote chain.
    function isSupportedChain(
        uint64 remoteChainSelector
    ) external view returns (bool);

    /// @notice Returns if the token pool supports the given token.
    /// @param token The address of the token.
    /// @return true if the token is supported by the pool.
    function isSupportedToken(address token) external view returns (bool);
}

pragma solidity ^0.8.0;

// End consumer library.
library Client {
    /// @dev RMN depends on this struct, if changing, please notify the RMN maintainers.
    struct EVMTokenAmount {
        address token; // token address on the local chain.
        uint256 amount; // Amount of tokens.
    }

    struct Any2EVMMessage {
        bytes32 messageId; // MessageId corresponding to ccipSend on source.
        uint64 sourceChainSelector; // Source chain selector.
        bytes sender; // abi.decode(sender) if coming from an EVM chain.
        bytes data; // payload sent in original message.
        EVMTokenAmount[] destTokenAmounts; // Tokens and their amounts in their destination chain representation.
    }

    // If extraArgs is empty bytes, the default is 200k gas limit.
    struct EVM2AnyMessage {
        bytes receiver; // abi.encode(receiver address) for dest EVM chains
        bytes data; // Data payload
        EVMTokenAmount[] tokenAmounts; // Token transfers
        address feeToken; // Address of feeToken. address(0) means you will send msg.value.
        bytes extraArgs; // Populate this with _argsToBytes(EVMExtraArgsV2)
    }

    // bytes4(keccak256("CCIP EVMExtraArgsV1"));
    bytes4 public constant EVM_EXTRA_ARGS_V1_TAG = 0x97a657c9;

    struct EVMExtraArgsV1 {
        uint256 gasLimit;
    }

    function _argsToBytes(
        EVMExtraArgsV1 memory extraArgs
    ) internal pure returns (bytes memory bts) {
        return abi.encodeWithSelector(EVM_EXTRA_ARGS_V1_TAG, extraArgs);
    }

    // bytes4(keccak256("CCIP EVMExtraArgsV2"));
    bytes4 public constant EVM_EXTRA_ARGS_V2_TAG = 0x181dcf10;

    /// @param gasLimit: gas limit for the callback on the destination chain.
    /// @param allowOutOfOrderExecution: if true, it indicates that the message can be executed in any order relative to other messages from the same sender.
    /// This value's default varies by chain. On some chains, a particular value is enforced, meaning if the expected value
    /// is not set, the message request will revert.
    struct EVMExtraArgsV2 {
        uint256 gasLimit;
        bool allowOutOfOrderExecution;
    }

    function _argsToBytes(
        EVMExtraArgsV2 memory extraArgs
    ) internal pure returns (bytes memory bts) {
        return abi.encodeWithSelector(EVM_EXTRA_ARGS_V2_TAG, extraArgs);
    }
}

pragma solidity ^0.8.0;

interface IRouter {
    error OnlyOffRamp();

    /// @notice Route the message to its intended receiver contract.
    /// @param message Client.Any2EVMMessage struct.
    /// @param gasForCallExactCheck of params for exec
    /// @param gasLimit set of params for exec
    /// @param receiver set of params for exec
    /// @dev if the receiver is a contracts that signals support for CCIP execution through EIP-165.
    /// the contract is called. If not, only tokens are transferred.
    /// @return success A boolean value indicating whether the ccip message was received without errors.
    /// @return retBytes A bytes array containing return data form CCIP receiver.
    /// @return gasUsed the gas used by the external customer call. Does not include any overhead.
    function routeMessage(
        Client.Any2EVMMessage calldata message,
        uint16 gasForCallExactCheck,
        uint256 gasLimit,
        address receiver
    ) external returns (bool success, bytes memory retBytes, uint256 gasUsed);

    /// @notice Returns the configured onramp for a specific destination chain.
    /// @param destChainSelector The destination chain Id to get the onRamp for.
    /// @return onRampAddress The address of the onRamp.
    function getOnRamp(
        uint64 destChainSelector
    ) external view returns (address onRampAddress);

    /// @notice Return true if the given offRamp is a configured offRamp for the given source chain.
    /// @param sourceChainSelector The source chain selector to check.
    /// @param offRamp The address of the offRamp to check.
    function isOffRamp(
        uint64 sourceChainSelector,
        address offRamp
    ) external view returns (bool isOffRamp);
}

pragma solidity ^0.8.0;

/// @notice This interface contains the only RMN-related functions that might be used on-chain by other CCIP contracts.
interface IRMN {
    /// @notice A Merkle root tagged with the address of the commit store contract it is destined for.
    struct TaggedRoot {
        address commitStore;
        bytes32 root;
    }

    /// @notice Callers MUST NOT cache the return value as a blessed tagged root could become unblessed.
    function isBlessed(
        TaggedRoot calldata taggedRoot
    ) external view returns (bool);

    /// @notice Iff there is an active global or legacy curse, this function returns true.
    function isCursed() external view returns (bool);

    /// @notice Iff there is an active global curse, or an active curse for `subject`, this function returns true.
    /// @param subject To check whether a particular chain is cursed, set to bytes16(uint128(chainSelector)).
    function isCursed(bytes16 subject) external view returns (bool);
}

pragma solidity ^0.8.4;

/// @notice Implements Token Bucket rate limiting.
/// @dev uint128 is safe for rate limiter state.
/// For USD value rate limiting, it can adequately store USD value in 18 decimals.
/// For ERC20 token amount rate limiting, all tokens that will be listed will have at most
/// a supply of uint128.max tokens, and it will therefore not overflow the bucket.
/// In exceptional scenarios where tokens consumed may be larger than uint128,
/// e.g. compromised issuer, an enabled RateLimiter will check and revert.
library RateLimiter {
    error BucketOverfilled();
    error OnlyCallableByAdminOrOwner();
    error TokenMaxCapacityExceeded(
        uint256 capacity,
        uint256 requested,
        address tokenAddress
    );
    error TokenRateLimitReached(
        uint256 minWaitInSeconds,
        uint256 available,
        address tokenAddress
    );
    error AggregateValueMaxCapacityExceeded(
        uint256 capacity,
        uint256 requested
    );
    error AggregateValueRateLimitReached(
        uint256 minWaitInSeconds,
        uint256 available
    );
    error InvalidRateLimitRate(Config rateLimiterConfig);
    error DisabledNonZeroRateLimit(Config config);
    error RateLimitMustBeDisabled();

    event TokensConsumed(uint256 tokens);
    event ConfigChanged(Config config);

    struct TokenBucket {
        uint128 tokens; // ──────╮ Current number of tokens that are in the bucket.
        uint32 lastUpdated; //   │ Timestamp in seconds of the last token refill, good for 100+ years.
        bool isEnabled; // ──────╯ Indication whether the rate limiting is enabled or not
        uint128 capacity; // ────╮ Maximum number of tokens that can be in the bucket.
        uint128 rate; // ────────╯ Number of tokens per second that the bucket is refilled.
    }

    struct Config {
        bool isEnabled; // Indication whether the rate limiting should be enabled
        uint128 capacity; // ────╮ Specifies the capacity of the rate limiter
        uint128 rate; //  ───────╯ Specifies the rate of the rate limiter
    }

    /// @notice _consume removes the given tokens from the pool, lowering the
    /// rate tokens allowed to be consumed for subsequent calls.
    /// @param requestTokens The total tokens to be consumed from the bucket.
    /// @param tokenAddress The token to consume capacity for, use 0x0 to indicate aggregate value capacity.
    /// @dev Reverts when requestTokens exceeds bucket capacity or available tokens in the bucket
    /// @dev emits removal of requestTokens if requestTokens is > 0
    function _consume(
        TokenBucket storage s_bucket,
        uint256 requestTokens,
        address tokenAddress
    ) internal {
        // If there is no value to remove or rate limiting is turned off, skip this step to reduce gas usage
        if (!s_bucket.isEnabled || requestTokens == 0) {
            return;
        }

        uint256 tokens = s_bucket.tokens;
        uint256 capacity = s_bucket.capacity;
        uint256 timeDiff = block.timestamp - s_bucket.lastUpdated;

        if (timeDiff != 0) {
            if (tokens > capacity) revert BucketOverfilled();

            // Refill tokens when arriving at a new block time
            tokens = _calculateRefill(
                capacity,
                tokens,
                timeDiff,
                s_bucket.rate
            );

            s_bucket.lastUpdated = uint32(block.timestamp);
        }

        if (capacity < requestTokens) {
            // Token address 0 indicates consuming aggregate value rate limit capacity.
            if (tokenAddress == address(0))
                revert AggregateValueMaxCapacityExceeded(
                    capacity,
                    requestTokens
                );
            revert TokenMaxCapacityExceeded(
                capacity,
                requestTokens,
                tokenAddress
            );
        }
        if (tokens < requestTokens) {
            uint256 rate = s_bucket.rate;
            // Wait required until the bucket is refilled enough to accept this value, round up to next higher second
            // Consume is not guaranteed to succeed after wait time passes if there is competing traffic.
            // This acts as a lower bound of wait time.
            uint256 minWaitInSeconds = ((requestTokens - tokens) + (rate - 1)) /
                rate;

            if (tokenAddress == address(0))
                revert AggregateValueRateLimitReached(minWaitInSeconds, tokens);
            revert TokenRateLimitReached(
                minWaitInSeconds,
                tokens,
                tokenAddress
            );
        }
        tokens -= requestTokens;

        // Downcast is safe here, as tokens is not larger than capacity
        s_bucket.tokens = uint128(tokens);
        emit TokensConsumed(requestTokens);
    }

    /// @notice Gets the token bucket with its values for the block it was requested at.
    /// @return The token bucket.
    function _currentTokenBucketState(
        TokenBucket memory bucket
    ) internal view returns (TokenBucket memory) {
        // We update the bucket to reflect the status at the exact time of the
        // call. This means we might need to refill a part of the bucket based
        // on the time that has passed since the last update.
        bucket.tokens = uint128(
            _calculateRefill(
                bucket.capacity,
                bucket.tokens,
                block.timestamp - bucket.lastUpdated,
                bucket.rate
            )
        );
        bucket.lastUpdated = uint32(block.timestamp);
        return bucket;
    }

    /// @notice Sets the rate limited config.
    /// @param s_bucket The token bucket
    /// @param config The new config
    function _setTokenBucketConfig(
        TokenBucket storage s_bucket,
        Config memory config
    ) internal {
        // First update the bucket to make sure the proper rate is used for all the time
        // up until the config change.
        uint256 timeDiff = block.timestamp - s_bucket.lastUpdated;
        if (timeDiff != 0) {
            s_bucket.tokens = uint128(
                _calculateRefill(
                    s_bucket.capacity,
                    s_bucket.tokens,
                    timeDiff,
                    s_bucket.rate
                )
            );

            s_bucket.lastUpdated = uint32(block.timestamp);
        }

        s_bucket.tokens = uint128(_min(config.capacity, s_bucket.tokens));
        s_bucket.isEnabled = config.isEnabled;
        s_bucket.capacity = config.capacity;
        s_bucket.rate = config.rate;

        emit ConfigChanged(config);
    }

    /// @notice Validates the token bucket config
    function _validateTokenBucketConfig(
        Config memory config,
        bool mustBeDisabled
    ) internal pure {
        if (config.isEnabled) {
            if (config.rate >= config.capacity || config.rate == 0) {
                revert InvalidRateLimitRate(config);
            }
            if (mustBeDisabled) {
                revert RateLimitMustBeDisabled();
            }
        } else {
            if (config.rate != 0 || config.capacity != 0) {
                revert DisabledNonZeroRateLimit(config);
            }
        }
    }

    /// @notice Calculate refilled tokens
    /// @param capacity bucket capacity
    /// @param tokens current bucket tokens
    /// @param timeDiff block time difference since last refill
    /// @param rate bucket refill rate
    /// @return the value of tokens after refill
    function _calculateRefill(
        uint256 capacity,
        uint256 tokens,
        uint256 timeDiff,
        uint256 rate
    ) private pure returns (uint256) {
        return _min(capacity, tokens + timeDiff * rate);
    }

    /// @notice Return the smallest of two integers
    /// @param a first int
    /// @param b second int
    /// @return smallest
    function _min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
}

pragma solidity ^0.8.0;

interface IOwnable {
    function owner() external returns (address);

    function transferOwnership(address recipient) external;

    function acceptOwnership() external;
}

pragma solidity ^0.8.4;

/// @notice A minimal contract that implements 2-step ownership transfer and nothing more. It's made to be minimal
/// to reduce the impact of the bytecode size on any contract that inherits from it.
contract Ownable2Step is IOwnable {
    /// @notice The pending owner is the address to which ownership may be transferred.
    address private s_pendingOwner;
    /// @notice The owner is the current owner of the contract.
    /// @dev The owner is the second storage variable so any implementing contract could pack other state with it
    /// instead of the much less used s_pendingOwner.
    address private s_owner;

    error OwnerCannotBeZero();
    error MustBeProposedOwner();
    error CannotTransferToSelf();
    error OnlyCallableByOwner();

    event OwnershipTransferRequested(address indexed from, address indexed to);
    event OwnershipTransferred(address indexed from, address indexed to);

    constructor(address newOwner, address pendingOwner) {
        if (newOwner == address(0)) {
            revert OwnerCannotBeZero();
        }

        s_owner = newOwner;
        if (pendingOwner != address(0)) {
            _transferOwnership(pendingOwner);
        }
    }

    /// @notice Get the current owner
    function owner() public view override returns (address) {
        return s_owner;
    }

    /// @notice Allows an owner to begin transferring ownership to a new address. The new owner needs to call
    /// `acceptOwnership` to accept the transfer before any permissions are changed.
    /// @param to The address to which ownership will be transferred.
    function transferOwnership(address to) public override onlyOwner {
        _transferOwnership(to);
    }

    /// @notice validate, transfer ownership, and emit relevant events
    /// @param to The address to which ownership will be transferred.
    function _transferOwnership(address to) private {
        if (to == msg.sender) {
            revert CannotTransferToSelf();
        }

        s_pendingOwner = to;

        emit OwnershipTransferRequested(s_owner, to);
    }

    /// @notice Allows an ownership transfer to be completed by the recipient.
    function acceptOwnership() external override {
        if (msg.sender != s_pendingOwner) {
            revert MustBeProposedOwner();
        }

        address oldOwner = s_owner;
        s_owner = msg.sender;
        s_pendingOwner = address(0);

        emit OwnershipTransferred(oldOwner, msg.sender);
    }

    /// @notice validate access
    function _validateOwnership() internal view {
        if (msg.sender != s_owner) {
            revert OnlyCallableByOwner();
        }
    }

    /// @notice Reverts if called by anyone other than the contract owner.
    modifier onlyOwner() {
        _validateOwnership();
        _;
    }
}

pragma solidity ^0.8.4;

/// @notice Sets the msg.sender to be the owner of the contract and does not set a pending owner.
contract Ownable2StepMsgSender is Ownable2Step {
    constructor() Ownable2Step(msg.sender, address(0)) {}
}

// OpenZeppelin Contracts (last updated v4.6.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.0;

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
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );

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
    function allowance(
        address owner,
        address spender
    ) external view returns (uint256);

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
    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) external returns (bool);
}

// OpenZeppelin Contracts v4.4.1 (token/ERC20/extensions/IERC20Metadata.sol)

pragma solidity ^0.8.0;

/**
 * @dev Interface for the optional metadata functions from the ERC20 standard.
 *
 * _Available since v4.1._
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

// OpenZeppelin Contracts (last updated v5.0.0) (utils/structs/EnumerableSet.sol)
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
    function _contains(
        Set storage set,
        bytes32 value
    ) private view returns (bool) {
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
    function _at(
        Set storage set,
        uint256 index
    ) private view returns (bytes32) {
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
    function add(
        Bytes32Set storage set,
        bytes32 value
    ) internal returns (bool) {
        return _add(set._inner, value);
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function remove(
        Bytes32Set storage set,
        bytes32 value
    ) internal returns (bool) {
        return _remove(set._inner, value);
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(
        Bytes32Set storage set,
        bytes32 value
    ) internal view returns (bool) {
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
    function at(
        Bytes32Set storage set,
        uint256 index
    ) internal view returns (bytes32) {
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
    function values(
        Bytes32Set storage set
    ) internal view returns (bytes32[] memory) {
        bytes32[] memory store = _values(set._inner);
        bytes32[] memory result;

        /// @solidity memory-safe-assembly
        assembly {
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
    function add(
        AddressSet storage set,
        address value
    ) internal returns (bool) {
        return _add(set._inner, bytes32(uint256(uint160(value))));
    }

    /**
     * @dev Removes a value from a set. O(1).
     *
     * Returns true if the value was removed from the set, that is if it was
     * present.
     */
    function remove(
        AddressSet storage set,
        address value
    ) internal returns (bool) {
        return _remove(set._inner, bytes32(uint256(uint160(value))));
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(
        AddressSet storage set,
        address value
    ) internal view returns (bool) {
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
    function at(
        AddressSet storage set,
        uint256 index
    ) internal view returns (address) {
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
    function values(
        AddressSet storage set
    ) internal view returns (address[] memory) {
        bytes32[] memory store = _values(set._inner);
        address[] memory result;

        /// @solidity memory-safe-assembly
        assembly {
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
    function remove(
        UintSet storage set,
        uint256 value
    ) internal returns (bool) {
        return _remove(set._inner, bytes32(value));
    }

    /**
     * @dev Returns true if the value is in the set. O(1).
     */
    function contains(
        UintSet storage set,
        uint256 value
    ) internal view returns (bool) {
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
    function at(
        UintSet storage set,
        uint256 index
    ) internal view returns (uint256) {
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
    function values(
        UintSet storage set
    ) internal view returns (uint256[] memory) {
        bytes32[] memory store = _values(set._inner);
        uint256[] memory result;

        /// @solidity memory-safe-assembly
        assembly {
            result := store
        }

        return result;
    }
}

pragma solidity 0.8.24;

/// @dev This pool supports different decimals on different chains but using this feature could impact the total number
/// of tokens in circulation. Since all of the tokens are locked/burned on the source, and a rounded amount is minted/released on the
/// destination, the number of tokens minted/released could be less than the number of tokens burned/locked. This is because the source
/// chain does not know about the destination token decimals. This is not a problem if the decimals are the same on both
/// chains.
///
/// Example:
/// Assume there is a token with 6 decimals on chain A and 3 decimals on chain B.
/// - 1.234567 tokens are burned on chain A.
/// - 1.234    tokens are minted on chain B.
/// When sending the 1.234 tokens back to chain A, you will receive 1.234000 tokens on chain A, effectively losing
/// 0.000567 tokens.
/// In the case of a burnMint pool on chain A, these funds are burned in the pool on chain A.
/// In the case of a lockRelease pool on chain A, these funds accumulate in the pool on chain A.
abstract contract TokenPool is IPoolV1, Ownable2StepMsgSender {
    using EnumerableSet for EnumerableSet.Bytes32Set;
    using EnumerableSet for EnumerableSet.AddressSet;
    using EnumerableSet for EnumerableSet.UintSet;
    using RateLimiter for RateLimiter.TokenBucket;

    error CallerIsNotARampOnRouter(address caller);
    error ZeroAddressNotAllowed();
    error SenderNotAllowed(address sender);
    error AllowListNotEnabled();
    error NonExistentChain(uint64 remoteChainSelector);
    error ChainNotAllowed(uint64 remoteChainSelector);
    error CursedByRMN();
    error ChainAlreadyExists(uint64 chainSelector);
    error InvalidSourcePoolAddress(bytes sourcePoolAddress);
    error InvalidToken(address token);
    error Unauthorized(address caller);
    error PoolAlreadyAdded(uint64 remoteChainSelector, bytes remotePoolAddress);
    error InvalidRemotePoolForChain(
        uint64 remoteChainSelector,
        bytes remotePoolAddress
    );
    error InvalidRemoteChainDecimals(bytes sourcePoolData);
    error OverflowDetected(
        uint8 remoteDecimals,
        uint8 localDecimals,
        uint256 remoteAmount
    );
    error InvalidDecimalArgs(uint8 expected, uint8 actual);

    event Locked(address indexed sender, uint256 amount);
    event Burned(address indexed sender, uint256 amount);
    event Released(
        address indexed sender,
        address indexed recipient,
        uint256 amount
    );
    event Minted(
        address indexed sender,
        address indexed recipient,
        uint256 amount
    );
    event ChainAdded(
        uint64 remoteChainSelector,
        bytes remoteToken,
        RateLimiter.Config outboundRateLimiterConfig,
        RateLimiter.Config inboundRateLimiterConfig
    );
    event ChainConfigured(
        uint64 remoteChainSelector,
        RateLimiter.Config outboundRateLimiterConfig,
        RateLimiter.Config inboundRateLimiterConfig
    );
    event ChainRemoved(uint64 remoteChainSelector);
    event RemotePoolAdded(
        uint64 indexed remoteChainSelector,
        bytes remotePoolAddress
    );
    event RemotePoolRemoved(
        uint64 indexed remoteChainSelector,
        bytes remotePoolAddress
    );
    event AllowListAdd(address sender);
    event AllowListRemove(address sender);
    event RouterUpdated(address oldRouter, address newRouter);
    event RateLimitAdminSet(address rateLimitAdmin);

    struct ChainUpdate {
        uint64 remoteChainSelector; // Remote chain selector
        bytes[] remotePoolAddresses; // Address of the remote pool, ABI encoded in the case of a remote EVM chain.
        bytes remoteTokenAddress; // Address of the remote token, ABI encoded in the case of a remote EVM chain.
        RateLimiter.Config outboundRateLimiterConfig; // Outbound rate limited config, meaning the rate limits for all of the onRamps for the given chain
        RateLimiter.Config inboundRateLimiterConfig; // Inbound rate limited config, meaning the rate limits for all of the offRamps for the given chain
    }

    struct RemoteChainConfig {
        RateLimiter.TokenBucket outboundRateLimiterConfig; // Outbound rate limited config, meaning the rate limits for all of the onRamps for the given chain
        RateLimiter.TokenBucket inboundRateLimiterConfig; // Inbound rate limited config, meaning the rate limits for all of the offRamps for the given chain
        bytes remoteTokenAddress; // Address of the remote token, ABI encoded in the case of a remote EVM chain.
        EnumerableSet.Bytes32Set remotePools; // Set of remote pool hashes, ABI encoded in the case of a remote EVM chain.
    }

    /// @dev The bridgeable token that is managed by this pool. Pools could support multiple tokens at the same time if
    /// required, but this implementation only supports one token.
    IERC20 internal immutable i_token;
    /// @dev The number of decimals of the token managed by this pool.
    uint8 internal immutable i_tokenDecimals;
    /// @dev The address of the RMN proxy
    address internal immutable i_rmnProxy;
    /// @dev The immutable flag that indicates if the pool is access-controlled.
    bool internal immutable i_allowlistEnabled;
    /// @dev A set of addresses allowed to trigger lockOrBurn as original senders.
    /// Only takes effect if i_allowlistEnabled is true.
    /// This can be used to ensure only token-issuer specified addresses can move tokens.
    EnumerableSet.AddressSet internal s_allowlist;
    /// @dev The address of the router
    IRouter internal s_router;
    /// @dev A set of allowed chain selectors. We want the allowlist to be enumerable to
    /// be able to quickly determine (without parsing logs) who can access the pool.
    /// @dev The chain selectors are in uint256 format because of the EnumerableSet implementation.
    EnumerableSet.UintSet internal s_remoteChainSelectors;
    mapping(uint64 remoteChainSelector => RemoteChainConfig)
        internal s_remoteChainConfigs;
    /// @notice A mapping of hashed pool addresses to their unhashed form. This is used to be able to find the actually
    /// configured pools and not just their hashed versions.
    mapping(bytes32 poolAddressHash => bytes poolAddress)
        internal s_remotePoolAddresses;
    /// @notice The address of the rate limiter admin.
    /// @dev Can be address(0) if none is configured.
    address internal s_rateLimitAdmin;

    constructor(
        IERC20 token,
        uint8 localTokenDecimals,
        address[] memory allowlist,
        address rmnProxy,
        address router
    ) {
        if (
            address(token) == address(0) ||
            router == address(0) ||
            rmnProxy == address(0)
        ) revert ZeroAddressNotAllowed();
        i_token = token;
        i_rmnProxy = rmnProxy;

        try IERC20Metadata(address(token)).decimals() returns (
            uint8 actualTokenDecimals
        ) {
            if (localTokenDecimals != actualTokenDecimals) {
                revert InvalidDecimalArgs(
                    localTokenDecimals,
                    actualTokenDecimals
                );
            }
        } catch {
            // The decimals function doesn't exist, which is possible since it's optional in the ERC20 spec. We skip the check and
            // assume the supplied token decimals are correct.
        }
        i_tokenDecimals = localTokenDecimals;

        s_router = IRouter(router);

        // Pool can be set as permissioned or permissionless at deployment time only to save hot-path gas.
        i_allowlistEnabled = allowlist.length > 0;
        if (i_allowlistEnabled) {
            _applyAllowListUpdates(new address[](0), allowlist);
        }
    }

    /// @inheritdoc IPoolV1
    function isSupportedToken(
        address token
    ) public view virtual returns (bool) {
        return token == address(i_token);
    }

    /// @notice Gets the IERC20 token that this pool can lock or burn.
    /// @return token The IERC20 token representation.
    function getToken() public view returns (IERC20 token) {
        return i_token;
    }

    /// @notice Get RMN proxy address
    /// @return rmnProxy Address of RMN proxy
    function getRmnProxy() public view returns (address rmnProxy) {
        return i_rmnProxy;
    }

    /// @notice Gets the pool's Router
    /// @return router The pool's Router
    function getRouter() public view returns (address router) {
        return address(s_router);
    }

    /// @notice Sets the pool's Router
    /// @param newRouter The new Router
    function setRouter(address newRouter) public onlyOwner {
        if (newRouter == address(0)) revert ZeroAddressNotAllowed();
        address oldRouter = address(s_router);
        s_router = IRouter(newRouter);

        emit RouterUpdated(oldRouter, newRouter);
    }

    /// @notice Signals which version of the pool interface is supported
    function supportsInterface(
        bytes4 interfaceId
    ) public pure virtual override returns (bool) {
        return
            interfaceId == Pool.CCIP_POOL_V1 ||
            interfaceId == type(IPoolV1).interfaceId ||
            interfaceId == type(IERC165).interfaceId;
    }

    // ================================================================
    // │                         Validation                           │
    // ================================================================

    /// @notice Validates the lock or burn input for correctness on
    /// - token to be locked or burned
    /// - RMN curse status
    /// - allowlist status
    /// - if the sender is a valid onRamp
    /// - rate limit status
    /// @param lockOrBurnIn The input to validate.
    /// @dev This function should always be called before executing a lock or burn. Not doing so would allow
    /// for various exploits.
    function _validateLockOrBurn(
        Pool.LockOrBurnInV1 calldata lockOrBurnIn
    ) internal {
        if (!isSupportedToken(lockOrBurnIn.localToken))
            revert InvalidToken(lockOrBurnIn.localToken);
        if (
            IRMN(i_rmnProxy).isCursed(
                bytes16(uint128(lockOrBurnIn.remoteChainSelector))
            )
        ) revert CursedByRMN();
        _checkAllowList(lockOrBurnIn.originalSender);

        _onlyOnRamp(lockOrBurnIn.remoteChainSelector);
        _consumeOutboundRateLimit(
            lockOrBurnIn.remoteChainSelector,
            lockOrBurnIn.amount
        );
    }

    /// @notice Validates the release or mint input for correctness on
    /// - token to be released or minted
    /// - RMN curse status
    /// - if the sender is a valid offRamp
    /// - if the source pool is valid
    /// - rate limit status
    /// @param releaseOrMintIn The input to validate.
    /// @dev This function should always be called before executing a release or mint. Not doing so would allow
    /// for various exploits.
    function _validateReleaseOrMint(
        Pool.ReleaseOrMintInV1 calldata releaseOrMintIn
    ) internal {
        if (!isSupportedToken(releaseOrMintIn.localToken))
            revert InvalidToken(releaseOrMintIn.localToken);
        if (
            IRMN(i_rmnProxy).isCursed(
                bytes16(uint128(releaseOrMintIn.remoteChainSelector))
            )
        ) revert CursedByRMN();
        _onlyOffRamp(releaseOrMintIn.remoteChainSelector);

        // Validates that the source pool address is configured on this pool.
        if (
            !isRemotePool(
                releaseOrMintIn.remoteChainSelector,
                releaseOrMintIn.sourcePoolAddress
            )
        ) {
            revert InvalidSourcePoolAddress(releaseOrMintIn.sourcePoolAddress);
        }

        _consumeInboundRateLimit(
            releaseOrMintIn.remoteChainSelector,
            releaseOrMintIn.amount
        );
    }

    // ================================================================
    // │                      Token decimals                          │
    // ================================================================

    /// @notice Gets the IERC20 token decimals on the local chain.
    function getTokenDecimals() public view virtual returns (uint8 decimals) {
        return i_tokenDecimals;
    }

    function _encodeLocalDecimals()
        internal
        view
        virtual
        returns (bytes memory)
    {
        return abi.encode(i_tokenDecimals);
    }

    function _parseRemoteDecimals(
        bytes memory sourcePoolData
    ) internal view virtual returns (uint8) {
        // Fallback to the local token decimals if the source pool data is empty. This allows for backwards compatibility.
        if (sourcePoolData.length == 0) {
            return i_tokenDecimals;
        }
        if (sourcePoolData.length != 32) {
            revert InvalidRemoteChainDecimals(sourcePoolData);
        }
        uint256 remoteDecimals = abi.decode(sourcePoolData, (uint256));
        if (remoteDecimals > type(uint8).max) {
            revert InvalidRemoteChainDecimals(sourcePoolData);
        }
        return uint8(remoteDecimals);
    }

    /// @notice Calculates the local amount based on the remote amount and decimals.
    /// @param remoteAmount The amount on the remote chain.
    /// @param remoteDecimals The decimals of the token on the remote chain.
    /// @return The local amount.
    /// @dev This function protects against overflows. If there is a transaction that hits the overflow check, it is
    /// probably incorrect as that means the amount cannot be represented on this chain. If the local decimals have been
    /// wrongly configured, the token issuer could redeploy the pool with the correct decimals and manually re-execute the
    /// CCIP tx to fix the issue.
    function _calculateLocalAmount(
        uint256 remoteAmount,
        uint8 remoteDecimals
    ) internal view virtual returns (uint256) {
        if (remoteDecimals == i_tokenDecimals) {
            return remoteAmount;
        }
        if (remoteDecimals > i_tokenDecimals) {
            uint8 decimalsDiff = remoteDecimals - i_tokenDecimals;
            if (decimalsDiff > 77) {
                // This is a safety check to prevent overflow in the next calculation.
                revert OverflowDetected(
                    remoteDecimals,
                    i_tokenDecimals,
                    remoteAmount
                );
            }
            // Solidity rounds down so there is no risk of minting more tokens than the remote chain sent.
            return remoteAmount / (10 ** decimalsDiff);
        }

        // This is a safety check to prevent overflow in the next calculation.
        // More than 77 would never fit in a uint256 and would cause an overflow. We also check if the resulting amount
        // would overflow.
        uint8 diffDecimals = i_tokenDecimals - remoteDecimals;
        if (
            diffDecimals > 77 ||
            remoteAmount > type(uint256).max / (10 ** diffDecimals)
        ) {
            revert OverflowDetected(
                remoteDecimals,
                i_tokenDecimals,
                remoteAmount
            );
        }

        return remoteAmount * (10 ** diffDecimals);
    }

    // ================================================================
    // │                     Chain permissions                        │
    // ================================================================

    /// @notice Gets the pool address on the remote chain.
    /// @param remoteChainSelector Remote chain selector.
    /// @dev To support non-evm chains, this value is encoded into bytes
    function getRemotePools(
        uint64 remoteChainSelector
    ) public view returns (bytes[] memory) {
        bytes32[] memory remotePoolHashes = s_remoteChainConfigs[
            remoteChainSelector
        ].remotePools.values();

        bytes[] memory remotePools = new bytes[](remotePoolHashes.length);
        for (uint256 i = 0; i < remotePoolHashes.length; ++i) {
            remotePools[i] = s_remotePoolAddresses[remotePoolHashes[i]];
        }

        return remotePools;
    }

    /// @notice Checks if the pool address is configured on the remote chain.
    /// @param remoteChainSelector Remote chain selector.
    /// @param remotePoolAddress The address of the remote pool.
    function isRemotePool(
        uint64 remoteChainSelector,
        bytes calldata remotePoolAddress
    ) public view returns (bool) {
        return
            s_remoteChainConfigs[remoteChainSelector].remotePools.contains(
                keccak256(remotePoolAddress)
            );
    }

    /// @notice Gets the token address on the remote chain.
    /// @param remoteChainSelector Remote chain selector.
    /// @dev To support non-evm chains, this value is encoded into bytes
    function getRemoteToken(
        uint64 remoteChainSelector
    ) public view returns (bytes memory) {
        return s_remoteChainConfigs[remoteChainSelector].remoteTokenAddress;
    }

    /// @notice Adds a remote pool for a given chain selector. This could be due to a pool being upgraded on the remote
    /// chain. We don't simply want to replace the old pool as there could still be valid inflight messages from the old
    /// pool. This function allows for multiple pools to be added for a single chain selector.
    /// @param remoteChainSelector The remote chain selector for which the remote pool address is being added.
    /// @param remotePoolAddress The address of the new remote pool.
    function addRemotePool(
        uint64 remoteChainSelector,
        bytes calldata remotePoolAddress
    ) external onlyOwner {
        if (!isSupportedChain(remoteChainSelector))
            revert NonExistentChain(remoteChainSelector);

        _setRemotePool(remoteChainSelector, remotePoolAddress);
    }

    /// @notice Removes the remote pool address for a given chain selector.
    /// @dev All inflight txs from the remote pool will be rejected after it is removed. To ensure no loss of funds, there
    /// should be no inflight txs from the given pool.
    function removeRemotePool(
        uint64 remoteChainSelector,
        bytes calldata remotePoolAddress
    ) external onlyOwner {
        if (!isSupportedChain(remoteChainSelector))
            revert NonExistentChain(remoteChainSelector);

        if (
            !s_remoteChainConfigs[remoteChainSelector].remotePools.remove(
                keccak256(remotePoolAddress)
            )
        ) {
            revert InvalidRemotePoolForChain(
                remoteChainSelector,
                remotePoolAddress
            );
        }

        emit RemotePoolRemoved(remoteChainSelector, remotePoolAddress);
    }

    /// @inheritdoc IPoolV1
    function isSupportedChain(
        uint64 remoteChainSelector
    ) public view returns (bool) {
        return s_remoteChainSelectors.contains(remoteChainSelector);
    }

    /// @notice Get list of allowed chains
    /// @return list of chains.
    function getSupportedChains() public view returns (uint64[] memory) {
        uint256[] memory uint256ChainSelectors = s_remoteChainSelectors
            .values();
        uint64[] memory chainSelectors = new uint64[](
            uint256ChainSelectors.length
        );
        for (uint256 i = 0; i < uint256ChainSelectors.length; ++i) {
            chainSelectors[i] = uint64(uint256ChainSelectors[i]);
        }

        return chainSelectors;
    }

    /// @notice Sets the permissions for a list of chains selectors. Actual senders for these chains
    /// need to be allowed on the Router to interact with this pool.
    /// @param remoteChainSelectorsToRemove A list of chain selectors to remove.
    /// @param chainsToAdd A list of chains and their new permission status & rate limits. Rate limits
    /// are only used when the chain is being added through `allowed` being true.
    /// @dev Only callable by the owner
    function applyChainUpdates(
        uint64[] calldata remoteChainSelectorsToRemove,
        ChainUpdate[] calldata chainsToAdd
    ) external virtual onlyOwner {
        for (uint256 i = 0; i < remoteChainSelectorsToRemove.length; ++i) {
            uint64 remoteChainSelectorToRemove = remoteChainSelectorsToRemove[
                i
            ];
            // If the chain doesn't exist, revert
            if (!s_remoteChainSelectors.remove(remoteChainSelectorToRemove)) {
                revert NonExistentChain(remoteChainSelectorToRemove);
            }

            // Remove all remote pool hashes for the chain
            bytes32[] memory remotePools = s_remoteChainConfigs[
                remoteChainSelectorToRemove
            ].remotePools.values();
            for (uint256 j = 0; j < remotePools.length; ++j) {
                s_remoteChainConfigs[remoteChainSelectorToRemove]
                    .remotePools
                    .remove(remotePools[j]);
            }

            delete s_remoteChainConfigs[remoteChainSelectorToRemove];

            emit ChainRemoved(remoteChainSelectorToRemove);
        }

        for (uint256 i = 0; i < chainsToAdd.length; ++i) {
            ChainUpdate memory newChain = chainsToAdd[i];
            RateLimiter._validateTokenBucketConfig(
                newChain.outboundRateLimiterConfig,
                false
            );
            RateLimiter._validateTokenBucketConfig(
                newChain.inboundRateLimiterConfig,
                false
            );

            if (newChain.remoteTokenAddress.length == 0) {
                revert ZeroAddressNotAllowed();
            }

            // If the chain already exists, revert
            if (!s_remoteChainSelectors.add(newChain.remoteChainSelector)) {
                revert ChainAlreadyExists(newChain.remoteChainSelector);
            }

            RemoteChainConfig storage remoteChainConfig = s_remoteChainConfigs[
                newChain.remoteChainSelector
            ];

            remoteChainConfig.outboundRateLimiterConfig = RateLimiter
                .TokenBucket({
                    rate: newChain.outboundRateLimiterConfig.rate,
                    capacity: newChain.outboundRateLimiterConfig.capacity,
                    tokens: newChain.outboundRateLimiterConfig.capacity,
                    lastUpdated: uint32(block.timestamp),
                    isEnabled: newChain.outboundRateLimiterConfig.isEnabled
                });
            remoteChainConfig.inboundRateLimiterConfig = RateLimiter
                .TokenBucket({
                    rate: newChain.inboundRateLimiterConfig.rate,
                    capacity: newChain.inboundRateLimiterConfig.capacity,
                    tokens: newChain.inboundRateLimiterConfig.capacity,
                    lastUpdated: uint32(block.timestamp),
                    isEnabled: newChain.inboundRateLimiterConfig.isEnabled
                });
            remoteChainConfig.remoteTokenAddress = newChain.remoteTokenAddress;

            for (uint256 j = 0; j < newChain.remotePoolAddresses.length; ++j) {
                _setRemotePool(
                    newChain.remoteChainSelector,
                    newChain.remotePoolAddresses[j]
                );
            }

            emit ChainAdded(
                newChain.remoteChainSelector,
                newChain.remoteTokenAddress,
                newChain.outboundRateLimiterConfig,
                newChain.inboundRateLimiterConfig
            );
        }
    }

    /// @notice Adds a pool address to the allowed remote token pools for a particular chain.
    /// @param remoteChainSelector The remote chain selector for which the remote pool address is being added.
    /// @param remotePoolAddress The address of the new remote pool.
    function _setRemotePool(
        uint64 remoteChainSelector,
        bytes memory remotePoolAddress
    ) internal {
        if (remotePoolAddress.length == 0) {
            revert ZeroAddressNotAllowed();
        }

        bytes32 poolHash = keccak256(remotePoolAddress);

        // Check if the pool already exists.
        if (
            !s_remoteChainConfigs[remoteChainSelector].remotePools.add(poolHash)
        ) {
            revert PoolAlreadyAdded(remoteChainSelector, remotePoolAddress);
        }

        // Add the pool to the mapping to be able to un-hash it later.
        s_remotePoolAddresses[poolHash] = remotePoolAddress;

        emit RemotePoolAdded(remoteChainSelector, remotePoolAddress);
    }

    // ================================================================
    // │                        Rate limiting                         │
    // ================================================================

    /// @dev The inbound rate limits should be slightly higher than the outbound rate limits. This is because many chains
    /// finalize blocks in batches. CCIP also commits messages in batches: the commit plugin bundles multiple messages in
    /// a single merkle root.
    /// Imagine the following scenario.
    /// - Chain A has an inbound and outbound rate limit of 100 tokens capacity and 1 token per second refill rate.
    /// - Chain B has an inbound and outbound rate limit of 100 tokens capacity and 1 token per second refill rate.
    ///
    /// At time 0:
    /// - Chain A sends 100 tokens to Chain B.
    /// At time 5:
    /// - Chain A sends 5 tokens to Chain B.
    /// At time 6:
    /// The epoch that contains blocks [0-5] is finalized.
    /// Both transactions will be included in the same merkle root and become executable at the same time. This means
    /// the token pool on chain B requires a capacity of 105 to successfully execute both messages at the same time.
    /// The exact additional capacity required depends on the refill rate and the size of the source chain epochs and the
    /// CCIP round time. For simplicity, a 5-10% buffer should be sufficient in most cases.

    /// @notice Sets the rate limiter admin address.
    /// @dev Only callable by the owner.
    /// @param rateLimitAdmin The new rate limiter admin address.
    function setRateLimitAdmin(address rateLimitAdmin) external onlyOwner {
        s_rateLimitAdmin = rateLimitAdmin;
        emit RateLimitAdminSet(rateLimitAdmin);
    }

    /// @notice Gets the rate limiter admin address.
    function getRateLimitAdmin() external view returns (address) {
        return s_rateLimitAdmin;
    }

    /// @notice Consumes outbound rate limiting capacity in this pool
    function _consumeOutboundRateLimit(
        uint64 remoteChainSelector,
        uint256 amount
    ) internal {
        s_remoteChainConfigs[remoteChainSelector]
            .outboundRateLimiterConfig
            ._consume(amount, address(i_token));
    }

    /// @notice Consumes inbound rate limiting capacity in this pool
    function _consumeInboundRateLimit(
        uint64 remoteChainSelector,
        uint256 amount
    ) internal {
        s_remoteChainConfigs[remoteChainSelector]
            .inboundRateLimiterConfig
            ._consume(amount, address(i_token));
    }

    /// @notice Gets the token bucket with its values for the block it was requested at.
    /// @return The token bucket.
    function getCurrentOutboundRateLimiterState(
        uint64 remoteChainSelector
    ) external view returns (RateLimiter.TokenBucket memory) {
        return
            s_remoteChainConfigs[remoteChainSelector]
                .outboundRateLimiterConfig
                ._currentTokenBucketState();
    }

    /// @notice Gets the token bucket with its values for the block it was requested at.
    /// @return The token bucket.
    function getCurrentInboundRateLimiterState(
        uint64 remoteChainSelector
    ) external view returns (RateLimiter.TokenBucket memory) {
        return
            s_remoteChainConfigs[remoteChainSelector]
                .inboundRateLimiterConfig
                ._currentTokenBucketState();
    }

    /// @notice Sets the chain rate limiter config.
    /// @param remoteChainSelector The remote chain selector for which the rate limits apply.
    /// @param outboundConfig The new outbound rate limiter config, meaning the onRamp rate limits for the given chain.
    /// @param inboundConfig The new inbound rate limiter config, meaning the offRamp rate limits for the given chain.
    function setChainRateLimiterConfig(
        uint64 remoteChainSelector,
        RateLimiter.Config memory outboundConfig,
        RateLimiter.Config memory inboundConfig
    ) external {
        if (msg.sender != s_rateLimitAdmin && msg.sender != owner())
            revert Unauthorized(msg.sender);

        _setRateLimitConfig(remoteChainSelector, outboundConfig, inboundConfig);
    }

    function _setRateLimitConfig(
        uint64 remoteChainSelector,
        RateLimiter.Config memory outboundConfig,
        RateLimiter.Config memory inboundConfig
    ) internal {
        if (!isSupportedChain(remoteChainSelector))
            revert NonExistentChain(remoteChainSelector);
        RateLimiter._validateTokenBucketConfig(outboundConfig, false);
        s_remoteChainConfigs[remoteChainSelector]
            .outboundRateLimiterConfig
            ._setTokenBucketConfig(outboundConfig);
        RateLimiter._validateTokenBucketConfig(inboundConfig, false);
        s_remoteChainConfigs[remoteChainSelector]
            .inboundRateLimiterConfig
            ._setTokenBucketConfig(inboundConfig);
        emit ChainConfigured(
            remoteChainSelector,
            outboundConfig,
            inboundConfig
        );
    }

    // ================================================================
    // │                           Access                             │
    // ================================================================

    /// @notice Checks whether remote chain selector is configured on this contract, and if the msg.sender
    /// is a permissioned onRamp for the given chain on the Router.
    function _onlyOnRamp(uint64 remoteChainSelector) internal view {
        if (!isSupportedChain(remoteChainSelector))
            revert ChainNotAllowed(remoteChainSelector);
        if (!(msg.sender == s_router.getOnRamp(remoteChainSelector)))
            revert CallerIsNotARampOnRouter(msg.sender);
    }

    /// @notice Checks whether remote chain selector is configured on this contract, and if the msg.sender
    /// is a permissioned offRamp for the given chain on the Router.
    function _onlyOffRamp(uint64 remoteChainSelector) internal view {
        if (!isSupportedChain(remoteChainSelector))
            revert ChainNotAllowed(remoteChainSelector);
        if (!s_router.isOffRamp(remoteChainSelector, msg.sender))
            revert CallerIsNotARampOnRouter(msg.sender);
    }

    // ================================================================
    // │                          Allowlist                           │
    // ================================================================

    function _checkAllowList(address sender) internal view {
        if (i_allowlistEnabled) {
            if (!s_allowlist.contains(sender)) {
                revert SenderNotAllowed(sender);
            }
        }
    }

    /// @notice Gets whether the allowlist functionality is enabled.
    /// @return true is enabled, false if not.
    function getAllowListEnabled() external view returns (bool) {
        return i_allowlistEnabled;
    }

    /// @notice Gets the allowed addresses.
    /// @return The allowed addresses.
    function getAllowList() external view returns (address[] memory) {
        return s_allowlist.values();
    }

    /// @notice Apply updates to the allow list.
    /// @param removes The addresses to be removed.
    /// @param adds The addresses to be added.
    function applyAllowListUpdates(
        address[] calldata removes,
        address[] calldata adds
    ) external onlyOwner {
        _applyAllowListUpdates(removes, adds);
    }

    /// @notice Internal version of applyAllowListUpdates to allow for reuse in the constructor.
    function _applyAllowListUpdates(
        address[] memory removes,
        address[] memory adds
    ) internal {
        if (!i_allowlistEnabled) revert AllowListNotEnabled();

        for (uint256 i = 0; i < removes.length; ++i) {
            address toRemove = removes[i];
            if (s_allowlist.remove(toRemove)) {
                emit AllowListRemove(toRemove);
            }
        }
        for (uint256 i = 0; i < adds.length; ++i) {
            address toAdd = adds[i];
            if (toAdd == address(0)) {
                continue;
            }
            if (s_allowlist.add(toAdd)) {
                emit AllowListAdd(toAdd);
            }
        }
    }
}

pragma solidity ^0.8.0;

interface IBurnMintERC20 is IERC20 {
    /// @notice Mints new tokens for a given address.
    /// @param account The address to mint the new tokens to.
    /// @param amount The number of tokens to be minted.
    /// @dev this function increases the total supply.
    function mint(address account, uint256 amount) external;

    /// @notice Burns tokens from the sender.
    /// @param amount The number of tokens to be burned.
    /// @dev this function decreases the total supply.
    function burn(uint256 amount) external;

    /// @notice Burns tokens from a given address..
    /// @param account The address to burn tokens from.
    /// @param amount The number of tokens to be burned.
    /// @dev this function decreases the total supply.
    function burn(address account, uint256 amount) external;

    /// @notice Burns tokens from a given address..
    /// @param account The address to burn tokens from.
    /// @param amount The number of tokens to be burned.
    /// @dev this function decreases the total supply.
    function burnFrom(address account, uint256 amount) external;
}

pragma solidity 0.8.24;

abstract contract BurnMintTokenPoolAbstract is TokenPool {
    /// @notice Contains the specific burn call for a pool.
    /// @dev overriding this method allows us to create pools with different burn signatures
    /// without duplicating the underlying logic.
    function _burn(uint256 amount) internal virtual;

    /// @notice Burn the token in the pool
    /// @dev The _validateLockOrBurn check is an essential security check
    function lockOrBurn(
        Pool.LockOrBurnInV1 calldata lockOrBurnIn
    ) external virtual override returns (Pool.LockOrBurnOutV1 memory) {
        _validateLockOrBurn(lockOrBurnIn);

        _burn(lockOrBurnIn.amount);

        emit Burned(msg.sender, lockOrBurnIn.amount);

        return
            Pool.LockOrBurnOutV1({
                destTokenAddress: getRemoteToken(
                    lockOrBurnIn.remoteChainSelector
                ),
                destPoolData: _encodeLocalDecimals()
            });
    }

    /// @notice Mint tokens from the pool to the recipient
    /// @dev The _validateReleaseOrMint check is an essential security check
    function releaseOrMint(
        Pool.ReleaseOrMintInV1 calldata releaseOrMintIn
    ) external virtual override returns (Pool.ReleaseOrMintOutV1 memory) {
        _validateReleaseOrMint(releaseOrMintIn);

        // Calculate the local amount
        uint256 localAmount = _calculateLocalAmount(
            releaseOrMintIn.amount,
            _parseRemoteDecimals(releaseOrMintIn.sourcePoolData)
        );

        // Mint to the receiver
        IBurnMintERC20(address(i_token)).mint(
            releaseOrMintIn.receiver,
            localAmount
        );

        emit Minted(msg.sender, releaseOrMintIn.receiver, localAmount);

        return Pool.ReleaseOrMintOutV1({destinationAmount: localAmount});
    }
}

pragma solidity ^0.8.0;

interface ITypeAndVersion {
    function typeAndVersion() external pure returns (string memory);
}

pragma solidity 0.8.24;

/// @notice This pool mints and burns a 3rd-party token.
/// @dev Pool whitelisting mode is set in the constructor and cannot be modified later.
/// It either accepts any address as originalSender, or only accepts whitelisted originalSender.
/// The only way to change whitelisting mode is to deploy a new pool.
/// If that is expected, please make sure the token's burner/minter roles are adjustable.
/// @dev This contract is a variant of BurnMintTokenPool that uses `burn(amount)`.
contract BurnMintTokenPool is BurnMintTokenPoolAbstract, ITypeAndVersion {
    string public constant override typeAndVersion = 'BurnMintTokenPool 1.5.1';

    constructor(
        IBurnMintERC20 token,
        uint8 localTokenDecimals,
        address[] memory allowlist,
        address rmnProxy,
        address router
    ) TokenPool(token, localTokenDecimals, allowlist, rmnProxy, router) {}

    /// @inheritdoc BurnMintTokenPoolAbstract
    function _burn(uint256 amount) internal virtual override {
        IBurnMintERC20(address(i_token)).burn(amount);
    }
}