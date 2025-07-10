// SPDX-License-Identifier: BUSL-1.1
pragma solidity =0.8.26 ^0.8.0 ^0.8.20;

// src/interfaces/IPausable.sol

/// @notice Interface for pausable & unpausable contracts
interface IPausable {
  /// @notice This function pauses the contract
  /// @dev Sets the pause flag to true
  function emergencyPause() external;

  /// @notice This function unpauses the contract
  /// @dev Sets the pause flag to false
  function emergencyUnpause() external;
}

// src/libraries/Common.sol

library Common {
  /// @notice Parameters to transfer assets
  struct AssetAmount {
    address asset; // asset address.
    uint256 amount; // Amount of assets.
  }
}

// src/libraries/EnumerableBytesSet.sol

/// @notice Library for managing sets of bytes. Reuses OpenZeppelin's EnumerableSet library logic but for bytes.
/* solhint-disable chainlink-solidity/prefix-internal-functions-with-underscore */
library EnumerableBytesSet {
  struct BytesSet {
    bytes[] _values;
    mapping(bytes value => uint256) _positions;
  }

  /// @dev Adds a value to a set. O(1).
  /// @param set The set to add the value to.
  /// @param value The value to add.
  /// @return True if the value was added to the set, false if the value was already in the set.
  function add(BytesSet storage set, bytes memory value) internal returns (bool) {
    return _add(set, value);
  }

  function _add(BytesSet storage set, bytes memory value) private returns (bool) {
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

  /// @dev Removes a value from a set. O(1).
  /// @param set The set to remove the value from.
  /// @param value The value to remove.
  /// @return True if the value was removed from the set, false if the value was not in the set.
  function remove(BytesSet storage set, bytes memory value) internal returns (bool) {
    return _remove(set, value);
  }

  function _remove(BytesSet storage set, bytes memory value) private returns (bool) {
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
        bytes memory lastValue = set._values[lastIndex];

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

  /// @dev Checks if a value is in a set. O(1).
  /// @param set The set to check the value in.
  /// @param value The value to check.
  /// @return True if the value is in the set, false otherwise.
  function contains(BytesSet storage set, bytes memory value) internal view returns (bool) {
    return _contains(set, value);
  }

  function _contains(BytesSet storage set, bytes memory value) private view returns (bool) {
    return set._positions[value] != 0;
  }

  /// @dev Returns the number of values in the set. O(1).
  /// @param set The set to count values in.
  /// @return The number of values in the set.
  function length(
    BytesSet storage set
  ) internal view returns (uint256) {
    return _length(set);
  }

  function _length(
    BytesSet storage set
  ) private view returns (uint256) {
    return set._values.length;
  }

  /// @dev Returns the value stored at position `index` in the set. O(1).
  /// Note that there are no guarantees on the ordering of values inside the array, and it may change when more values
  /// are added or removed.
  /// @dev precondition - `index` must be strictly less than {length}.
  /// @param set The set to get the value from.
  /// @param index The index to get the value at.
  /// @return The value stored at the specified index.
  function at(BytesSet storage set, uint256 index) internal view returns (bytes memory) {
    return _at(set, index);
  }

  function _at(BytesSet storage set, uint256 index) private view returns (bytes memory) {
    return set._values[index];
  }

  /// @dev Return the entire set in an array
  ///
  /// WARNING: This operation will copy the entire storage to memory, which can be quite expensive. This is designed to
  /// mostly be used by view accessors that are queried without any gas fees. Developers should keep in mind that this
  /// function has an unbounded cost, and using it as part of a state-changing function may render the function
  /// uncallable if the set grows to a point where copying to memory consumes too much gas to fit in a block.
  /// @param set The set to get the values from.
  ///
  /// @return An array containing all the values in the set.
  function values(
    BytesSet storage set
  ) internal view returns (bytes[] memory) {
    bytes[] memory store = _values(set);
    bytes[] memory result;

    assembly ("memory-safe") {
      result := store
    }

    return result;
  }

  function _values(
    BytesSet storage set
  ) private view returns (bytes[] memory) {
    return set._values;
  }
}

// src/libraries/Errors.sol

/// @notice Library for common custom errors used across multiple contracts
library Errors {
  /// @notice This error is thrown whenever a zero-address is supplied when
  /// a non-zero address is required
  error InvalidZeroAddress();
  /// @notice This error is thrown when trying to pass in an empty list as an argument
  error EmptyList();
  /// @notice This error is thrown when the data returned by the price feed is zero
  error ZeroFeedData();
  /// @notice This error is thrown when the data returned by the price feed is older than the set
  /// threshold
  error StaleFeedData();
  /// @notice This error is thrown when an unauthorized caller tries to call a function
  /// another address than that caller
  error AccessForbidden();
  /// @notice This error is thrown when passing in a zero amount as a function parameter
  error InvalidZeroAmount();
  /// @notice This error is thrown when attempting to remove an asset that is
  /// not on the allowlist
  /// @param asset The asset that is not allowlisted
  error AssetNotAllowlisted(address asset);
  /// @notice This error is thrown when trying to withdraw an asset that is allowlisted
  error AssetAllowlisted(address asset);
  /// @notice This error is thrown when a value is not updated e.g. when trying to configure a state variable the same
  /// value as the one already configured
  error ValueNotUpdated();
  /// @notice This error is thrown when setting a fee aggregator that does not support the IFeeAggregator interface
  error InvalidFeeAggregator(address feeAggregator);
}

// src/libraries/Roles.sol

/// @notice Library for payment abstraction contract roles IDs to use with the OpenZeppelin AccessControl contracts.
library Roles {
  /// @notice This is the ID for the pauser role, which is given to the addresses that can pause and
  /// the contract.
  /// @dev Hash: 0x65d7a28e3265b37a6474929f336521b332c1681b933f6cb9f3376673440d862a
  bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
  /// @notice This is the ID for the unpauser role, which is given to the addresses that can unpause
  /// the contract.
  /// @dev Hash: 0x427da25fe773164f88948d3e215c94b6554e2ed5e5f203a821c9f2f6131cf75a
  bytes32 public constant UNPAUSER_ROLE = keccak256("UNPAUSER_ROLE");
  /// @notice This is the ID for the asset admin role, which is given to the addresses that can:
  /// - Add and remove assets from the allowlist
  /// - Set the swap parameters for an asset
  /// @dev Hash: 0x5e608239aadc5f1e750186f22bbac828160fb6191c4a7b9eee6b9432b1eac59e
  bytes32 public constant ASSET_ADMIN_ROLE = keccak256("ASSET_ADMIN_ROLE");
  /// @notice This is the ID for the bridger role, which is given to addresses that are able to
  /// bridge assets
  /// @dev Hash: 0xc809a7fd521f10cdc3c068621a1c61d5fd9bb3f1502a773e53811bc248d919a8
  bytes32 public constant BRIDGER_ROLE = keccak256("BRIDGER_ROLE");
  /// @notice This is the ID for earmark manager role, which is given to addresses that are able to
  /// set earmarks
  /// @dev Hash: 0xa1ccbd74bc39a2421c04f3b35fcdea6a99019423855b3e642ec1ef8e448afb97
  bytes32 public constant EARMARK_MANAGER_ROLE = keccak256("EARMARK_MANAGER_ROLE");
  /// @notice This is the ID for the withdrawer role, which is given to addresses that are able to able to withdraw non
  /// allowlisted assets
  /// @dev Hash: 0x10dac8c06a04bec0b551627dad28bc00d6516b0caacd1c7b345fcdb5211334e4
  bytes32 public constant WITHDRAWER_ROLE = keccak256("WITHDRAWER_ROLE");
  /// @notice This is the ID for the swapper role, which is given to addresses that are able to
  /// call the transferForSwap function on the FeeAggregator contract
  /// @dev Hash: 0x724f6a44d576143e18c60911798b2b15551ca96bd8f7cb7524b8fa36253a26d8
  bytes32 public constant SWAPPER_ROLE = keccak256("SWAPPER_ROLE");
  /// @notice This is the ID for the token manager role, which is given to addresses that are able to
  /// add and remove tokens from the allowlist on the PaymentTokenOnRamp contract
  /// @dev Hash: 0x74f7a545c65c11839a48d7453738b30c295408df2d944516167556759ddc6d06
  bytes32 public constant TOKEN_MANAGER_ROLE = keccak256("TOKEN_MANAGER_ROLE");
}

// src/vendor/@chainlink/contracts/src/v0.8/shared/interfaces/ITypeAndVersion.sol

interface ITypeAndVersion {
  function typeAndVersion() external pure returns (string memory);
}

// src/vendor/@chainlink/contracts/src/v0.8/shared/interfaces/IWERC20.sol

interface IWERC20 {
  function deposit() external payable;

  function withdraw(
    uint256
  ) external;
}

// src/vendor/@chainlink/contracts/src/v0.8/shared/token/ERC677/IERC677Receiver.sol

interface IERC677Receiver {
  function onTokenTransfer(address sender, uint256 amount, bytes calldata data) external;
}

// src/vendor/@chainlink/contracts-ccip/src/v0.8/ccip/interfaces/automation/ILinkAvailable.sol

/// @notice Implement this contract so that a keeper-compatible contract can monitor
/// and fund the implementation contract with LINK if it falls below a defined threshold.
interface ILinkAvailable {
  function linkAvailableForPayment() external view returns (int256 availableBalance);
}

// src/vendor/@chainlink/contracts-ccip/src/v0.8/ccip/libraries/Client.sol

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
    bytes extraArgs; // Populate this with _argsToBytes(EVMExtraArgsV1)
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
}

// src/vendor/@openzeppelin/contracts/access/IAccessControl.sol

// OpenZeppelin Contracts (last updated v5.0.0) (access/IAccessControl.sol)

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
  function getRoleAdmin(
    bytes32 role
  ) external view returns (bytes32);

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

// src/vendor/@openzeppelin/contracts/interfaces/IERC5313.sol

// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC5313.sol)

/**
 * @dev Interface for the Light Contract Ownership Standard.
 *
 * A standardized minimal interface required to identify an account that controls a contract
 */
interface IERC5313 {
  /**
   * @dev Gets the address of the owner.
   */
  function owner() external view returns (address);
}

// src/vendor/@openzeppelin/contracts/token/ERC20/IERC20.sol

// OpenZeppelin Contracts (last updated v5.0.0) (token/ERC20/IERC20.sol)

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
  function balanceOf(
    address account
  ) external view returns (uint256);

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

// src/vendor/@openzeppelin/contracts/token/ERC20/extensions/IERC20Permit.sol

// OpenZeppelin Contracts (last updated v5.0.0) (token/ERC20/extensions/IERC20Permit.sol)

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
interface IERC20Permit {
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
  function nonces(
    address owner
  ) external view returns (uint256);

  /**
   * @dev Returns the domain separator used in the encoding of the signature for {permit}, as defined by {EIP712}.
   */
  // solhint-disable-next-line func-name-mixedcase
  function DOMAIN_SEPARATOR() external view returns (bytes32);
}

// src/vendor/@openzeppelin/contracts/utils/Address.sol

// OpenZeppelin Contracts (last updated v5.0.0) (utils/Address.sol)

/**
 * @dev Collection of functions related to the address type
 */
library Address {
  /**
   * @dev The ETH balance of the account is not enough to perform the operation.
   */
  error AddressInsufficientBalance(address account);

  /**
   * @dev There's no code at `target` (it is not a contract).
   */
  error AddressEmptyCode(address target);

  /**
   * @dev A call to an address target failed. The target may have reverted.
   */
  error FailedInnerCall();

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
   * https://solidity.readthedocs.io/en/v0.8.20/security-considerations.html#use-the-checks-effects-interactions-pattern[checks-effects-interactions
   * pattern].
   */
  function sendValue(address payable recipient, uint256 amount) internal {
    if (address(this).balance < amount) {
      revert AddressInsufficientBalance(address(this));
    }

    (bool success,) = recipient.call{value: amount}("");
    if (!success) {
      revert FailedInnerCall();
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
   * {FailedInnerCall} error.
   *
   * Returns the raw returned data. To convert to the expected return value,
   * use
   * https://solidity.readthedocs.io/en/latest/units-and-global-variables.html?highlight=abi.decode#abi-encoding-and-decoding-functions[`abi.decode`].
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
      revert AddressInsufficientBalance(address(this));
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
   * was not a contract or bubbling up the revert reason (falling back to {FailedInnerCall}) in case of an
   * unsuccessful call.
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
   * revert reason or with a default {FailedInnerCall} error.
   */
  function verifyCallResult(bool success, bytes memory returndata) internal pure returns (bytes memory) {
    if (!success) {
      _revert(returndata);
    } else {
      return returndata;
    }
  }

  /**
   * @dev Reverts with returndata if present. Otherwise reverts with {FailedInnerCall}.
   */
  function _revert(
    bytes memory returndata
  ) private pure {
    // Look for revert reason and bubble it up if present
    if (returndata.length > 0) {
      // The easiest way to bubble the revert reason is using memory via assembly
      /// @solidity memory-safe-assembly
      assembly {
        let returndata_size := mload(returndata)
        revert(add(32, returndata), returndata_size)
      }
    } else {
      revert FailedInnerCall();
    }
  }
}

// src/vendor/@openzeppelin/contracts/utils/Context.sol

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

// src/vendor/@openzeppelin/contracts/utils/introspection/IERC165.sol

// OpenZeppelin Contracts (last updated v5.0.0) (utils/introspection/IERC165.sol)

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
  function supportsInterface(
    bytes4 interfaceId
  ) external view returns (bool);
}

// src/vendor/@openzeppelin/contracts/utils/math/Math.sol

// OpenZeppelin Contracts (last updated v5.0.0) (utils/math/Math.sol)

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
  function sqrt(
    uint256 a
  ) internal pure returns (uint256) {
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
  function log2(
    uint256 value
  ) internal pure returns (uint256) {
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
  function log10(
    uint256 value
  ) internal pure returns (uint256) {
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
  function log256(
    uint256 value
  ) internal pure returns (uint256) {
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
  function unsignedRoundsUp(
    Rounding rounding
  ) internal pure returns (bool) {
    return uint8(rounding) % 2 == 1;
  }
}

// src/vendor/@openzeppelin/contracts/utils/math/SafeCast.sol

// OpenZeppelin Contracts (last updated v5.0.0) (utils/math/SafeCast.sol)
// This file was procedurally generated from scripts/generate/templates/SafeCast.js.

/**
 * @dev Wrappers over Solidity's uintXX/intXX casting operators with added overflow
 * checks.
 *
 * Downcasting from uint256/int256 in Solidity does not revert on overflow. This can
 * easily result in undesired exploitation or bugs, since developers usually
 * assume that overflows raise errors. `SafeCast` restores this intuition by
 * reverting the transaction when such an operation overflows.
 *
 * Using this library instead of the unchecked operations eliminates an entire
 * class of bugs, so it's recommended to use it always.
 */
library SafeCast {
  /**
   * @dev Value doesn't fit in an uint of `bits` size.
   */
  error SafeCastOverflowedUintDowncast(uint8 bits, uint256 value);

  /**
   * @dev An int value doesn't fit in an uint of `bits` size.
   */
  error SafeCastOverflowedIntToUint(int256 value);

  /**
   * @dev Value doesn't fit in an int of `bits` size.
   */
  error SafeCastOverflowedIntDowncast(uint8 bits, int256 value);

  /**
   * @dev An uint value doesn't fit in an int of `bits` size.
   */
  error SafeCastOverflowedUintToInt(uint256 value);

  /**
   * @dev Returns the downcasted uint248 from uint256, reverting on
   * overflow (when the input is greater than largest uint248).
   *
   * Counterpart to Solidity's `uint248` operator.
   *
   * Requirements:
   *
   * - input must fit into 248 bits
   */
  function toUint248(
    uint256 value
  ) internal pure returns (uint248) {
    if (value > type(uint248).max) {
      revert SafeCastOverflowedUintDowncast(248, value);
    }
    return uint248(value);
  }

  /**
   * @dev Returns the downcasted uint240 from uint256, reverting on
   * overflow (when the input is greater than largest uint240).
   *
   * Counterpart to Solidity's `uint240` operator.
   *
   * Requirements:
   *
   * - input must fit into 240 bits
   */
  function toUint240(
    uint256 value
  ) internal pure returns (uint240) {
    if (value > type(uint240).max) {
      revert SafeCastOverflowedUintDowncast(240, value);
    }
    return uint240(value);
  }

  /**
   * @dev Returns the downcasted uint232 from uint256, reverting on
   * overflow (when the input is greater than largest uint232).
   *
   * Counterpart to Solidity's `uint232` operator.
   *
   * Requirements:
   *
   * - input must fit into 232 bits
   */
  function toUint232(
    uint256 value
  ) internal pure returns (uint232) {
    if (value > type(uint232).max) {
      revert SafeCastOverflowedUintDowncast(232, value);
    }
    return uint232(value);
  }

  /**
   * @dev Returns the downcasted uint224 from uint256, reverting on
   * overflow (when the input is greater than largest uint224).
   *
   * Counterpart to Solidity's `uint224` operator.
   *
   * Requirements:
   *
   * - input must fit into 224 bits
   */
  function toUint224(
    uint256 value
  ) internal pure returns (uint224) {
    if (value > type(uint224).max) {
      revert SafeCastOverflowedUintDowncast(224, value);
    }
    return uint224(value);
  }

  /**
   * @dev Returns the downcasted uint216 from uint256, reverting on
   * overflow (when the input is greater than largest uint216).
   *
   * Counterpart to Solidity's `uint216` operator.
   *
   * Requirements:
   *
   * - input must fit into 216 bits
   */
  function toUint216(
    uint256 value
  ) internal pure returns (uint216) {
    if (value > type(uint216).max) {
      revert SafeCastOverflowedUintDowncast(216, value);
    }
    return uint216(value);
  }

  /**
   * @dev Returns the downcasted uint208 from uint256, reverting on
   * overflow (when the input is greater than largest uint208).
   *
   * Counterpart to Solidity's `uint208` operator.
   *
   * Requirements:
   *
   * - input must fit into 208 bits
   */
  function toUint208(
    uint256 value
  ) internal pure returns (uint208) {
    if (value > type(uint208).max) {
      revert SafeCastOverflowedUintDowncast(208, value);
    }
    return uint208(value);
  }

  /**
   * @dev Returns the downcasted uint200 from uint256, reverting on
   * overflow (when the input is greater than largest uint200).
   *
   * Counterpart to Solidity's `uint200` operator.
   *
   * Requirements:
   *
   * - input must fit into 200 bits
   */
  function toUint200(
    uint256 value
  ) internal pure returns (uint200) {
    if (value > type(uint200).max) {
      revert SafeCastOverflowedUintDowncast(200, value);
    }
    return uint200(value);
  }

  /**
   * @dev Returns the downcasted uint192 from uint256, reverting on
   * overflow (when the input is greater than largest uint192).
   *
   * Counterpart to Solidity's `uint192` operator.
   *
   * Requirements:
   *
   * - input must fit into 192 bits
   */
  function toUint192(
    uint256 value
  ) internal pure returns (uint192) {
    if (value > type(uint192).max) {
      revert SafeCastOverflowedUintDowncast(192, value);
    }
    return uint192(value);
  }

  /**
   * @dev Returns the downcasted uint184 from uint256, reverting on
   * overflow (when the input is greater than largest uint184).
   *
   * Counterpart to Solidity's `uint184` operator.
   *
   * Requirements:
   *
   * - input must fit into 184 bits
   */
  function toUint184(
    uint256 value
  ) internal pure returns (uint184) {
    if (value > type(uint184).max) {
      revert SafeCastOverflowedUintDowncast(184, value);
    }
    return uint184(value);
  }

  /**
   * @dev Returns the downcasted uint176 from uint256, reverting on
   * overflow (when the input is greater than largest uint176).
   *
   * Counterpart to Solidity's `uint176` operator.
   *
   * Requirements:
   *
   * - input must fit into 176 bits
   */
  function toUint176(
    uint256 value
  ) internal pure returns (uint176) {
    if (value > type(uint176).max) {
      revert SafeCastOverflowedUintDowncast(176, value);
    }
    return uint176(value);
  }

  /**
   * @dev Returns the downcasted uint168 from uint256, reverting on
   * overflow (when the input is greater than largest uint168).
   *
   * Counterpart to Solidity's `uint168` operator.
   *
   * Requirements:
   *
   * - input must fit into 168 bits
   */
  function toUint168(
    uint256 value
  ) internal pure returns (uint168) {
    if (value > type(uint168).max) {
      revert SafeCastOverflowedUintDowncast(168, value);
    }
    return uint168(value);
  }

  /**
   * @dev Returns the downcasted uint160 from uint256, reverting on
   * overflow (when the input is greater than largest uint160).
   *
   * Counterpart to Solidity's `uint160` operator.
   *
   * Requirements:
   *
   * - input must fit into 160 bits
   */
  function toUint160(
    uint256 value
  ) internal pure returns (uint160) {
    if (value > type(uint160).max) {
      revert SafeCastOverflowedUintDowncast(160, value);
    }
    return uint160(value);
  }

  /**
   * @dev Returns the downcasted uint152 from uint256, reverting on
   * overflow (when the input is greater than largest uint152).
   *
   * Counterpart to Solidity's `uint152` operator.
   *
   * Requirements:
   *
   * - input must fit into 152 bits
   */
  function toUint152(
    uint256 value
  ) internal pure returns (uint152) {
    if (value > type(uint152).max) {
      revert SafeCastOverflowedUintDowncast(152, value);
    }
    return uint152(value);
  }

  /**
   * @dev Returns the downcasted uint144 from uint256, reverting on
   * overflow (when the input is greater than largest uint144).
   *
   * Counterpart to Solidity's `uint144` operator.
   *
   * Requirements:
   *
   * - input must fit into 144 bits
   */
  function toUint144(
    uint256 value
  ) internal pure returns (uint144) {
    if (value > type(uint144).max) {
      revert SafeCastOverflowedUintDowncast(144, value);
    }
    return uint144(value);
  }

  /**
   * @dev Returns the downcasted uint136 from uint256, reverting on
   * overflow (when the input is greater than largest uint136).
   *
   * Counterpart to Solidity's `uint136` operator.
   *
   * Requirements:
   *
   * - input must fit into 136 bits
   */
  function toUint136(
    uint256 value
  ) internal pure returns (uint136) {
    if (value > type(uint136).max) {
      revert SafeCastOverflowedUintDowncast(136, value);
    }
    return uint136(value);
  }

  /**
   * @dev Returns the downcasted uint128 from uint256, reverting on
   * overflow (when the input is greater than largest uint128).
   *
   * Counterpart to Solidity's `uint128` operator.
   *
   * Requirements:
   *
   * - input must fit into 128 bits
   */
  function toUint128(
    uint256 value
  ) internal pure returns (uint128) {
    if (value > type(uint128).max) {
      revert SafeCastOverflowedUintDowncast(128, value);
    }
    return uint128(value);
  }

  /**
   * @dev Returns the downcasted uint120 from uint256, reverting on
   * overflow (when the input is greater than largest uint120).
   *
   * Counterpart to Solidity's `uint120` operator.
   *
   * Requirements:
   *
   * - input must fit into 120 bits
   */
  function toUint120(
    uint256 value
  ) internal pure returns (uint120) {
    if (value > type(uint120).max) {
      revert SafeCastOverflowedUintDowncast(120, value);
    }
    return uint120(value);
  }

  /**
   * @dev Returns the downcasted uint112 from uint256, reverting on
   * overflow (when the input is greater than largest uint112).
   *
   * Counterpart to Solidity's `uint112` operator.
   *
   * Requirements:
   *
   * - input must fit into 112 bits
   */
  function toUint112(
    uint256 value
  ) internal pure returns (uint112) {
    if (value > type(uint112).max) {
      revert SafeCastOverflowedUintDowncast(112, value);
    }
    return uint112(value);
  }

  /**
   * @dev Returns the downcasted uint104 from uint256, reverting on
   * overflow (when the input is greater than largest uint104).
   *
   * Counterpart to Solidity's `uint104` operator.
   *
   * Requirements:
   *
   * - input must fit into 104 bits
   */
  function toUint104(
    uint256 value
  ) internal pure returns (uint104) {
    if (value > type(uint104).max) {
      revert SafeCastOverflowedUintDowncast(104, value);
    }
    return uint104(value);
  }

  /**
   * @dev Returns the downcasted uint96 from uint256, reverting on
   * overflow (when the input is greater than largest uint96).
   *
   * Counterpart to Solidity's `uint96` operator.
   *
   * Requirements:
   *
   * - input must fit into 96 bits
   */
  function toUint96(
    uint256 value
  ) internal pure returns (uint96) {
    if (value > type(uint96).max) {
      revert SafeCastOverflowedUintDowncast(96, value);
    }
    return uint96(value);
  }

  /**
   * @dev Returns the downcasted uint88 from uint256, reverting on
   * overflow (when the input is greater than largest uint88).
   *
   * Counterpart to Solidity's `uint88` operator.
   *
   * Requirements:
   *
   * - input must fit into 88 bits
   */
  function toUint88(
    uint256 value
  ) internal pure returns (uint88) {
    if (value > type(uint88).max) {
      revert SafeCastOverflowedUintDowncast(88, value);
    }
    return uint88(value);
  }

  /**
   * @dev Returns the downcasted uint80 from uint256, reverting on
   * overflow (when the input is greater than largest uint80).
   *
   * Counterpart to Solidity's `uint80` operator.
   *
   * Requirements:
   *
   * - input must fit into 80 bits
   */
  function toUint80(
    uint256 value
  ) internal pure returns (uint80) {
    if (value > type(uint80).max) {
      revert SafeCastOverflowedUintDowncast(80, value);
    }
    return uint80(value);
  }

  /**
   * @dev Returns the downcasted uint72 from uint256, reverting on
   * overflow (when the input is greater than largest uint72).
   *
   * Counterpart to Solidity's `uint72` operator.
   *
   * Requirements:
   *
   * - input must fit into 72 bits
   */
  function toUint72(
    uint256 value
  ) internal pure returns (uint72) {
    if (value > type(uint72).max) {
      revert SafeCastOverflowedUintDowncast(72, value);
    }
    return uint72(value);
  }

  /**
   * @dev Returns the downcasted uint64 from uint256, reverting on
   * overflow (when the input is greater than largest uint64).
   *
   * Counterpart to Solidity's `uint64` operator.
   *
   * Requirements:
   *
   * - input must fit into 64 bits
   */
  function toUint64(
    uint256 value
  ) internal pure returns (uint64) {
    if (value > type(uint64).max) {
      revert SafeCastOverflowedUintDowncast(64, value);
    }
    return uint64(value);
  }

  /**
   * @dev Returns the downcasted uint56 from uint256, reverting on
   * overflow (when the input is greater than largest uint56).
   *
   * Counterpart to Solidity's `uint56` operator.
   *
   * Requirements:
   *
   * - input must fit into 56 bits
   */
  function toUint56(
    uint256 value
  ) internal pure returns (uint56) {
    if (value > type(uint56).max) {
      revert SafeCastOverflowedUintDowncast(56, value);
    }
    return uint56(value);
  }

  /**
   * @dev Returns the downcasted uint48 from uint256, reverting on
   * overflow (when the input is greater than largest uint48).
   *
   * Counterpart to Solidity's `uint48` operator.
   *
   * Requirements:
   *
   * - input must fit into 48 bits
   */
  function toUint48(
    uint256 value
  ) internal pure returns (uint48) {
    if (value > type(uint48).max) {
      revert SafeCastOverflowedUintDowncast(48, value);
    }
    return uint48(value);
  }

  /**
   * @dev Returns the downcasted uint40 from uint256, reverting on
   * overflow (when the input is greater than largest uint40).
   *
   * Counterpart to Solidity's `uint40` operator.
   *
   * Requirements:
   *
   * - input must fit into 40 bits
   */
  function toUint40(
    uint256 value
  ) internal pure returns (uint40) {
    if (value > type(uint40).max) {
      revert SafeCastOverflowedUintDowncast(40, value);
    }
    return uint40(value);
  }

  /**
   * @dev Returns the downcasted uint32 from uint256, reverting on
   * overflow (when the input is greater than largest uint32).
   *
   * Counterpart to Solidity's `uint32` operator.
   *
   * Requirements:
   *
   * - input must fit into 32 bits
   */
  function toUint32(
    uint256 value
  ) internal pure returns (uint32) {
    if (value > type(uint32).max) {
      revert SafeCastOverflowedUintDowncast(32, value);
    }
    return uint32(value);
  }

  /**
   * @dev Returns the downcasted uint24 from uint256, reverting on
   * overflow (when the input is greater than largest uint24).
   *
   * Counterpart to Solidity's `uint24` operator.
   *
   * Requirements:
   *
   * - input must fit into 24 bits
   */
  function toUint24(
    uint256 value
  ) internal pure returns (uint24) {
    if (value > type(uint24).max) {
      revert SafeCastOverflowedUintDowncast(24, value);
    }
    return uint24(value);
  }

  /**
   * @dev Returns the downcasted uint16 from uint256, reverting on
   * overflow (when the input is greater than largest uint16).
   *
   * Counterpart to Solidity's `uint16` operator.
   *
   * Requirements:
   *
   * - input must fit into 16 bits
   */
  function toUint16(
    uint256 value
  ) internal pure returns (uint16) {
    if (value > type(uint16).max) {
      revert SafeCastOverflowedUintDowncast(16, value);
    }
    return uint16(value);
  }

  /**
   * @dev Returns the downcasted uint8 from uint256, reverting on
   * overflow (when the input is greater than largest uint8).
   *
   * Counterpart to Solidity's `uint8` operator.
   *
   * Requirements:
   *
   * - input must fit into 8 bits
   */
  function toUint8(
    uint256 value
  ) internal pure returns (uint8) {
    if (value > type(uint8).max) {
      revert SafeCastOverflowedUintDowncast(8, value);
    }
    return uint8(value);
  }

  /**
   * @dev Converts a signed int256 into an unsigned uint256.
   *
   * Requirements:
   *
   * - input must be greater than or equal to 0.
   */
  function toUint256(
    int256 value
  ) internal pure returns (uint256) {
    if (value < 0) {
      revert SafeCastOverflowedIntToUint(value);
    }
    return uint256(value);
  }

  /**
   * @dev Returns the downcasted int248 from int256, reverting on
   * overflow (when the input is less than smallest int248 or
   * greater than largest int248).
   *
   * Counterpart to Solidity's `int248` operator.
   *
   * Requirements:
   *
   * - input must fit into 248 bits
   */
  function toInt248(
    int256 value
  ) internal pure returns (int248 downcasted) {
    downcasted = int248(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(248, value);
    }
  }

  /**
   * @dev Returns the downcasted int240 from int256, reverting on
   * overflow (when the input is less than smallest int240 or
   * greater than largest int240).
   *
   * Counterpart to Solidity's `int240` operator.
   *
   * Requirements:
   *
   * - input must fit into 240 bits
   */
  function toInt240(
    int256 value
  ) internal pure returns (int240 downcasted) {
    downcasted = int240(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(240, value);
    }
  }

  /**
   * @dev Returns the downcasted int232 from int256, reverting on
   * overflow (when the input is less than smallest int232 or
   * greater than largest int232).
   *
   * Counterpart to Solidity's `int232` operator.
   *
   * Requirements:
   *
   * - input must fit into 232 bits
   */
  function toInt232(
    int256 value
  ) internal pure returns (int232 downcasted) {
    downcasted = int232(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(232, value);
    }
  }

  /**
   * @dev Returns the downcasted int224 from int256, reverting on
   * overflow (when the input is less than smallest int224 or
   * greater than largest int224).
   *
   * Counterpart to Solidity's `int224` operator.
   *
   * Requirements:
   *
   * - input must fit into 224 bits
   */
  function toInt224(
    int256 value
  ) internal pure returns (int224 downcasted) {
    downcasted = int224(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(224, value);
    }
  }

  /**
   * @dev Returns the downcasted int216 from int256, reverting on
   * overflow (when the input is less than smallest int216 or
   * greater than largest int216).
   *
   * Counterpart to Solidity's `int216` operator.
   *
   * Requirements:
   *
   * - input must fit into 216 bits
   */
  function toInt216(
    int256 value
  ) internal pure returns (int216 downcasted) {
    downcasted = int216(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(216, value);
    }
  }

  /**
   * @dev Returns the downcasted int208 from int256, reverting on
   * overflow (when the input is less than smallest int208 or
   * greater than largest int208).
   *
   * Counterpart to Solidity's `int208` operator.
   *
   * Requirements:
   *
   * - input must fit into 208 bits
   */
  function toInt208(
    int256 value
  ) internal pure returns (int208 downcasted) {
    downcasted = int208(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(208, value);
    }
  }

  /**
   * @dev Returns the downcasted int200 from int256, reverting on
   * overflow (when the input is less than smallest int200 or
   * greater than largest int200).
   *
   * Counterpart to Solidity's `int200` operator.
   *
   * Requirements:
   *
   * - input must fit into 200 bits
   */
  function toInt200(
    int256 value
  ) internal pure returns (int200 downcasted) {
    downcasted = int200(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(200, value);
    }
  }

  /**
   * @dev Returns the downcasted int192 from int256, reverting on
   * overflow (when the input is less than smallest int192 or
   * greater than largest int192).
   *
   * Counterpart to Solidity's `int192` operator.
   *
   * Requirements:
   *
   * - input must fit into 192 bits
   */
  function toInt192(
    int256 value
  ) internal pure returns (int192 downcasted) {
    downcasted = int192(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(192, value);
    }
  }

  /**
   * @dev Returns the downcasted int184 from int256, reverting on
   * overflow (when the input is less than smallest int184 or
   * greater than largest int184).
   *
   * Counterpart to Solidity's `int184` operator.
   *
   * Requirements:
   *
   * - input must fit into 184 bits
   */
  function toInt184(
    int256 value
  ) internal pure returns (int184 downcasted) {
    downcasted = int184(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(184, value);
    }
  }

  /**
   * @dev Returns the downcasted int176 from int256, reverting on
   * overflow (when the input is less than smallest int176 or
   * greater than largest int176).
   *
   * Counterpart to Solidity's `int176` operator.
   *
   * Requirements:
   *
   * - input must fit into 176 bits
   */
  function toInt176(
    int256 value
  ) internal pure returns (int176 downcasted) {
    downcasted = int176(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(176, value);
    }
  }

  /**
   * @dev Returns the downcasted int168 from int256, reverting on
   * overflow (when the input is less than smallest int168 or
   * greater than largest int168).
   *
   * Counterpart to Solidity's `int168` operator.
   *
   * Requirements:
   *
   * - input must fit into 168 bits
   */
  function toInt168(
    int256 value
  ) internal pure returns (int168 downcasted) {
    downcasted = int168(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(168, value);
    }
  }

  /**
   * @dev Returns the downcasted int160 from int256, reverting on
   * overflow (when the input is less than smallest int160 or
   * greater than largest int160).
   *
   * Counterpart to Solidity's `int160` operator.
   *
   * Requirements:
   *
   * - input must fit into 160 bits
   */
  function toInt160(
    int256 value
  ) internal pure returns (int160 downcasted) {
    downcasted = int160(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(160, value);
    }
  }

  /**
   * @dev Returns the downcasted int152 from int256, reverting on
   * overflow (when the input is less than smallest int152 or
   * greater than largest int152).
   *
   * Counterpart to Solidity's `int152` operator.
   *
   * Requirements:
   *
   * - input must fit into 152 bits
   */
  function toInt152(
    int256 value
  ) internal pure returns (int152 downcasted) {
    downcasted = int152(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(152, value);
    }
  }

  /**
   * @dev Returns the downcasted int144 from int256, reverting on
   * overflow (when the input is less than smallest int144 or
   * greater than largest int144).
   *
   * Counterpart to Solidity's `int144` operator.
   *
   * Requirements:
   *
   * - input must fit into 144 bits
   */
  function toInt144(
    int256 value
  ) internal pure returns (int144 downcasted) {
    downcasted = int144(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(144, value);
    }
  }

  /**
   * @dev Returns the downcasted int136 from int256, reverting on
   * overflow (when the input is less than smallest int136 or
   * greater than largest int136).
   *
   * Counterpart to Solidity's `int136` operator.
   *
   * Requirements:
   *
   * - input must fit into 136 bits
   */
  function toInt136(
    int256 value
  ) internal pure returns (int136 downcasted) {
    downcasted = int136(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(136, value);
    }
  }

  /**
   * @dev Returns the downcasted int128 from int256, reverting on
   * overflow (when the input is less than smallest int128 or
   * greater than largest int128).
   *
   * Counterpart to Solidity's `int128` operator.
   *
   * Requirements:
   *
   * - input must fit into 128 bits
   */
  function toInt128(
    int256 value
  ) internal pure returns (int128 downcasted) {
    downcasted = int128(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(128, value);
    }
  }

  /**
   * @dev Returns the downcasted int120 from int256, reverting on
   * overflow (when the input is less than smallest int120 or
   * greater than largest int120).
   *
   * Counterpart to Solidity's `int120` operator.
   *
   * Requirements:
   *
   * - input must fit into 120 bits
   */
  function toInt120(
    int256 value
  ) internal pure returns (int120 downcasted) {
    downcasted = int120(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(120, value);
    }
  }

  /**
   * @dev Returns the downcasted int112 from int256, reverting on
   * overflow (when the input is less than smallest int112 or
   * greater than largest int112).
   *
   * Counterpart to Solidity's `int112` operator.
   *
   * Requirements:
   *
   * - input must fit into 112 bits
   */
  function toInt112(
    int256 value
  ) internal pure returns (int112 downcasted) {
    downcasted = int112(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(112, value);
    }
  }

  /**
   * @dev Returns the downcasted int104 from int256, reverting on
   * overflow (when the input is less than smallest int104 or
   * greater than largest int104).
   *
   * Counterpart to Solidity's `int104` operator.
   *
   * Requirements:
   *
   * - input must fit into 104 bits
   */
  function toInt104(
    int256 value
  ) internal pure returns (int104 downcasted) {
    downcasted = int104(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(104, value);
    }
  }

  /**
   * @dev Returns the downcasted int96 from int256, reverting on
   * overflow (when the input is less than smallest int96 or
   * greater than largest int96).
   *
   * Counterpart to Solidity's `int96` operator.
   *
   * Requirements:
   *
   * - input must fit into 96 bits
   */
  function toInt96(
    int256 value
  ) internal pure returns (int96 downcasted) {
    downcasted = int96(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(96, value);
    }
  }

  /**
   * @dev Returns the downcasted int88 from int256, reverting on
   * overflow (when the input is less than smallest int88 or
   * greater than largest int88).
   *
   * Counterpart to Solidity's `int88` operator.
   *
   * Requirements:
   *
   * - input must fit into 88 bits
   */
  function toInt88(
    int256 value
  ) internal pure returns (int88 downcasted) {
    downcasted = int88(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(88, value);
    }
  }

  /**
   * @dev Returns the downcasted int80 from int256, reverting on
   * overflow (when the input is less than smallest int80 or
   * greater than largest int80).
   *
   * Counterpart to Solidity's `int80` operator.
   *
   * Requirements:
   *
   * - input must fit into 80 bits
   */
  function toInt80(
    int256 value
  ) internal pure returns (int80 downcasted) {
    downcasted = int80(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(80, value);
    }
  }

  /**
   * @dev Returns the downcasted int72 from int256, reverting on
   * overflow (when the input is less than smallest int72 or
   * greater than largest int72).
   *
   * Counterpart to Solidity's `int72` operator.
   *
   * Requirements:
   *
   * - input must fit into 72 bits
   */
  function toInt72(
    int256 value
  ) internal pure returns (int72 downcasted) {
    downcasted = int72(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(72, value);
    }
  }

  /**
   * @dev Returns the downcasted int64 from int256, reverting on
   * overflow (when the input is less than smallest int64 or
   * greater than largest int64).
   *
   * Counterpart to Solidity's `int64` operator.
   *
   * Requirements:
   *
   * - input must fit into 64 bits
   */
  function toInt64(
    int256 value
  ) internal pure returns (int64 downcasted) {
    downcasted = int64(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(64, value);
    }
  }

  /**
   * @dev Returns the downcasted int56 from int256, reverting on
   * overflow (when the input is less than smallest int56 or
   * greater than largest int56).
   *
   * Counterpart to Solidity's `int56` operator.
   *
   * Requirements:
   *
   * - input must fit into 56 bits
   */
  function toInt56(
    int256 value
  ) internal pure returns (int56 downcasted) {
    downcasted = int56(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(56, value);
    }
  }

  /**
   * @dev Returns the downcasted int48 from int256, reverting on
   * overflow (when the input is less than smallest int48 or
   * greater than largest int48).
   *
   * Counterpart to Solidity's `int48` operator.
   *
   * Requirements:
   *
   * - input must fit into 48 bits
   */
  function toInt48(
    int256 value
  ) internal pure returns (int48 downcasted) {
    downcasted = int48(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(48, value);
    }
  }

  /**
   * @dev Returns the downcasted int40 from int256, reverting on
   * overflow (when the input is less than smallest int40 or
   * greater than largest int40).
   *
   * Counterpart to Solidity's `int40` operator.
   *
   * Requirements:
   *
   * - input must fit into 40 bits
   */
  function toInt40(
    int256 value
  ) internal pure returns (int40 downcasted) {
    downcasted = int40(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(40, value);
    }
  }

  /**
   * @dev Returns the downcasted int32 from int256, reverting on
   * overflow (when the input is less than smallest int32 or
   * greater than largest int32).
   *
   * Counterpart to Solidity's `int32` operator.
   *
   * Requirements:
   *
   * - input must fit into 32 bits
   */
  function toInt32(
    int256 value
  ) internal pure returns (int32 downcasted) {
    downcasted = int32(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(32, value);
    }
  }

  /**
   * @dev Returns the downcasted int24 from int256, reverting on
   * overflow (when the input is less than smallest int24 or
   * greater than largest int24).
   *
   * Counterpart to Solidity's `int24` operator.
   *
   * Requirements:
   *
   * - input must fit into 24 bits
   */
  function toInt24(
    int256 value
  ) internal pure returns (int24 downcasted) {
    downcasted = int24(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(24, value);
    }
  }

  /**
   * @dev Returns the downcasted int16 from int256, reverting on
   * overflow (when the input is less than smallest int16 or
   * greater than largest int16).
   *
   * Counterpart to Solidity's `int16` operator.
   *
   * Requirements:
   *
   * - input must fit into 16 bits
   */
  function toInt16(
    int256 value
  ) internal pure returns (int16 downcasted) {
    downcasted = int16(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(16, value);
    }
  }

  /**
   * @dev Returns the downcasted int8 from int256, reverting on
   * overflow (when the input is less than smallest int8 or
   * greater than largest int8).
   *
   * Counterpart to Solidity's `int8` operator.
   *
   * Requirements:
   *
   * - input must fit into 8 bits
   */
  function toInt8(
    int256 value
  ) internal pure returns (int8 downcasted) {
    downcasted = int8(value);
    if (downcasted != value) {
      revert SafeCastOverflowedIntDowncast(8, value);
    }
  }

  /**
   * @dev Converts an unsigned uint256 into a signed int256.
   *
   * Requirements:
   *
   * - input must be less than or equal to maxInt256.
   */
  function toInt256(
    uint256 value
  ) internal pure returns (int256) {
    // Note: Unsafe cast below is okay because `type(int256).max` is guaranteed to be positive
    if (value > uint256(type(int256).max)) {
      revert SafeCastOverflowedUintToInt(value);
    }
    return int256(value);
  }
}

// src/vendor/@openzeppelin/contracts/utils/structs/EnumerableSet.sol

// OpenZeppelin Contracts (last updated v5.0.0) (utils/structs/EnumerableSet.sol)
// This file was procedurally generated from scripts/generate/templates/EnumerableSet.js.

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
  function _length(
    Set storage set
  ) private view returns (uint256) {
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
  function _values(
    Set storage set
  ) private view returns (bytes32[] memory) {
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
  function length(
    Bytes32Set storage set
  ) internal view returns (uint256) {
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
  function length(
    AddressSet storage set
  ) internal view returns (uint256) {
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
  function length(
    UintSet storage set
  ) internal view returns (uint256) {
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

// src/interfaces/IFeeAggregator.sol

/// @notice Interface for FeeAggregator contracts, which accrue assets, and allow transferring out allowlisted assets.
interface IFeeAggregator {
  /// @notice Transfers a list of allowlisted assets to the target recipient. Can only be called by addresses with the
  /// SWAPPER role.
  /// @param to The address to transfer the assets to
  /// @param assetAmounts List of assets  and amounts to transfer
  function transferForSwap(address to, Common.AssetAmount[] calldata assetAmounts) external;

  /// @notice Getter function to retrieve the list of allowlisted assets
  /// @return allowlistedAssets List of allowlisted assets
  function getAllowlistedAssets() external view returns (address[] memory allowlistedAssets);

  /// @notice Checks if an asset is in the allow list
  /// @param asset The asset to check
  /// @return isAllowlisted Returns true if asset is in the allow list, false if not
  function isAssetAllowlisted(
    address asset
  ) external view returns (bool isAllowlisted);
}

// src/vendor/@chainlink/contracts-ccip/src/v0.8/ccip/interfaces/IRouterClient.sol

interface IRouterClient {
  error UnsupportedDestinationChain(uint64 destChainSelector);
  error InsufficientFeeTokenAmount();
  error InvalidMsgValue();

  /// @notice Checks if the given chain ID is supported for sending/receiving.
  /// @param chainSelector The chain to check.
  /// @return supported is true if it is supported, false if not.
  function isChainSupported(
    uint64 chainSelector
  ) external view returns (bool supported);

  /// @notice Gets a list of all supported tokens which can be sent or received
  /// to/from a given chain id.
  /// @param chainSelector The chainSelector.
  /// @return tokens The addresses of all tokens that are supported.
  function getSupportedTokens(
    uint64 chainSelector
  ) external view returns (address[] memory tokens);

  /// @param destinationChainSelector The destination chainSelector
  /// @param message The cross-chain CCIP message including data and/or tokens
  /// @return fee returns execution fee for the message
  /// delivery to destination chain, denominated in the feeToken specified in the message.
  /// @dev Reverts with appropriate reason upon invalid message.
  function getFee(
    uint64 destinationChainSelector,
    Client.EVM2AnyMessage memory message
  ) external view returns (uint256 fee);

  /// @notice Request a message to be sent to the destination chain
  /// @param destinationChainSelector The destination chain ID
  /// @param message The cross-chain CCIP message including data and/or tokens
  /// @return messageId The message ID
  /// @dev Note if msg.value is larger than the required fee (from getFee) we accept
  /// the overpayment with no refund.
  /// @dev Reverts with appropriate reason upon invalid message.
  function ccipSend(
    uint64 destinationChainSelector,
    Client.EVM2AnyMessage calldata message
  ) external payable returns (bytes32);
}

// src/vendor/@openzeppelin/contracts/access/extensions/IAccessControlDefaultAdminRules.sol

// OpenZeppelin Contracts (last updated v5.0.0) (access/extensions/IAccessControlDefaultAdminRules.sol)

/**
 * @dev External interface of AccessControlDefaultAdminRules declared to support ERC165 detection.
 */
interface IAccessControlDefaultAdminRules is IAccessControl {
  /**
   * @dev The new default admin is not a valid default admin.
   */
  error AccessControlInvalidDefaultAdmin(address defaultAdmin);

  /**
   * @dev At least one of the following rules was violated:
   *
   * - The `DEFAULT_ADMIN_ROLE` must only be managed by itself.
   * - The `DEFAULT_ADMIN_ROLE` must only be held by one account at the time.
   * - Any `DEFAULT_ADMIN_ROLE` transfer must be in two delayed steps.
   */
  error AccessControlEnforcedDefaultAdminRules();

  /**
   * @dev The delay for transferring the default admin delay is enforced and
   * the operation must wait until `schedule`.
   *
   * NOTE: `schedule` can be 0 indicating there's no transfer scheduled.
   */
  error AccessControlEnforcedDefaultAdminDelay(uint48 schedule);

  /**
   * @dev Emitted when a {defaultAdmin} transfer is started, setting `newAdmin` as the next
   * address to become the {defaultAdmin} by calling {acceptDefaultAdminTransfer} only after `acceptSchedule`
   * passes.
   */
  event DefaultAdminTransferScheduled(address indexed newAdmin, uint48 acceptSchedule);

  /**
   * @dev Emitted when a {pendingDefaultAdmin} is reset if it was never accepted, regardless of its schedule.
   */
  event DefaultAdminTransferCanceled();

  /**
   * @dev Emitted when a {defaultAdminDelay} change is started, setting `newDelay` as the next
   * delay to be applied between default admin transfer after `effectSchedule` has passed.
   */
  event DefaultAdminDelayChangeScheduled(uint48 newDelay, uint48 effectSchedule);

  /**
   * @dev Emitted when a {pendingDefaultAdminDelay} is reset if its schedule didn't pass.
   */
  event DefaultAdminDelayChangeCanceled();

  /**
   * @dev Returns the address of the current `DEFAULT_ADMIN_ROLE` holder.
   */
  function defaultAdmin() external view returns (address);

  /**
   * @dev Returns a tuple of a `newAdmin` and an accept schedule.
   *
   * After the `schedule` passes, the `newAdmin` will be able to accept the {defaultAdmin} role
   * by calling {acceptDefaultAdminTransfer}, completing the role transfer.
   *
   * A zero value only in `acceptSchedule` indicates no pending admin transfer.
   *
   * NOTE: A zero address `newAdmin` means that {defaultAdmin} is being renounced.
   */
  function pendingDefaultAdmin() external view returns (address newAdmin, uint48 acceptSchedule);

  /**
   * @dev Returns the delay required to schedule the acceptance of a {defaultAdmin} transfer started.
   *
   * This delay will be added to the current timestamp when calling {beginDefaultAdminTransfer} to set
   * the acceptance schedule.
   *
   * NOTE: If a delay change has been scheduled, it will take effect as soon as the schedule passes, making this
   * function returns the new delay. See {changeDefaultAdminDelay}.
   */
  function defaultAdminDelay() external view returns (uint48);

  /**
   * @dev Returns a tuple of `newDelay` and an effect schedule.
   *
   * After the `schedule` passes, the `newDelay` will get into effect immediately for every
   * new {defaultAdmin} transfer started with {beginDefaultAdminTransfer}.
   *
   * A zero value only in `effectSchedule` indicates no pending delay change.
   *
   * NOTE: A zero value only for `newDelay` means that the next {defaultAdminDelay}
   * will be zero after the effect schedule.
   */
  function pendingDefaultAdminDelay() external view returns (uint48 newDelay, uint48 effectSchedule);

  /**
   * @dev Starts a {defaultAdmin} transfer by setting a {pendingDefaultAdmin} scheduled for acceptance
   * after the current timestamp plus a {defaultAdminDelay}.
   *
   * Requirements:
   *
   * - Only can be called by the current {defaultAdmin}.
   *
   * Emits a DefaultAdminRoleChangeStarted event.
   */
  function beginDefaultAdminTransfer(
    address newAdmin
  ) external;

  /**
   * @dev Cancels a {defaultAdmin} transfer previously started with {beginDefaultAdminTransfer}.
   *
   * A {pendingDefaultAdmin} not yet accepted can also be cancelled with this function.
   *
   * Requirements:
   *
   * - Only can be called by the current {defaultAdmin}.
   *
   * May emit a DefaultAdminTransferCanceled event.
   */
  function cancelDefaultAdminTransfer() external;

  /**
   * @dev Completes a {defaultAdmin} transfer previously started with {beginDefaultAdminTransfer}.
   *
   * After calling the function:
   *
   * - `DEFAULT_ADMIN_ROLE` should be granted to the caller.
   * - `DEFAULT_ADMIN_ROLE` should be revoked from the previous holder.
   * - {pendingDefaultAdmin} should be reset to zero values.
   *
   * Requirements:
   *
   * - Only can be called by the {pendingDefaultAdmin}'s `newAdmin`.
   * - The {pendingDefaultAdmin}'s `acceptSchedule` should've passed.
   */
  function acceptDefaultAdminTransfer() external;

  /**
   * @dev Initiates a {defaultAdminDelay} update by setting a {pendingDefaultAdminDelay} scheduled for getting
   * into effect after the current timestamp plus a {defaultAdminDelay}.
   *
   * This function guarantees that any call to {beginDefaultAdminTransfer} done between the timestamp this
   * method is called and the {pendingDefaultAdminDelay} effect schedule will use the current {defaultAdminDelay}
   * set before calling.
   *
   * The {pendingDefaultAdminDelay}'s effect schedule is defined in a way that waiting until the schedule and then
   * calling {beginDefaultAdminTransfer} with the new delay will take at least the same as another {defaultAdmin}
   * complete transfer (including acceptance).
   *
   * The schedule is designed for two scenarios:
   *
   * - When the delay is changed for a larger one the schedule is `block.timestamp + newDelay` capped by
   * {defaultAdminDelayIncreaseWait}.
   * - When the delay is changed for a shorter one, the schedule is `block.timestamp + (current delay - new delay)`.
   *
   * A {pendingDefaultAdminDelay} that never got into effect will be canceled in favor of a new scheduled change.
   *
   * Requirements:
   *
   * - Only can be called by the current {defaultAdmin}.
   *
   * Emits a DefaultAdminDelayChangeScheduled event and may emit a DefaultAdminDelayChangeCanceled event.
   */
  function changeDefaultAdminDelay(
    uint48 newDelay
  ) external;

  /**
   * @dev Cancels a scheduled {defaultAdminDelay} change.
   *
   * Requirements:
   *
   * - Only can be called by the current {defaultAdmin}.
   *
   * May emit a DefaultAdminDelayChangeCanceled event.
   */
  function rollbackDefaultAdminDelay() external;

  /**
   * @dev Maximum time in seconds for an increase to {defaultAdminDelay} (that is scheduled using
   * {changeDefaultAdminDelay})
   * to take effect. Default to 5 days.
   *
   * When the {defaultAdminDelay} is scheduled to be increased, it goes into effect after the new delay has passed with
   * the purpose of giving enough time for reverting any accidental change (i.e. using milliseconds instead of seconds)
   * that may lock the contract. However, to avoid excessive schedules, the wait is capped by this function and it can
   * be overrode for a custom {defaultAdminDelay} increase scheduling.
   *
   * IMPORTANT: Make sure to add a reasonable amount of time while overriding this value, otherwise,
   * there's a risk of setting a high new delay that goes into effect almost immediately without the
   * possibility of human intervention in the case of an input error (eg. set milliseconds instead of seconds).
   */
  function defaultAdminDelayIncreaseWait() external view returns (uint48);
}

// src/vendor/@openzeppelin/contracts/access/extensions/IAccessControlEnumerable.sol

// OpenZeppelin Contracts (last updated v5.0.0) (access/extensions/IAccessControlEnumerable.sol)

/**
 * @dev External interface of AccessControlEnumerable declared to support ERC165 detection.
 */
interface IAccessControlEnumerable is IAccessControl {
  /**
   * @dev Returns one of the accounts that have `role`. `index` must be a
   * value between 0 and {getRoleMemberCount}, non-inclusive.
   *
   * Role bearers are not sorted in any particular way, and their ordering may
   * change at any point.
   *
   * WARNING: When using {getRoleMember} and {getRoleMemberCount}, make sure
   * you perform all queries on the same block. See the following
   * https://forum.openzeppelin.com/t/iterating-over-elements-on-enumerableset-in-openzeppelin-contracts/2296[forum
   * post]
   * for more information.
   */
  function getRoleMember(bytes32 role, uint256 index) external view returns (address);

  /**
   * @dev Returns the number of accounts that have `role`. Can be used
   * together with {getRoleMember} to enumerate all bearers of a role.
   */
  function getRoleMemberCount(
    bytes32 role
  ) external view returns (uint256);
}

// src/vendor/@openzeppelin/contracts/interfaces/IERC20.sol

// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC20.sol)

// src/vendor/@openzeppelin/contracts/utils/Pausable.sol

// OpenZeppelin Contracts (last updated v5.0.0) (utils/Pausable.sol)

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

// src/vendor/@openzeppelin/contracts/utils/introspection/ERC165.sol

// OpenZeppelin Contracts (last updated v5.0.0) (utils/introspection/ERC165.sol)

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
abstract contract ERC165 is IERC165 {
  /**
   * @dev See {IERC165-supportsInterface}.
   */
  function supportsInterface(
    bytes4 interfaceId
  ) public view virtual returns (bool) {
    return interfaceId == type(IERC165).interfaceId;
  }
}

// src/NativeTokenReceiver.sol

/// @notice Native token reciever contract that handles native token wrapping.
abstract contract NativeTokenReceiver {
  /// @notice This event is emitted when the wrapped native token is set.
  /// @param wrappedNativeToken The wrapped native token address.
  event WrappedNativeTokenSet(address wrappedNativeToken);

  /// @notice This error is thrown when trying to wrap native tokens without any outstanding balance.
  error ZeroBalance();

  /// @notice The minimum gas left in the call to perform a wrapping call on receive.
  uint256 public constant MIN_GAS_FOR_RECEIVE = 2300;

  /// @notice The wrapped native token.
  IWERC20 internal s_wrappedNativeToken;

  constructor(
    address wrappedNativeToken
  ) {
    if (wrappedNativeToken != address(0)) {
      _setWrappedNativeToken(wrappedNativeToken);
    }
  }

  // ================================================================
  // |                    Native Token Handling                     |
  // ================================================================

  /// @notice Wraps the outstanding native token balance.
  function deposit() external virtual {
    if (address(this).balance == 0) {
      revert ZeroBalance();
    }

    s_wrappedNativeToken.deposit{value: address(this).balance}();
  }

  /// @dev Receive function that autowraps native tokens on receive if the gas left is greater than 2300 which
  /// indicates a low level call. Otherwise, transfer method has been used which won't allow for a wrapping call so the
  /// contracts simply receives the msg.value.
  receive() external payable {
    if (gasleft() > MIN_GAS_FOR_RECEIVE) {
      if (address(s_wrappedNativeToken) != address(0)) {
        // We try catch the deposit call as some chain's wrapped native token may not support the deposit function
        try s_wrappedNativeToken.deposit{value: msg.value}() {} catch {}
      }
    }
  }

  // ================================================================
  // |                            Config                            |
  // ================================================================

  /// @dev Sets the wrapped native token.
  /// @dev We allow setting to the zero address for chains that may not have a wrapped native token.
  /// @param wrappedNativeToken The wrapped native token address.
  function _setWrappedNativeToken(
    address wrappedNativeToken
  ) internal {
    if (wrappedNativeToken == address(s_wrappedNativeToken)) {
      revert Errors.ValueNotUpdated();
    }

    s_wrappedNativeToken = IWERC20(wrappedNativeToken);

    emit WrappedNativeTokenSet(wrappedNativeToken);
  }

  /// @notice Getter function to retrieve the configured wrapped native token.
  /// @return wrappedNativeToken The configured wrapped native token.
  function getWrappedNativeToken() external view returns (IWERC20 wrappedNativeToken) {
    return s_wrappedNativeToken;
  }
}

// src/vendor/@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol

// OpenZeppelin Contracts (last updated v5.0.0) (token/ERC20/utils/SafeERC20.sol)

/**
 * @title SafeERC20
 * @dev Wrappers around ERC20 operations that throw on failure (when the token
 * contract returns false). Tokens that return no value (and instead revert or
 * throw on failure) are also supported, non-reverting calls are assumed to be
 * successful.
 * To use this library you can add a `using SafeERC20 for IERC20;` statement to your contract,
 * which allows you to call the safe operations as `token.safeTransfer(...)`, etc.
 */
library SafeERC20 {
  using Address for address;

  /**
   * @dev An operation with an ERC20 token failed.
   */
  error SafeERC20FailedOperation(address token);

  /**
   * @dev Indicates a failed `decreaseAllowance` request.
   */
  error SafeERC20FailedDecreaseAllowance(address spender, uint256 currentAllowance, uint256 requestedDecrease);

  /**
   * @dev Transfer `value` amount of `token` from the calling contract to `to`. If `token` returns no value,
   * non-reverting calls are assumed to be successful.
   */
  function safeTransfer(IERC20 token, address to, uint256 value) internal {
    _callOptionalReturn(token, abi.encodeCall(token.transfer, (to, value)));
  }

  /**
   * @dev Transfer `value` amount of `token` from `from` to `to`, spending the approval given by `from` to the
   * calling contract. If `token` returns no value, non-reverting calls are assumed to be successful.
   */
  function safeTransferFrom(IERC20 token, address from, address to, uint256 value) internal {
    _callOptionalReturn(token, abi.encodeCall(token.transferFrom, (from, to, value)));
  }

  /**
   * @dev Increase the calling contract's allowance toward `spender` by `value`. If `token` returns no value,
   * non-reverting calls are assumed to be successful.
   */
  function safeIncreaseAllowance(IERC20 token, address spender, uint256 value) internal {
    uint256 oldAllowance = token.allowance(address(this), spender);
    forceApprove(token, spender, oldAllowance + value);
  }

  /**
   * @dev Decrease the calling contract's allowance toward `spender` by `requestedDecrease`. If `token` returns no
   * value, non-reverting calls are assumed to be successful.
   */
  function safeDecreaseAllowance(IERC20 token, address spender, uint256 requestedDecrease) internal {
    unchecked {
      uint256 currentAllowance = token.allowance(address(this), spender);
      if (currentAllowance < requestedDecrease) {
        revert SafeERC20FailedDecreaseAllowance(spender, currentAllowance, requestedDecrease);
      }
      forceApprove(token, spender, currentAllowance - requestedDecrease);
    }
  }

  /**
   * @dev Set the calling contract's allowance toward `spender` to `value`. If `token` returns no value,
   * non-reverting calls are assumed to be successful. Meant to be used with tokens that require the approval
   * to be set to zero before setting it to a non-zero value, such as USDT.
   */
  function forceApprove(IERC20 token, address spender, uint256 value) internal {
    bytes memory approvalCall = abi.encodeCall(token.approve, (spender, value));

    if (!_callOptionalReturnBool(token, approvalCall)) {
      _callOptionalReturn(token, abi.encodeCall(token.approve, (spender, 0)));
      _callOptionalReturn(token, approvalCall);
    }
  }

  /**
   * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
   * on the return value: the return value is optional (but if data is returned, it must not be false).
   * @param token The token targeted by the call.
   * @param data The call data (encoded using abi.encode or one of its variants).
   */
  function _callOptionalReturn(IERC20 token, bytes memory data) private {
    // We need to perform a low level call here, to bypass Solidity's return data size checking mechanism, since
    // we're implementing it ourselves. We use {Address-functionCall} to perform this call, which verifies that
    // the target address contains contract code and also asserts for success in the low-level call.

    bytes memory returndata = address(token).functionCall(data);
    if (returndata.length != 0 && !abi.decode(returndata, (bool))) {
      revert SafeERC20FailedOperation(address(token));
    }
  }

  /**
   * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
   * on the return value: the return value is optional (but if data is returned, it must not be false).
   * @param token The token targeted by the call.
   * @param data The call data (encoded using abi.encode or one of its variants).
   *
   * This is a variant of {_callOptionalReturn} that silents catches all reverts and returns a bool instead.
   */
  function _callOptionalReturnBool(IERC20 token, bytes memory data) private returns (bool) {
    // We need to perform a low level call here, to bypass Solidity's return data size checking mechanism, since
    // we're implementing it ourselves. We cannot use {Address-functionCall} here since this should return false
    // and not revert is the subcall reverts.

    (bool success, bytes memory returndata) = address(token).call(data);
    return success && (returndata.length == 0 || abi.decode(returndata, (bool))) && address(token).code.length > 0;
  }
}

// src/LinkReceiver.sol

/// @notice Base contract that adds ERC677 transferAndCall receiver functionality scoped to the LINK token
abstract contract LinkReceiver is IERC677Receiver {
  /// @notice This event is emitted when the LINK token address is set
  /// @param linkToken The LINK token address
  event LinkTokenSet(address indexed linkToken);

  /// @notice This error is thrown when the sender is not the LINK token
  /// @param sender The address of the sender
  error SenderNotLinkToken(address sender);

  /// @notice The link token
  IERC20 internal immutable i_linkToken;

  constructor(
    address linkToken
  ) {
    if (linkToken == address(0)) {
      revert Errors.InvalidZeroAddress();
    }

    i_linkToken = IERC20(linkToken);
    emit LinkTokenSet(linkToken);
  }

  // ================================================================
  // │                            LINK Token                        │
  // ================================================================

  /// @inheritdoc IERC677Receiver
  /// @dev Implementing onTokenTransfer only to maximize Link receiving compatibility. No extra logic added.
  /// @dev precondition The sender must be the LINK token
  function onTokenTransfer(address, uint256, bytes calldata) external view {
    if (msg.sender != address(i_linkToken)) {
      revert SenderNotLinkToken(msg.sender);
    }
  }

  /// @notice Getter function to retrieve the LINK token address
  /// @return linkToken The LINK token address
  function getLinkToken() external view returns (IERC20 linkToken) {
    return i_linkToken;
  }
}

// src/vendor/@openzeppelin/contracts/access/AccessControl.sol

// OpenZeppelin Contracts (last updated v5.0.0) (access/AccessControl.sol)

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
  modifier onlyRole(
    bytes32 role
  ) {
    _checkRole(role);
    _;
  }

  /**
   * @dev See {IERC165-supportsInterface}.
   */
  function supportsInterface(
    bytes4 interfaceId
  ) public view virtual override returns (bool) {
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
  function _checkRole(
    bytes32 role
  ) internal view virtual {
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
  function getRoleAdmin(
    bytes32 role
  ) public view virtual returns (bytes32) {
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

// src/vendor/@openzeppelin/contracts/access/extensions/AccessControlDefaultAdminRules.sol

// OpenZeppelin Contracts (last updated v5.0.0) (access/extensions/AccessControlDefaultAdminRules.sol)

/**
 * @dev Extension of {AccessControl} that allows specifying special rules to manage
 * the `DEFAULT_ADMIN_ROLE` holder, which is a sensitive role with special permissions
 * over other roles that may potentially have privileged rights in the system.
 *
 * If a specific role doesn't have an admin role assigned, the holder of the
 * `DEFAULT_ADMIN_ROLE` will have the ability to grant it and revoke it.
 *
 * This contract implements the following risk mitigations on top of {AccessControl}:
 *
 * * Only one account holds the `DEFAULT_ADMIN_ROLE` since deployment until it's potentially renounced.
 * * Enforces a 2-step process to transfer the `DEFAULT_ADMIN_ROLE` to another account.
 * * Enforces a configurable delay between the two steps, with the ability to cancel before the transfer is accepted.
 * * The delay can be changed by scheduling, see {changeDefaultAdminDelay}.
 * * It is not possible to use another role to manage the `DEFAULT_ADMIN_ROLE`.
 *
 * Example usage:
 *
 * ```solidity
 * contract MyToken is AccessControlDefaultAdminRules {
 *   constructor() AccessControlDefaultAdminRules(
 *     3 days,
 *     msg.sender // Explicit initial `DEFAULT_ADMIN_ROLE` holder
 *    ) {}
 * }
 * ```
 */
abstract contract AccessControlDefaultAdminRules is IAccessControlDefaultAdminRules, IERC5313, AccessControl {
  // pending admin pair read/written together frequently
  address private _pendingDefaultAdmin;
  uint48 private _pendingDefaultAdminSchedule; // 0 == unset

  uint48 private _currentDelay;
  address private _currentDefaultAdmin;

  // pending delay pair read/written together frequently
  uint48 private _pendingDelay;
  uint48 private _pendingDelaySchedule; // 0 == unset

  /**
   * @dev Sets the initial values for {defaultAdminDelay} and {defaultAdmin} address.
   */
  constructor(uint48 initialDelay, address initialDefaultAdmin) {
    if (initialDefaultAdmin == address(0)) {
      revert AccessControlInvalidDefaultAdmin(address(0));
    }
    _currentDelay = initialDelay;
    _grantRole(DEFAULT_ADMIN_ROLE, initialDefaultAdmin);
  }

  /**
   * @dev See {IERC165-supportsInterface}.
   */
  function supportsInterface(
    bytes4 interfaceId
  ) public view virtual override returns (bool) {
    return interfaceId == type(IAccessControlDefaultAdminRules).interfaceId || super.supportsInterface(interfaceId);
  }

  /**
   * @dev See {IERC5313-owner}.
   */
  function owner() public view virtual returns (address) {
    return defaultAdmin();
  }

  ///
  /// Override AccessControl role management
  ///

  /**
   * @dev See {AccessControl-grantRole}. Reverts for `DEFAULT_ADMIN_ROLE`.
   */
  function grantRole(bytes32 role, address account) public virtual override(AccessControl, IAccessControl) {
    if (role == DEFAULT_ADMIN_ROLE) {
      revert AccessControlEnforcedDefaultAdminRules();
    }
    super.grantRole(role, account);
  }

  /**
   * @dev See {AccessControl-revokeRole}. Reverts for `DEFAULT_ADMIN_ROLE`.
   */
  function revokeRole(bytes32 role, address account) public virtual override(AccessControl, IAccessControl) {
    if (role == DEFAULT_ADMIN_ROLE) {
      revert AccessControlEnforcedDefaultAdminRules();
    }
    super.revokeRole(role, account);
  }

  /**
   * @dev See {AccessControl-renounceRole}.
   *
   * For the `DEFAULT_ADMIN_ROLE`, it only allows renouncing in two steps by first calling
   * {beginDefaultAdminTransfer} to the `address(0)`, so it's required that the {pendingDefaultAdmin} schedule
   * has also passed when calling this function.
   *
   * After its execution, it will not be possible to call `onlyRole(DEFAULT_ADMIN_ROLE)` functions.
   *
   * NOTE: Renouncing `DEFAULT_ADMIN_ROLE` will leave the contract without a {defaultAdmin},
   * thereby disabling any functionality that is only available for it, and the possibility of reassigning a
   * non-administrated role.
   */
  function renounceRole(bytes32 role, address account) public virtual override(AccessControl, IAccessControl) {
    if (role == DEFAULT_ADMIN_ROLE && account == defaultAdmin()) {
      (address newDefaultAdmin, uint48 schedule) = pendingDefaultAdmin();
      if (newDefaultAdmin != address(0) || !_isScheduleSet(schedule) || !_hasSchedulePassed(schedule)) {
        revert AccessControlEnforcedDefaultAdminDelay(schedule);
      }
      delete _pendingDefaultAdminSchedule;
    }
    super.renounceRole(role, account);
  }

  /**
   * @dev See {AccessControl-_grantRole}.
   *
   * For `DEFAULT_ADMIN_ROLE`, it only allows granting if there isn't already a {defaultAdmin} or if the
   * role has been previously renounced.
   *
   * NOTE: Exposing this function through another mechanism may make the `DEFAULT_ADMIN_ROLE`
   * assignable again. Make sure to guarantee this is the expected behavior in your implementation.
   */
  function _grantRole(bytes32 role, address account) internal virtual override returns (bool) {
    if (role == DEFAULT_ADMIN_ROLE) {
      if (defaultAdmin() != address(0)) {
        revert AccessControlEnforcedDefaultAdminRules();
      }
      _currentDefaultAdmin = account;
    }
    return super._grantRole(role, account);
  }

  /**
   * @dev See {AccessControl-_revokeRole}.
   */
  function _revokeRole(bytes32 role, address account) internal virtual override returns (bool) {
    if (role == DEFAULT_ADMIN_ROLE && account == defaultAdmin()) {
      delete _currentDefaultAdmin;
    }
    return super._revokeRole(role, account);
  }

  /**
   * @dev See {AccessControl-_setRoleAdmin}. Reverts for `DEFAULT_ADMIN_ROLE`.
   */
  function _setRoleAdmin(bytes32 role, bytes32 adminRole) internal virtual override {
    if (role == DEFAULT_ADMIN_ROLE) {
      revert AccessControlEnforcedDefaultAdminRules();
    }
    super._setRoleAdmin(role, adminRole);
  }

  ///
  /// AccessControlDefaultAdminRules accessors
  ///

  /**
   * @inheritdoc IAccessControlDefaultAdminRules
   */
  function defaultAdmin() public view virtual returns (address) {
    return _currentDefaultAdmin;
  }

  /**
   * @inheritdoc IAccessControlDefaultAdminRules
   */
  function pendingDefaultAdmin() public view virtual returns (address newAdmin, uint48 schedule) {
    return (_pendingDefaultAdmin, _pendingDefaultAdminSchedule);
  }

  /**
   * @inheritdoc IAccessControlDefaultAdminRules
   */
  function defaultAdminDelay() public view virtual returns (uint48) {
    uint48 schedule = _pendingDelaySchedule;
    return (_isScheduleSet(schedule) && _hasSchedulePassed(schedule)) ? _pendingDelay : _currentDelay;
  }

  /**
   * @inheritdoc IAccessControlDefaultAdminRules
   */
  function pendingDefaultAdminDelay() public view virtual returns (uint48 newDelay, uint48 schedule) {
    schedule = _pendingDelaySchedule;
    return (_isScheduleSet(schedule) && !_hasSchedulePassed(schedule)) ? (_pendingDelay, schedule) : (0, 0);
  }

  /**
   * @inheritdoc IAccessControlDefaultAdminRules
   */
  function defaultAdminDelayIncreaseWait() public view virtual returns (uint48) {
    return 5 days;
  }

  ///
  /// AccessControlDefaultAdminRules public and internal setters for defaultAdmin/pendingDefaultAdmin
  ///

  /**
   * @inheritdoc IAccessControlDefaultAdminRules
   */
  function beginDefaultAdminTransfer(
    address newAdmin
  ) public virtual onlyRole(DEFAULT_ADMIN_ROLE) {
    _beginDefaultAdminTransfer(newAdmin);
  }

  /**
   * @dev See {beginDefaultAdminTransfer}.
   *
   * Internal function without access restriction.
   */
  function _beginDefaultAdminTransfer(
    address newAdmin
  ) internal virtual {
    uint48 newSchedule = SafeCast.toUint48(block.timestamp) + defaultAdminDelay();
    _setPendingDefaultAdmin(newAdmin, newSchedule);
    emit DefaultAdminTransferScheduled(newAdmin, newSchedule);
  }

  /**
   * @inheritdoc IAccessControlDefaultAdminRules
   */
  function cancelDefaultAdminTransfer() public virtual onlyRole(DEFAULT_ADMIN_ROLE) {
    _cancelDefaultAdminTransfer();
  }

  /**
   * @dev See {cancelDefaultAdminTransfer}.
   *
   * Internal function without access restriction.
   */
  function _cancelDefaultAdminTransfer() internal virtual {
    _setPendingDefaultAdmin(address(0), 0);
  }

  /**
   * @inheritdoc IAccessControlDefaultAdminRules
   */
  function acceptDefaultAdminTransfer() public virtual {
    (address newDefaultAdmin,) = pendingDefaultAdmin();
    if (_msgSender() != newDefaultAdmin) {
      // Enforce newDefaultAdmin explicit acceptance.
      revert AccessControlInvalidDefaultAdmin(_msgSender());
    }
    _acceptDefaultAdminTransfer();
  }

  /**
   * @dev See {acceptDefaultAdminTransfer}.
   *
   * Internal function without access restriction.
   */
  function _acceptDefaultAdminTransfer() internal virtual {
    (address newAdmin, uint48 schedule) = pendingDefaultAdmin();
    if (!_isScheduleSet(schedule) || !_hasSchedulePassed(schedule)) {
      revert AccessControlEnforcedDefaultAdminDelay(schedule);
    }
    _revokeRole(DEFAULT_ADMIN_ROLE, defaultAdmin());
    _grantRole(DEFAULT_ADMIN_ROLE, newAdmin);
    delete _pendingDefaultAdmin;
    delete _pendingDefaultAdminSchedule;
  }

  ///
  /// AccessControlDefaultAdminRules public and internal setters for defaultAdminDelay/pendingDefaultAdminDelay
  ///

  /**
   * @inheritdoc IAccessControlDefaultAdminRules
   */
  function changeDefaultAdminDelay(
    uint48 newDelay
  ) public virtual onlyRole(DEFAULT_ADMIN_ROLE) {
    _changeDefaultAdminDelay(newDelay);
  }

  /**
   * @dev See {changeDefaultAdminDelay}.
   *
   * Internal function without access restriction.
   */
  function _changeDefaultAdminDelay(
    uint48 newDelay
  ) internal virtual {
    uint48 newSchedule = SafeCast.toUint48(block.timestamp) + _delayChangeWait(newDelay);
    _setPendingDelay(newDelay, newSchedule);
    emit DefaultAdminDelayChangeScheduled(newDelay, newSchedule);
  }

  /**
   * @inheritdoc IAccessControlDefaultAdminRules
   */
  function rollbackDefaultAdminDelay() public virtual onlyRole(DEFAULT_ADMIN_ROLE) {
    _rollbackDefaultAdminDelay();
  }

  /**
   * @dev See {rollbackDefaultAdminDelay}.
   *
   * Internal function without access restriction.
   */
  function _rollbackDefaultAdminDelay() internal virtual {
    _setPendingDelay(0, 0);
  }

  /**
   * @dev Returns the amount of seconds to wait after the `newDelay` will
   * become the new {defaultAdminDelay}.
   *
   * The value returned guarantees that if the delay is reduced, it will go into effect
   * after a wait that honors the previously set delay.
   *
   * See {defaultAdminDelayIncreaseWait}.
   */
  function _delayChangeWait(
    uint48 newDelay
  ) internal view virtual returns (uint48) {
    uint48 currentDelay = defaultAdminDelay();

    // When increasing the delay, we schedule the delay change to occur after a period of "new delay" has passed, up
    // to a maximum given by defaultAdminDelayIncreaseWait, by default 5 days. For example, if increasing from 1 day
    // to 3 days, the new delay will come into effect after 3 days. If increasing from 1 day to 10 days, the new
    // delay will come into effect after 5 days. The 5 day wait period is intended to be able to fix an error like
    // using milliseconds instead of seconds.
    //
    // When decreasing the delay, we wait the difference between "current delay" and "new delay". This guarantees
    // that an admin transfer cannot be made faster than "current delay" at the time the delay change is scheduled.
    // For example, if decreasing from 10 days to 3 days, the new delay will come into effect after 7 days.
    return newDelay > currentDelay
      ? uint48(Math.min(newDelay, defaultAdminDelayIncreaseWait())) // no need to safecast, both inputs are uint48
      : currentDelay - newDelay;
  }

  ///
  /// Private setters
  ///

  /**
   * @dev Setter of the tuple for pending admin and its schedule.
   *
   * May emit a DefaultAdminTransferCanceled event.
   */
  function _setPendingDefaultAdmin(address newAdmin, uint48 newSchedule) private {
    (, uint48 oldSchedule) = pendingDefaultAdmin();

    _pendingDefaultAdmin = newAdmin;
    _pendingDefaultAdminSchedule = newSchedule;

    // An `oldSchedule` from `pendingDefaultAdmin()` is only set if it hasn't been accepted.
    if (_isScheduleSet(oldSchedule)) {
      // Emit for implicit cancellations when another default admin was scheduled.
      emit DefaultAdminTransferCanceled();
    }
  }

  /**
   * @dev Setter of the tuple for pending delay and its schedule.
   *
   * May emit a DefaultAdminDelayChangeCanceled event.
   */
  function _setPendingDelay(uint48 newDelay, uint48 newSchedule) private {
    uint48 oldSchedule = _pendingDelaySchedule;

    if (_isScheduleSet(oldSchedule)) {
      if (_hasSchedulePassed(oldSchedule)) {
        // Materialize a virtual delay
        _currentDelay = _pendingDelay;
      } else {
        // Emit for implicit cancellations when another delay was scheduled.
        emit DefaultAdminDelayChangeCanceled();
      }
    }

    _pendingDelay = newDelay;
    _pendingDelaySchedule = newSchedule;
  }

  ///
  /// Private helpers
  ///

  /**
   * @dev Defines if an `schedule` is considered set. For consistency purposes.
   */
  function _isScheduleSet(
    uint48 schedule
  ) private pure returns (bool) {
    return schedule != 0;
  }

  /**
   * @dev Defines if an `schedule` is considered passed. For consistency purposes.
   */
  function _hasSchedulePassed(
    uint48 schedule
  ) private view returns (bool) {
    return schedule < block.timestamp;
  }
}

// src/PausableWithAccessControl.sol

/// @notice Base contract that adds pausing and access control functionality.
abstract contract PausableWithAccessControl is
  Pausable,
  AccessControlDefaultAdminRules,
  IPausable,
  IAccessControlEnumerable
{
  using EnumerableSet for EnumerableSet.AddressSet;

  /// @notice The set of members in each role
  mapping(bytes32 role => EnumerableSet.AddressSet) private s_roleMembers;

  constructor(
    uint48 adminRoleTransferDelay,
    address admin
  ) AccessControlDefaultAdminRules(adminRoleTransferDelay, admin) {}

  /// @notice This function pauses the contract
  /// @dev Sets the pause flag to true
  function emergencyPause() external onlyRole(Roles.PAUSER_ROLE) {
    _pause();
  }

  /// @inheritdoc AccessControlDefaultAdminRules
  function supportsInterface(
    bytes4 interfaceId
  ) public view virtual override returns (bool) {
    return interfaceId == type(IAccessControlEnumerable).interfaceId || super.supportsInterface(interfaceId);
  }

  /// @notice This function unpauses the contract
  /// @dev Sets the pause flag to false
  function emergencyUnpause() external onlyRole(Roles.UNPAUSER_ROLE) {
    _unpause();
  }

  /// @inheritdoc IAccessControlEnumerable
  function getRoleMember(bytes32 role, uint256 index) external view override returns (address) {
    return s_roleMembers[role].at(index);
  }

  /// @inheritdoc IAccessControlEnumerable
  function getRoleMemberCount(
    bytes32 role
  ) external view override returns (uint256) {
    return s_roleMembers[role].length();
  }

  /// @notice This function returns the members of a role
  /// @param role The role to get the members of
  /// @return roleMembers members of the role
  function getRoleMembers(
    bytes32 role
  ) public view virtual returns (address[] memory roleMembers) {
    return s_roleMembers[role].values();
  }

  /// @inheritdoc AccessControlDefaultAdminRules
  function _grantRole(bytes32 role, address account) internal virtual override returns (bool) {
    bool granted = super._grantRole(role, account);
    if (granted) {
      s_roleMembers[role].add(account);
    }
    return granted;
  }

  /// @inheritdoc AccessControlDefaultAdminRules
  function _revokeRole(bytes32 role, address account) internal virtual override returns (bool) {
    bool revoked = super._revokeRole(role, account);
    if (revoked) {
      s_roleMembers[role].remove(account);
    }
    return revoked;
  }
}

// src/EmergencyWithdrawer.sol

/// @notice Base contract that adds ERC20 emergencyWithdraw functionality.
abstract contract EmergencyWithdrawer is PausableWithAccessControl {
  using SafeERC20 for IERC20;

  /// @notice This event is emitted when an asset is withdrawn from the contract by the admin during
  /// an emergency
  /// @param to The address of the admin
  /// @param asset The address of the asset that was withdrawn
  /// @param amount The amount of assets that was withdrawn
  event AssetEmergencyWithdrawn(address indexed to, address indexed asset, uint256 amount);

  /// @notice This error is thrown when a native token transfer fails
  /// @param to The address of the recipient
  /// @param amount The amount of native token transferred - address(0) is used for native token
  /// @param data The bubbled up revert data
  error FailedNativeTokenTransfer(address to, uint256 amount, bytes data);

  constructor(uint48 adminRoleTransferDelay, address admin) PausableWithAccessControl(adminRoleTransferDelay, admin) {}

  /// @notice Withdraws assets from the contract to the specfied address
  /// @dev precondition - The contract must be paused
  /// @dev precondition - The caller must have the DEFAULT_ADMIN_ROLE
  /// @dev precondition - The assetAmounts list must not be empty
  /// @param to The address to transfer the assets to
  /// @param assetAmounts The list of assets and amounts to transfer
  function emergencyWithdraw(
    address to,
    Common.AssetAmount[] calldata assetAmounts
  ) external whenPaused onlyRole(DEFAULT_ADMIN_ROLE) {
    if (assetAmounts.length == 0) {
      revert Errors.EmptyList();
    }

    for (uint256 i; i < assetAmounts.length; ++i) {
      address asset = assetAmounts[i].asset;
      uint256 amount = assetAmounts[i].amount;
      _transferAsset(to, asset, amount);
      emit AssetEmergencyWithdrawn(to, asset, amount);
    }
  }

  /// @notice Withdraws native token from the contract to the specfied address
  /// @dev precondition The contract must be paused
  /// @dev precondition The caller must have the DEFAULT_ADMIN_ROLE
  /// @param amount The amount of native token to transfer
  function emergencyWithdrawNative(address payable to, uint256 amount) external whenPaused onlyRole(DEFAULT_ADMIN_ROLE) {
    _transferNative(to, amount);
    emit AssetEmergencyWithdrawn(to, address(0), amount);
  }

  /// @dev Helper function to withdraw native tokens and perform sanity checks
  /// @dev precondition The recipient must not be the zero address
  /// @dev precondition The amount must be greater than zero
  /// @param to The address to transfer the native tokens to
  /// @param amount The amount of native tokens to transfer
  function _transferNative(address payable to, uint256 amount) internal {
    if (to == address(0)) {
      revert Errors.InvalidZeroAddress();
    }
    if (amount == 0) {
      revert Errors.InvalidZeroAmount();
    }

    (bool success, bytes memory data) = to.call{value: amount}("");

    if (!success) {
      revert FailedNativeTokenTransfer(to, amount, data);
    }
  }

  /// @dev Helper function to transfer a list of assets
  /// @dev precondition The transferred assets must not be the zero address
  /// @dev precondition The amounts must be greater than zero
  /// @param to The address to transfer the asset to
  /// @param asset The asset to transfer
  /// @param amount The amount of asset to transfer
  function _transferAsset(address to, address asset, uint256 amount) internal {
    if (to == address(0) || asset == address(0)) {
      revert Errors.InvalidZeroAddress();
    }
    if (amount == 0) {
      revert Errors.InvalidZeroAmount();
    }

    IERC20(asset).safeTransfer(to, amount);
  }
}

// src/FeeAggregator.sol

/// @notice Contract which accrues assets and enables transferring out assets for swapping and further settlement to
/// swapper roles.
/// The contract enables opt-in support to receive assets from other chains via CCIP,
/// as well as bridge assets to other chains (to allowlisted receivers).
contract FeeAggregator is
  IFeeAggregator,
  EmergencyWithdrawer,
  LinkReceiver,
  ITypeAndVersion,
  ILinkAvailable,
  NativeTokenReceiver
{
  using EnumerableSet for EnumerableSet.AddressSet;
  using SafeERC20 for IERC20;
  using EnumerableSet for EnumerableSet.UintSet;
  using EnumerableBytesSet for EnumerableBytesSet.BytesSet;

  /// @notice This event is emitted when an asset is removed from the allowlist
  /// @param asset The address of the asset that was removed from the allowlist
  event AssetRemovedFromAllowlist(address asset);
  /// @notice This event is emitted when an asset is added to the allow list
  /// @param asset The address of the asset that was added to the allow list
  event AssetAddedToAllowlist(address asset);
  /// @notice This event is emitted when the CCIP Router Client address is set
  /// @param ccipRouter The address of the CCIP Router Client
  event CCIPRouterClientSet(address indexed ccipRouter);
  /// @notice This event is emitted when a destination chain is added to the allowlist
  /// @param chainSelector The selector of the destination chain that was added to the allowlist
  event DestinationChainAddedToAllowlist(uint64 chainSelector);
  /// @notice This event is emitted when a destination chain is removed from the allowlist
  /// @param chainSelector The selector of the destination chain that was removed from the allowlist
  event DestinationChainRemovedFromAllowlist(uint64 chainSelector);
  /// @notice This event is emitted when a receiver is added to the allowlist
  /// @param chainSelector The destination chain selector
  /// @param receiver The encoded address of the receiver that was added
  event ReceiverAddedToAllowlist(uint64 indexed chainSelector, bytes receiver);
  /// @notice This event is emitted when a receiver is removed from the allowlist
  /// @param chainSelector The destination chain selector
  /// @param receiver The encoded address of the receiver that was removed
  event ReceiverRemovedFromAllowlist(uint64 indexed chainSelector, bytes receiver);
  /// @notice This event is emitted when an asset is transferred for swapping
  /// @param to The address to which the asset was sent
  /// @param asset The address of the asset that was transferred
  /// @param amount The amount of asset that was transferred
  event AssetTransferredForSwap(address indexed to, address indexed asset, uint256 amount);
  /// @notice This event is emitted when a non allowlisted asset is withdrawn
  /// @param to The address that received the withdrawn asset
  /// @param asset The address of the asset that was withdrawn - address(0) is used for native token
  /// @param amount The amount of assets that was withdrawn
  event NonAllowlistedAssetWithdrawn(address indexed to, address indexed asset, uint256 amount);
  /// @notice This event is emitted when a bridgeAssets call is successfully initiated
  /// @param messageId CCIP Message ID
  /// @param message Message contents
  event BridgeAssetsMessageSent(bytes32 indexed messageId, Client.EVM2AnyMessage message);

  /// @notice This error is thrown when the contract's balance is not
  /// enough to pay bridging fees
  /// @param currentBalance The contract's balance in juels
  /// @param fee The minimum amount of juels required to bridge assets
  error InsufficientBalance(uint256 currentBalance, uint256 fee);
  /// @notice This error is thrown when an asset is being allow listed while
  /// already allow listed
  /// @param asset The asset that is already allowlisted
  error AssetAlreadyAllowlisted(address asset);
  /// @notice This error is thrown when attempting to remove a receiver that is
  /// not on the allowlist
  /// @param receiver The receiver that was not allowlisted
  /// @param chainSelector The destination chain selector that the receiver was not allowlisted for
  error ReceiverNotAllowlisted(uint64 chainSelector, bytes receiver);
  /// @notice This error is thrown when a receiver being added to the allowlist is already in the
  /// allowlist
  /// @param receiver The receiver that was already allowlisted
  /// @param chainSelector The destination chain selector that the receiver was already allowlisted for
  error ReceiverAlreadyAllowlisted(uint64 chainSelector, bytes receiver);
  /// @notice This error is thrown when attempting to add a 0 destination or source chain selector
  error InvalidChainSelector();

  /// @notice Parameters to instantiate the contract in the constructor
  // solhint-disable-next-line gas-struct-packing
  struct ConstructorParams {
    address admin; // ──────────────────╮ The initial contract admin
    uint48 adminRoleTransferDelay; // ──╯ The min seconds before the admin address can be transferred
    address linkToken; // The LINK token
    address ccipRouterClient; // The CCIP Router client
    address wrappedNativeToken; // The wrapped native token
  }

  /// @notice This struct contains the parameters to allowlist remote receivers on a given chain
  struct AllowlistedReceivers {
    uint64 remoteChainSelector; // ──╮ The remote chain selector to allowlist
    bytes[] receivers; // ───────────╯ The list of encoded remote receivers
  }

  /// @inheritdoc ITypeAndVersion
  string public constant override typeAndVersion = "Fee Aggregator 1.0.0";

  /// @dev Hash of encoded address(0) used for empty bytes32 address checks
  bytes32 internal constant EMPTY_ENCODED_BYTES32_ADDRESS_HASH = keccak256(abi.encode(address(0)));

  /// @notice CCIP Router client
  IRouterClient internal immutable i_ccipRouter;

  /// @notice The set of assets that are allowed to be bridged
  EnumerableSet.AddressSet internal s_allowlistedAssets;
  /// @notice The set of destination chain selectors that are allowed to receiver assets to the contract
  EnumerableSet.UintSet private s_allowlistedDestinationChains;

  /// @notice Mapping of chain selectors to the set of encoded addresses that are allowed to receive assets
  /// @dev We use bytes to store the addresses because CCIP transmits addresses as raw bytes.
  mapping(uint64 chainSelector => EnumerableBytesSet.BytesSet receivers) private s_allowlistedReceivers;

  constructor(
    ConstructorParams memory params
  )
    EmergencyWithdrawer(params.adminRoleTransferDelay, params.admin)
    LinkReceiver(params.linkToken)
    NativeTokenReceiver(params.wrappedNativeToken)
  {
    if (params.ccipRouterClient == address(0)) {
      revert Errors.InvalidZeroAddress();
    }

    i_ccipRouter = IRouterClient(params.ccipRouterClient);
    emit CCIPRouterClientSet(params.ccipRouterClient);
  }

  /// @inheritdoc IERC165
  function supportsInterface(
    bytes4 interfaceId
  ) public view override(PausableWithAccessControl) returns (bool) {
    return (interfaceId == type(IFeeAggregator).interfaceId || PausableWithAccessControl.supportsInterface(interfaceId));
  }

  // ================================================================
  // │                     Receive & Swap Assets                    │
  // ================================================================

  /// @inheritdoc IFeeAggregator
  /// @dev precondition - the caller must have the SWAPPER_ROLE
  /// @dev precondition - the assetAmounts list must not be empty
  /// @dev precondition - the assets must be allowlisted
  /// @dev precondition - the amounts must be greater than 0
  function transferForSwap(
    address to,
    Common.AssetAmount[] calldata assetAmounts
  ) external whenNotPaused onlyRole(Roles.SWAPPER_ROLE) {
    if (assetAmounts.length == 0) {
      revert Errors.EmptyList();
    }

    for (uint256 i; i < assetAmounts.length; ++i) {
      address asset = assetAmounts[i].asset;
      uint256 amount = assetAmounts[i].amount;

      if (!s_allowlistedAssets.contains(asset)) {
        revert Errors.AssetNotAllowlisted(asset);
      }

      _transferAsset(to, asset, amount);
      emit AssetTransferredForSwap(to, asset, amount);
    }
  }

  /// @inheritdoc IFeeAggregator
  function isAssetAllowlisted(
    address asset
  ) external view returns (bool isAllowlisted) {
    return s_allowlistedAssets.contains(asset);
  }

  /// @notice Getter function to retrieve the list of allowlisted assets
  /// @return allowlistedAssets List of allowlisted assets
  function getAllowlistedAssets() external view returns (address[] memory allowlistedAssets) {
    return s_allowlistedAssets.values();
  }

  /// @notice Getter function to retrieve the list of allowlisted destination chains
  /// @return allowlistedDestinationChains List of allowlisted destination chains
  function getAllowlistedDestinationChains() external view returns (uint256[] memory allowlistedDestinationChains) {
    return s_allowlistedDestinationChains.values();
  }

  // ================================================================
  // │                           Bridging                           │
  // ================================================================

  /// @notice Bridges assets from the source chain to a receiving
  /// address on the destination chain
  /// @dev precondition The caller must have the BRIDGER_ROLE
  /// @dev precondition The contract must not be paused
  /// @dev precondition The contract must have sufficient LINK to pay
  /// the bridging fee
  /// @param bridgeAssetAmounts The amount of assets to bridge
  /// @param destinationChainSelector The chain to receive funds
  /// @param bridgeReceiver The address to receive funds
  /// @param extraArgs Extra arguments to pass to the CCIP
  /// @return messageId The bridging message ID
  function bridgeAssets(
    Client.EVMTokenAmount[] calldata bridgeAssetAmounts,
    uint64 destinationChainSelector,
    bytes calldata bridgeReceiver,
    bytes calldata extraArgs
  ) external whenNotPaused onlyRole(Roles.BRIDGER_ROLE) returns (bytes32 messageId) {
    if (!s_allowlistedReceivers[destinationChainSelector].contains(bridgeReceiver)) {
      revert ReceiverNotAllowlisted(destinationChainSelector, bridgeReceiver);
    }

    if (bridgeAssetAmounts.length == 0) {
      revert Errors.EmptyList();
    }

    Client.EVM2AnyMessage memory evm2AnyMessage =
      _buildBridgeAssetsMessage(bridgeAssetAmounts, bridgeReceiver, extraArgs);

    uint256 fees = i_ccipRouter.getFee(destinationChainSelector, evm2AnyMessage);

    uint256 currentBalance = i_linkToken.balanceOf(address(this));

    if (fees > currentBalance) {
      revert InsufficientBalance(currentBalance, fees);
    }

    IERC20(address(i_linkToken)).safeIncreaseAllowance(address(i_ccipRouter), fees);

    messageId = i_ccipRouter.ccipSend(destinationChainSelector, evm2AnyMessage);
    emit BridgeAssetsMessageSent(messageId, evm2AnyMessage);

    return messageId;
  }

  /// @notice Builds the CCIP message to bridge assets from the source chain
  /// to the destination chain
  /// @param bridgeAssetAmounts The assets to bridge and their amounts
  /// @param bridgeReceiver The address to receive bridged funds
  /// @param extraArgs Extra arguments to pass to the CCIP
  /// @return message The constructed CCIP message
  function _buildBridgeAssetsMessage(
    Client.EVMTokenAmount[] memory bridgeAssetAmounts,
    bytes memory bridgeReceiver,
    bytes calldata extraArgs
  ) internal returns (Client.EVM2AnyMessage memory message) {
    for (uint256 i; i < bridgeAssetAmounts.length; ++i) {
      address asset = bridgeAssetAmounts[i].token;
      if (!s_allowlistedAssets.contains(asset)) {
        revert Errors.AssetNotAllowlisted(asset);
      }

      IERC20(asset).safeIncreaseAllowance(address(i_ccipRouter), bridgeAssetAmounts[i].amount);
    }

    return Client.EVM2AnyMessage({
      receiver: bridgeReceiver,
      data: "",
      tokenAmounts: bridgeAssetAmounts,
      extraArgs: extraArgs,
      feeToken: address(i_linkToken)
    });
  }

  /// @notice Getter function to retrieve the list of allowlisted receivers for a chain
  /// @param destChainSelector The destination chain selector
  /// @return allowlistedReceivers List of encoded receiver addresses
  function getAllowlistedReceivers(
    uint64 destChainSelector
  ) external view returns (bytes[] memory allowlistedReceivers) {
    return s_allowlistedReceivers[destChainSelector].values();
  }

  /// @inheritdoc ILinkAvailable
  function linkAvailableForPayment() external view returns (int256 linkBalance) {
    // LINK balance is returned as an int256 to match the interface
    // It will never be negative and will always fit in an int256 since the max
    // supply of LINK is 1e27
    return int256(i_linkToken.balanceOf(address(this)));
  }

  /// @notice Return the current router
  /// @return ccipRouter CCIP router address
  function getRouter() public view returns (address ccipRouter) {
    return address(i_ccipRouter);
  }

  // ================================================================
  // │                   Asset administration                       │
  // ================================================================

  /// @notice Adds and removes assets from the allowlist
  /// @dev precondition The caller must have the ASSET_ADMIN_ROLE
  /// @dev precondition The contract must not be paused
  /// @dev precondition The assets to add must not be the zero address
  /// @dev precondition The assets to remove must be already allowlisted
  /// @dev precondition The assets to add must not already be allowlisted
  /// @param assetsToRemove The list of assets to remove from the allowlist
  /// @param assetsToAdd The list of assets to add to the allowlist
  function applyAllowlistedAssetUpdates(
    address[] calldata assetsToRemove,
    address[] calldata assetsToAdd
  ) external onlyRole(Roles.ASSET_ADMIN_ROLE) whenNotPaused {
    for (uint256 i; i < assetsToRemove.length; ++i) {
      address asset = assetsToRemove[i];
      if (!s_allowlistedAssets.remove(asset)) {
        revert Errors.AssetNotAllowlisted(asset);
      }
      emit AssetRemovedFromAllowlist(asset);
    }

    for (uint256 i; i < assetsToAdd.length; ++i) {
      address asset = assetsToAdd[i];
      if (asset == address(0)) {
        revert Errors.InvalidZeroAddress();
      }
      if (!s_allowlistedAssets.add(asset)) {
        revert AssetAlreadyAllowlisted(asset);
      }
      emit AssetAddedToAllowlist(asset);
    }
  }

  /// @notice Withdraws non allowlisted assets from the contract
  /// @dev precondition - The contract must not be paused
  /// @dev precondition - The caller must have the WITHDRAWER_ROLE
  /// @dev precondition - The list of assetAmounts must not be empty
  /// @dev precondition - The asset must not be the zero address
  /// @dev precondition - The amount must be greater than 0
  /// @dev precondition - The asset must not be allowlisted
  /// @param to The address to transfer the assets to
  /// @param assetAmounts The list of assets and amounts to withdraw
  function withdrawNonAllowlistedAssets(
    address to,
    Common.AssetAmount[] calldata assetAmounts
  ) external whenNotPaused onlyRole(Roles.WITHDRAWER_ROLE) {
    if (assetAmounts.length == 0) {
      revert Errors.EmptyList();
    }

    for (uint256 i; i < assetAmounts.length; ++i) {
      address asset = assetAmounts[i].asset;
      uint256 amount = assetAmounts[i].amount;

      if (s_allowlistedAssets.contains(asset)) {
        revert Errors.AssetAllowlisted(asset);
      }

      _transferAsset(to, asset, amount);
      emit NonAllowlistedAssetWithdrawn(to, asset, amount);
    }
  }

  /// @notice Withdraws native tokens from the contract to the specified address
  /// @dev precondition - The contract must not be paused
  /// @dev precondition - The caller must have the WITHDRAWER_ROLE
  /// @dev precondition - The wrapped native token must not be allowlisted
  /// @param to The address to transfer the native tokens to
  /// @param amount The amount of native tokens to transfer
  function withdrawNative(address payable to, uint256 amount) external whenNotPaused onlyRole(Roles.WITHDRAWER_ROLE) {
    address wrappedNativeToken = address(s_wrappedNativeToken);

    if (s_allowlistedAssets.contains(wrappedNativeToken)) {
      revert Errors.AssetAllowlisted(wrappedNativeToken);
    }

    _transferNative(to, amount);
    emit NonAllowlistedAssetWithdrawn(to, address(0), amount);
  }

  /// @notice Adds and removes receivers from the allowlist for specified chains
  /// @dev The caller must have the DEFAULT_ADMIN_ROLE
  /// @dev precondition The contract must not be paused
  /// @param receiversToRemove The list of receivers to remove from the allowlist
  /// @param receiversToAdd The list of receivers to add to the allowlist
  function applyAllowlistedReceiverUpdates(
    AllowlistedReceivers[] calldata receiversToRemove,
    AllowlistedReceivers[] calldata receiversToAdd
  ) external onlyRole(DEFAULT_ADMIN_ROLE) whenNotPaused {
    for (uint256 i; i < receiversToRemove.length; ++i) {
      uint64 destChainSelector = receiversToRemove[i].remoteChainSelector;
      bytes[] memory receivers = receiversToRemove[i].receivers;

      for (uint256 j; j < receivers.length; ++j) {
        bytes memory receiver = receivers[j];
        if (!s_allowlistedReceivers[destChainSelector].remove(receiver)) {
          revert ReceiverNotAllowlisted(destChainSelector, receiver);
        }
        emit ReceiverRemovedFromAllowlist(destChainSelector, receiver);
      }

      if (s_allowlistedReceivers[destChainSelector].length() == 0) {
        s_allowlistedDestinationChains.remove(destChainSelector);
        emit DestinationChainRemovedFromAllowlist(destChainSelector);
      }
    }

    // Process additions next
    for (uint256 i; i < receiversToAdd.length; ++i) {
      uint64 destChainSelector = receiversToAdd[i].remoteChainSelector;
      if (destChainSelector == 0) {
        revert InvalidChainSelector();
      }

      bytes[] memory receivers = receiversToAdd[i].receivers;

      for (uint256 j; j < receivers.length; ++j) {
        bytes memory receiver = receivers[j];
        if (receiver.length == 0 || keccak256(receiver) == EMPTY_ENCODED_BYTES32_ADDRESS_HASH) {
          revert Errors.InvalidZeroAddress();
        }

        if (!s_allowlistedReceivers[destChainSelector].add(receiver)) {
          revert ReceiverAlreadyAllowlisted(destChainSelector, receiver);
        }
        emit ReceiverAddedToAllowlist(destChainSelector, receiver);
      }

      if (s_allowlistedDestinationChains.add(destChainSelector)) {
        emit DestinationChainAddedToAllowlist(destChainSelector);
      }
    }
  }

  /// @dev Sets the wrapped native token.
  /// @dev precondition The caller must have the DEFAULT_ADMIN_ROLE
  /// @param wrappedNativeToken The wrapped native token address.
  function setWrappedNativeToken(
    address wrappedNativeToken
  ) external onlyRole(DEFAULT_ADMIN_ROLE) {
    _setWrappedNativeToken(wrappedNativeToken);
  }
}