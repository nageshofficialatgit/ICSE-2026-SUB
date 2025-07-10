// SPDX-License-Identifier: MIT
pragma solidity =0.8.26 >=0.5.0 >=0.7.5 ^0.8.0 ^0.8.20;
pragma abicoder v2;

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

// src/vendor/@aave/core-v3/contracts/protocol/libraries/math/PercentageMath.sol

/**
 * @title PercentageMath library
 * @author Aave
 * @notice Provides functions to perform percentage calculations
 * @dev Percentages are defined by default with 2 decimals of precision (100.00). The precision is indicated by
 * PERCENTAGE_FACTOR
 * @dev Operations are rounded. If a value is >=.5, will be rounded up, otherwise rounded down.
 */
library PercentageMath {
  // Maximum percentage factor (100.00%)
  uint256 internal constant PERCENTAGE_FACTOR = 1e4;

  // Half percentage factor (50.00%)
  uint256 internal constant HALF_PERCENTAGE_FACTOR = 0.5e4;

  /**
   * @notice Executes a percentage multiplication
   * @dev assembly optimized for improved gas savings, see
   * https://twitter.com/transmissions11/status/1451131036377571328
   * @param value The value of which the percentage needs to be calculated
   * @param percentage The percentage of the value to be calculated
   * @return result value percentmul percentage
   */
  function percentMul(uint256 value, uint256 percentage) internal pure returns (uint256 result) {
    // to avoid overflow, value <= (type(uint256).max - HALF_PERCENTAGE_FACTOR) / percentage
    assembly {
      if iszero(or(iszero(percentage), iszero(gt(value, div(sub(not(0), HALF_PERCENTAGE_FACTOR), percentage))))) {
        revert(0, 0)
      }

      result := div(add(mul(value, percentage), HALF_PERCENTAGE_FACTOR), PERCENTAGE_FACTOR)
    }
  }

  /**
   * @notice Executes a percentage division
   * @dev assembly optimized for improved gas savings, see
   * https://twitter.com/transmissions11/status/1451131036377571328
   * @param value The value of which the percentage needs to be calculated
   * @param percentage The percentage of the value to be calculated
   * @return result value percentdiv percentage
   */
  function percentDiv(uint256 value, uint256 percentage) internal pure returns (uint256 result) {
    // to avoid overflow, value <= (type(uint256).max - halfPercentage) / PERCENTAGE_FACTOR
    assembly {
      if or(iszero(percentage), iszero(iszero(gt(value, div(sub(not(0), div(percentage, 2)), PERCENTAGE_FACTOR))))) {
        revert(0, 0)
      }

      result := div(add(mul(value, PERCENTAGE_FACTOR), div(percentage, 2)), percentage)
    }
  }
}

// src/vendor/@chainlink/contracts/src/v0.8/automation/AutomationBase.sol

contract AutomationBase {
  error OnlySimulatedBackend();

  /**
   * @notice method that allows it to be simulated via eth_call by checking that
   * the sender is the zero address.
   */
  function preventExecution() internal view {
    if (tx.origin != address(0)) {
      revert OnlySimulatedBackend();
    }
  }

  /**
   * @notice modifier that allows it to be simulated via eth_call by checking
   * that the sender is the zero address.
   */
  modifier cannotExecute() {
    preventExecution();
    _;
  }
}

// src/vendor/@chainlink/contracts/src/v0.8/automation/interfaces/AutomationCompatibleInterface.sol

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
  function performUpkeep(
    bytes calldata performData
  ) external;
}

// src/vendor/@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol

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

// src/vendor/@chainlink/contracts/src/v0.8/shared/interfaces/ITypeAndVersion.sol

interface ITypeAndVersion {
  function typeAndVersion() external pure returns (string memory);
}

// src/vendor/@chainlink/contracts/src/v0.8/shared/interfaces/LinkTokenInterface.sol

interface LinkTokenInterface {
  function allowance(address owner, address spender) external view returns (uint256 remaining);

  function approve(address spender, uint256 value) external returns (bool success);

  function balanceOf(
    address owner
  ) external view returns (uint256 balance);

  function decimals() external view returns (uint8 decimalPlaces);

  function decreaseApproval(address spender, uint256 addedValue) external returns (bool success);

  function increaseApproval(address spender, uint256 subtractedValue) external;

  function name() external view returns (string memory tokenName);

  function symbol() external view returns (string memory tokenSymbol);

  function totalSupply() external view returns (uint256 totalTokensIssued);

  function transfer(address to, uint256 value) external returns (bool success);

  function transferAndCall(address to, uint256 value, bytes calldata data) external returns (bool success);

  function transferFrom(address from, address to, uint256 value) external returns (bool success);
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

// src/vendor/@uniswap/v3-core/contracts/interfaces/callback/IUniswapV3SwapCallback.sol

/// @title Callback for IUniswapV3PoolActions#swap
/// @notice Any contract that calls IUniswapV3PoolActions#swap must implement this interface
interface IUniswapV3SwapCallback {
  /// @notice Called to `msg.sender` after executing a swap via IUniswapV3Pool#swap.
  /// @dev In the implementation you must pay the pool tokens owed for the swap.
  /// The caller of this method must be checked to be a UniswapV3Pool deployed by the canonical UniswapV3Factory.
  /// amount0Delta and amount1Delta can both be 0 if no tokens were swapped.
  /// @param amount0Delta The amount of token0 that was sent (negative) or must be received (positive) by the pool by
  /// the end of the swap. If positive, the callback must send that amount of token0 to the pool.
  /// @param amount1Delta The amount of token1 that was sent (negative) or must be received (positive) by the pool by
  /// the end of the swap. If positive, the callback must send that amount of token1 to the pool.
  /// @param data Any data passed through by the caller via the IUniswapV3PoolActions#swap call
  function uniswapV3SwapCallback(int256 amount0Delta, int256 amount1Delta, bytes calldata data) external;
}

// src/vendor/@uniswap/v3-periphery/contracts/interfaces/IQuoterV2.sol

/// @title QuoterV2 Interface
/// @notice Supports quoting the calculated amounts from exact input or exact output swaps.
/// @notice For each pool also tells you the number of initialized ticks crossed and the sqrt price of the pool after
/// the swap.
/// @dev These functions are not marked view because they rely on calling non-view functions and reverting
/// to compute the result. They are also not gas efficient and should not be called on-chain.
interface IQuoterV2 {
  /// @notice Returns the amount out received for a given exact input swap without executing the swap
  /// @param path The path of the swap, i.e. each token pair and the pool fee
  /// @param amountIn The amount of the first token to swap
  /// @return amountOut The amount of the last token that would be received
  /// @return sqrtPriceX96AfterList List of the sqrt price after the swap for each pool in the path
  /// @return initializedTicksCrossedList List of the initialized ticks that the swap crossed for each pool in the path
  /// @return gasEstimate The estimate of the gas that the swap consumes
  function quoteExactInput(
    bytes memory path,
    uint256 amountIn
  )
    external
    returns (
      uint256 amountOut,
      uint160[] memory sqrtPriceX96AfterList,
      uint32[] memory initializedTicksCrossedList,
      uint256 gasEstimate
    );

  struct QuoteExactInputSingleParams {
    address tokenIn;
    address tokenOut;
    uint256 amountIn;
    uint24 fee;
    uint160 sqrtPriceLimitX96;
  }

  /// @notice Returns the amount out received for a given exact input but for a swap of a single pool
  /// @param params The params for the quote, encoded as `QuoteExactInputSingleParams`
  /// tokenIn The token being swapped in
  /// tokenOut The token being swapped out
  /// fee The fee of the token pool to consider for the pair
  /// amountIn The desired input amount
  /// sqrtPriceLimitX96 The price limit of the pool that cannot be exceeded by the swap
  /// @return amountOut The amount of `tokenOut` that would be received
  /// @return sqrtPriceX96After The sqrt price of the pool after the swap
  /// @return initializedTicksCrossed The number of initialized ticks that the swap crossed
  /// @return gasEstimate The estimate of the gas that the swap consumes
  function quoteExactInputSingle(
    QuoteExactInputSingleParams memory params
  )
    external
    returns (uint256 amountOut, uint160 sqrtPriceX96After, uint32 initializedTicksCrossed, uint256 gasEstimate);

  /// @notice Returns the amount in required for a given exact output swap without executing the swap
  /// @param path The path of the swap, i.e. each token pair and the pool fee. Path must be provided in reverse order
  /// @param amountOut The amount of the last token to receive
  /// @return amountIn The amount of first token required to be paid
  /// @return sqrtPriceX96AfterList List of the sqrt price after the swap for each pool in the path
  /// @return initializedTicksCrossedList List of the initialized ticks that the swap crossed for each pool in the path
  /// @return gasEstimate The estimate of the gas that the swap consumes
  function quoteExactOutput(
    bytes memory path,
    uint256 amountOut
  )
    external
    returns (
      uint256 amountIn,
      uint160[] memory sqrtPriceX96AfterList,
      uint32[] memory initializedTicksCrossedList,
      uint256 gasEstimate
    );

  struct QuoteExactOutputSingleParams {
    address tokenIn;
    address tokenOut;
    uint256 amount;
    uint24 fee;
    uint160 sqrtPriceLimitX96;
  }

  /// @notice Returns the amount in required to receive the given exact output amount but for a swap of a single pool
  /// @param params The params for the quote, encoded as `QuoteExactOutputSingleParams`
  /// tokenIn The token being swapped in
  /// tokenOut The token being swapped out
  /// fee The fee of the token pool to consider for the pair
  /// amountOut The desired output amount
  /// sqrtPriceLimitX96 The price limit of the pool that cannot be exceeded by the swap
  /// @return amountIn The amount required as the input for the swap in order to receive `amountOut`
  /// @return sqrtPriceX96After The sqrt price of the pool after the swap
  /// @return initializedTicksCrossed The number of initialized ticks that the swap crossed
  /// @return gasEstimate The estimate of the gas that the swap consumes
  function quoteExactOutputSingle(
    QuoteExactOutputSingleParams memory params
  ) external returns (uint256 amountIn, uint160 sqrtPriceX96After, uint32 initializedTicksCrossed, uint256 gasEstimate);
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

// src/vendor/@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol

// OpenZeppelin Contracts (last updated v5.0.0) (token/ERC20/extensions/IERC20Metadata.sol)

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

// src/vendor/@uniswap/swap-router-contracts/contracts/interfaces/IV3SwapRouter.sol

/// @title Router token swapping functionality
/// @notice Functions for swapping tokens via Uniswap V3
interface IV3SwapRouter is IUniswapV3SwapCallback {
  struct ExactInputSingleParams {
    address tokenIn;
    address tokenOut;
    uint24 fee;
    address recipient;
    uint256 amountIn;
    uint256 amountOutMinimum;
    uint160 sqrtPriceLimitX96;
  }

  /// @notice Swaps `amountIn` of one token for as much as possible of another token
  /// @dev Setting `amountIn` to 0 will cause the contract to look up its own balance,
  /// and swap the entire amount, enabling contracts to send tokens before calling this function.
  /// @param params The parameters necessary for the swap, encoded as `ExactInputSingleParams` in calldata
  /// @return amountOut The amount of the received token
  function exactInputSingle(
    ExactInputSingleParams calldata params
  ) external payable returns (uint256 amountOut);

  struct ExactInputParams {
    bytes path;
    address recipient;
    uint256 amountIn;
    uint256 amountOutMinimum;
  }

  /// @notice Swaps `amountIn` of one token for as much as possible of another along the specified path
  /// @dev Setting `amountIn` to 0 will cause the contract to look up its own balance,
  /// and swap the entire amount, enabling contracts to send tokens before calling this function.
  /// @param params The parameters necessary for the multi-hop swap, encoded as `ExactInputParams` in calldata
  /// @return amountOut The amount of the received token
  function exactInput(
    ExactInputParams calldata params
  ) external payable returns (uint256 amountOut);

  struct ExactOutputSingleParams {
    address tokenIn;
    address tokenOut;
    uint24 fee;
    address recipient;
    uint256 amountOut;
    uint256 amountInMaximum;
    uint160 sqrtPriceLimitX96;
  }

  /// @notice Swaps as little as possible of one token for `amountOut` of another token
  /// that may remain in the router after the swap.
  /// @param params The parameters necessary for the swap, encoded as `ExactOutputSingleParams` in calldata
  /// @return amountIn The amount of the input token
  function exactOutputSingle(
    ExactOutputSingleParams calldata params
  ) external payable returns (uint256 amountIn);

  struct ExactOutputParams {
    bytes path;
    address recipient;
    uint256 amountOut;
    uint256 amountInMaximum;
  }

  /// @notice Swaps as little as possible of one token for `amountOut` of another along the specified path (reversed)
  /// that may remain in the router after the swap.
  /// @param params The parameters necessary for the multi-hop swap, encoded as `ExactOutputParams` in calldata
  /// @return amountIn The amount of the input token
  function exactOutput(
    ExactOutputParams calldata params
  ) external payable returns (uint256 amountIn);
}

// src/vendor/@chainlink/contracts/src/v0.8/automation/AutomationCompatible.sol

abstract contract AutomationCompatible is AutomationBase, AutomationCompatibleInterface {}

// src/vendor/@openzeppelin/contracts/interfaces/IERC20Metadata.sol

// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC20Metadata.sol)

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

// src/SwapAutomator.sol

/// @notice Chainlink Automation upkeep implementation contract that automates swapping of FeeAggregator assets
/// into LINK by utilising Uniswap V3.
contract SwapAutomator is ITypeAndVersion, PausableWithAccessControl, AutomationCompatible {
  using PercentageMath for uint256;
  using SafeCast for int256;
  using SafeERC20 for IERC20;

  /// @notice This event is emitted when the LINK token address is set
  /// @param linkToken The LINK token address
  event LinkTokenSet(address indexed linkToken);
  /// @notice This event is emitted when the LINK/USD price feed address is set
  /// @param linkUsdFeed The address of the LINK/USD price feed
  event LINKUsdFeedSet(address indexed linkUsdFeed);
  /// @notice This event is emitted when the Uniswap Router address is set
  /// @param uniswapRouter The address of the Uniswap Router
  event UniswapRouterSet(address indexed uniswapRouter);
  /// @notice This event is emitted when the Uniswap Quoter V2 address is set
  /// @param uniswapQuoterV2 The address of the Uniswap QuoterV2
  event UniswapQuoterV2Set(address indexed uniswapQuoterV2);
  /// @notice This event is emitted when a new forwarder is set
  /// @param forwarder The address of the new forwarder
  event ForwarderSet(address forwarder);
  /// @notice This event is emitted when a new fee aggregator receiver
  /// is set
  /// @param feeAggregator The address of the fee aggregator
  event FeeAggregatorSet(address feeAggregator);
  /// @notice This event is emitted when an asset is converted to LINK
  /// @param recipient The address that received the swapped LINK
  /// @param asset The address of the asset
  /// @param amountIn The amount of assets converted to LINK
  /// @param amountOut The amount of LINK received after swapping
  event AssetSwapped(address indexed recipient, address indexed asset, uint256 amountIn, uint256 amountOut);
  /// @notice This event is emitted when new swap parameters are set for an asset
  /// @param asset The address of the asset
  /// @param params The swap parameters
  event AssetSwapParamsUpdated(address asset, SwapParams params);
  /// @notice This event is emitted when swap parameters are removed for an asset
  /// @param asset The address of the asset
  event AssetSwapParamsRemoved(address asset);
  /// @notice This event is emitted when a new deadline delay is set
  /// @param newDeadlinDelay The new deadline delay
  event DeadlineDelaySet(uint96 newDeadlinDelay);
  /// @notice This event is emitted when as swap fails
  /// @param asset The address of the asset that failed to swap
  /// @param swapInput The swap input that failed
  event AssetSwapFailure(address indexed asset, IV3SwapRouter.ExactInputParams swapInput);
  /// @notice This event is emitted when the address that will receive swapped
  /// LINK is set
  /// @param linkReceiver The address that will receive swapped LINK
  event LinkReceiverSet(address indexed linkReceiver);
  /// @notice This event is emitted when the LINK token decimals are set in the constructor
  /// @param decimals The LINK token decimals
  event LinkDecimalsSet(uint256 decimals);
  /// @notice This event is emitted when the LINK/USD feed decimals are set in the constructor
  /// @param decimals The LINK/USD feed decimals
  event LinkUsdFeedDecimalsSet(uint256 decimals);
  /// @notice This event is emitted when the maximum size of the perform data is set
  /// @param maxPerformDataSize The maximum size of the perform data
  event MaxPerformDataSizeSet(uint256 maxPerformDataSize);

  /// @notice This error is thrown when max slippage parameter set is 0, or above 100%
  /// @param maxSlippage value for max slippage passed into function
  error InvalidSlippage(uint16 maxSlippage);
  /// @notice This error is thrown when the max price deviation is set below the max slippage, or above 100%
  /// @param maxPriceDeviation value for max price deviation passed into function
  error InvalidMaxPriceDeviation(uint16 maxPriceDeviation);
  /// @notice This error is thrown when the min swap size is zero or greater than the max swap size
  error InvalidMinSwapSizeUsd();
  /// @notice This error is thrown when trying to set an empty swap path
  error EmptySwapPath();
  /// @notice This error is thrown when trying to set the deadline delay to a value lower than the
  /// minimum threshold
  error DeadlineDelayTooLow(uint96 deadlineDelay, uint96 minDeadlineDelay);
  /// @notice This error is thrown when trying to set the deadline delay to a value higher than the
  /// maximum threshold
  error DeadlineDelayTooHigh(uint96 deadlineDelay, uint96 maxDeadlineDelay);
  /// @notice This error is thrown when the transaction timestamp is greater than the deadline
  error TransactionTooOld(uint256 timestamp, uint256 deadline);
  /// @notice This error is thrown when the swap path is invalid as compared to the swap path set by
  /// the Admin.
  error InvalidSwapPath();
  /// @notice This error is thrown when the recipent of the swap param does not match the receiver's
  /// @param feeRecipient address of the fee recipient passed into function
  error FeeRecipientMismatch(address feeRecipient);
  /// @notice This error is thrown when all performed swaps have failed
  error AllSwapsFailed();
  /// @notice This error is thrown when the amount received from a swap is less than the minimum
  /// @param amountOut Uniswap extracted amount out
  /// @param minAmount Minimum amount required for swap
  error InsufficientAmountReceived(uint256 amountOut, uint256 minAmount);

  /// @notice Parameters to instantiate the contract in the constructor
  /* solhint-disable-next-line gas-struct-packing */
  struct ConstructorParams {
    uint48 adminRoleTransferDelay; // ─╮ The minimum amount of seconds that must pass before the admin address can be
    //                                 │ transferred
    address admin; // ─────────────────╯ The initial contract admin
    uint96 deadlineDelay; // ──────────╮ The maximum amount of seconds the swap transaction is valid for
    address linkToken; // ─────────────╯ The Link token
    address feeAggregator; //            The Fee Aggregator
    address linkUsdFeed; //              The link usd feed
    address uniswapRouter; //            The address of the Uniswap router
    address uniswapQuoterV2; //          The address of the Uniswap QuoterV2
    address linkReceiver; //             The address that will receive converted LINK
    uint256 maxPerformDataSize; //       The maximum size of the perform data passed to the performUpkeep function
  }

  /// @notice The parameters to perform a swap
  struct SwapParams {
    AggregatorV3Interface usdFeed; // ─╮ The asset usd feed
    uint16 maxSlippage; //             │ The maximum allowed slippage for the swap in basis points
    uint16 maxPriceDeviation; //       │ The maximum allowed one-side deviation of actual swapped out amount
    //                                 │ vs CLprice feed estimated amount, in basis points
    uint32 swapInterval; //            │ The minimum interval between swaps
    uint32 stalenessThreshold; // ─────╯ The staleness threshold for price feed data
    uint128 minSwapSizeUsd; // ────────╮ The minimum swap size expressed in USD feed decimals
    uint128 maxSwapSizeUsd; // ────────╯ The maximum swap size expressed in USD feed decimals
    bytes path; // The swap path
  }

  /// @notice Contains the swap parameters for an asset
  struct AssetSwapParamsArgs {
    address asset; // The asset
    SwapParams swapParams; // The asset's swap parameters
  }

  /// @inheritdoc ITypeAndVersion
  string public constant override typeAndVersion = "Uniswap V3 Swap Automator 1.0.0";
  /// @notice The lower bound for the deadline delay
  uint96 private constant MIN_DEADLINE_DELAY = 1 minutes;
  /// @notice The upper bound for the deadline delay
  uint96 private constant MAX_DEADLINE_DELAY = 1 hours;

  /// @notice The link token
  LinkTokenInterface private immutable i_linkToken;
  /// @notice The address of the chainlink USD feed
  AggregatorV3Interface private immutable i_linkUsdFeed;
  /// @notice The address of the Uniswap router
  IV3SwapRouter private immutable i_uniswapRouter;
  /// @notice The address of the Uniswap QuoterV2
  IQuoterV2 private immutable i_uniswapQuoterV2;
  /// @notice The number of decimals for the LINK token
  uint256 private immutable i_linkDecimals;
  /// @notice The number of decimals for the LINK/USD feed
  uint256 private immutable i_linkUsdFeedDecimals;

  /// @notice The address will execute the automation job
  address private s_forwarder;
  /// @notice The maximum amount of seconds the swap transaction is valid for
  uint96 private s_deadlineDelay;

  /// @notice The fee aggregator
  IFeeAggregator private s_feeAggregator;
  /// @notice The receiver of LINK tokens
  address private s_linkReceiver;
  /// @notice The maximum size of the perform data passed to the performUpkeep function
  uint256 private s_maxPerformDataSize;

  /// @notice Mapping of assets to their swap parameters
  mapping(address asset => SwapParams swapParams) private s_assetSwapParams;
  /// @notice Mapping of assets to their lastest swap timestamp
  mapping(address asset => uint256 latestSwapTimestamp) private s_latestSwapTimestamp;
  /// @notice Mapping of assets to their hashed swap path
  mapping(address asset => bytes32 hashedSwapPath) private s_assetHashedSwapPath;

  constructor(
    ConstructorParams memory params
  ) PausableWithAccessControl(params.adminRoleTransferDelay, params.admin) {
    if (
      params.linkToken == address(0) || params.linkUsdFeed == address(0) || params.uniswapRouter == address(0)
        || params.uniswapQuoterV2 == address(0)
    ) {
      revert Errors.InvalidZeroAddress();
    }

    i_linkToken = LinkTokenInterface(params.linkToken);
    i_linkUsdFeed = AggregatorV3Interface(params.linkUsdFeed);
    i_uniswapRouter = IV3SwapRouter(params.uniswapRouter);
    i_uniswapQuoterV2 = IQuoterV2(params.uniswapQuoterV2);
    i_linkDecimals = IERC20Metadata(params.linkToken).decimals();
    i_linkUsdFeedDecimals = AggregatorV3Interface(params.linkUsdFeed).decimals();

    emit LinkTokenSet(params.linkToken);
    emit LINKUsdFeedSet(params.linkUsdFeed);
    emit UniswapRouterSet(params.uniswapRouter);
    emit UniswapQuoterV2Set(params.uniswapQuoterV2);
    emit LinkDecimalsSet(i_linkDecimals);
    emit LinkUsdFeedDecimalsSet(i_linkUsdFeedDecimals);

    _setFeeAggregator(params.feeAggregator);
    _setDeadlineDelay(params.deadlineDelay);
    _setLinkReceiver(params.linkReceiver);
    _setMaxPerformDataSize(params.maxPerformDataSize);
  }

  /// @notice Set the address that `performUpkeep` is called from
  /// @dev precondition The caller must have the DEFAULT_ADMIN_ROLE
  /// @dev precondition The contract must not be paused
  /// @dev precondition The forwarder address must not be the zero address
  /// @param forwarder the address to set
  function setForwarder(
    address forwarder
  ) external onlyRole(DEFAULT_ADMIN_ROLE) {
    if (forwarder == address(0)) {
      revert Errors.InvalidZeroAddress();
    }
    if (s_forwarder == forwarder) {
      revert Errors.ValueNotUpdated();
    }
    s_forwarder = forwarder;
    emit ForwarderSet(forwarder);
  }

  /// @notice Sets and removes swap parameters for assets
  /// @dev precondition The caller must have the ASSET_ADMIN_ROLE
  /// @dev precondition The assets must be allowlisted on the FeeAggregator
  /// @dev precondition The asset list length must match the params list length
  /// @dev precondition The assets feed addresses must not be the zero address
  /// @dev precondition The assets token address must not be the zero address
  /// @dev precondition The assets maxSlippage must be greater than 0
  /// @param assetsToRemove The list of assets to remove swap parameters
  /// @param assetSwapParamsArgs The asset swap parameters arguments
  function applyAssetSwapParamsUpdates(
    address[] calldata assetsToRemove,
    AssetSwapParamsArgs[] calldata assetSwapParamsArgs
  ) external onlyRole(Roles.ASSET_ADMIN_ROLE) {
    // process removals first
    for (uint256 i; i < assetsToRemove.length; ++i) {
      delete s_assetSwapParams[assetsToRemove[i]];
      delete s_assetHashedSwapPath[assetsToRemove[i]];

      emit AssetSwapParamsRemoved(assetsToRemove[i]);
    }

    IFeeAggregator feeAggregator = s_feeAggregator;

    for (uint256 i; i < assetSwapParamsArgs.length; ++i) {
      SwapParams memory assetSwapParams = assetSwapParamsArgs[i].swapParams;
      address assetAddress = assetSwapParamsArgs[i].asset;

      if (!feeAggregator.isAssetAllowlisted(assetAddress)) {
        revert Errors.AssetNotAllowlisted(assetAddress);
      }
      if (address(assetSwapParams.usdFeed) == address(0)) {
        revert Errors.InvalidZeroAddress();
      }
      if (assetSwapParams.maxSlippage == 0 || assetSwapParams.maxSlippage >= PercentageMath.PERCENTAGE_FACTOR) {
        revert InvalidSlippage(assetSwapParams.maxSlippage);
      }
      if (
        assetSwapParams.maxPriceDeviation < assetSwapParams.maxSlippage
          || assetSwapParams.maxPriceDeviation >= PercentageMath.PERCENTAGE_FACTOR
      ) {
        revert InvalidMaxPriceDeviation(assetSwapParams.maxPriceDeviation);
      }
      if (assetSwapParams.stalenessThreshold == 0) {
        revert Errors.InvalidZeroAmount();
      }
      if (assetSwapParams.minSwapSizeUsd == 0 || assetSwapParams.minSwapSizeUsd > assetSwapParams.maxSwapSizeUsd) {
        revert InvalidMinSwapSizeUsd();
      }
      if (assetSwapParams.path.length == 0) {
        revert EmptySwapPath();
      }

      s_assetSwapParams[assetAddress] = assetSwapParams;
      s_assetHashedSwapPath[assetAddress] = keccak256(assetSwapParams.path);

      emit AssetSwapParamsUpdated(assetAddress, assetSwapParams);
    }
  }

  /// @notice Gets the swap params for an asset
  /// @param asset The address of the asset
  /// @return swapParams The swap parameters for the asset
  function getAssetSwapParams(
    address asset
  ) external view returns (SwapParams memory swapParams) {
    return s_assetSwapParams[asset];
  }

  /// @notice Sets the fee aggregator receiver
  /// @dev precondition The caller must have the DEFAULT_ADMIN_ROLE
  /// @dev precondition The new fee aggregator address must
  /// not be the zero address
  /// @dev precondition The new fee aggregator address must be
  /// different from the already configured fee aggregator
  function setFeeAggregator(
    address feeAggregator
  ) external onlyRole(DEFAULT_ADMIN_ROLE) {
    _setFeeAggregator(feeAggregator);
  }

  /// @notice Sets the fee aggregator
  /// @param feeAggregator The new fee aggregator
  function _setFeeAggregator(
    address feeAggregator
  ) internal {
    if (feeAggregator == address(0)) {
      revert Errors.InvalidZeroAddress();
    }
    if (address(s_feeAggregator) == feeAggregator) {
      revert Errors.ValueNotUpdated();
    }
    if (!IERC165(feeAggregator).supportsInterface(type(IFeeAggregator).interfaceId)) {
      revert Errors.InvalidFeeAggregator(feeAggregator);
    }

    s_feeAggregator = IFeeAggregator(feeAggregator);

    emit FeeAggregatorSet(feeAggregator);
  }

  /// @notice Sets a new deadline delay
  /// @dev precondition The caller must have the ASSET_ADMIN_ROLE
  /// @dev precondition The new deadline delay must be lower or equal than the maximum deadline
  /// delay
  /// @dev precondition The new deadline delay must be different from the already set deadline delay
  /// @param deadlineDelay The new deadline delay
  function setDeadlineDelay(
    uint96 deadlineDelay
  ) external onlyRole(Roles.ASSET_ADMIN_ROLE) {
    _setDeadlineDelay(deadlineDelay);
  }

  /// @notice Sets the deadline delay
  /// @param deadlineDelay The new deadline delay
  function _setDeadlineDelay(
    uint96 deadlineDelay
  ) internal {
    if (s_deadlineDelay == deadlineDelay) {
      revert Errors.ValueNotUpdated();
    }
    if (deadlineDelay < MIN_DEADLINE_DELAY) {
      revert DeadlineDelayTooLow(deadlineDelay, MIN_DEADLINE_DELAY);
    }
    if (deadlineDelay > MAX_DEADLINE_DELAY) {
      revert DeadlineDelayTooHigh(deadlineDelay, MAX_DEADLINE_DELAY);
    }

    s_deadlineDelay = deadlineDelay;
    emit DeadlineDelaySet(deadlineDelay);
  }

  /// @notice Sets the maximum size of the perform data passed to the performUpkeep function
  /// @dev precondition - The caller must have the DEFAULT_ADMIN_ROLE
  /// @param maxPerformDataSize The maximum size of the perform data
  function setMaxPerformDataSize(
    uint256 maxPerformDataSize
  ) external onlyRole(DEFAULT_ADMIN_ROLE) {
    _setMaxPerformDataSize(maxPerformDataSize);
  }

  /// @notice Sets the maximum size of the perform data passed to the performUpkeep function
  /// @dev precondition - The new maximum size must be greater than 0
  /// @dev precondition - The new maximum size must different than the old one
  /// @param maxPerformDataSize The maximum size of the perform data
  function _setMaxPerformDataSize(
    uint256 maxPerformDataSize
  ) internal {
    if (maxPerformDataSize == 0) {
      revert Errors.InvalidZeroAmount();
    }
    if (maxPerformDataSize == s_maxPerformDataSize) {
      revert Errors.ValueNotUpdated();
    }

    s_maxPerformDataSize = maxPerformDataSize;
    emit MaxPerformDataSizeSet(maxPerformDataSize);
  }

  /// @notice Getter function to retrieve the maximum perform data size
  /// @return maxPrformDataSize the maximum data size that the performUpkeep function accepts
  function getMaxPerformDataSize() external view returns (uint256 maxPrformDataSize) {
    return s_maxPerformDataSize;
  }

  /// @notice Getter function to retrieve the LINK/USD feed
  /// @return linkUsdFeed The LINK/USD feed
  function getLINKUsdFeed() external view returns (AggregatorV3Interface linkUsdFeed) {
    return i_linkUsdFeed;
  }

  /// @notice Getter function to retrieve the address that `performUpkeep` is called from
  /// @return forwarder The address that `performUpkeep` is called from
  function getForwarder() external view returns (address forwarder) {
    return s_forwarder;
  }

  /// @notice Getter function to retrieve the LINK token used
  /// @return linkToken The LINK token
  function getLinkToken() external view returns (LinkTokenInterface linkToken) {
    return i_linkToken;
  }

  /// @notice Getter function to retrieve the Uniswap Router used for swaps
  /// @return uniswapRouter The Uniswap Router
  function getUniswapRouter() external view returns (IV3SwapRouter uniswapRouter) {
    return i_uniswapRouter;
  }

  /// @notice Getter function to retrieve the Uniswap QuoterV2 used for quotes
  /// @return uniswapQuoter The Uniswap QuoterV2
  function getUniswapQuoterV2() external view returns (IQuoterV2 uniswapQuoter) {
    return i_uniswapQuoterV2;
  }

  /// @notice Getter function to retrieve the configured fee aggregator
  /// @return feeAggregator The configured fee aggregator
  function getFeeAggregator() external view returns (IFeeAggregator feeAggregator) {
    return s_feeAggregator;
  }

  /// @notice Getter function to retrieve the latest swap timestamp for an asset
  /// @param asset The address of the asset
  /// @return latestSwapTimestamp Latest swap timestamp for an asset, or 0 if never swapped
  function getLatestSwapTimestamp(
    address asset
  ) external view returns (uint256 latestSwapTimestamp) {
    return s_latestSwapTimestamp[asset];
  }

  /// @notice Getter function to retrieve the deadline delay
  /// @return deadlineDelay The deadline delay
  function getDeadlineDelay() external view returns (uint96 deadlineDelay) {
    return s_deadlineDelay;
  }

  /// @notice Getter function to retrieve the hash of the registered swap path given an asset
  /// @return hashedSwapPath The hashed swap path, 0 if asset is unregistered.
  function getHashedSwapPath(
    address asset
  ) external view returns (bytes32 hashedSwapPath) {
    return s_assetHashedSwapPath[asset];
  }

  /// @notice Getter function to retrieve the configured LINK receiver
  /// @return linkReceiver The address of the receiver
  function getLinkReceiver() external view returns (address linkReceiver) {
    return s_linkReceiver;
  }

  /// @notice Sets the address that will receive swapped LINK
  /// @dev precondition The caller must have the DEFAULT_ADMIN_ROLE
  /// @dev precondition The LINK receiver address must not be the zero address
  /// @dev precondition The LINK receiver address must be different from the already configured one
  /// @param linkReceiver The address of the address that will
  /// receive swapped LINK
  function setLinkReceiver(
    address linkReceiver
  ) external onlyRole(DEFAULT_ADMIN_ROLE) {
    _setLinkReceiver(linkReceiver);
  }

  /// @notice Sets the address that will receive swapped LINK
  /// @param linkReceiver The address of the address that will
  /// receive swapped LINK
  function _setLinkReceiver(
    address linkReceiver
  ) internal {
    if (linkReceiver == address(0)) {
      revert Errors.InvalidZeroAddress();
    }
    if (linkReceiver == s_linkReceiver) {
      revert Errors.ValueNotUpdated();
    }

    s_linkReceiver = linkReceiver;

    emit LinkReceiverSet(linkReceiver);
  }

  // ================================================================
  // │                Swap Logic And Automation                     │
  // ================================================================

  /// @inheritdoc AutomationCompatibleInterface
  /* solhint-disable-next-line chainlink-solidity/explicit-returns */
  function checkUpkeep(
    bytes calldata
  ) external whenNotPaused cannotExecute returns (bool upkeepNeeded, bytes memory performData) {
    address[] memory allowlistedAssets = s_feeAggregator.getAllowlistedAssets();
    IV3SwapRouter.ExactInputParams[] memory swapInputs = new IV3SwapRouter.ExactInputParams[](allowlistedAssets.length);
    address receiver = s_linkReceiver;
    uint256 idx;
    uint256 linkUSDPrice = _getValidatedAssetPrice(address(i_linkToken), i_linkUsdFeed);
    // The fixed size of the performData is 3 * 32 = 96 bytes corresponding to:
    // - slot 0: the offset to the encoded data
    // - slot 1: the deadlineDelay
    // - slot 2: the length of the swapInputs array
    uint256 performDataSize = 96;

    for (uint256 i; i < allowlistedAssets.length; ++i) {
      address asset = allowlistedAssets[i];

      SwapParams memory swapParams = s_assetSwapParams[asset];

      if (swapParams.usdFeed == AggregatorV3Interface(address(0))) {
        continue;
      }

      (uint256 assetPrice, uint256 updatedAt) = _getAssetPrice(swapParams.usdFeed);

      if (assetPrice == 0 || updatedAt < block.timestamp - swapParams.stalenessThreshold) {
        continue;
      }

      uint256 assetUnit = 10 ** IERC20Metadata(asset).decimals();

      // 1) Get the current asset value in USD available in the FeeAggregator
      uint256 availableAssetUsdValue = IERC20(asset).balanceOf(address(s_feeAggregator)) * assetPrice;

      // 2) Don't swap asset if the asset's current USD balance on this FeeAggregator is
      // below the minimum swap amount or if insufficient time has elapsed since the last swap
      if (
        availableAssetUsdValue >= swapParams.minSwapSizeUsd * assetUnit
          && block.timestamp - s_latestSwapTimestamp[asset] >= swapParams.swapInterval
      ) {
        // 3) Determine the swap amountIn
        uint256 swapAmountIn = Math.min(swapParams.maxSwapSizeUsd * assetUnit, availableAssetUsdValue) / assetPrice;

        // 4) Quote the amountOut from both Uniswap V3 quoter and CL price feed for all ADTs
        // except LINK
        uint256 amountOutUniswapQuote;
        uint256 amountOutCLPriceFeedQuote =
          _convertToLink(swapAmountIn, assetPrice, swapParams.usdFeed.decimals(), linkUSDPrice, IERC20Metadata(asset));

        if (asset != address(i_linkToken)) {
          (amountOutUniswapQuote,,,) = i_uniswapQuoterV2.quoteExactInput(swapParams.path, swapAmountIn);

          // 5) If amountOutUniswapQuote is below the amountOutPriceFeed with slippage, do not
          // perform swap for this asset.
          if (
            amountOutUniswapQuote
              < amountOutCLPriceFeedQuote.percentMul(PercentageMath.PERCENTAGE_FACTOR - swapParams.maxSlippage)
          ) {
            continue;
          }
        }

        // We increment the performDataSize by:
        // - 6 * 32 = 192 bytes corresponding to:
        //    - slot 3: the offset to the struct data
        //    - slot 4: the offset to the path
        //    - slot 5: the recipient
        //    - slot 6: the amountIn
        //    - slot 7: the amountOutMinimum
        //    - slot 8: the path length
        // - The number of slots required for the path:
        //    - path.length / 32 * 32 -> the rounded down number of slots required
        //    - path.length % 32 > 0 ? 32 : 0 -> +1 slot if the path length is not a multiple of 32
        performDataSize += 192 + (swapParams.path.length / 32) * 32 + (swapParams.path.length % 32 > 0 ? 32 : 0);

        // 6) If the performDataSize exceeds the maximum size, break out of the loop
        if (performDataSize > s_maxPerformDataSize) {
          break;
        }

        swapInputs[idx++] = IV3SwapRouter.ExactInputParams({
          path: swapParams.path,
          recipient: receiver,
          amountIn: swapAmountIn,
          // 7) Determine the minimum amount of juels we expect to get back by applying slippage to
          // the greater of two quotes.
          amountOutMinimum: Math.max(amountOutUniswapQuote, amountOutCLPriceFeedQuote).percentMul(
            PercentageMath.PERCENTAGE_FACTOR - swapParams.maxSlippage
          )
        });
      }
    }

    if (idx != allowlistedAssets.length) {
      assembly {
        // Update executeSwapData length
        mstore(swapInputs, idx)
      }
    }

    // Using if/else here to avoid abi.encoding empty bytes when idx = 0
    if (idx > 0) {
      return (true, abi.encode(swapInputs, block.timestamp + s_deadlineDelay));
    } else {
      return (false, "");
    }
  }

  /// @inheritdoc AutomationCompatibleInterface
  /// @dev precondition The caller must be the forwarder
  function performUpkeep(
    bytes calldata performData
  ) external whenNotPaused {
    if (msg.sender != s_forwarder) {
      revert Errors.AccessForbidden();
    }

    (IV3SwapRouter.ExactInputParams[] memory swapInputs, uint256 deadline) =
      abi.decode(performData, (IV3SwapRouter.ExactInputParams[], uint256));

    if (deadline < block.timestamp) {
      revert TransactionTooOld(block.timestamp, deadline);
    }

    bool success;
    address linkReceiver = s_linkReceiver;
    uint256 linkPriceFromFeed = _getValidatedAssetPrice(address(i_linkToken), i_linkUsdFeed);

    Common.AssetAmount[] memory assetAmounts = new Common.AssetAmount[](swapInputs.length);

    for (uint256 i; i < swapInputs.length; ++i) {
      assetAmounts[i] =
        Common.AssetAmount({asset: address(bytes20(swapInputs[i].path)), amount: swapInputs[i].amountIn});
    }

    IFeeAggregator feeAggregator = s_feeAggregator;

    feeAggregator.transferForSwap(address(this), assetAmounts);

    // This may run into out of gas errors but the likelihood is low as there
    // will not be too many assets to swap to LINK
    for (uint256 i; i < swapInputs.length; ++i) {
      bytes memory assetSwapPath = swapInputs[i].path;
      address asset = assetAmounts[i].asset;

      if (keccak256(assetSwapPath) != s_assetHashedSwapPath[asset]) {
        revert InvalidSwapPath();
      }

      if (swapInputs[i].recipient != linkReceiver) {
        revert FeeRecipientMismatch(swapInputs[i].recipient);
      }

      // Pull tokens from the FeeAggregator
      uint256 amountIn = swapInputs[i].amountIn;

      // NOTE: LINK is expected to be configured with static values:
      // pool: LINK -> LINK
      // maxSlippage: 1
      // maxSwapSizeUsd: type(uint128).max
      // swapInterval: 0
      if (asset == address(i_linkToken)) {
        IERC20(asset).safeTransfer(linkReceiver, amountIn);
        success = true;
      } else {
        IERC20(asset).safeIncreaseAllowance(address(i_uniswapRouter), amountIn);
        // For multiple swaps we don't want to revert the whole transaction if only some of the
        // swaps
        // fail so we catch the revert and continue with the next swap
        try this.swapWithPriceFeedValidation(swapInputs[i], asset, linkPriceFromFeed) returns (uint256 amountOut) {
          s_latestSwapTimestamp[asset] = block.timestamp;
          success = true;
          emit AssetSwapped(swapInputs[i].recipient, asset, amountIn, amountOut);
        } catch {
          IERC20(asset).safeDecreaseAllowance(address(i_uniswapRouter), amountIn);

          // Transfer failed swap amount back to the FeeAggregator
          IERC20(asset).safeTransfer(address(feeAggregator), amountIn);

          emit AssetSwapFailure(asset, swapInputs[i]);
        }
      }
    }

    // If all swaps have failed, revert the transaction
    if (!success) {
      revert AllSwapsFailed();
    }
  }

  /// @notice Helper function that executes the swap and check the swap amountOut against ADT & LINK
  /// price feed.
  /// @param swapInput The swapInput for Uniswap Router
  /// @param asset The address of the asset to be swapped.
  /// @param linkPriceFromFeed The price of Link from price feed
  /// @return amountOut Swapped out token amount
  function swapWithPriceFeedValidation(
    IV3SwapRouter.ExactInputParams calldata swapInput,
    address asset,
    uint256 linkPriceFromFeed
  ) external returns (uint256 amountOut) {
    if (msg.sender != address(this)) {
      revert Errors.AccessForbidden();
    }
    amountOut = i_uniswapRouter.exactInput(swapInput);

    SwapParams memory swapParams = s_assetSwapParams[asset];
    uint256 assetPriceFromPriceFeed = _getValidatedAssetPrice(asset, swapParams.usdFeed);
    uint256 linkAmountOutFromPriceFeed = _convertToLink(
      swapInput.amountIn,
      assetPriceFromPriceFeed,
      swapParams.usdFeed.decimals(),
      linkPriceFromFeed,
      IERC20Metadata(asset)
    );

    if (
      amountOut < linkAmountOutFromPriceFeed.percentMul(PercentageMath.PERCENTAGE_FACTOR - swapParams.maxPriceDeviation)
    ) {
      revert InsufficientAmountReceived(
        amountOut,
        linkAmountOutFromPriceFeed.percentMul(PercentageMath.PERCENTAGE_FACTOR - swapParams.maxPriceDeviation)
      );
    }
    return amountOut;
  }

  /// @notice Helper function to fetch an asset price
  /// @param usdFeed The USD price feed to fetch the price from
  /// @return assetPrice The asset price
  /// @return updatedAtTimestamp Timestamp at which the price was last updated
  function _getAssetPrice(
    AggregatorV3Interface usdFeed
  ) private view returns (uint256 assetPrice, uint256 updatedAtTimestamp) {
    (, int256 answer,, uint256 updatedAt,) = usdFeed.latestRoundData();
    return (answer.toUint256(), updatedAt);
  }

  /// @notice Helper function to fetch the LINK price, with feed staleness & answer validation
  /// @param asset The asset to fetch the price for
  /// @param usdFeed The USD price feed to fetch the price from
  /// @return assetPrice The asset price
  function _getValidatedAssetPrice(
    address asset,
    AggregatorV3Interface usdFeed
  ) private view returns (uint256 assetPrice) {
    (uint256 answer, uint256 updatedAt) = _getAssetPrice(usdFeed);

    if (answer == 0) {
      revert Errors.ZeroFeedData();
    }
    if (updatedAt < block.timestamp - s_assetSwapParams[asset].stalenessThreshold) {
      revert Errors.StaleFeedData();
    }

    return answer;
  }

  /// @notice Helper function to convert an asset amount to Juels denomination
  /// @param assetAmount The amount to convert
  /// @param asset The asset to convert
  /// @param assetPrice The asset price in USD
  /// @param assetFeedDecimals The asset feed decimals
  /// @param linkUSDPrice The LINK price in USD
  /// @return linkAmount The converted amount in Juels
  /* solhint-disable-next-line chainlink-solidity/explicit-returns */
  function _convertToLink(
    uint256 assetAmount,
    uint256 assetPrice,
    uint256 assetFeedDecimals,
    uint256 linkUSDPrice,
    IERC20Metadata asset
  ) private view returns (uint256 linkAmount) {
    // Scale feed decimals
    // In order to account for different decimals between the asset and the LINK/USD feed and avoid losing precision, we
    // scale the smallest decimal feed to the largest one.
    if (assetFeedDecimals > i_linkUsdFeedDecimals) {
      linkUSDPrice = linkUSDPrice * 10 ** (assetFeedDecimals - i_linkUsdFeedDecimals);
    } else if (assetFeedDecimals < i_linkUsdFeedDecimals) {
      assetPrice = assetPrice * 10 ** (i_linkUsdFeedDecimals - assetFeedDecimals);
    }

    uint256 tokenDecimals = asset.decimals();
    // Once prices are scaled, we can convert the asset amount to LINK.
    // Since the returned ammount is in LINK, token decimals must also be taken into consideration to scale the result
    // up or down.
    // Note: asset price & link USD price are normalized to the same units from the previous step.
    if (tokenDecimals < i_linkDecimals) {
      // X = linkDecimals
      // Y = tokenDecimals
      // Z = decimals for assetPrice & linkPrice
      // AA = assetAmount
      // LP = linkPrice
      // AP = assetPrice

      // (AA * 10**Y * AP * 10**Z * 10**(X - Y)) / (LP * 10**Z)
      // (AA * 10**(Y + X - Y) * AP * 10**Z) / (LP * 10**Z)
      // (AA * 10**X * AP * 10**Z) / (LP * 10**Z)
      // (AA * 10**X) * (AP * 10**Z) / (LP * 10**Z)
      // (AA * 10**X) * (AP / LP)
      return (assetAmount * assetPrice * 10 ** (i_linkDecimals - tokenDecimals)) / linkUSDPrice;
    } else {
      // X = linkDecimals
      // Y = tokenDecimals
      // Z = decimals for assetPrice & linkPrice
      // AA = assetAmount
      // LP = linkPrice
      // AP = assetPrice

      // ((AA * 10**Y * AP * 10**Z) / (LP * 10**Z)) / (10**(Y - X))
      // ((AA * 10**Y * AP) / LP) / (10**(Y - X))
      // ((AA * 10**Y) * (AP / LP)) / (10**(Y - X))
      // ((AA * (AP / LP)) * 10**Y) / (10**(Y - X))
      // (AA * (AP / LP) * 10**(Y - (Y - X)))
      // (AA * (AP / LP)) * 10**(Y - Y + X)
      // (AA * (AP / LP)) * 10**X
      // (AA * 10**X) * (AP / LP)
      return ((assetAmount * assetPrice) / linkUSDPrice) / (10 ** (tokenDecimals - i_linkDecimals));
    }
  }
}