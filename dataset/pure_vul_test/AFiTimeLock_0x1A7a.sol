// File: contracts/Interfaces/IERC20.sol


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
    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) external returns (bool);
}
// File: contracts/Interfaces/Context.sol


// OpenZeppelin Contracts v4.4.1 (utils/Context.sol)

pragma solidity ^0.8.0;

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
}
// File: contracts/Interfaces/Ownable.sol


// OpenZeppelin Contracts (last updated v4.7.0) (access/Ownable.sol)

pragma solidity ^0.8.0;


/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * By default, the owner account will be the one that deploys the contract. This
 * can later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    /**
     * @dev Initializes the contract setting the deployer as the initial owner.
     */
    constructor() {
        _transferOwnership(_msgSender());
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
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions anymore. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby removing any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(
            newOwner != address(0),
            "Ownable: new owner is the zero address"
        );
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

// File: contracts/Interfaces/Ownable2Step.sol


// OpenZeppelin Contracts (last updated v4.8.0) (access/Ownable2Step.sol)

pragma solidity ^0.8.0;


/**
 * @dev Contract module which provides access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * By default, the owner account will be the one that deploys the contract. This
 * can later be changed with {transferOwnership} and {acceptOwnership}.
 *
 * This module is used through inheritance. It will make available all functions
 * from parent (Ownable).
 */
abstract contract Ownable2Step is Ownable {
    address internal _pendingOwner;

    event OwnershipTransferStarted(
        address indexed previousOwner,
        address indexed newOwner
    );

    /**
     * @dev Returns the address of the pending owner.
     */
    function pendingOwner() public view virtual returns (address) {
        return _pendingOwner;
    }

    // /**
    //  * @dev Starts the ownership transfer of the contract to a new account. Replaces the pending transfer if there is one.
    //  * Can only be called by the current owner.
    //  */
    // function transferOwnership(address newOwner) public virtual override onlyOwner {
    //     _pendingOwner = newOwner;
    //     emit OwnershipTransferStarted(owner(), newOwner);
    // }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`) and deletes any pending owner.
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual override {
        delete _pendingOwner;
        super._transferOwnership(newOwner);
    }

    /**
     * @dev The new owner accepts the ownership transfer.
     */
    function acceptOwnership() external {
        address sender = _msgSender();
        require(
            pendingOwner() == sender,
            "Ownable2Step: caller is not the new owner"
        );
        _transferOwnership(sender);
    }
}

// File: contracts/Interfaces/OwnableDelayModule.sol


pragma solidity ^0.8.0;


contract OwnableDelayModule is Ownable2Step {
  address internal delayModule;

  constructor() {
    delayModule = msg.sender;
  }

  function isDelayModule() internal view {
    require(msg.sender == delayModule, "NA");
  }

  function setDelayModule(address _delayModule) external {
    isDelayModule();
    require(_delayModule != address(0), "ODZ");
    delayModule = _delayModule;
  }

  function getDelayModule() external view returns (address) {
    return delayModule;
  }

  /**
   * @dev Starts the ownership transfer of the contract to a new account. Replaces the pending transfer if there is one.
   * Can only be called by the current owner.
   */
  function transferOwnership(address newOwner) public override {
    isDelayModule();
    _pendingOwner = newOwner;
    emit OwnershipTransferStarted(owner(), newOwner);
  }
}

// File: contracts/Interfaces/Pausable.sol


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
// File: contracts/Interfaces/draft-IERC20Permit.sol


// OpenZeppelin Contracts v4.4.1 (token/ERC20/extensions/draft-IERC20Permit.sol)

pragma solidity ^0.8.0;

/**
 * @dev Interface of the ERC20 Permit extension allowing approvals to be made via signatures, as defined in
 * https://eips.ethereum.org/EIPS/eip-2612[EIP-2612].
 *
 * Adds the {permit} method, which can be used to change an account's ERC20 allowance (see {IERC20-allowance}) by
 * presenting a message signed by the account. By not relying on {IERC20-approve}, the token holder account doesn't
 * need to send a transaction, and thus is not required to hold Ether at all.
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
// File: contracts/Interfaces/Address.sol


// OpenZeppelin Contracts (last updated v4.8.0) (utils/Address.sol)

pragma solidity ^0.8.0;

/**
 * @dev Collection of functions related to the address type
 */
library Address {
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
     * https://diligence.consensys.net/posts/2019/09/stop-using-soliditys-transfer-now/[Learn more].
     *
     * IMPORTANT: because control is transferred to `recipient`, care must be
     * taken to not create reentrancy vulnerabilities. Consider using
     * {ReentrancyGuard} or the
     * https://solidity.readthedocs.io/en/v0.5.11/security-considerations.html#use-the-checks-effects-interactions-pattern[checks-effects-interactions pattern].
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
    function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value
    ) internal returns (bytes memory) {
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
// File: contracts/Interfaces/SafeERC20.sol


    // OpenZeppelin Contracts (last updated v4.8.0) (token/ERC20/utils/SafeERC20.sol)

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
    library SafeERC20 {
        using Address for address;

        function safeTransfer(
            IERC20 token,
            address to,
            uint256 value
        ) internal {
            _callOptionalReturn(token, abi.encodeWithSelector(token.transfer.selector, to, value));
        }

        function safeTransferFrom(
            IERC20 token,
            address from,
            address to,
            uint256 value
        ) internal {
            _callOptionalReturn(token, abi.encodeWithSelector(token.transferFrom.selector, from, to, value));
        }

        /**
        * @dev Deprecated. This function has issues similar to the ones found in
        * {IERC20-approve}, and its usage is discouraged.
        *
        * Whenever possible, use {safeIncreaseAllowance} and
        * {safeDecreaseAllowance} instead.
        */
        function safeApprove(
            IERC20 token,
            address spender,
            uint256 value
        ) internal {
            // safeApprove should only be called when setting an initial allowance,
            // or when resetting it to zero. To increase and decrease it, use
            // 'safeIncreaseAllowance' and 'safeDecreaseAllowance'
            require(
                (value == 0) || (token.allowance(address(this), spender) == 0),
                "SafeERC20: approve from non-zero to non-zero allowance"
            );
            _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, value));
        }

        function safeIncreaseAllowance(
            IERC20 token,
            address spender,
            uint256 value
        ) internal {
            uint256 newAllowance = token.allowance(address(this), spender) + value;
            _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, newAllowance));
        }

        function safeDecreaseAllowance(
            IERC20 token,
            address spender,
            uint256 value
        ) internal {
            unchecked {
                uint256 oldAllowance = token.allowance(address(this), spender);
                require(oldAllowance >= value, "SafeERC20: decreased allowance below zero");
                uint256 newAllowance = oldAllowance - value;
                _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, newAllowance));
            }
        }

        function safePermit(
            IERC20Permit token,
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
        function _callOptionalReturn(IERC20 token, bytes memory data) private {
            // We need to perform a low level call here, to bypass Solidity's return data size checking mechanism, since
            // we're implementing it ourselves. We use {Address-functionCall} to perform this call, which verifies that
            // the target address contains contract code and also asserts for success in the low-level call.

            bytes memory returndata = address(token).functionCall(data, "SafeERC20: low-level call failed");
            if (returndata.length > 0) {
                // Return data is optional
                require(abi.decode(returndata, (bool)), "SafeERC20: ERC20 operation did not succeed");
            }
        }
    }
// File: contracts/AFiTimeLock.sol


pragma solidity ^0.8.0;





/**
 * @title IAFi.
 * @notice Interface of the AToken.
 */
interface IAFi {
    function depositUserNav(address user) external returns(uint256);
    function stakeShares(address user, uint256 amount, bool lock) external;
}

contract AFiTimeLock is OwnableDelayModule, Pausable {
    using SafeERC20 for IERC20;

    IERC20 public rewardToken;
    uint256 public cap;   // The maximum total rewards to distribute
    uint256 public totalRewardsDistributed;
    uint256 public numLockDates;
    uint256 public numLockPeriods;
    uint256 public baseRate; // 100 for 1% base rate

    uint256 public constant MAX_BASE_RATE = 1000; // Maximum base rate can be set at 10%

    mapping (address => bool) public isAFiToken;  // The token being staked
    mapping (address => uint256) public totalStaked;
    mapping (uint256 => uint256) public lockDates; // First lockDate will be the launch date
    mapping (uint256 => uint256) public lockDateFactor; // 1000 for 1% and first factor will be 0
    mapping (uint256 => uint256) public lockPeriods;
    mapping (uint256 => uint256) public lockPeriodFactor; // 1000 for 1%
    mapping (address => mapping (uint256 => Stake)) public stakingDetails;
    mapping (address => uint256) public numStakes;
    mapping (address => Freeze) public frozen;

    struct Stake {
        uint256 amount;
        uint256 startTime;
        uint256 durationIndex;
        address token;
        uint256 lockdate;
        uint256 lockdateFactor;
        uint256 rewardAmount;
        bool claimed;
    }

    struct Freeze {
        bool isFrozen;
        uint256 freezingDate;
    }

    event CapSet(uint256 _cap);
    event BaseRateSet(uint256 _rate);
    event RewardTokenSet(address indexed _rewardToken);
    event AFiTokenAdded(address indexed _afiToken);
    event AFiTokenRemoved(address indexed _afiToken);
    event AFiTokenFrozen(address indexed _afiToken, uint256 _freezingDate);
    event Staked(
        address indexed user,
        address indexed _afiToken,
        uint256 _amount,
        uint256 _lockPeriod,
        uint256 _userStakeCounter
    );
    event Unstaked(
        address indexed user,
        uint256 _reward,
        uint256 _userStakeCounter
    );
    event UnclaimedRewardsWithdrawn(
        address _owner,
        uint256 _unclaimedRewards
    );

    constructor (address token) {
        require(token != address(0), "TimeLock: Please enter a valid address");

        rewardToken = IERC20(token);

        emit RewardTokenSet(token);
    }

    function setCap(uint256 _cap) external onlyOwner whenPaused {
        require(_cap > cap, "TimeLock: Cap must be greater than current cap");

        uint256 trnasferFromAmount;

        if (cap > 0) trnasferFromAmount = _cap - cap;

        cap = _cap;

        if (trnasferFromAmount == 0) trnasferFromAmount = _cap;

        emit CapSet(cap);

        rewardToken.safeTransferFrom(msg.sender, address(this), trnasferFromAmount);
    }

    function pause() external onlyOwner {
       _pause();
    }

    function unpause() external onlyOwner {
       _unpause();
    }

    function setBaseRate(uint256 _rate) external onlyOwner whenPaused {
        require(_rate > 0, "TimeLock: base rate must be greater than 0");
        require(_rate <= MAX_BASE_RATE, "TimeLock: base rate cannot exceed 10%");
        require(baseRate == 0, "TimeLock: base rate initialized already");

        baseRate = _rate;

        emit BaseRateSet(_rate);
    }

    function addAFiToken(address token) external onlyOwner {

        require(token != address(0), "TimeLock: Please enter a valid address");
        require(!isAFiToken[token], "TimeLock: Already added");

        isAFiToken[token] = true;

        emit AFiTokenAdded(token);

    }

    function removeAFiToken(address token) external onlyOwner {

        require(token != address(0), "TimeLock: Please enter a valid address");
        require(isAFiToken[token], "TimeLock: Not added yet");

        delete isAFiToken[token];

        emit AFiTokenRemoved(token);

    }

    function freezeRewardsForAFiToken(address token) external onlyOwner {

        require(token != address(0), "TimeLock: Please enter a valid address");
        require(isAFiToken[token], "TimeLock: Not an AFi token");
        require(!frozen[token].isFrozen, "TimeLock: Token already frozen");

        frozen[token].isFrozen = true;
        frozen[token].freezingDate = block.timestamp;

        emit AFiTokenFrozen(token, frozen[token].freezingDate);

    }

    function stake(uint256 amount, address token, uint256 lockPeriodIndex) external whenNotPaused {

        require(
            block.timestamp >= lockDates[0] && block.timestamp <= lockDates[numLockDates - 1],
            "TimeLock: Not the time to stake"
        );
        require(isAFiToken[token], "TimeLock: This token is not stakable");
        require(amount > 0, "TimeLock: Amount must be greater than 0");
        require(!frozen[token].isFrozen, "TimeLock: Staking is frozen for this token");

        uint256 i;

        for(i = 0; i < numLockDates; i++) {
            if (block.timestamp >= lockDates[i] && block.timestamp < lockDates[i + 1]) {
                break;
            }
        }

        uint256 rewardAmount = (
            (
                lockDateFactor[i] * lockPeriodFactor[lockPeriodIndex] * baseRate *
                amount * (IAFi(token).depositUserNav(msg.sender)) * lockPeriods[lockPeriodIndex]
            ) / (
                (365 days) * 10000 * 1000 * 1000 * 100 * 100
            )
        );

        totalRewardsDistributed += rewardAmount;

        require(totalRewardsDistributed <= cap, "TimeLock: Reward cap reached");

        stakingDetails[msg.sender][numStakes[msg.sender]] = Stake(
            amount,
            block.timestamp,
            lockPeriodIndex,
            token,
            lockDates[i],
            lockDateFactor[i],
            rewardAmount,
            false
        );

        totalStaked[token] += amount;

        numStakes[msg.sender]++;

        IAFi(token).stakeShares(msg.sender, amount, true);

        emit Staked(
            msg.sender,
            token,
            amount,
            lockPeriods[lockPeriodIndex],
            numStakes[msg.sender]
        );

    }

    function unstake(uint256 stakeIndex) external {

        require(numStakes[msg.sender] > stakeIndex, "TimeLock: Invalid index");

        Stake storage userStake = stakingDetails[msg.sender][stakeIndex];

        require(!userStake.claimed, "TimeLock: Already claimed");

        uint256 j;
        uint256 reward;
        uint256 stakingDuration;

        if (frozen[userStake.token].isFrozen) {
            stakingDuration = frozen[userStake.token].freezingDate - userStake.startTime;
        } else {
            stakingDuration = block.timestamp - userStake.startTime;
        }

        uint256 durationIndex = userStake.durationIndex;

        if (stakingDuration >= lockPeriods[0] && stakingDuration < lockPeriods[durationIndex]) {

            for (j = 0; j < durationIndex; j++) {

                if (stakingDuration >= lockPeriods[j] && stakingDuration < lockPeriods[j + 1]) {
                    durationIndex = j;

                    break;
                }

            }

            reward = (
                (userStake.rewardAmount * lockPeriodFactor[durationIndex] * lockPeriods[durationIndex]) / 
                (lockPeriodFactor[userStake.durationIndex] * lockPeriods[userStake.durationIndex])
            );

            totalRewardsDistributed -= (userStake.rewardAmount - reward);

            userStake.rewardAmount = reward;

        } else if (stakingDuration < lockPeriods[0]) {

            totalRewardsDistributed -= userStake.rewardAmount;

            reward = 0;

        } else {

            reward = userStake.rewardAmount;

        }

        totalStaked[userStake.token] -= userStake.amount;
        userStake.claimed = true;

        IAFi(userStake.token).stakeShares(msg.sender, userStake.amount, false);

        emit Unstaked(
            msg.sender,
            reward,
            stakeIndex
        );

        if (reward > 0) rewardToken.safeTransfer(msg.sender, reward);

    }

    function setLockDateDetails(
        uint256[] calldata _lockDates,
        uint256[] calldata _lockDateFactors
    ) external  onlyOwner whenPaused {

        require(_lockDates.length == (_lockDateFactors.length + 1), "TimeLock: Array lengths not appropriate");
        require(numLockDates == 0, "TimeLock: Lock dates initialized already");

        for(uint i = 0; i < _lockDates.length; i++) {
            lockDates[i] = _lockDates[i];
            if (i < _lockDateFactors.length) {
                lockDateFactor[i] = _lockDateFactors[i];
            }
        }

        numLockDates = _lockDates.length;

    }

    function setLockPeriodDetails(
        uint256[] calldata _lockPeriods,
        uint256[] calldata _lockPeriodFactors
    ) external  onlyOwner whenPaused {

        require(_lockPeriods.length == _lockPeriodFactors.length, "TimeLock: Array lengths should be equal");
        require(numLockPeriods == 0, "TimeLock: Lock periods initialized already");

        for(uint i = 0; i < _lockPeriods.length; i++) {
            lockPeriods[i] = _lockPeriods[i];
            lockPeriodFactor[i] = _lockPeriodFactors[i];
        }

        numLockPeriods = _lockPeriods.length;

    }

    function withdrawUnclaimedRewards() external onlyOwner {

        require(block.timestamp > lockDates[numLockDates - 1], "TimeLock: Staking is in progress");

        uint256 unclaimedRewards = cap - totalRewardsDistributed;

        require(unclaimedRewards > 0, "TimeLock: No unclaimed rewards");

        cap = totalRewardsDistributed; // Set the cap to the current total staked to prevent further rewards distribution

        emit UnclaimedRewardsWithdrawn(owner(), unclaimedRewards);

        rewardToken.safeTransfer(owner(), unclaimedRewards);

    }

    function withdrawStrayToken(address token) external onlyOwner {
        require(token != address(rewardToken), "TimeLock: cannot withdraw reward token");

        IERC20(token).safeTransfer(owner(), IERC20(token).balanceOf(address(this)));
    }

}