// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.19;




/**
 * @dev Interface of the ERC4626 "Tokenized Vault Standard", as defined in
 * https://eips.ethereum.org/EIPS/eip-4626[ERC-4626].
 *
 * _Available since v4.7._
 */
interface IERC4626 {
    /// @notice Triggers when an account deposits funds in the contract
    event Deposit(address indexed sender, address indexed owner, uint256 assets, uint256 shares);

    event Withdraw(
        address indexed sender,
        address indexed receiver,
        address indexed owner,
        uint256 assets,
        uint256 shares
    );

    /**
     * @dev Returns the address of the underlying token used for the Vault for accounting, depositing, and withdrawing.
     *
     * - MUST be an ERC-20 token contract.
     * - MUST NOT revert.
     */
    function asset() external view returns (address assetTokenAddress);

    /**
     * @dev Returns the total amount of the underlying asset that is “managed” by Vault.
     *
     * - SHOULD include any compounding that occurs from yield.
     * - MUST be inclusive of any fees that are charged against assets in the Vault.
     * - MUST NOT revert.
     */
    function totalAssets() external view returns (uint256 totalManagedAssets);

    /**
     * @dev Returns the amount of shares that the Vault would exchange for the amount of assets provided, in an ideal
     * scenario where all the conditions are met.
     *
     * - MUST NOT be inclusive of any fees that are charged against assets in the Vault.
     * - MUST NOT show any variations depending on the caller.
     * - MUST NOT reflect slippage or other on-chain conditions, when performing the actual exchange.
     * - MUST NOT revert.
     *
     * NOTE: This calculation MAY NOT reflect the “per-user” price-per-share, and instead should reflect the
     * “average-user’s” price-per-share, meaning what the average user should expect to see when exchanging to and
     * from.
     */
    function convertToShares(uint256 assets) external view returns (uint256 shares);

    /**
     * @dev Returns the amount of assets that the Vault would exchange for the amount of shares provided, in an ideal
     * scenario where all the conditions are met.
     *
     * - MUST NOT be inclusive of any fees that are charged against assets in the Vault.
     * - MUST NOT show any variations depending on the caller.
     * - MUST NOT reflect slippage or other on-chain conditions, when performing the actual exchange.
     * - MUST NOT revert.
     *
     * NOTE: This calculation MAY NOT reflect the “per-user” price-per-share, and instead should reflect the
     * “average-user’s” price-per-share, meaning what the average user should expect to see when exchanging to and
     * from.
     */
    function convertToAssets(uint256 shares) external view returns (uint256 assets);

    /**
     * @dev Returns the maximum amount of the underlying asset that can be deposited into the Vault for the receiver,
     * through a deposit call.
     *
     * - MUST return a limited value if receiver is subject to some deposit limit.
     * - MUST return 2 ** 256 - 1 if there is no limit on the maximum amount of assets that may be deposited.
     * - MUST NOT revert.
     */
    function maxDeposit(address receiver) external view returns (uint256 maxAssets);

    /**
     * @dev Allows an on-chain or off-chain user to simulate the effects of their deposit at the current block, given
     * current on-chain conditions.
     *
     * - MUST return as close to and no more than the exact amount of Vault shares that would be minted in a deposit
     *   call in the same transaction. I.e. deposit should return the same or more shares as previewDeposit if called
     *   in the same transaction.
     * - MUST NOT account for deposit limits like those returned from maxDeposit and should always act as though the
     *   deposit would be accepted, regardless if the user has enough tokens approved, etc.
     * - MUST be inclusive of deposit fees. Integrators should be aware of the existence of deposit fees.
     * - MUST NOT revert.
     *
     * NOTE: any unfavorable discrepancy between convertToShares and previewDeposit SHOULD be considered slippage in
     * share price or some other type of condition, meaning the depositor will lose assets by depositing.
     */
    function previewDeposit(uint256 assets) external view returns (uint256 shares);

    /**
     * @dev Mints shares Vault shares to receiver by depositing exactly amount of underlying tokens.
     *
     * - MUST emit the Deposit event.
     * - MAY support an additional flow in which the underlying tokens are owned by the Vault contract before the
     *   deposit execution, and are accounted for during deposit.
     * - MUST revert if all of assets cannot be deposited (due to deposit limit being reached, slippage, the user not
     *   approving enough underlying tokens to the Vault contract, etc).
     *
     * NOTE: most implementations will require pre-approval of the Vault with the Vault’s underlying asset token.
     */
    function deposit(uint256 assets, address receiver) external returns (uint256 shares);

    /**
     * @dev Returns the maximum amount of the Vault shares that can be minted for the receiver, through a mint call.
     * - MUST return a limited value if receiver is subject to some mint limit.
     * - MUST return 2 ** 256 - 1 if there is no limit on the maximum amount of shares that may be minted.
     * - MUST NOT revert.
     */
    function maxMint(address receiver) external view returns (uint256 maxShares);

    /**
     * @dev Allows an on-chain or off-chain user to simulate the effects of their mint at the current block, given
     * current on-chain conditions.
     *
     * - MUST return as close to and no fewer than the exact amount of assets that would be deposited in a mint call
     *   in the same transaction. I.e. mint should return the same or fewer assets as previewMint if called in the
     *   same transaction.
     * - MUST NOT account for mint limits like those returned from maxMint and should always act as though the mint
     *   would be accepted, regardless if the user has enough tokens approved, etc.
     * - MUST be inclusive of deposit fees. Integrators should be aware of the existence of deposit fees.
     * - MUST NOT revert.
     *
     * NOTE: any unfavorable discrepancy between convertToAssets and previewMint SHOULD be considered slippage in
     * share price or some other type of condition, meaning the depositor will lose assets by minting.
     */
    function previewMint(uint256 shares) external view returns (uint256 assets);

    /**
     * @dev Mints exactly shares Vault shares to receiver by depositing amount of underlying tokens.
     *
     * - MUST emit the Deposit event.
     * - MAY support an additional flow in which the underlying tokens are owned by the Vault contract before the mint
     *   execution, and are accounted for during mint.
     * - MUST revert if all of shares cannot be minted (due to deposit limit being reached, slippage, the user not
     *   approving enough underlying tokens to the Vault contract, etc).
     *
     * NOTE: most implementations will require pre-approval of the Vault with the Vault’s underlying asset token.
     */
    function mint(uint256 shares, address receiver) external returns (uint256 assets);

    /**
     * @dev Returns the maximum amount of the underlying asset that can be withdrawn from the owner balance in the
     * Vault, through a withdraw call.
     *
     * - MUST return a limited value if owner is subject to some withdrawal limit or timelock.
     * - MUST NOT revert.
     */
    function maxWithdraw(address owner) external view returns (uint256 maxAssets);

    /**
     * @dev Allows an on-chain or off-chain user to simulate the effects of their withdrawal at the current block,
     * given current on-chain conditions.
     *
     * - MUST return as close to and no fewer than the exact amount of Vault shares that would be burned in a withdraw
     *   call in the same transaction. I.e. withdraw should return the same or fewer shares as previewWithdraw if
     *   called
     *   in the same transaction.
     * - MUST NOT account for withdrawal limits like those returned from maxWithdraw and should always act as though
     *   the withdrawal would be accepted, regardless if the user has enough shares, etc.
     * - MUST be inclusive of withdrawal fees. Integrators should be aware of the existence of withdrawal fees.
     * - MUST NOT revert.
     *
     * NOTE: any unfavorable discrepancy between convertToShares and previewWithdraw SHOULD be considered slippage in
     * share price or some other type of condition, meaning the depositor will lose assets by depositing.
     */
    function previewWithdraw(uint256 assets) external view returns (uint256 shares);

    /**
     * @dev Burns shares from owner and sends exactly assets of underlying tokens to receiver.
     *
     * - MUST emit the Withdraw event.
     * - MAY support an additional flow in which the underlying tokens are owned by the Vault contract before the
     *   withdraw execution, and are accounted for during withdraw.
     * - MUST revert if all of assets cannot be withdrawn (due to withdrawal limit being reached, slippage, the owner
     *   not having enough shares, etc).
     *
     * Note that some implementations will require pre-requesting to the Vault before a withdrawal may be performed.
     * Those methods should be performed separately.
     */
    function withdraw(
        uint256 assets,
        address receiver,
        address owner
    ) external returns (uint256 shares);

    /**
     * @dev Returns the maximum amount of Vault shares that can be redeemed from the owner balance in the Vault,
     * through a redeem call.
     *
     * - MUST return a limited value if owner is subject to some withdrawal limit or timelock.
     * - MUST return balanceOf(owner) if owner is not subject to any withdrawal limit or timelock.
     * - MUST NOT revert.
     */
    function maxRedeem(address owner) external view returns (uint256 maxShares);

    /**
     * @dev Allows an on-chain or off-chain user to simulate the effects of their redeemption at the current block,
     * given current on-chain conditions.
     *
     * - MUST return as close to and no more than the exact amount of assets that would be withdrawn in a redeem call
     *   in the same transaction. I.e. redeem should return the same or more assets as previewRedeem if called in the
     *   same transaction.
     * - MUST NOT account for redemption limits like those returned from maxRedeem and should always act as though the
     *   redemption would be accepted, regardless if the user has enough shares, etc.
     * - MUST be inclusive of withdrawal fees. Integrators should be aware of the existence of withdrawal fees.
     * - MUST NOT revert.
     *
     * NOTE: any unfavorable discrepancy between convertToAssets and previewRedeem SHOULD be considered slippage in
     * share price or some other type of condition, meaning the depositor will lose assets by redeeming.
     */
    function previewRedeem(uint256 shares) external view returns (uint256 assets);

    /**
     * @dev Burns exactly shares from owner and sends assets of underlying tokens to receiver.
     *
     * - MUST emit the Withdraw event.
     * - MAY support an additional flow in which the underlying tokens are owned by the Vault contract before the
     *   redeem execution, and are accounted for during redeem.
     * - MUST revert if all of shares cannot be redeemed (due to withdrawal limit being reached, slippage, the owner
     *   not having enough shares, etc).
     *
     * NOTE: some implementations will require pre-requesting to the Vault before a withdrawal may be performed.
     * Those methods should be performed separately.
     */
    function redeem(
        uint256 shares,
        address receiver,
        address owner
    ) external returns (uint256 assets);
}




interface IUniswapV3SwapRouter {
    struct ExactInputSingleParams {
        address tokenIn;            // The contract address of the inbound token
        address tokenOut;           // The contract address of the outbound token
        uint24 fee;                 // The fee tier of the pool, used to determine the correct pool contract in which to execute the swap
        address recipient;          // The destination address of the outbound token
        uint256 deadline;           // The unix time after which a swap will fail, to protect against long-pending transactions and wild swings in prices
        uint256 amountIn;           // The input amount
        uint256 amountOutMinimum;   // The minimum amount out
        uint160 sqrtPriceLimitX96;  // The price limit
    }

    function exactInputSingle(ExactInputSingleParams calldata params) external payable returns (uint256 amountOut);
}





// OpenZeppelin Contracts (last updated v4.8.0) (token/ERC20/utils/SafeERC20.sol)




// OpenZeppelin Contracts (last updated v4.6.0) (token/ERC20/IERC20.sol)



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
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}


// OpenZeppelin Contracts v4.4.1 (token/ERC20/extensions/IERC20Permit.sol)



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


// OpenZeppelin Contracts (last updated v4.8.0) (utils/Address.sol)



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

    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transfer.selector, to, value));
    }

    function safeTransferFrom(IERC20 token, address from, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transferFrom.selector, from, to, value));
    }

    /**
     * @dev Deprecated. This function has issues similar to the ones found in
     * {IERC20-approve}, and its usage is discouraged.
     *
     * Whenever possible, use {safeIncreaseAllowance} and
     * {safeDecreaseAllowance} instead.
     */
    function safeApprove(IERC20 token, address spender, uint256 value) internal {
        // safeApprove should only be called when setting an initial allowance,
        // or when resetting it to zero. To increase and decrease it, use
        // 'safeIncreaseAllowance' and 'safeDecreaseAllowance'
        require(
            (value == 0) || (token.allowance(address(this), spender) == 0),
            "SafeERC20: approve from non-zero to non-zero allowance"
        );
        _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, value));
    }

    function safeIncreaseAllowance(IERC20 token, address spender, uint256 value) internal {
        uint256 newAllowance = token.allowance(address(this), spender) + value;
        _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, newAllowance));
    }

    function safeDecreaseAllowance(IERC20 token, address spender, uint256 value) internal {
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







/**
 * @notice Defines the interface for whitelisting tokens.
 */
interface ITokensWhitelist {
    /**
     * @notice Whitelists the address specified.
     * @param addr The address to enable
     */
    function enableToken (address addr) external;

    /**
     * @notice Whitelists the addresses specified.
     * @param arr The addresses to enable
     */
    function enableTokens (address[] calldata arr) external;

    /**
     * @notice Disables the address specified.
     * @param addr The address to disable
     */
    function disableToken (address addr) external;

    /**
     * @notice Disables the addresses specified.
     * @param arr The addresses to disable
     */
    function disableTokens (address[] calldata arr) external;

    /**
     * @notice Indicates if the address is whitelisted or not.
     * @param addr The address to disable
     * @return Returns 1 if the address is whitelisted
     */
    function isWhitelistedToken (address addr) external view returns (bool);

    /**
     * This event is triggered when a new address is whitelisted.
     * @param addr The address that was whitelisted
     */
    event OnTokenEnabled(address addr);

    /**
     * This event is triggered when an address is disabled.
     * @param addr The address that was disabled
     */
    event OnTokenDisabled(address addr);
}







interface IOwnable {
    function transferOwnership(address newOwner) external;
    function owner() external view returns (address);
}




/**
 * @title Base reentrancy guard. This is constructor-less implementation for both proxies and standalone contracts.
 */
abstract contract BaseReentrancyGuard {
    error ReentrantCall();
    
    uint256 internal constant _REENTRANCY_NOT_ENTERED = 1;
    uint256 internal constant _REENTRANCY_ENTERED = 2;

    uint256 internal _reentrancyStatus;

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
        if (_reentrancyStatus == _REENTRANCY_ENTERED) revert ReentrantCall();

        // Any calls to nonReentrant after this point will fail
        _reentrancyStatus = _REENTRANCY_ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _reentrancyStatus = _REENTRANCY_NOT_ENTERED;
    }
}




abstract contract BaseOwnable {
    error OwnerOnly();

    address internal _owner;

    /**
     * @notice Triggers when contract ownership changes.
     * @param previousOwner The previous owner of the contract.
     * @param newOwner The new owner of the contract.
     */
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        if (msg.sender != _owner) revert OwnerOnly();
        _;
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}


/**
 * @title Lightweight version of the ownership contract. This contract has a reentrancy guard.
 */
abstract contract LightweightOwnable is IOwnable, BaseReentrancyGuard, BaseOwnable {
    /**
     * @notice Transfers ownership of the contract to the account specified.
     * @param newOwner The address of the new owner.
     */
    function transferOwnership(address newOwner) external virtual nonReentrant onlyOwner {
        _transferOwnership(newOwner);
    }

    /**
     * @notice Gets the owner of the contract.
     * @return address The address who owns the contract.
     */
    function owner() external view virtual returns (address) {
        return _owner;
    }    
}


/**
 * @title Standalone contract for whitelisting token addresses.
 */
contract TokensWhitelist is ITokensWhitelist, LightweightOwnable {
    mapping (address => bool) public _whitelistedTokens;

    constructor(address ownerAddr) {
        require(ownerAddr != address(0), "Owner required");
        _owner = ownerAddr;
    }

    /**
     * @notice Whitelists the address specified.
     * @param addr The address to enable
     */
    function enableToken (address addr) external override nonReentrant onlyOwner {
        require(!_whitelistedTokens[addr], "Already enabled");
        _whitelistedTokens[addr] = true;
        emit OnTokenEnabled(addr);
    }

    /**
     * @notice Whitelists the addresses specified.
     * @param arr The addresses to enable
     */
    function enableTokens (address[] calldata arr) external override nonReentrant onlyOwner {
        require(arr.length > 0, "Addresses required");

        for (uint256 i; i < arr.length; i++) {
            require(arr[i] != address(0), "Invalid address");
            require(!_whitelistedTokens[arr[i]], "Already enabled");
            _whitelistedTokens[arr[i]] = true;
            emit OnTokenEnabled(arr[i]);
        }
    }

    /**
     * @notice Disables the address specified.
     * @param addr The address to disable
     */
    function disableToken (address addr) external override nonReentrant onlyOwner {
        require(_whitelistedTokens[addr], "Already disabled");
        _whitelistedTokens[addr] = false;
        emit OnTokenDisabled(addr);
    }

    /**
     * @notice Disables the addresses specified.
     * @param arr The addresses to disable
     */
    function disableTokens (address[] calldata arr) external override nonReentrant onlyOwner {
        for (uint256 i; i < arr.length; i++) {
            require(_whitelistedTokens[arr[i]], "Already disabled");
            _whitelistedTokens[arr[i]] = false;
            emit OnTokenDisabled(arr[i]);
        }
    }

    /**
     * @notice Indicates if the address is whitelisted or not.
     * @param addr The address to evaluate.
     * @return Returns true if the address is whitelisted.
     */
    function isWhitelistedToken (address addr) external view override returns (bool) {
        return _whitelistedTokens[addr];
    }
}


/**
 * @title Primitive swapper for ERC-20 tokens.
 */
abstract contract BasePoolSwapper is TokensWhitelist {
    // ---------------------------------------------------------------
    // Structures
    // ---------------------------------------------------------------
    // Represents a bridge like Paraswap, Lifi, etc.
    struct SwapProvider {
        uint8 id;
        bool enabled;
        address routerAddress;
        address tokenTransferProxy;
    }

    // Represents the parameters of a swap
    struct SwapInfo {
        uint256 amountIn;
        uint256 minAmountOut;
        IERC20 srcToken;
        IERC20 dstToken;
        uint8 bridgeId;
        bytes quoteData;
    }

    struct SingleHopItem {
        uint256 amountIn;
        uint256 minAmountOut;
        address tokenIn;
        uint24 fee;
        uint160 sqrtPriceLimitX96;
    }

    // ---------------------------------------------------------------
    // Storage layout
    // ---------------------------------------------------------------
    /// @notice The swap providers (LiFi, Paraswap, etc)
    mapping (uint8 => SwapProvider) public swapProviders;

    // ---------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------
    /// @notice Triggers when an atomic swap is processed
    event SwapProcessed(address srcToken, address dstToken, uint256 srcAmount, uint256 dstAmount);


    // ---------------------------------------------------------------
    // Functions
    // ---------------------------------------------------------------
    function _updateSwapProviders(SwapProvider[] memory newSwapProviders) internal virtual {
        for (uint256 i; i < newSwapProviders.length; i++) {
            require(newSwapProviders[i].id > 0, "Provider ID required");
            require(newSwapProviders[i].routerAddress != address(0), "Router required");
            require(newSwapProviders[i].tokenTransferProxy != address(0), "Transfer contract required");

            swapProviders[newSwapProviders[i].id] = newSwapProviders[i];
        }
    }

    function _singleSwap(SwapInfo memory item, address collectorAddr) internal virtual {
        require(swapProviders[item.bridgeId].enabled, "Invalid bridge");

        // Make sure the input token is whitelisted
        require(_whitelistedTokens[address(item.srcToken)], "Token not whitelisted");
        
        uint256 dstTokenBalanceBeforeSwap = item.dstToken.balanceOf(address(this));

        // We need this check because we do not control the off-chain payload.
        // Without this check, an attacker could trade an amount bigger than the one specified.
        uint256 srcTokenBalanceBeforeSwap = item.srcToken.balanceOf(address(this));
        if ((srcTokenBalanceBeforeSwap > 0) && (collectorAddr != address(0))) {
            SafeERC20.safeTransfer(item.srcToken, collectorAddr, srcTokenBalanceBeforeSwap);
        }

        // Transfer the funds
        SafeERC20.safeTransferFrom(item.srcToken, msg.sender, address(this), item.amountIn);

        // Determine the addresses of the bridge
        address routerAddr = swapProviders[item.bridgeId].routerAddress;
        address tokenTransferProxy = swapProviders[item.bridgeId].tokenTransferProxy;

        // Approve the bridge as a spender
        SafeERC20.safeApprove(item.srcToken, tokenTransferProxy, item.amountIn);

        // Run the swap through the router
        (bool success, ) = routerAddr.call{value: 0}(item.quoteData);
        require(success, "Swap failed");

        // Slippage check
        uint256 amountReceived = item.dstToken.balanceOf(address(this)) - dstTokenBalanceBeforeSwap;
        require(amountReceived >= item.minAmountOut, "Slippage check failed");

        // Revoke the approval
        SafeERC20.safeApprove(item.srcToken, tokenTransferProxy, 0);

        // Refund the user, if needed
        uint256 remainingInputTokens = item.srcToken.balanceOf(address(this));
        if (remainingInputTokens > 0) {
            SafeERC20.safeTransfer(item.srcToken, msg.sender, remainingInputTokens);
        }

        // Log the event
        emit SwapProcessed(address(item.srcToken), address(item.dstToken), item.amountIn, amountReceived);
    }
}


/**
 * @title Swap adapter for ERC-4626 pools. Supports deposits in multiple tokens. Tokens must be whitelisted in advance.
 * @dev This contract assumes the pool is not a proxy.
 */
contract StandalonePoolAdapter is BasePoolSwapper {
    error RouterNotSet();
    error InputTokenNotAllowed();
    error TokenNotWhitelisted();

    // ---------------------------------------------------------------
    // Storage layout
    // ---------------------------------------------------------------
    /// @notice The pool associated to this adapter.
    IERC4626 public immutable pool;

    /// @notice The underlying asset of the ERC-4626 pool.
    IERC20 public immutable poolAsset;

    /// @notice The swap fee, expressed with 2 decimal places.
    uint256 public swapFee;

    /// @notice The address of the fees collector, if any.
    address public feesCollector;

    /// @notice Indicates if the adapter is paused.
    bool public isPaused;

    /// @notice The address of the Uniswap V3 router
    address public routerAddress;

    mapping(address => bool) public isBlacklisted;

    // ---------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------
    /// @notice Triggers when the contract collects swap fees
    event SwapFeeApplied(uint256 swapAmount, uint256 applicableFee, address tokenAddr);


    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------
    constructor(
        uint256 newSwapFee,
        address ownerAddr, 
        address newFeesCollectorAddr,
        address routerAddr,
        SwapProvider[] memory newSwapProviders,
        IERC4626 newPool
    ) TokensWhitelist(ownerAddr) {
        // Checks
        require(newFeesCollectorAddr != address(0), "Fees collector required");
        require(newSwapFee < 10000, "Swap fee too high");
        require(newPool.asset() != address(0), "Invalid pool");

        // State changes
        feesCollector = newFeesCollectorAddr;
        swapFee = newSwapFee;
        routerAddress = routerAddr;
        pool = newPool;
        poolAsset = IERC20(newPool.asset());

        // Define the swap providers
        _updateSwapProviders(newSwapProviders);
    }

    // ---------------------------------------------------------------
    // Modifiers
    // ---------------------------------------------------------------
    modifier ifNotPaused() {
        require(!isPaused, "Contract paused");
        _;
    }

    // ---------------------------------------------------------------
    // Swap functions
    // ---------------------------------------------------------------
    /**
     * @notice Swaps one or more tokens via Uniswap and deposits the outcome in the pool.
     * @param items The swaps to perform.
     * @return shares The number of shares deposited in the pool.
     */
    function swapAndDeposit(
        SingleHopItem[] memory items
    ) external nonReentrant ifNotPaused returns (uint256 shares) {
        if (routerAddress == address(0)) revert RouterNotSet();
       
        // Transfer any existing underlying assets to the fee collector prior running the swap(s)
        SafeERC20.safeTransfer(poolAsset, feesCollector, poolAsset.balanceOf(address(this)));

        uint256 surplusBalance;

        for (uint256 i; i < items.length; i++) {
            // Make sure the input token is whitelisted
            if (!_whitelistedTokens[items[i].tokenIn]) revert TokenNotWhitelisted();

            if (items[i].tokenIn == address(poolAsset)) revert InputTokenNotAllowed();

            SafeERC20.safeTransferFrom(IERC20(items[i].tokenIn), msg.sender, address(this), items[i].amountIn);

            // Approve the router
            SafeERC20.safeApprove(IERC20(items[i].tokenIn), routerAddress, items[i].amountIn);

            IUniswapV3SwapRouter.ExactInputSingleParams memory swapParams = IUniswapV3SwapRouter.ExactInputSingleParams({
                tokenIn: items[i].tokenIn, 
                tokenOut: address(poolAsset),
                recipient: address(this),
                amountIn: items[i].amountIn,
                amountOutMinimum: items[i].minAmountOut,

                // The fee tier of the pool, used to determine the correct pool contract in which to execute the swap
                fee: items[i].fee,

                // The unix time after which a swap will fail, to protect against long-pending transactions and wild swings in prices
                deadline: block.timestamp,

                // The price limit
                sqrtPriceLimitX96: items[i].sqrtPriceLimitX96
            });

            // Run the swap via Uniswap V3
            IUniswapV3SwapRouter(routerAddress).exactInputSingle(swapParams);

            surplusBalance = IERC20(items[i].tokenIn).balanceOf(address(this));
            if (surplusBalance > 0) {
                SafeERC20.safeTransfer(IERC20(items[i].tokenIn), msg.sender, surplusBalance);
            }
        }

        // Apply fees
        uint256 effectiveDepositAmount = _applyFees();

        // Deposit funds in the pool and return the respective number of shares
        SafeERC20.safeApprove(poolAsset, address(pool), effectiveDepositAmount);
        shares = pool.deposit(effectiveDepositAmount, msg.sender);
        SafeERC20.safeApprove(poolAsset, address(pool), 0);
    }

    /**
     * @notice Swaps a single token and deposits the output token into the pool.
     * @param item The swap parameters.
     * @return shares The number of shares deposited in the pool.
     */
    function swapAndDeposit(SwapInfo memory item) external nonReentrant ifNotPaused returns (uint256 shares) {
        require(address(item.srcToken) != address(poolAsset), "Invalid asset");
        require(!isBlacklisted[msg.sender], "Address blacklisted");

        // Force the destination asset to match the one defined by the pool
        item.dstToken = poolAsset;

        SafeERC20.safeTransfer(poolAsset, feesCollector, poolAsset.balanceOf(address(this)));

        // Run the atomic swap through the bridge specified.
        _singleSwap(item, feesCollector);

        // Apply fees
        uint256 effectiveDepositAmount = _applyFees();

        // Deposit funds in the pool and return the respective number of shares
        SafeERC20.safeApprove(poolAsset, address(pool), effectiveDepositAmount);
        shares = pool.deposit(effectiveDepositAmount, msg.sender);
        SafeERC20.safeApprove(poolAsset, address(pool), 0);
    }

    /**
     * @notice Swaps multiple tokens and deposits the output token into the pool.
     * @dev Allows the use of multiple bridges for each swap.
     * @param items The parameters of each swap.
     * @return shares The number of shares deposited in the pool.
     */
    function swapAndDeposit(SwapInfo[] memory items) external nonReentrant ifNotPaused returns (uint256 shares) {
        require(items.length > 0, "Tokens required");
        require(!isBlacklisted[msg.sender], "Address blacklisted");

        SafeERC20.safeTransfer(poolAsset, feesCollector, poolAsset.balanceOf(address(this)));

        // Run all swaps
        for (uint256 i; i < items.length; i++) {
            require(address(items[i].srcToken) != address(poolAsset), "Invalid asset");

            // Force the destination asset to match the one defined by the pool
            items[i].dstToken = poolAsset;

            // Run the atomic swap through the bridge specified.
            _singleSwap(items[i], feesCollector);
        }

        // Apply fees
        uint256 effectiveDepositAmount = _applyFees();

        // Deposit funds in the pool and return the respective number of shares
        SafeERC20.safeApprove(poolAsset, address(pool), effectiveDepositAmount);
        shares = pool.deposit(effectiveDepositAmount, msg.sender);
        SafeERC20.safeApprove(poolAsset, address(pool), 0);
    }

    // ---------------------------------------------------------------
    // Maintenance functions
    // ---------------------------------------------------------------
    /**
     * @notice Sets the swap providers supported by this contract.
     * @param newSwapProviders The array of swap providers.
     */
    function updateSwapProviders(SwapProvider[] memory newSwapProviders) external nonReentrant onlyOwner ifNotPaused {
        _updateSwapProviders(newSwapProviders);
    }

    /**
     * @notice Updates the address of the fees collector.
     * @dev The fees collector cannot be the zero address.
     * @param newFeesCollectorAddr The address of the new fees collector.
     */
    function updateFeesCollector(address newFeesCollectorAddr) external nonReentrant onlyOwner ifNotPaused {
        require(newFeesCollectorAddr != address(0), "Fees collector required");
        feesCollector = newFeesCollectorAddr;
    }

    /**
     * @notice Updates the swap fee.
     * @dev The fee is applied in the currency defined by the pool (ERC-4626). The fee can be zero.
     * @param newSwapFee The new fee, expressed as a percentage with 2 decimal places.
     */
    function updateSwapFee(uint256 newSwapFee) external nonReentrant onlyOwner ifNotPaused {
        require(newSwapFee < 10000, "Swap fee too high");
        swapFee = newSwapFee;
    }

    /**
     * @notice Pauses the contract.
     */
    function pause() external nonReentrant onlyOwner ifNotPaused {
        isPaused = true;
    }

    /**
     * @notice Resumes the contract.
     */
    function resume() external nonReentrant onlyOwner {
        require(isPaused, "Already resumed");
        isPaused = false;
    }

    function addToBlacklist(address addr) external nonReentrant onlyOwner {
        require(addr != _owner, "Cannot blacklist owner");
        isBlacklisted[addr] = true;
    }

    function removeFromBlacklist(address addr) external nonReentrant onlyOwner {
        isBlacklisted[addr] = false;
    }

    // ---------------------------------------------------------------
    // Internal functions
    // ---------------------------------------------------------------
    function _applyFees() internal returns (uint256 effectiveDepositAmount) {
        // Balance of destination tokens after running the swap
        uint256 amountReceivedInUnderlyingTokens = poolAsset.balanceOf(address(this));
        uint256 applicableFee = (swapFee > 0) ? (swapFee * amountReceivedInUnderlyingTokens) / 1e4 : 0;
        effectiveDepositAmount = amountReceivedInUnderlyingTokens - applicableFee;

        // Transfer fees to the collector, if needed
        if (applicableFee > 0) {
            SafeERC20.safeTransfer(poolAsset, feesCollector, applicableFee);
            emit SwapFeeApplied(amountReceivedInUnderlyingTokens, applicableFee, address(poolAsset));
        }
    }
}