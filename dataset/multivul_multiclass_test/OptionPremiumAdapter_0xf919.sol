// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.19;


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




    error UnauthorizedDeployer();
    error NotConfigured();
    error AlreadyConfigured();
    error MissingConfigKey();
    error AddressAlreadyTaken();
    error ProxyAdminDeploymentFailed();
    error ProxyDeploymentFailed();

    error AccountNotInitialized();
    error AccountDeployerOnly();
    error InvalidSubAccount();
    error InvalidUpgradeVersion();
    error InvalidOwnershipTransfer();
    error InvalidInterfaceAddress();

    error PermissionDenied();
    error AbstractCallFailed();
    error FeesTransferFailed();
    error NotAllowed();
    error InvalidRole();
    error InvalidAccount();
    error InvalidInputValue();
    error InvalidLengths();

    error MustBeBorrrower();
    error ActiveLoansInPlace();
    error AlreadyRegistered();
    error NativeTransferFailed();

    error ModuleAlreadyExists();
    error InvalidModule();
    error AddressAlreadySet();
    error ModuleInquiryFailed();
    error ModuleCallFailed();
    error BalanceCheckFailed();


    error InvalidChainID();
    error WithdrawalsDisabled();
    error NoData();








struct MetaTx {
    address msgTo;
    uint256 msgValue;
    bytes payload;
}




struct ModuleFee {
    address tokenAddress;
    uint256 dstAmount;
    uint256 dstPercent;
}

struct CallCheck {
    uint8 checkType;
    address contractAddr;
    uint256 numericVal;
    address contractAddr2;
    uint256 numericVal2;
}

struct ModuleResponse {
    uint256[] targetCallValues;
    address[] targetAddresses;
    bytes[] targetPayloads;
    CallCheck[] checks;
    ModuleFee[] feesBefore;
    ModuleFee[] feesAfter;
}


interface IUniversalAccount {
    event AccountConfigured(bytes configPayload);
    event NativeTokensReceived(address indexed senderAddr, uint256 amount);
    event OnDeposit(address indexed tokenAddr, address indexed senderAddr, uint256 amount);
    event OnWithdrawal(address indexed tokenAddr, uint256 amount, address indexed receivingAddr, address indexed requestedByAddr);
    event AbstractCallsProcessed(address indexed requestedBy, address[] targets, uint256[] callValues, bytes[] payloads);
    event ModuleExecuted(bytes32 indexed moduleKey, address indexed senderAddr, bytes payload);

    function configure(bytes calldata configPayload) external;

    function runBatchWithOutputs(
        uint256[] calldata callValues, 
        address[] calldata targetAddresses, 
        bytes[] calldata payloads        
    ) external payable returns (bytes[] memory);

    function executeModule(bytes32 moduleKey, bytes calldata payload) external payable returns (bytes memory);

    function deposit(
        address tokenAddr, 
        uint256 amount
    ) external;

    function depositNative() external payable;

    function withdraw(
        address tokenAddr, 
        uint256 amount,
        address payable receivingAddr
    ) external;

    function getSubAccountVersion() external view returns (uint8);
    function getDomainSeparator() external view returns (bytes32);
    function getDomainName() external view returns (string memory);
    function getDomainVersion() external view returns (string memory);

    function getRoles() external view returns (bytes32[] memory);
    function isValidSignature(bytes32 hash, bytes memory signature) external view returns (bytes4 magicValue);
}




//import { LoanDeploymentParams } from "../../structs/LoanDeploymentParams.sol";
//import "./IUniversalLoansDeployer.sol";

interface IMasterSecurityScheme {
    struct Endpoint {
        bytes4 selector;
        address contractAddr;
    }

    struct Permission {
        bytes32 roleId;
        address subAccountAddr;
        bytes4 selector;
        address contractAddr;
    }

    function assignRole(
        bytes32 roleId,
        address subAccountAddr,
        address userAddr
    ) external;

    function revokeRole(
        bytes32 roleId,
        address subAccountAddr,
        address userAddr
    ) external;

    function grantPermission(
        bytes32 roleId,
        address subAccountAddr,
        bytes4 selector,
        address contractAddr
    ) external;

    function grantPermissions(Permission[] calldata items) external;

    function revokePermission(
        bytes32 roleId,
        address subAccountAddr,
        bytes4 selector,
        address contractAddr
    ) external;

    function revokePermissions(Permission[] calldata items) external;

    function grantClientAdmin(address loanAddr) external;
    function revokeClientAdmin(address loanAddr) external;

    //function deployLoan(LoanDeploymentParams calldata loanParams) external returns (address);

    function getOraclePrice(address token) external view returns (uint256);

    function hasPermission(
        bytes32 roleId,
        address subAccountAddr,
        bytes4 selector,
        address contractAddr
    ) external view returns (bool);

    function hasPermission(
        bytes32 roleId,
        address subAccountAddr,
        bytes[] calldata payloads,
        address[] calldata targetAddresses
    ) external view returns (bool);

    function getPermissionsByRoleAndAddress(
        bytes32 roleId,
        address subAccountAddr
    ) external view returns (Endpoint[] memory);

    function getRoleOf(
        address userAddr,
        address subAccountAddr
    ) external view returns (bytes32);

    function whoIsRole(
        bytes32 roleId,
        address subAccountAddr
    ) external view returns (address);

    function isPlatformRole(bytes32 roleId) external view returns (bool);

    function getModulesRegistry() external view returns (address);

    function getWithdrawalStatus(address) external view returns (uint8);
}




interface IMasterAccountsRegistry {
    /**
     * @notice This event is triggered when a new Universal Account is deployed.
     */
    event UniversalAccountDeployed(address proxyAddr, address proxyAdminAddr);

    /**
     * @notice This event is triggered when the Universal Account specified is upgraded to a new version.
     */
    event UniversalAccountUpgraded(address proxyAddr, uint8 newVersionNumber);


    function setReferenceImplementation(address newImplementationAddr) external;
    function setDeployersWhitelist(address addr) external;

    function deploySubAccount(
        address subAccountOwnerAddr,
        string memory newSubAccountName,
        bytes calldata configPayload
    ) external returns (
        address adminContractAddr, 
        address proxyContractAddr
    );

    function getSubAccountDeploymentAddress (
        bytes32 adminSalt, 
        bytes32 proxySalt, 
        address implementationAddr, 
        bytes memory initData
    ) external view returns (
        address adminContractAddr, 
        address proxyContractAddr
    );

    function isValidSubAccount(address subAccountAddr) external view returns (bool);
    function getSubAccountByName(string calldata subAccountName) external view returns (address);
    function getAccountProxyAdmin(address proxyAddr) external view returns (address);
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


contract OptionPremiumAdapter is LightweightOwnable {
    address public securitySchemeAddress;

    mapping(address => mapping(address => int256)) internal _settledPremium;

    event SettleOption(
        address fromAddress,
        address toAddress,
        uint256 amount,
        address tokenAddress
    );

    constructor(address ownerAddr, address securitySchemeAddr) {
        _owner = ownerAddr;
        securitySchemeAddress = securitySchemeAddr;
    }

    function settleOption (
        address fromAddress,
        address payable toAddress,
        uint256 amount,
        address tokenAddress
    ) external nonReentrant onlyOwner {
        if (!IMasterAccountsRegistry(securitySchemeAddress).isValidSubAccount(fromAddress)) revert InvalidAccount();
        if (!IMasterAccountsRegistry(securitySchemeAddress).isValidSubAccount(toAddress)) revert InvalidAccount();

        _settledPremium[fromAddress][tokenAddress] -= int256(amount);
        _settledPremium[toAddress][tokenAddress] += int256(amount);

        IUniversalAccount(fromAddress).withdraw(tokenAddress, amount, toAddress);

        emit SettleOption(fromAddress, toAddress, amount, tokenAddress);
    }

    function getSettledPremium(
        address addr,
        address tokenAddress
    ) external view returns (int256) {
        return _settledPremium[addr][tokenAddress];
    }
}