// File: @openzeppelin/contracts/security/ReentrancyGuard.sol

// SPDX-License-Identifier: MIT

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

// File: @openzeppelin/contracts/token/ERC20/IERC20.sol


// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC-20 standard as defined in the ERC.
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

// File: @openzeppelin/contracts/interfaces/IERC20.sol


// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC20.sol)

pragma solidity ^0.8.20;


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

// File: @openzeppelin/contracts/interfaces/IERC165.sol


// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC165.sol)

pragma solidity ^0.8.20;


// File: @openzeppelin/contracts/interfaces/IERC1363.sol


// OpenZeppelin Contracts (last updated v5.1.0) (interfaces/IERC1363.sol)

pragma solidity ^0.8.20;



/**
 * @title IERC1363
 * @dev Interface of the ERC-1363 standard as defined in the https://eips.ethereum.org/EIPS/eip-1363[ERC-1363].
 *
 * Defines an extension interface for ERC-20 tokens that supports executing code on a recipient contract
 * after `transfer` or `transferFrom`, or code on a spender contract after `approve`, in a single transaction.
 */
interface IERC1363 is IERC20, IERC165 {
    /*
     * Note: the ERC-165 identifier for this interface is 0xb0202a11.
     * 0xb0202a11 ===
     *   bytes4(keccak256('transferAndCall(address,uint256)')) ^
     *   bytes4(keccak256('transferAndCall(address,uint256,bytes)')) ^
     *   bytes4(keccak256('transferFromAndCall(address,address,uint256)')) ^
     *   bytes4(keccak256('transferFromAndCall(address,address,uint256,bytes)')) ^
     *   bytes4(keccak256('approveAndCall(address,uint256)')) ^
     *   bytes4(keccak256('approveAndCall(address,uint256,bytes)'))
     */

    /**
     * @dev Moves a `value` amount of tokens from the caller's account to `to`
     * and then calls {IERC1363Receiver-onTransferReceived} on `to`.
     * @param to The address which you want to transfer to.
     * @param value The amount of tokens to be transferred.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function transferAndCall(address to, uint256 value) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from the caller's account to `to`
     * and then calls {IERC1363Receiver-onTransferReceived} on `to`.
     * @param to The address which you want to transfer to.
     * @param value The amount of tokens to be transferred.
     * @param data Additional data with no specified format, sent in call to `to`.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function transferAndCall(address to, uint256 value, bytes calldata data) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to` using the allowance mechanism
     * and then calls {IERC1363Receiver-onTransferReceived} on `to`.
     * @param from The address which you want to send tokens from.
     * @param to The address which you want to transfer to.
     * @param value The amount of tokens to be transferred.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function transferFromAndCall(address from, address to, uint256 value) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to` using the allowance mechanism
     * and then calls {IERC1363Receiver-onTransferReceived} on `to`.
     * @param from The address which you want to send tokens from.
     * @param to The address which you want to transfer to.
     * @param value The amount of tokens to be transferred.
     * @param data Additional data with no specified format, sent in call to `to`.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function transferFromAndCall(address from, address to, uint256 value, bytes calldata data) external returns (bool);

    /**
     * @dev Sets a `value` amount of tokens as the allowance of `spender` over the
     * caller's tokens and then calls {IERC1363Spender-onApprovalReceived} on `spender`.
     * @param spender The address which will spend the funds.
     * @param value The amount of tokens to be spent.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function approveAndCall(address spender, uint256 value) external returns (bool);

    /**
     * @dev Sets a `value` amount of tokens as the allowance of `spender` over the
     * caller's tokens and then calls {IERC1363Spender-onApprovalReceived} on `spender`.
     * @param spender The address which will spend the funds.
     * @param value The amount of tokens to be spent.
     * @param data Additional data with no specified format, sent in call to `spender`.
     * @return A boolean value indicating whether the operation succeeded unless throwing.
     */
    function approveAndCall(address spender, uint256 value, bytes calldata data) external returns (bool);
}

// File: @openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol


// OpenZeppelin Contracts (last updated v5.2.0) (token/ERC20/utils/SafeERC20.sol)

pragma solidity ^0.8.20;



/**
 * @title SafeERC20
 * @dev Wrappers around ERC-20 operations that throw on failure (when the token
 * contract returns false). Tokens that return no value (and instead revert or
 * throw on failure) are also supported, non-reverting calls are assumed to be
 * successful.
 * To use this library you can add a `using SafeERC20 for IERC20;` statement to your contract,
 * which allows you to call the safe operations as `token.safeTransfer(...)`, etc.
 */
library SafeERC20 {
    /**
     * @dev An operation with an ERC-20 token failed.
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
     *
     * IMPORTANT: If the token implements ERC-7674 (ERC-20 with temporary allowance), and if the "client"
     * smart contract uses ERC-7674 to set temporary allowances, then the "client" smart contract should avoid using
     * this function. Performing a {safeIncreaseAllowance} or {safeDecreaseAllowance} operation on a token contract
     * that has a non-zero temporary allowance (for that particular owner-spender) will result in unexpected behavior.
     */
    function safeIncreaseAllowance(IERC20 token, address spender, uint256 value) internal {
        uint256 oldAllowance = token.allowance(address(this), spender);
        forceApprove(token, spender, oldAllowance + value);
    }

    /**
     * @dev Decrease the calling contract's allowance toward `spender` by `requestedDecrease`. If `token` returns no
     * value, non-reverting calls are assumed to be successful.
     *
     * IMPORTANT: If the token implements ERC-7674 (ERC-20 with temporary allowance), and if the "client"
     * smart contract uses ERC-7674 to set temporary allowances, then the "client" smart contract should avoid using
     * this function. Performing a {safeIncreaseAllowance} or {safeDecreaseAllowance} operation on a token contract
     * that has a non-zero temporary allowance (for that particular owner-spender) will result in unexpected behavior.
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
     *
     * NOTE: If the token implements ERC-7674, this function will not modify any temporary allowance. This function
     * only sets the "standard" allowance. Any temporary allowance will remain active, in addition to the value being
     * set here.
     */
    function forceApprove(IERC20 token, address spender, uint256 value) internal {
        bytes memory approvalCall = abi.encodeCall(token.approve, (spender, value));

        if (!_callOptionalReturnBool(token, approvalCall)) {
            _callOptionalReturn(token, abi.encodeCall(token.approve, (spender, 0)));
            _callOptionalReturn(token, approvalCall);
        }
    }

    /**
     * @dev Performs an {ERC1363} transferAndCall, with a fallback to the simple {ERC20} transfer if the target has no
     * code. This can be used to implement an {ERC721}-like safe transfer that rely on {ERC1363} checks when
     * targeting contracts.
     *
     * Reverts if the returned value is other than `true`.
     */
    function transferAndCallRelaxed(IERC1363 token, address to, uint256 value, bytes memory data) internal {
        if (to.code.length == 0) {
            safeTransfer(token, to, value);
        } else if (!token.transferAndCall(to, value, data)) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    /**
     * @dev Performs an {ERC1363} transferFromAndCall, with a fallback to the simple {ERC20} transferFrom if the target
     * has no code. This can be used to implement an {ERC721}-like safe transfer that rely on {ERC1363} checks when
     * targeting contracts.
     *
     * Reverts if the returned value is other than `true`.
     */
    function transferFromAndCallRelaxed(
        IERC1363 token,
        address from,
        address to,
        uint256 value,
        bytes memory data
    ) internal {
        if (to.code.length == 0) {
            safeTransferFrom(token, from, to, value);
        } else if (!token.transferFromAndCall(from, to, value, data)) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    /**
     * @dev Performs an {ERC1363} approveAndCall, with a fallback to the simple {ERC20} approve if the target has no
     * code. This can be used to implement an {ERC721}-like safe transfer that rely on {ERC1363} checks when
     * targeting contracts.
     *
     * NOTE: When the recipient address (`to`) has no code (i.e. is an EOA), this function behaves as {forceApprove}.
     * Opposedly, when the recipient address (`to`) has code, this function only attempts to call {ERC1363-approveAndCall}
     * once without retrying, and relies on the returned value to be true.
     *
     * Reverts if the returned value is other than `true`.
     */
    function approveAndCallRelaxed(IERC1363 token, address to, uint256 value, bytes memory data) internal {
        if (to.code.length == 0) {
            forceApprove(token, to, value);
        } else if (!token.approveAndCall(to, value, data)) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    /**
     * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
     * on the return value: the return value is optional (but if data is returned, it must not be false).
     * @param token The token targeted by the call.
     * @param data The call data (encoded using abi.encode or one of its variants).
     *
     * This is a variant of {_callOptionalReturnBool} that reverts if call fails to meet the requirements.
     */
    function _callOptionalReturn(IERC20 token, bytes memory data) private {
        uint256 returnSize;
        uint256 returnValue;
        assembly ("memory-safe") {
            let success := call(gas(), token, 0, add(data, 0x20), mload(data), 0, 0x20)
            // bubble errors
            if iszero(success) {
                let ptr := mload(0x40)
                returndatacopy(ptr, 0, returndatasize())
                revert(ptr, returndatasize())
            }
            returnSize := returndatasize()
            returnValue := mload(0)
        }

        if (returnSize == 0 ? address(token).code.length == 0 : returnValue != 1) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    /**
     * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
     * on the return value: the return value is optional (but if data is returned, it must not be false).
     * @param token The token targeted by the call.
     * @param data The call data (encoded using abi.encode or one of its variants).
     *
     * This is a variant of {_callOptionalReturn} that silently catches all reverts and returns a bool instead.
     */
    function _callOptionalReturnBool(IERC20 token, bytes memory data) private returns (bool) {
        bool success;
        uint256 returnSize;
        uint256 returnValue;
        assembly ("memory-safe") {
            success := call(gas(), token, 0, add(data, 0x20), mload(data), 0, 0x20)
            returnSize := returndatasize()
            returnValue := mload(0)
        }
        return success && (returnSize == 0 ? address(token).code.length > 0 : returnValue == 1);
    }
}

// File: contracts/openOracleL1.sol


// openOracle is an attempt at a trust-free price oracle that uses an escalating auction.
// This contract is for researching if the economic incentives in the design work.
// With appropriate oracle parameters, expiration is evidence of a good price.
// https://openprices.gitbook.io/openoracle-docs
// v0.1.6

pragma solidity ^0.8.26;


using SafeERC20 for IERC20;

contract openOracle is ReentrancyGuard {

    constructor() ReentrancyGuard() {
        //
    }

    struct ReportMeta {
        address token1;
        address token2;
        uint256 feePercentage;
        uint256 multiplier;
        uint256 settlementTime;
        uint256 exactToken1Report;
        uint256 fee;
        uint256 escalationHalt;
        uint256 disputeDelay;
        uint256 protocolFee;
        uint256 settlerReward;
    }

    struct ReportStatus {
        uint256 currentAmount1;
        uint256 currentAmount2;
        address payable currentReporter;
        address payable initialReporter;
        uint256 reportTimestamp;
        uint256 settlementTimestamp;
        uint256 price; // price scaled by 1e18
        bool isSettled;
        bool disputeOccurred;

        bool isDistributed;
        uint256 lastDisputeBlock; // Added to track block of last dispute

    }

    uint256 public nextReportId = 1;
//    address constant arbSysAddress = 0x0000000000000000000000000000000000000064;

    mapping(uint256 => ReportMeta) public reportMeta;
    mapping(uint256 => ReportStatus) public reportStatus;

    mapping(address => uint256) public protocolFees;

    event ReportInstanceCreated(uint256 indexed reportId, address indexed token1Address, address indexed token2Address, uint256 feePercentage, uint256 multiplier, uint256 exactToken1Report, uint256 ethFee, address creator, uint256 settlementTime, uint256 escalationHalt, uint256 disputeDelay, uint256 protocolFee, uint256 settlerReward);
    event InitialReportSubmitted(uint256 indexed reportId, address reporter, uint256 amount1, uint256 amount2, address indexed token1Address, address indexed token2Address, uint256 swapFee, uint256 protocolFee, uint256 settlementTime, uint256 disputeDelay, uint256 escalationHalt);
    event ReportDisputed(uint256 indexed reportId, address disputer, uint256 newAmount1, uint256 newAmount2, address indexed token1Address, address indexed token2Address, uint256 swapFee, uint256 protocolFee, uint256 settlementTime, uint256 disputeDelay, uint256 escalationHalt);
    event ReportSettled(uint256 indexed reportId, uint256 price, uint256 settlementTimestamp);

    function createReportInstance(
        address token1Address,
        address token2Address,
        uint256 exactToken1Report,
        uint256 feePercentage, // in thousandths of a basis point i.e. 3000 means 3bps.
        uint256 multiplier, //in percentage points i.e. 110 means multiplier of 1.1x
        uint256 settlementTime,
        uint256 escalationHalt, // when exactToken1Report passes this, the multiplier drops to 100 after
        uint256 disputeDelay, // seconds, increase free option cost for self dispute games
        uint256 protocolFee, //in thousandths of a basis point
        uint256 settlerReward // in wei
    ) external payable returns (uint256 reportId) {
        require(msg.value > 100, "Fee must be greater than 100 wei");
        require(exactToken1Report > 0, "exactToken1Report must be greater than zero");
        require(token1Address != token2Address, "Tokens must be different");
        require(settlementTime > disputeDelay);
        require(msg.value > settlerReward);

        reportId = nextReportId++;
        ReportMeta storage meta = reportMeta[reportId];
        meta.token1 = token1Address;
        meta.token2 = token2Address;
        meta.exactToken1Report = exactToken1Report;
        meta.feePercentage = feePercentage;
        meta.multiplier = multiplier;
        meta.settlementTime = settlementTime;
        meta.fee = msg.value - settlerReward;
        meta.escalationHalt = escalationHalt;
        meta.disputeDelay = disputeDelay;
        meta.protocolFee = protocolFee;
        meta.settlerReward = settlerReward;

        emit ReportInstanceCreated(reportId, token1Address, token2Address, feePercentage, multiplier, exactToken1Report, msg.value, msg.sender, settlementTime, escalationHalt, disputeDelay, protocolFee, settlerReward);

    }

    function submitInitialReport(uint256 reportId, uint256 amount1, uint256 amount2) external nonReentrant {
        ReportMeta storage meta = reportMeta[reportId];
        ReportStatus storage status = reportStatus[reportId];

        require(reportId <= nextReportId);
        require(status.currentReporter == address(0), "Report already submitted");
        require(amount1 == meta.exactToken1Report, "Amount1 equals exact amount");
        require(amount2 > 0);

        _transferTokens(meta.token1, msg.sender, address(this), amount1);
        _transferTokens(meta.token2, msg.sender, address(this), amount2);

        status.currentAmount1 = amount1;
        status.currentAmount2 = amount2;
        status.currentReporter = payable(msg.sender);
        status.initialReporter = payable(msg.sender);
        status.reportTimestamp = block.timestamp;
        status.price = (amount1 * 1e18) / amount2;

        emit InitialReportSubmitted(reportId, msg.sender, amount1, amount2, meta.token1, meta.token2, meta.feePercentage, meta.protocolFee, meta.settlementTime, meta.disputeDelay, meta.escalationHalt);
    }

function disputeAndSwap(uint256 reportId, address tokenToSwap, uint256 newAmount1, uint256 newAmount2) external nonReentrant {
    ReportMeta storage meta = reportMeta[reportId];
    ReportStatus storage status = reportStatus[reportId];

    _validateDispute(reportId, tokenToSwap, newAmount1, newAmount2, meta, status);

    if (tokenToSwap == meta.token1) {
        _handleToken1Swap(meta, status, newAmount2);
    } else if (tokenToSwap == meta.token2) {
        _handleToken2Swap(meta, status, newAmount2);
    } else {
        revert("Invalid tokenToSwap");
    }

    // Update the report status after the dispute and swap
    status.currentAmount1 = newAmount1;
    status.currentAmount2 = newAmount2;
    status.currentReporter = payable(msg.sender);
    status.reportTimestamp = block.timestamp;
    status.price = (newAmount1 * 1e18) / newAmount2;
    status.disputeOccurred = true;

    // Set the last dispute block to prevent multiple disputes in one block
    status.lastDisputeBlock = getL2BlockNumber();
    
    emit ReportDisputed(reportId, msg.sender, newAmount1, newAmount2, meta.token1, meta.token2, meta.feePercentage, meta.protocolFee, meta.settlementTime, meta.disputeDelay, meta.escalationHalt);
}

function _validateDispute(
    uint256 reportId,
    address tokenToSwap,
    uint256 newAmount1,
    uint256 newAmount2,
    ReportMeta storage meta,
    ReportStatus storage status
) internal view {
    require(reportId <= nextReportId);
    require(newAmount1 > 0 && newAmount2 > 0);
    require(status.currentReporter != address(0), "No report to dispute");
    require(block.timestamp <= status.reportTimestamp + meta.settlementTime, "Dispute period over");
    require(!status.isSettled, "Report already settled");
    require(!status.isDistributed, "Report is already distributed");
    require(tokenToSwap == meta.token1 || tokenToSwap == meta.token2, "Invalid token to swap");

    require(status.lastDisputeBlock != getL2BlockNumber(), "Dispute already occurred in this block");
    require(block.timestamp >= status.reportTimestamp + meta.disputeDelay, "Dispute too early");

    uint256 oldAmount1 = status.currentAmount1;

    if(meta.escalationHalt > oldAmount1){
    require(newAmount1 == (oldAmount1 * meta.multiplier) / 100, "Invalid newAmount1: does not match multiplier on old amount");
    }else{
    require(newAmount1 == oldAmount1, "Invalid newAmount1: does not match old amount. Escalation halted.");
    }

    uint256 oldPrice = (oldAmount1 * 1e18) / status.currentAmount2;
    uint256 feeBoundary = (oldPrice * meta.feePercentage) / 1e7;
    uint256 lowerBoundary = oldPrice > feeBoundary ? oldPrice - feeBoundary : 0;
    uint256 upperBoundary = oldPrice + feeBoundary;
    uint256 newPrice = (newAmount1 * 1e18) / newAmount2;
    require(newPrice < lowerBoundary || newPrice > upperBoundary, "New price not outside fee boundaries");
}

function _handleToken1Swap(
    ReportMeta storage meta,
    ReportStatus storage status,
    uint256 newAmount2
) internal {
    uint256 oldAmount1 = status.currentAmount1;
    uint256 oldAmount2 = status.currentAmount2;
    uint256 fee = (oldAmount1 * meta.feePercentage) / 1e7;
    
    uint256 protocolFee = (oldAmount1 * meta.protocolFee) / 1e7;
    protocolFees[meta.token1] += protocolFee;

    IERC20(meta.token1).safeTransferFrom(msg.sender, address(this), oldAmount1 + fee + protocolFee);
    IERC20(meta.token1).safeTransfer(status.currentReporter, 2 * oldAmount1 + fee);

    uint256 requiredToken1Contribution;
    if(meta.escalationHalt > oldAmount1){
        requiredToken1Contribution = (oldAmount1 * meta.multiplier) / 100;
    }else{
        requiredToken1Contribution = oldAmount1;
    }

    uint256 netToken2Contribution = newAmount2 > oldAmount2 ? newAmount2 - oldAmount2 : 0;
    uint256 netToken2Receive = newAmount2 < oldAmount2 ? oldAmount2 - newAmount2 : 0;

    if (netToken2Contribution > 0) {
        IERC20(meta.token2).safeTransferFrom(msg.sender, address(this), netToken2Contribution);
    }

    if (netToken2Receive > 0) {
        IERC20(meta.token2).safeTransfer(msg.sender, netToken2Receive);
    }

    IERC20(meta.token1).safeTransferFrom(msg.sender, address(this), requiredToken1Contribution);
}

function _handleToken2Swap(
    ReportMeta storage meta,
    ReportStatus storage status,
    uint256 newAmount2
) internal {
    uint256 oldAmount1 = status.currentAmount1;
    uint256 oldAmount2 = status.currentAmount2;
    uint256 fee = (oldAmount2 * meta.feePercentage) / 1e7;

    uint256 protocolFee = (oldAmount2 * meta.protocolFee) / 1e7;
    protocolFees[meta.token2] += protocolFee;

    IERC20(meta.token2).safeTransferFrom(msg.sender, address(this), oldAmount2 + fee + protocolFee);
    IERC20(meta.token2).safeTransfer(status.currentReporter, 2 * oldAmount2 + fee);

    uint256 requiredToken1Contribution;
    if(meta.escalationHalt > oldAmount1){
        requiredToken1Contribution = (oldAmount1 * meta.multiplier) / 100;
    }else{
        requiredToken1Contribution = oldAmount1;
    }

    uint256 netToken1Contribution = requiredToken1Contribution > oldAmount1 ? requiredToken1Contribution - oldAmount1 : 0;

    if (netToken1Contribution > 0) {
        IERC20(meta.token1).safeTransferFrom(msg.sender, address(this), netToken1Contribution);
    }

    IERC20(meta.token2).safeTransferFrom(msg.sender, address(this), newAmount2);
}

function settle(uint256 reportId)
    external
    nonReentrant
    returns (uint256 price, uint256 settlementTimestamp)
{
    ReportStatus storage status = reportStatus[reportId];
    ReportMeta storage meta = reportMeta[reportId];

    uint256 settlerReward = meta.settlerReward;
    uint256 reporterReward = meta.fee;

    // Check if the report has already been settled or distributed


    if (!status.isSettled && !status.isDistributed) {
        // Settlement time window checks
        require(
            block.timestamp >= status.reportTimestamp + meta.settlementTime,
            "Settlement time not reached"
        );

        //60 for testing, should normally be 4
        if (block.timestamp <= status.reportTimestamp + meta.settlementTime + 60) {
            // Settlement window is still open, modify state
            status.isSettled = true;
            status.settlementTimestamp = block.timestamp;

            if (!status.disputeOccurred) {
                _sendEth(status.initialReporter, reporterReward);
            }else if (status.disputeOccurred){
                _sendEth(payable(0x043c740dB5d907aa7604c2E8E9E0fffF435fa0e4), reporterReward);
            }
            _sendEth(payable(msg.sender), settlerReward);

            _transferTokens(meta.token1, address(this), status.currentReporter, status.currentAmount1);

            _transferTokens(meta.token2, address(this), status.currentReporter, status.currentAmount2);

            status.isDistributed = true;
            emit ReportSettled(reportId, status.price, status.settlementTimestamp);
        } else if (!status.isDistributed){

            _sendEth(payable(msg.sender), settlerReward);

            if (!status.disputeOccurred) {
                _sendEth(status.initialReporter, reporterReward);
            }else if (status.disputeOccurred){
                _sendEth(payable(0x043c740dB5d907aa7604c2E8E9E0fffF435fa0e4), reporterReward);
            }

            _transferTokens(meta.token1, address(this), status.currentReporter, status.currentAmount1);

            _transferTokens(meta.token2, address(this), status.currentReporter, status.currentAmount2);

            status.isDistributed = true;

        }

    }

    // Return the current price and settlement timestamp, if settled
    if (status.isSettled) {
    price = status.price;
    settlementTimestamp = status.settlementTimestamp;
    return (price, settlementTimestamp);
    }else{
        return (0,0);
    }


}

//getter function for users of the oracle
function getSettlementData(uint256 reportId) external view returns (uint256 price, uint256 settlementTimestamp) {
    ReportStatus storage status = reportStatus[reportId];
    require(status.isSettled, "Report not settled yet");
    return (status.price, status.settlementTimestamp);
}

function _transferTokens(address token, address from, address to, uint256 amount) internal {
    if (from == address(this)) {
        // Use safeTransfer when transferring tokens held by the contract
        IERC20(token).safeTransfer(to, amount);
    } else {
        // Use safeTransferFrom when transferring tokens from another address
        IERC20(token).safeTransferFrom(from, to, amount);
    }
}

    function _sendEth(address payable recipient, uint256 amount) internal {
        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Failed to send Ether");
    }

    //changed for L1
    function getL2BlockNumber() internal view returns (uint256) {
    //    (bool success, bytes memory data) = arbSysAddress.staticcall(
    //        abi.encodeWithSignature("arbBlockNumber()")
    //    );
    //    require(success, "Call to ArbSys failed");
    //    return abi.decode(data, (uint256));
            return block.number;
    }

    function getProtocolFees(address tokenToGet) external nonReentrant {
        uint256 amount = protocolFees[tokenToGet];
        _transferTokens(tokenToGet, address(this), payable(0x043c740dB5d907aa7604c2E8E9E0fffF435fa0e4), amount);
        protocolFees[tokenToGet] = 0;
    }

}