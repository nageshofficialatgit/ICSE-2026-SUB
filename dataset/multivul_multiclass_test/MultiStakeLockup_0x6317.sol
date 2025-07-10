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

// File: contracts/MultiStake.sol


pragma solidity ^0.8.17;



/**
 * @title Multiple-Stake Lockup Contract with Tier Logic
 *
 * @notice
 *  This contract allows users to stake an ERC20 token (e.g. ONDOAI) multiple
 *  times, choosing between three lockup durations (3, 6, or 12 months).
 *  Longer lockups get higher weight (useful for off-chain reward calculations).
 *  The contract also implements a simple tier system (Bronze/Silver/Gold)
 *  based on the user's total staked balance.
 *
 *  There are no token rewards distributed on-chain; off-chain logic or scripts
 *  can query total staked and weight to compute user rewards.
 */
contract MultiStakeLockup {
    using SafeERC20 for IERC20;

    // ---------------------------------------
    // State Variables
    // ---------------------------------------

    /// @notice The ERC20 token to be staked (e.g. ONDOAI)
    IERC20 public immutable stakingToken;

    /// @notice Approximate durations for lockups
    uint256 public constant THREE_MONTHS = 90 days;
    uint256 public constant SIX_MONTHS   = 180 days;
    uint256 public constant TWELVE_MONTHS= 360 days;

    /// @notice Weight multipliers for each lockup
    uint256 public constant WEIGHT_3  = 1; // 3-month stake weight
    uint256 public constant WEIGHT_6  = 2; // 6-month stake weight
    uint256 public constant WEIGHT_12 = 4; // 12-month stake weight

    /// @notice Tier thresholds (example values, can be modified as needed)
    uint256 public constant BRONZE_THRESHOLD = 1_000_000 ether; // 1M tokens
    uint256 public constant SILVER_THRESHOLD = 5_000_000 ether; // 5M tokens
    uint256 public constant GOLD_THRESHOLD   = 10_000_000 ether; // 10M tokens

    /// @notice Total staked across all users (for reference)
    uint256 public totalStaked;

    /// @notice Total weight across all users (for reference)
    uint256 public totalWeight;

    /// @notice Information about an individual stake
    struct StakeInfo {
        uint256 amount;       // Amount of tokens staked
        uint256 startTime;    // Timestamp when the stake was created
        uint256 lockupPeriod; // How many seconds tokens are locked
        uint256 weight;       // Amount * weightFactor
        bool withdrawn;       // Track if stake has been withdrawn
    }

    /// @notice Mapping from user => array of all stakes they created
    mapping(address => StakeInfo[]) public userStakes;

    // ---------------------------------------
    // Constructor
    // ---------------------------------------

    /**
     * @dev Sets the staking token address once upon deployment.
     * @param _stakingToken The ERC20 token to be staked (e.g. ONDOAI).
     */
    constructor(IERC20 _stakingToken) {
        stakingToken = _stakingToken;
    }

    // ---------------------------------------
    // Public Functions
    // ---------------------------------------

    /**
     * @notice Stake tokens with a chosen lockup period (3, 6, or 12 months).
     * @param amount The number of tokens to stake.
     * @param lockupChoice 1 => 3 months, 2 => 6 months, 3 => 12 months.
     */
    function stake(uint256 amount, uint8 lockupChoice) external {
        require(amount > 0, "Amount must be > 0");

        (uint256 chosenPeriod, uint256 weightFactor) = _getLockupData(lockupChoice);

        // Transfer tokens from user to this contract
        stakingToken.safeTransferFrom(msg.sender, address(this), amount);

        // Calculate stake weight
        uint256 stakeWeight = amount * weightFactor;

        // Create a new StakeInfo record
        StakeInfo memory newStake = StakeInfo({
            amount: amount,
            startTime: block.timestamp,
            lockupPeriod: chosenPeriod,
            weight: stakeWeight,
            withdrawn: false
        });

        // Add this stake to the user's array
        userStakes[msg.sender].push(newStake);

        // Update global totals
        totalStaked += amount;
        totalWeight += stakeWeight;
    }

    /**
     * @notice Withdraw a specific stake (by index in the user's stake array),
     *         but only if the lockup period has passed.
     * @param stakeIndex The index of the stake in userStakes[msg.sender].
     */
    function withdraw(uint256 stakeIndex) external {
        require(stakeIndex < userStakes[msg.sender].length, "Invalid stake index");
        StakeInfo storage currentStake = userStakes[msg.sender][stakeIndex];

        require(!currentStake.withdrawn, "Stake already withdrawn");
        require(
            block.timestamp >= (currentStake.startTime + currentStake.lockupPeriod),
            "Lockup period not finished"
        );

        // Mark stake as withdrawn
        currentStake.withdrawn = true;

        // Update global totals
        totalStaked -= currentStake.amount;
        totalWeight -= currentStake.weight;

        // Transfer tokens back to the user
        stakingToken.safeTransfer(msg.sender, currentStake.amount);
    }

    // ---------------------------------------
    // View Functions
    // ---------------------------------------

    /**
     * @notice Returns the number of stakes a user has.
     */
    function getUserStakeCount(address user) external view returns (uint256) {
        return userStakes[user].length;
    }

    /**
     * @notice Get info about a specific stake.
     * @param user The address whose stakes you want to examine.
     * @param stakeIndex The index in userStakes[user].
     */
    function getUserStakeInfo(address user, uint256 stakeIndex)
        external
        view
        returns (
            uint256 amount,
            uint256 startTime,
            uint256 lockupPeriod,
            uint256 weight,
            bool withdrawn
        )
    {
        require(stakeIndex < userStakes[user].length, "Invalid stake index");
        StakeInfo storage st = userStakes[user][stakeIndex];
        return (st.amount, st.startTime, st.lockupPeriod, st.weight, st.withdrawn);
    }

    /**
     * @notice Calculates the sum of all active stakes for a user (not withdrawn).
     * @param user The address to check.
     */
    function getUserTotalStaked(address user) public view returns (uint256) {
        uint256 total = 0;
        StakeInfo[] storage stakesArray = userStakes[user];
        for (uint256 i = 0; i < stakesArray.length; i++) {
            if (!stakesArray[i].withdrawn) {
                total += stakesArray[i].amount;
            }
        }
        return total;
    }

    /**
     * @notice Calculates the sum of all active stake weights for a user.
     * @param user The address to check.
     */
    function getUserTotalWeight(address user) external view returns (uint256) {
        uint256 w = 0;
        StakeInfo[] storage stakesArray = userStakes[user];
        for (uint256 i = 0; i < stakesArray.length; i++) {
            if (!stakesArray[i].withdrawn) {
                w += stakesArray[i].weight;
            }
        }
        return w;
    }

    /**
     * @notice Gets the user's current tier (Bronze, Silver, Gold, or None).
     *         Based on the sum of all *active* staked tokens.
     * @param user The address to check.
     */
    function getUserTier(address user) external view returns (string memory) {
        uint256 userActiveStake = getUserTotalStaked(user);

        if (userActiveStake >= GOLD_THRESHOLD) {
            return "Gold";
        } else if (userActiveStake >= SILVER_THRESHOLD) {
            return "Silver";
        } else if (userActiveStake >= BRONZE_THRESHOLD) {
            return "Bronze";
        } else {
            return "None";
        }
    }

    // ---------------------------------------
    // Internal Helpers
    // ---------------------------------------

    /**
     * @dev Returns the lockup period in seconds and weight factor
     *      for a given choice: 1 => 3 months, 2 => 6 months, 3 => 12 months.
     */
    function _getLockupData(uint8 choice) internal pure returns (uint256 period, uint256 factor) {
        if (choice == 1) {
            return (THREE_MONTHS, WEIGHT_3);
        } else if (choice == 2) {
            return (SIX_MONTHS, WEIGHT_6);
        } else if (choice == 3) {
            return (TWELVE_MONTHS, WEIGHT_12);
        }
        revert("Invalid lockup choice");
    }
}