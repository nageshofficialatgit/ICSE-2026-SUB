// SPDX-License-Identifier: MIT
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

// File: @openzeppelin/contracts/security/Pausable.sol


// OpenZeppelin Contracts (last updated v4.7.0) (security/Pausable.sol)

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
    /**
     * @dev Emitted when the pause is triggered by `account`.
     */
    event Paused(address account);

    /**
     * @dev Emitted when the pause is lifted by `account`.
     */
    event Unpaused(address account);

    bool private _paused;

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
        require(!paused(), "Pausable: paused");
    }

    /**
     * @dev Throws if the contract is not paused.
     */
    function _requirePaused() internal view virtual {
        require(paused(), "Pausable: not paused");
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

// File: @openzeppelin/contracts/utils/math/SafeMath.sol


// OpenZeppelin Contracts (last updated v4.9.0) (utils/math/SafeMath.sol)

pragma solidity ^0.8.0;

// CAUTION
// This version of SafeMath should only be used with Solidity 0.8 or later,
// because it relies on the compiler's built in overflow checks.

/**
 * @dev Wrappers over Solidity's arithmetic operations.
 *
 * NOTE: `SafeMath` is generally not needed starting with Solidity 0.8, since the compiler
 * now has built in overflow checking.
 */
library SafeMath {
    /**
     * @dev Returns the addition of two unsigned integers, with an overflow flag.
     *
     * _Available since v3.4._
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
     *
     * _Available since v3.4._
     */
    function trySub(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b > a) return (false, 0);
            return (true, a - b);
        }
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, with an overflow flag.
     *
     * _Available since v3.4._
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
     *
     * _Available since v3.4._
     */
    function tryDiv(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a / b);
        }
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers, with a division by zero flag.
     *
     * _Available since v3.4._
     */
    function tryMod(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a % b);
        }
    }

    /**
     * @dev Returns the addition of two unsigned integers, reverting on
     * overflow.
     *
     * Counterpart to Solidity's `+` operator.
     *
     * Requirements:
     *
     * - Addition cannot overflow.
     */
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, reverting on
     * overflow (when the result is negative).
     *
     * Counterpart to Solidity's `-` operator.
     *
     * Requirements:
     *
     * - Subtraction cannot overflow.
     */
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return a - b;
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, reverting on
     * overflow.
     *
     * Counterpart to Solidity's `*` operator.
     *
     * Requirements:
     *
     * - Multiplication cannot overflow.
     */
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        return a * b;
    }

    /**
     * @dev Returns the integer division of two unsigned integers, reverting on
     * division by zero. The result is rounded towards zero.
     *
     * Counterpart to Solidity's `/` operator.
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return a / b;
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers. (unsigned integer modulo),
     * reverting when dividing by zero.
     *
     * Counterpart to Solidity's `%` operator. This function uses a `revert`
     * opcode (which leaves remaining gas untouched) while Solidity uses an
     * invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return a % b;
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, reverting with custom message on
     * overflow (when the result is negative).
     *
     * CAUTION: This function is deprecated because it requires allocating memory for the error
     * message unnecessarily. For custom revert reasons use {trySub}.
     *
     * Counterpart to Solidity's `-` operator.
     *
     * Requirements:
     *
     * - Subtraction cannot overflow.
     */
    function sub(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        unchecked {
            require(b <= a, errorMessage);
            return a - b;
        }
    }

    /**
     * @dev Returns the integer division of two unsigned integers, reverting with custom message on
     * division by zero. The result is rounded towards zero.
     *
     * Counterpart to Solidity's `/` operator. Note: this function uses a
     * `revert` opcode (which leaves remaining gas untouched) while Solidity
     * uses an invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function div(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a / b;
        }
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers. (unsigned integer modulo),
     * reverting with custom message when dividing by zero.
     *
     * CAUTION: This function is deprecated because it requires allocating memory for the error
     * message unnecessarily. For custom revert reasons use {tryMod}.
     *
     * Counterpart to Solidity's `%` operator. This function uses a `revert`
     * opcode (which leaves remaining gas untouched) while Solidity uses an
     * invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function mod(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a % b;
        }
    }
}

// File: contracts/ECMCoinICO.sol


pragma solidity ^0.8.26;






/// @title ECM Coin ICO Contract
/// @notice Manages ICO stages, purchases, referrals, and bonuses for ECM token
contract ECMCoinICO is ReentrancyGuard, Pausable, Ownable(msg.sender) {
    using SafeMath for uint256;

    // Core token contracts
    IERC20 public immutable ecmCoin;    // ECM Coin contract
    IERC20 public immutable usdtToken;   // USDT token contract
    address payable public treasuryWallet;

    enum PaymentType { ETH, USDT }

    // Stage structure for ICO phases
    struct Stage {
        uint256 target;           // Total tokens to sell in this stage
        uint256 ethPrice;         // Price in ETH
        uint256 usdtPrice;        // Price in USDT
        uint256 ecmRefBonus;      // Referral bonus in ECM (%)
        uint256 paymentRefBonus;  // Referral bonus in payment token (%)
        uint256 ecmSold;          // Amount sold in this stage
        bool isCompleted;         // Stage completion status
    }

    Stage[] public stages;

    // State variables
    uint256 public currentStage;
    uint256 public totalEcmSold;
    uint256 public totalBonusDistributed;
    uint256 public totalECMReferralDistributed;
    uint256 public totalETHReferralDistributed;
    uint256 public totalUSDTReferralDistributed;
    uint256 public totalEcmSoldByETH;
    uint256 public totalEcmSoldByUSDT;

    bool public isBonusEnabled = false;  // Default enabled for bonus distribution
    uint256 public minBonusAmount = 300 * 1e18;  // Minimum amount for bonus eligibility
    uint256 public bonusPercentage = 5;         // Bonus percentage

    // Events
    event ECMPurchased(address indexed buyer, uint256 amount, uint256 stage, PaymentType paymentType, uint256 paymentAmount);
    event ReferralRewardPaid(address indexed referrer, uint256 ecmAmount, uint256 paymentAmount, PaymentType paymentType);
    event BonusTokensAwarded(address indexed buyer, uint256 bonusAmount);
    event TreasuryWalletUpdated(address indexed newWallet);
    event TokensWithdrawn(address indexed to, uint256 amount);
    event FundsWithdrawn(address indexed to, uint256 amount, PaymentType paymentType);
    event EmergencyWithdraw(address indexed treasury, uint256 ethAmount, uint256 usdtAmount, uint256 ecmAmount);
    event StageCompleted(uint256 stageIndex);
    event StageUpdated(uint256 stageIndex);
    event CurrentStageUpdated(uint256 newStage);

    /// @notice Initialize contract with token addresses
    constructor(address _ecmCoin, address _usdtToken) {
        require(_ecmCoin != address(0) && _usdtToken != address(0), "Invalid addresses");
        ecmCoin = IERC20(_ecmCoin);
        usdtToken = IERC20(_usdtToken);
        treasuryWallet = payable(owner());
        
        // Initialize ICO stages
        stages.push(Stage(200000 * 1e18, 0.00040 ether, 1.20 * 1e6, 2, 3, 0, false));
        stages.push(Stage(200000 * 1e18, 0.00046 ether, 1.20 * 1e6, 2, 3, 0, false));
        stages.push(Stage(200000 * 1e18, 0.00047 ether, 1.20 * 1e6, 2, 3, 0, false));
        stages.push(Stage(100000 * 1e18, 0.00048 ether, 1.20 * 1e6, 2, 3, 0, false));
        stages.push(Stage(100000 * 1e18, 0.00049 ether, 1.20 * 1e6, 2, 3, 0, false));
        stages.push(Stage(50000 * 1e18, 0.00050 ether, 1.20 * 1e6, 2, 3, 0, false));
        stages.push(Stage(50000 * 1e18, 0.00051 ether, 1.20 * 1e6, 2, 3, 0, false));
        stages.push(Stage(50000 * 1e18, 0.00052 ether, 1.20 * 1e6, 2, 3, 0, false));
        stages.push(Stage(50000 * 1e18, 0.00053 ether, 1.20 * 1e6, 2, 3, 0, false));
    }

    /// @notice Handle direct ETH transfers
    receive() external payable whenNotPaused {
        require(currentStage < stages.length, "ICO has ended");
        require(msg.value > 0, "Invalid amount");
        _buyECM(msg.sender, address(0), PaymentType.ETH, msg.value);
    }

    /// @notice Buy ECM with ETH using referral
    function buyECMWithETH(address referrer) external payable whenNotPaused nonReentrant {
        require(currentStage < stages.length, "ICO has ended");
        require(msg.value > 0, "Invalid amount");
        _buyECM(msg.sender, referrer, PaymentType.ETH, msg.value);
    }

    /// @notice Buy ECM with USDT using referral
    function buyECMWithUSDT(uint256 amount, address referrer) external whenNotPaused nonReentrant {
        require(currentStage < stages.length, "ICO has ended");
        require(amount > 0, "Invalid amount");
        require(block.chainid == 1, "Only Ethereum mainnet supported");
        require(usdtToken.balanceOf(msg.sender) >= amount, "Insufficient USDT balance");
        require(usdtToken.allowance(msg.sender, address(this)) >= amount, "Insufficient USDT allowance");
        _buyECM(msg.sender, referrer, PaymentType.USDT, amount);
    }

    /// @notice Internal function to handle ECM purchases
    function _buyECM(address buyer, address referrer, PaymentType paymentType, uint256 paymentAmount) internal {
        Stage storage stage = stages[currentStage];
        require(paymentAmount > 0, "Payment amount must be greater than 0");

        uint256 ecmToBuy;
        if (paymentType == PaymentType.ETH) {
            require(stage.ethPrice > 0, "Invalid ETH price");
            ecmToBuy = paymentAmount.mul(1e18).div(stage.ethPrice);
            totalEcmSoldByETH = totalEcmSoldByETH.add(ecmToBuy);
        } else {
            require(stage.usdtPrice > 0, "Invalid USDT price"); 
            ecmToBuy = paymentAmount.mul(1e18).div(stage.usdtPrice);
            totalEcmSoldByUSDT = totalEcmSoldByUSDT.add(ecmToBuy);
            require(usdtToken.transferFrom(buyer, address(this), paymentAmount), "USDT transfer failed");
        }

        require(ecmToBuy > 0, "ECM amount to buy must be greater than 0");
        require(stage.ecmSold.add(ecmToBuy) <= stage.target, "Exceeds stage target");

        if (referrer != address(0) && referrer != buyer) {
            _handleReferral(referrer, paymentAmount, ecmToBuy, paymentType);
        }

        // Update state and handle bonus
        stage.ecmSold = stage.ecmSold.add(ecmToBuy);
        totalEcmSold = totalEcmSold.add(ecmToBuy);

        if (ecmToBuy >= minBonusAmount && bonusPercentage > 0 && isBonusEnabled) {
            uint256 bonusTokens = ecmToBuy.mul(bonusPercentage).div(100);
            ecmToBuy = ecmToBuy.add(bonusTokens);
            totalBonusDistributed = totalBonusDistributed.add(bonusTokens);
            emit BonusTokensAwarded(buyer, bonusTokens);
        }

        require(ecmCoin.balanceOf(address(this)) >= ecmToBuy, "Insufficient ECM balance in contract");
        require(ecmCoin.transfer(buyer, ecmToBuy), "ECM transfer failed");

        emit ECMPurchased(buyer, ecmToBuy, currentStage, paymentType, paymentAmount);

        if (stage.ecmSold >= stage.target) {
            stage.isCompleted = true;
            emit StageCompleted(currentStage);
            progressToNextStage();
        }
    }

    /// @notice Handle referral rewards
    function _handleReferral(address referrer, uint256 paymentAmount, uint256 ecmAmount, PaymentType paymentType) internal {
        Stage storage stage = stages[currentStage];

        uint256 paymentReferralAmount = paymentAmount.mul(stage.paymentRefBonus).div(100);
        if (paymentReferralAmount > 0) {
            if (paymentType == PaymentType.USDT) {
                require(usdtToken.transfer(referrer, paymentReferralAmount), "USDT referral transfer failed");
                totalUSDTReferralDistributed = totalUSDTReferralDistributed.add(paymentReferralAmount);
            } else {
                (bool success, ) = payable(referrer).call{value: paymentReferralAmount}("");
                require(success, "ETH referral transfer failed");
                totalETHReferralDistributed = totalETHReferralDistributed.add(paymentReferralAmount);
            }
        }

        uint256 ecmReferralAmount = ecmAmount.mul(stage.ecmRefBonus).div(100);
        if (ecmReferralAmount > 0) {
            require(ecmCoin.balanceOf(address(this)) >= ecmReferralAmount, "Insufficient ECM balance for referral");
            require(ecmCoin.transfer(referrer, ecmReferralAmount), "ECM referral transfer failed");
            totalECMReferralDistributed = totalECMReferralDistributed.add(ecmReferralAmount);
        }

        emit ReferralRewardPaid(referrer, ecmReferralAmount, paymentReferralAmount, paymentType);
    }

    // Admin Functions

    /// @notice Toggle bonus token distribution on/off
    /// @dev Only owner can call this function
    function toggleBonus() external onlyOwner {
        isBonusEnabled = !isBonusEnabled;
    }

    /// @notice Set bonus parameters
    function setBonusParameters(uint256 _minAmount, uint256 _percentage) external onlyOwner {
        require(_minAmount > 0, "Invalid minimum amount");
        require(_percentage > 0, "Invalid percentage");
        minBonusAmount = _minAmount;
        bonusPercentage = _percentage;
    }

    /// @notice Progress to next available stage
    function progressToNextStage() internal {
        for (uint256 i = currentStage + 1; i < stages.length; i++) {
            if (stages[i].target > 0) {
                currentStage = i;
                emit CurrentStageUpdated(i);
                return;
            }
        }
        currentStage = stages.length;
    }

    /// @notice Admin control to set current stage
    function setCurrentStage(uint256 newStage) external onlyOwner {
        require(newStage < stages.length, "Invalid stage index");
        currentStage = newStage;
        emit CurrentStageUpdated(newStage);
    }

    /// @notice Get current stage information
    function currentStageInfo() external view returns (
        uint256 stageIndex,
        uint256 target,
        uint256 ethPrice,
        uint256 usdtPrice,
        uint256 ecmRefBonus,
        uint256 paymentRefBonus,
        uint256 ecmSold,
        bool isCompleted
    ) {
        require(currentStage < stages.length, "ICO has ended");
        Stage storage stage = stages[currentStage];
        return (
            currentStage,
            stage.target,
            stage.ethPrice,
            stage.usdtPrice,
            stage.ecmRefBonus,
            stage.paymentRefBonus,
            stage.ecmSold,
            stage.isCompleted
        );
    }

    /// @notice Update stage target
    function updateStageTarget(uint256 stageIndex, uint256 target) external onlyOwner {
        require(stageIndex < stages.length, "Invalid stage index");
        require(target > stages[stageIndex].ecmSold, "Target below sold amount");
        stages[stageIndex].target = target;
        emit StageUpdated(stageIndex);
    }

    /// @notice Update stage ecm sold
    function updateStageSold(uint256 stageIndex, uint256 soldAmount) external onlyOwner {
        require(stageIndex < stages.length, "Invalid stage index");
        require(soldAmount <= stages[stageIndex].target, "Sold amount exceeds stage target");
        stages[stageIndex].ecmSold = soldAmount;
        emit StageUpdated(stageIndex);
    }

    /// @notice Update stage prices
    function updateStagePrices(uint256 stageIndex, uint256 ethPrice, uint256 usdtPrice) external onlyOwner {
        require(stageIndex < stages.length, "Invalid stage index");
        require(ethPrice > 0 && usdtPrice > 0, "Invalid prices");
        Stage storage stage = stages[stageIndex];
        stage.ethPrice = ethPrice;
        stage.usdtPrice = usdtPrice;
        emit StageUpdated(stageIndex);
    }

    // Withdrawal Functions

    /// @notice Withdraw ECM tokens
    function withdrawECM(uint256 amount) external onlyOwner {
        require(amount > 0 && amount <= ecmCoin.balanceOf(address(this)), "Invalid amount");
        require(ecmCoin.transfer(owner(), amount), "Token transfer failed");
        emit TokensWithdrawn(owner(), amount);
    }

    /// @notice Withdraw ETH
    function withdrawETH(uint256 amount) external onlyOwner {
        require(amount > 0 && amount <= address(this).balance, "Invalid amount");
        (bool success, ) = treasuryWallet.call{value: amount}("");
        require(success, "ETH transfer failed");
        emit FundsWithdrawn(treasuryWallet, amount, PaymentType.ETH);
    }

    /// @notice Withdraw USDT
    function withdrawUSDT(uint256 amount) external onlyOwner {
       require(amount > 0 && amount <= usdtToken.balanceOf(address(this)), "Invalid amount");
       require(usdtToken.transfer(treasuryWallet, amount), "USDT transfer failed");
       emit FundsWithdrawn(treasuryWallet, amount, PaymentType.USDT);
    }

    /// @notice Emergency withdrawal of all funds
    function emergencyWithdraw() external nonReentrant onlyOwner {
        require(paused(), "Contract must be paused");
        require(treasuryWallet != address(0), "Treasury not set");

        uint256 ethBalance = address(this).balance;
        uint256 usdtBalance = usdtToken.balanceOf(address(this));
        uint256 ecmBalance = ecmCoin.balanceOf(address(this));
        
        if (ethBalance > 0) {
            (bool success, ) = treasuryWallet.call{value: ethBalance}("");
            require(success, "ETH transfer failed");
        }
        
        if (usdtBalance > 0) {
            require(usdtToken.transfer(treasuryWallet, usdtBalance), "USDT transfer failed");
        }
        
        if (ecmBalance > 0) {
            require(ecmCoin.transfer(treasuryWallet, ecmBalance), "ECM transfer failed");
        }

        emit EmergencyWithdraw(treasuryWallet, ethBalance, usdtBalance, ecmBalance);
    }

    /// @notice Update treasury wallet address
    function setTreasuryWallet(address payable _newWallet) external onlyOwner {
        require(_newWallet != address(0), "Invalid address");
        treasuryWallet = _newWallet;
        emit TreasuryWalletUpdated(_newWallet);
    }

    /// @notice Pause the contract
    function pause() external onlyOwner {
        _pause();
    }

    /// @notice Unpause the contract
    function unpause() external onlyOwner {
        _unpause();
    }
}