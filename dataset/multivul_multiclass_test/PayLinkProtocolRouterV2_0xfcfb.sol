// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

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

contract PayLinkProtocolRouterV2 is ReentrancyGuard, Ownable {

    // Finals
    address constant DEAD_ADDRESS = address(0x000000000000000000000000000000000000dEaD);
    address constant WETH = address(0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2);
    uint256 constant PURCHASE_TYPE_HOLD = 0;

    // Variables
    address providerFeeRecipientAddress;
    uint256 providerFeePercentage;
    mapping(uint256 => bool) private appIdFeeWhitelist;
    address ecosystemTokenAddress;
    uint256 ecosystemTokenMinimumBeneficiaryBalance;
    uint256 ecosystemTokenBeneficiaryFeePercentage;

    // Models
    struct PurchaseData {
        uint256 appId;
        uint256 userId;
        uint256 purchaseAmount;
        address purchaseTokenAddress;
        address clientWalletAddress;
        uint256 purchaseType;
        uint256 expirationTimestamp;
        uint256 burnPercentage;
    }

    // Events
    event Purchase(
        uint256 indexed appId,
        uint256 indexed userId,
        address indexed userWalletAddress,
        uint256 purchaseType,
        address purchaseTokenAddress,
        uint256 purchaseAmount,
        uint256 expirationTimestamp
    );

    constructor() Ownable(msg.sender) {
        providerFeeRecipientAddress = address(0xA4A52B2BF9AB8a5e20650b0A7326DC4E2A4c3Fa1);
        ecosystemTokenAddress = DEAD_ADDRESS;
        providerFeePercentage = 10;
        ecosystemTokenBeneficiaryFeePercentage = 5;
    }

    function purchase(PurchaseData calldata data) external payable nonReentrant {
        if (data.purchaseTokenAddress == WETH) {
            purchaseWithETH(data.appId, msg.value, data.clientWalletAddress, msg.sender);
        } else {
            if (data.purchaseType == PURCHASE_TYPE_HOLD) {
                verifyHolding(data.purchaseAmount, data.purchaseTokenAddress);
            } else {
                purchaseWithTokens(data.appId, data.purchaseAmount, data.purchaseTokenAddress, data.clientWalletAddress, data.burnPercentage, msg.sender);
            }
        }
        // Emit the Event to persist the Purchase via PLP on the Blockchain
        emit Purchase(data.appId, data.userId, msg.sender, data.purchaseType, data.purchaseTokenAddress, data.purchaseAmount, data.expirationTimestamp);
    }
    
    function verifyHolding(uint256 purchaseAmount, address purchaseTokenAddress) view private {
        // Define Payment Token
        IERC20 paymentToken = IERC20(purchaseTokenAddress);
        // Check current token balance of user
        uint256 currentTokenBalance = paymentToken.balanceOf(msg.sender);
        require(currentTokenBalance >= purchaseAmount, "Minimum token balance not reached");
    }

    function purchaseWithTokens(uint256 appId, uint256 purchaseAmount, address purchaseTokenAddress, address clientWalletAddress, uint256 burnPercentage, address purchaseMaker) private {
        // Define Payment Token
        IERC20 paymentToken = IERC20(purchaseTokenAddress);
        // Check Token Balance
        require(paymentToken.balanceOf(msg.sender) >= purchaseAmount, "Insufficient Token Balance");
        // Check Token Allowance
        uint256 allowance = paymentToken.allowance(msg.sender, address(this));
        require(allowance >= purchaseAmount, "Insufficient allowance");
        // Transfer tokens from sender to contract
        require(paymentToken.transferFrom(msg.sender, address(this), purchaseAmount), "Token transfer to contract failed");
        // Calculate provision and transfer to PLP fee address
        uint256 currentProviderFeePercentage = providerFeePercentage;
        uint256 plpFee = 0;
        uint256 purchaseMakerBenefit = 0;
        if (appIdFeeWhitelist[appId] == false) {
            if (isBeneficiaryStatus(purchaseMaker)) {
                // Reduce Provider Fee & Send Cashback to PurchaseMaker
                currentProviderFeePercentage -= ecosystemTokenBeneficiaryFeePercentage;
                purchaseMakerBenefit = (purchaseAmount * ecosystemTokenBeneficiaryFeePercentage) / 100;
                require(paymentToken.transfer(purchaseMaker, purchaseMakerBenefit), "PurchaseMaker Benefit transfer failed");
            }
            // Deduct Provider Fee
            plpFee = (purchaseAmount * currentProviderFeePercentage) / 100;
            require(paymentToken.transfer(providerFeeRecipientAddress, plpFee), "PLP fee transfer failed");
        }
        // Calculate Remaining Tokens after deducting Cashback and Provider Fee
        uint256 remainingTokens = purchaseAmount - purchaseMakerBenefit - plpFee;
        // Optionally burn tokens
        uint256 burnAmount = (remainingTokens * burnPercentage) / 100;
        if (burnAmount != 0) {
            require(paymentToken.transfer(DEAD_ADDRESS, burnAmount), "Token burn failed");
        }
        // Send remaining ETH to Client Wallet
        remainingTokens -= burnAmount;
        require(paymentToken.transfer(clientWalletAddress, remainingTokens), "Token transfer to target wallet failed");
    }

    function purchaseWithETH(uint256 appId, uint256 purchaseAmount, address clientWalletAddress, address purchaseMaker) private {
        // Check ETH Balance
        require(msg.value == purchaseAmount, "Incorrect ETH Amount");
        // Calculate provision and transfer to PLP fee address
        uint256 currentProviderFeePercentage = providerFeePercentage;
        uint256 plpFee = 0;
        uint256 purchaseMakerBenefit = 0;
        if (appIdFeeWhitelist[appId] == false) {
            if (isBeneficiaryStatus(purchaseMaker)) {
                // Reduce Provider Fee & Send Cashback to PurchaseMaker
                currentProviderFeePercentage -= ecosystemTokenBeneficiaryFeePercentage;
                purchaseMakerBenefit = (purchaseAmount * ecosystemTokenBeneficiaryFeePercentage) / 100;
                (bool purchaseMakerBenefitTransfer, ) = purchaseMaker.call{value: purchaseMakerBenefit}("");
                require(purchaseMakerBenefitTransfer, "PurchaseMaker Benefit transfer failed");
            }
            // Deduct Provider Fee
            plpFee = (purchaseAmount * currentProviderFeePercentage) / 100;
            (bool feeTransfer, ) = providerFeeRecipientAddress.call{value: plpFee}("");
            require(feeTransfer, "PLP fee transfer failed");
        }
        // Send remaining ETH to Client Wallet
        uint256 remainingETH = purchaseAmount - plpFee - purchaseMakerBenefit;
        (bool ethTransfer, ) = clientWalletAddress.call{value: remainingETH}("");
        require(ethTransfer, "ETH transfer to target wallet failed");
    }

    function isBeneficiaryStatus(address purchaseMaker) view private returns (bool) {
        if (ecosystemTokenAddress == DEAD_ADDRESS) {
            return false;
        } else {
            return IERC20(ecosystemTokenAddress).balanceOf(purchaseMaker) >= ecosystemTokenMinimumBeneficiaryBalance;
        }
    }

    function setProviderFeeRecipientAddress(address newProviderFeeRecipientAddress) external onlyOwner {
        providerFeeRecipientAddress = newProviderFeeRecipientAddress;
    }

    function setProviderFeePercentage(uint256 newProviderFeePercentage) external onlyOwner {
        require(newProviderFeePercentage <= 100, "Invalid percentage values");
        providerFeePercentage = newProviderFeePercentage;
    }

    function addAppIdToFeeWhitelist(uint256 appId) external onlyOwner {
        appIdFeeWhitelist[appId] = true;
    }

    function removeAppIdFromFeeWhitelist(uint256 appId) external onlyOwner {
        appIdFeeWhitelist[appId] = false;
    }

    function setEcosystemTokenAddress(address newEcosystemTokenAddress) external onlyOwner {
        ecosystemTokenAddress = newEcosystemTokenAddress;
    }

    function setEcosystemTokenMinimumBeneficiaryBalance(uint256 newEcosystemTokenMinimumBeneficiaryBalance) external onlyOwner {
        ecosystemTokenMinimumBeneficiaryBalance = newEcosystemTokenMinimumBeneficiaryBalance;
    }

    function setEcosystemTokenBeneficiaryFeePercentage(uint256 newEcosystemTokenBeneficiaryFeePercentage) external onlyOwner {
        require(newEcosystemTokenBeneficiaryFeePercentage <= providerFeePercentage, "Beneficiary Fee must be smaller than Provider Fee");
        ecosystemTokenBeneficiaryFeePercentage = newEcosystemTokenBeneficiaryFeePercentage;
    }

}