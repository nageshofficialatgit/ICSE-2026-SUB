// SPDX-License-Identifier: MIT
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

// File: @chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol


pragma solidity ^0.8.0;

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

// File: YoutHNesT/Private Sale Contract Mar25.sol


pragma solidity ^0.8.28;






/**
* @title YoutHNesT Private Sale Round Contract
* @dev This contract manages the YoutHNesT Private Sale of Tokens.
*
* ---- *** YoutHNesT *** More than a meme, a vision! ----
*
* --- The World's First Meme-Social Cryptocurrency. ---
*
* -- Official launch price on TGE: $0.06 per token. --
*
* ---------- Token distribution conditions: ----------
* 100% of the purchased tokens will be minted and distributed to their owners in the TGE (Token Generation Event) during Q4 2025 according to the acquisition schedule: 30% will be released initially and the remaining 70% will be done in a cliff period of 3 months followed by a linear Vesting for 6 months.
*
* - Contact: info€youthnest.info
* - Website: https://www.youthnest.info
*/

contract YoutHNesTPrivateSale is Ownable, Pausable, ReentrancyGuard { 

AggregatorV3Interface internal priceFeedPrimary; 
AggregatorV3Interface internal priceFeedSecondary; 

uint256 public manualEthPrice; 
bool public useManualPrice; 

uint8 public constant DECIMALS = 4;
uint256 public constant MAX_TOKENS_FOR_SALE = 1000000000 * 10 ** DECIMALS;
    
uint256 public constant PRICE_VARIATION_LIMIT = 2; // 2% margin for price variation
uint256 public totalTokensSold; // Track total tokens sold to avoid costly iteration

struct Investor {
     uint256 amountPurchased;
     uint256 ethSpent;
    }

IERC20 public immutable token;
address public immutable FUNDS_WALLET = 0xef5266F47F260d0DB67d3fEE2312C8eE35fA0e3a;
mapping(address => Investor) private investors;
address[] private privateInvestorList;

// Predefined token purchase options
uint256[6] public tokenOptions = [
    60000000 * 10 ** DECIMALS,  // $240,000  // $0,004 per Token
    20000000 * 10 ** DECIMALS,  // $120,000  // $0,006 per Token
    7500000 * 10 ** DECIMALS,   // $60,000   // $0,008 per Token
    1500000 * 10 ** DECIMALS,   // $15,000   // $0,010 per Token
    300000 * 10 ** DECIMALS,     // $6,000    // $0,020 per Token
    100000 * 10 ** DECIMALS     // $3,000    // $0,030 per Token
    ];

// Custom Errors
    error SaleEnded();
    error InsufficientETHSent();
    error ExceedsMaxTokensForSale();
    error InvalidAddress();
    error InvalidAmount();
    error UnauthorizedContractCall(address caller);

// Events
    event TokensPurchased(address indexed investor, uint256 amount, uint256 cost, uint256 ethPrice, string message);
    
// Constructor that sets the token contract address directly 
constructor() Ownable(msg.sender) {
token = IERC20(0x928d066C6d6b2b0DCd4D7F731c6E48b361d52b10);
    priceFeedPrimary = AggregatorV3Interface(0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419); 
    priceFeedSecondary = AggregatorV3Interface(0xF79D6aFBb6dA890132F9D7c355e3015f15F3406F); 
    useManualPrice = false;
    }

// Function to update the ETH price manually
function setManualEthPrice(uint256 _manualPrice) external onlyOwner { 
manualEthPrice = _manualPrice;
    }
function setUseManualPrice(bool _useManual) external onlyOwner { 
useManualPrice = _useManual; 
    }

function getLatestEthPrice() public view returns (uint256 price) {
    // Primero, intenta obtener el precio del oráculo primario
    (, int256 primaryPrice, , , ) = priceFeedPrimary.latestRoundData();
    if (primaryPrice > 0) {
        return uint256(primaryPrice);
    }

    // Si el oráculo primario falla, intenta obtener el precio del oráculo secundario
    (, int256 secondaryPrice, , , ) = priceFeedSecondary.latestRoundData();
    if (secondaryPrice > 0) {
        return uint256(secondaryPrice);
    }

    // Si ambos oráculos fallan, usa el precio manual
    if (useManualPrice) {
        return manualEthPrice;
    }

    revert("No valid price available from oracles or manual price.");
}

// Function to calculate cost in ETH for a given amount of tokens
    function getCostInETH(uint256 amount) public view returns (uint256) {
    uint256 usdCost;
        
        if (amount == tokenOptions[0]) {
            usdCost = 240000;
        } else if (amount == tokenOptions[1]) {
            usdCost = 120000;
        } else if (amount == tokenOptions[2]) {
            usdCost = 60000;
        } else if (amount == tokenOptions[3]) {
            usdCost = 15000;
        } else if (amount == tokenOptions[4]) {
            usdCost = 6000;
        } else if (amount == tokenOptions[5]) {
            usdCost = 3000;
        } else {
            revert InvalidAmount();
        }

        return (usdCost * 1 ether) / getLatestEthPrice();
    }

// Function to purchase tokens with a 2% price variation margin
    function buyTokens(uint256 amount) external payable whenNotPaused nonReentrant {
    if (amount == 0) revert InvalidAmount();

uint256 costInETH = getCostInETH(amount);
    uint256 minAcceptableETH = costInETH - (costInETH * PRICE_VARIATION_LIMIT / 100);
    uint256 maxAcceptableETH = costInETH + (costInETH * PRICE_VARIATION_LIMIT / 100);

if (msg.value < minAcceptableETH || msg.value > maxAcceptableETH) revert InsufficientETHSent();
    if (totalTokensSold + amount > MAX_TOKENS_FOR_SALE) revert ExceedsMaxTokensForSale();

if (investors[msg.sender].amountPurchased == 0) {
    privateInvestorList.push(msg.sender);
    }
        
investors[msg.sender].amountPurchased += amount;
investors[msg.sender].ethSpent += msg.value;

// Update total tokens sold
totalTokensSold += amount;

// Transfer the ETH directly to the funds wallet
    payable(FUNDS_WALLET).transfer(msg.value);

// Emit the TokensPurchased event with a thank you message emit     
    uint256 ethPrice = getLatestEthPrice();
    emit TokensPurchased(msg.sender, amount, costInETH, ethPrice, "Your purchase of YoutHNesT Tokens has been completed successfully. Thank you for joining the YoutHNesT project!");
    }

// Function to retrieve the amount purchased by an investor
    function getAllocation(address investor) external view onlyOwner returns (uint256) {
    return investors[investor].amountPurchased;
    }

// Function to get the number of private investors
    function getPrivateInvestorCount() external view onlyOwner returns (uint256) {
    return privateInvestorList.length;
   }

// Function to get the address of a private investor by index
    function getPrivateInvestor(uint256 index) external view onlyOwner returns (address) {
    require(index < privateInvestorList.length, "Invalid index");
    return privateInvestorList[index];
    }

// Function to pause the contract in case of emergency
    function pause() external onlyOwner {
    _pause();
    }

// Function to unpause the contract
    function unpause() external onlyOwner {
    _unpause();
    }

// Function to withdraw ETH accidentally sent to the contract 
function withdrawETH() external onlyOwner { 
    uint256 balance = address(this).balance; 
    require(balance > 0, "No ETH to withdraw"); 
    payable(FUNDS_WALLET).transfer(balance); 
    }
  }