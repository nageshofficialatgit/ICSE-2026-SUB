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

// File: contracts/CrypturalPresale.sol


pragma solidity 0.8.26;






contract CrypturalPresale is Ownable, ReentrancyGuard {
    IERC20 public usdtContract;
    IERC20 public crypturalContract;
    AggregatorV3Interface public priceFeed;

    uint private numberOfTokenSold = 0;
    
    struct Stage {
        uint tokenPrice;
        uint tokenAvailable;
    }

    Stage[] public stages;
    uint8 currentStage;

    event TokenPurchased(address from, address to, uint amount);
    event StageAdvanced(uint newStage);

    constructor(address _crypturalTokenAddress, address _usdtAddress, address _priceFeedAddress) Ownable(msg.sender) {
        crypturalContract = IERC20(_crypturalTokenAddress);
        usdtContract = IERC20(_usdtAddress);
        priceFeed = AggregatorV3Interface(_priceFeedAddress);
        currentStage = 1;

        stages.push(Stage(0.02 ether, 50_000_000));
        stages.push(Stage(0.04 ether, 150_000_000));
        stages.push(Stage(0.06 ether, 150_000_000));
        stages.push(Stage(0.085 ether, 50_000_000));
        stages.push(Stage(0.105 ether, 50_000_000));
        stages.push(Stage(0.135 ether, 50_000_000));
    }

    function buyTokensByEth() public payable {
        require(msg.value > 0, "Must send ETH to buy tokens");

        int price = getEthPriceNow();
        require(price > 0, "Invalid price");

        uint256 tokenAmount = (msg.value * uint256(price)) / getCurrentStageTokenPrice() / (10 ** priceFeed.decimals());
        require(tokenAmount <= getCurrentStageTokenAvailable(), "Insufficient tokens available");

        bool success = crypturalContract.transfer(msg.sender, tokenAmount * 1e18);
        require(success, "Transaction Failed");

        stages[currentStage].tokenAvailable = stages[currentStage].tokenAvailable - tokenAmount;
        numberOfTokenSold = numberOfTokenSold + tokenAmount;
        emit TokenPurchased(address(this), msg.sender, tokenAmount);
    }

    function buyTokensByUsdt(uint _usdtAmount) public payable {
        require(_usdtAmount > 0, "Insufficient Amount");

        uint256 tokenAmount = _usdtAmount * 1e18 / getCurrentStageTokenPrice() / 1e6;
        require(tokenAmount <= getCurrentStageTokenAvailable(), "Insufficient tokens available");

        // uint usdtBalance = usdtContract.balanceOf(msg.sender);
        // require(usdtBalance >= _usdtAmount * 1e6, "Insufficient Balance");

        uint256 allowance = usdtContract.allowance(msg.sender, address(this));
        require(allowance >= _usdtAmount, "ERC20: transfer amount exceeds allowance");

        bool usdtSuccess = usdtContract.transferFrom(msg.sender, owner(), _usdtAmount);
        require(usdtSuccess, "USDT Transaction Failed");

        bool tokenSuccess = crypturalContract.transfer(msg.sender, tokenAmount * 1e18);
        require(tokenSuccess, "Cryptural Transaction Failed");

        stages[currentStage].tokenAvailable = stages[currentStage].tokenAvailable - tokenAmount;
        numberOfTokenSold = numberOfTokenSold + tokenAmount;
        emit TokenPurchased(address(this), msg.sender, tokenAmount);
    }

    function getNumberOfTokenSold() public view onlyOwner returns(uint) {
        return numberOfTokenSold;
    }

    function getEthPriceNow() public view returns(int) {
        (, int256 price, , , ) = priceFeed.latestRoundData();
        return price;
    }

    function moveToNextStage() public onlyOwner {
        currentStage ++;
        emit StageAdvanced(currentStage);
    }

    function addStage(uint _tokenPrice, uint _tokenAvailable) public onlyOwner {
        stages.push(Stage(_tokenPrice, _tokenAvailable));
    }

    function updateStage(uint8 _stage, uint _tokenPrice, uint _tokenAvailable) public onlyOwner {
        stages[_stage].tokenPrice = _tokenPrice;
        stages[_stage].tokenAvailable = _tokenAvailable;
    }

    function deleteStage(uint _stage) public onlyOwner {
        require(_stage < stages.length, "Index out of bounds");

        for (uint256 i = _stage; i < stages.length - 1; i++) {
            stages[i] = stages[i + 1];
        }
        stages.pop();
    }

    function getStageTokenAvailable(uint _stage) public view returns(uint) {
        return stages[_stage].tokenAvailable;
    }

    function getStageTokenPrice(uint8 _stage) public view returns(uint) {
        return stages[_stage].tokenPrice;
    }

    function getCurrentStageTokenAvailable() public view returns(uint) {
        return stages[currentStage].tokenAvailable;
    }

    function getCurrentStageTokenPrice() public view returns(uint) {
        return stages[currentStage].tokenPrice;
    }

    function getCurrentStage() public view returns(uint8) {
        return currentStage;
    }

    function getStages() public view returns(Stage[] memory) {
        return stages;
    }

    fallback() external payable { }

    receive() external payable { }

    function withdraw() public payable onlyOwner nonReentrant {
        payable(owner()).transfer(address(this).balance);
    }
}