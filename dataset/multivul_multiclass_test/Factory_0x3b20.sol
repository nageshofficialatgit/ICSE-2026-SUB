// SPDX-License-Identifier: MIT
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

// Interface for managing the casino token address
interface ICasinoTokenStorage {
    function setCasinoToken(address _casinoToken) external;
    function getCasinoToken() external view returns (address);
}

contract Factory is Ownable, ICasinoTokenStorage {
    IERC20 public lpToken;
    IERC20 public casinoToken;

    uint256 public totalLiquidity;
    uint256 public casinoBalance;

    struct User {
        uint256 depositedLP;
        uint256 pendingRewards;
        bool exists;
    }

    mapping(address => User) public users;
    address[] public userList;

    event LiquidityAdded(address indexed user, uint256 amount);
    event LiquidityRemoved(address indexed user, uint256 amount);
    event CasinoFundDeposited(uint256 amount);
    event RewardsDistributed(uint256 totalReward);
    event RewardClaimed(address indexed user, uint256 amount);
    event CasinoTokenUpdated(address newToken);

    constructor(address _lpToken, address _casinoToken) Ownable(msg.sender) {
        lpToken = IERC20(_lpToken);
        casinoToken = IERC20(_casinoToken);
    }

    function addLiquidity(uint256 amount) external {
        require(amount > 0, "Invalid amount");
        require(lpToken.transferFrom(msg.sender, address(this), amount), "LP transfer failed");

        if (!users[msg.sender].exists) {
            users[msg.sender].exists = true;
            userList.push(msg.sender);
        }

        users[msg.sender].depositedLP += amount;
        totalLiquidity += amount;

        emit LiquidityAdded(msg.sender, amount);
    }

    function removeLiquidity(uint256 amount) external {
        require(users[msg.sender].depositedLP >= amount, "Insufficient balance");

        users[msg.sender].depositedLP -= amount;
        totalLiquidity -= amount;

        if (users[msg.sender].depositedLP == 0) {
            users[msg.sender].exists = false;
        }

        require(lpToken.transfer(msg.sender, amount), "LP transfer failed");

        emit LiquidityRemoved(msg.sender, amount);
    }

    function depositCasinoFund(uint256 amount) external onlyOwner {
        require(amount > 0, "Invalid amount");
        require(casinoToken.transferFrom(msg.sender, address(this), amount), "Casino token transfer failed");

        casinoBalance += amount;

        emit CasinoFundDeposited(amount);
    }

    function distributeRewards() external onlyOwner {
        require(totalLiquidity > 0, "No liquidity available");
        require(casinoBalance > 0, "No casino funds");

        uint256 totalReward = casinoBalance;
        casinoBalance = 0;

        for (uint256 i = 0; i < userList.length; i++) {
            address userAddr = userList[i];

            if (users[userAddr].exists && users[userAddr].depositedLP > 0) {
                uint256 userShare = (users[userAddr].depositedLP * totalReward) / totalLiquidity;
                users[userAddr].pendingRewards += userShare;
            }
        }

        emit RewardsDistributed(totalReward);
    }

    function claimRewards() external {
        uint256 reward = users[msg.sender].pendingRewards;
        require(reward > 0, "No rewards available");

        users[msg.sender].pendingRewards = 0;
        require(casinoToken.transfer(msg.sender, reward), "Reward transfer failed");

        emit RewardClaimed(msg.sender, reward);
    }

    function withdrawcasinoToken() external onlyOwner {
        require(casinoToken.transfer(owner(), casinoToken.balanceOf(address(this))), "transfer failed");
    }

    function getAllUsers() external view returns (address[] memory) {
        return userList;
    }

    // --- Interface Implementation ---

    function setCasinoToken(address _casinoToken) external override onlyOwner {
        require(_casinoToken != address(0), "Invalid token address");
        casinoToken = IERC20(_casinoToken);
        emit CasinoTokenUpdated(_casinoToken);
    }

    function getCasinoToken() external view override returns (address) {
        return address(casinoToken);
    }
}