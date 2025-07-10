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

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

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

// File: @openzeppelin/contracts/token/ERC20/IERC20.sol

// OpenZeppelin Contracts (last updated v5.0.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.20;

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
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );

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
    function allowance(
        address owner,
        address spender
    ) external view returns (uint256);

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
    function transferFrom(
        address from,
        address to,
        uint256 value
    ) external returns (bool);
}

// File: contracts/RewardDistributor.sol

//SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract AwardDistributor is Ownable {
    address public immutable ur369gifthAddress_30 =
        0x70C819445c6Bb5a144954818DE138b4A713408dC; // 30%
    address public immutable ur369impactAddress_30 =
        0x22357B3034DF4a65a00E5887aFB09e94Df17B7B9; // 30%
    address public immutable ur369Address_30 =
        0x4eb401801b42139737faC676C5da5e43F6A1A828; // 30%
    address public immutable ur369devsncomAddress_10 =
        0xDB0ccF145A929c48277a4431004D633E9D84258a; // 10%

    constructor() Ownable(msg.sender) {}

    function distributeEth() external payable onlyOwner {
        uint256 nativeCurrency = msg.value;

        if (nativeCurrency > 0) {
            uint256 thirtyPercent = (nativeCurrency * 30) / 100;
            uint256 remaining = nativeCurrency - (thirtyPercent * 3);

            payable(ur369gifthAddress_30).transfer(thirtyPercent); // 30%
            payable(ur369impactAddress_30).transfer(thirtyPercent); // 30%
            payable(ur369Address_30).transfer(thirtyPercent); // 30%
            payable(ur369devsncomAddress_10).transfer(remaining); // 10%
        }
    }

    function distributeERC20(
        address[] memory tokenAddresses,
        uint256[] memory amounts
    ) external onlyOwner {
        address sender = msg.sender;
        // Distribute ERC20 tokens
        require(
            tokenAddresses.length == amounts.length,
            "RewardDistributor: Amount for each token in not entered"
        );
        for (uint i = 0; i < tokenAddresses.length; i++) {
            uint256 amountToDistribute = amounts[i];
            address tokenAddress = tokenAddresses[i];

            require(
                amountToDistribute > 0,
                "RewardDistributor: Invalid amount"
            );

            uint256 thirtyPercent = (amountToDistribute * 30) / 100;
            uint256 remaining = amountToDistribute - (thirtyPercent * 3);

            require(
                IERC20(tokenAddress).transferFrom(
                    sender,
                    ur369gifthAddress_30,
                    thirtyPercent
                ),
                "RewardDistributor: TransferFrom Failed."
            ); // 30%
            require(
                IERC20(tokenAddress).transferFrom(
                    sender,
                    ur369impactAddress_30,
                    thirtyPercent
                ),
                "RewardDistributor: TransferFrom Failed."
            ); // 30%
            require(
                IERC20(tokenAddress).transferFrom(
                    sender,
                    ur369Address_30,
                    thirtyPercent
                ),
                "RewardDistributor: TransferFrom Failed."
            ); // remaining
            require(
                IERC20(tokenAddress).transferFrom(
                    sender,
                    ur369devsncomAddress_10,
                    remaining
                ),
                "RewardDistributor: TransferFrom Failed."
            ); // 10%
        }
    }

    function donateAndDistribute() external payable {
        uint256 nativeCurrency = msg.value;

        if (nativeCurrency > 0) {
            uint256 thirtyPercent = (nativeCurrency * 30) / 100;
            uint256 remaining = nativeCurrency - (thirtyPercent * 3);

            payable(ur369gifthAddress_30).transfer(thirtyPercent); // 30%
            payable(ur369impactAddress_30).transfer(thirtyPercent); // 30%
            payable(ur369Address_30).transfer(thirtyPercent); // 30%
            payable(ur369devsncomAddress_10).transfer(remaining); // 10%
        }
    }

    function donateAndDistributeERC20(
        address tokenAddress,
        uint256 _amount
    ) external {
        require(_amount != 0, "RewardDistributor: Invalid Amount");
        address sender = msg.sender;

        uint256 thirtyPercent = (_amount * 30) / 100;
        uint256 remaining = _amount - (thirtyPercent * 3);

        require(
            IERC20(tokenAddress).transferFrom(
                sender,
                ur369gifthAddress_30,
                thirtyPercent
            ),
            "RewardDistributor: TransferFrom Failed."
        ); // 30%
        require(
            IERC20(tokenAddress).transferFrom(
                sender,
                ur369impactAddress_30,
                thirtyPercent
            ),
            "RewardDistributor: TransferFrom Failed."
        ); // 30%
        require(
            IERC20(tokenAddress).transferFrom(
                sender,
                ur369Address_30,
                thirtyPercent
            ),
            "RewardDistributor: TransferFrom Failed."
        ); // remaining
        require(
            IERC20(tokenAddress).transferFrom(
                sender,
                ur369devsncomAddress_10,
                remaining
            ),
            "RewardDistributor: TransferFrom Failed."
        ); // 10%
    }
}