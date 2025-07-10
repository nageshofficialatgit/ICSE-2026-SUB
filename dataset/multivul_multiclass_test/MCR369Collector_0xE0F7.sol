// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @notice Simple single owner authorization mixin.
/// @author Solmate (https://github.com/transmissions11/solmate/blob/main/src/auth/Owned.sol)
/// @dev Modified to have a 2-step ownership
abstract contract Owned {
    /*//////////////////////////////////////////////////////////////
                                 EVENTS
    //////////////////////////////////////////////////////////////*/

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );
    event OwnershipTransferInitiated(
        address indexed previousOwner,
        address indexed newOwner
    );

    /*//////////////////////////////////////////////////////////////
                            OWNERSHIP STORAGE
    //////////////////////////////////////////////////////////////*/

    address public owner;
    address public pendingOwner;

    modifier onlyOwner() virtual {
        require(msg.sender == owner, "UNAUTHORIZED");

        _;
    }

    modifier onlyPendingOwner() virtual {
        require(msg.sender == pendingOwner, "UNAUTHORIZED");

        _;
    }

    /*//////////////////////////////////////////////////////////////
                               CONSTRUCTOR
    //////////////////////////////////////////////////////////////*/

    constructor(address _owner) {
        owner = _owner;

        emit OwnershipTransferred(address(0), _owner);
    }

    /*//////////////////////////////////////////////////////////////
                             OWNERSHIP LOGIC
    //////////////////////////////////////////////////////////////*/

    function transferOwnership(address newOwner) public virtual onlyOwner {
        pendingOwner = newOwner;

        emit OwnershipTransferInitiated(msg.sender, newOwner);
    }

    function acceptOwnership() public virtual onlyPendingOwner {
        emit OwnershipTransferred(owner, msg.sender);

        owner = pendingOwner;
        delete pendingOwner;
    }
}

/// @title Interface for ERC20 Token Standard
interface IERC20 {
    /// @notice Returns the total amount of tokens in existence
    /// @return The total supply of tokens
    function totalSupply() external view returns (uint256);

    /// @notice Returns the token balance of a specific account
    /// @param account The address to query the balance of
    /// @return The amount of tokens owned by the account
    function balanceOf(address account) external view returns (uint256);

    /// @notice Transfers tokens from the caller's account to another account
    /// @param recipient The address to transfer tokens to
    /// @param amount The number of tokens to transfer
    /// @return A boolean indicating whether the transfer was successful
    function transfer(
        address recipient,
        uint256 amount
    ) external returns (bool);

    /// @notice Returns the remaining number of tokens that a spender is allowed to spend on behalf of the token owner
    /// @param owner The address owning the tokens
    /// @param spender The address spending the tokens
    /// @return The number of tokens still available for the spender
    function allowance(
        address owner,
        address spender
    ) external view returns (uint256);

    /// @notice Sets an allowance for a spender over the caller's tokens
    /// @param spender The address authorized to spend the tokens
    /// @param amount The number of tokens the spender is allowed to use
    /// @return A boolean indicating whether the operation succeeded
    function approve(address spender, uint256 amount) external returns (bool);

    /// @notice Transfers tokens on behalf of an account to another account, based on a previously granted allowance
    /// @param sender The address of the account providing the tokens
    /// @param recipient The address of the account receiving the tokens
    /// @param amount The number of tokens to transfer
    /// @return A boolean indicating whether the operation succeeded
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);
}

/// @title Interface for Wrapped PLS Token
interface IWPLS {
    /// @notice Deposits Ether and mints WPLS tokens
    function deposit() external payable;

    /// @notice Burns WPLS tokens and withdraws Ether
    /// @param wad The amount of WPLS to burn
    function withdraw(uint wad) external;

    /// @notice Transfers WPLS tokens to another address
    /// @param recipient The address to transfer tokens to
    /// @param amount The number of tokens to transfer
    /// @return A boolean indicating whether the transfer was successful
    function transfer(
        address recipient,
        uint256 amount
    ) external returns (bool);
}

/// @title Interface for Staker
interface IStaker {
    function distributeReward(address rewardAsset, uint rewardAmount) external;
}

/// @title Fee Collector Contract
/// @notice Contract for collecting fees from individual dapps
contract MCR369Collector is Owned(msg.sender) {
    /// @notice Event emitted when rewards are distributed
    event RewardDistribution(address indexed rewardAsset, uint rewardAmount);

    /// @notice List of Reward Tokens
    address[] public rewardToken;
    /// @notice Mapping of rewards distributed for a token
    mapping(address asset => uint amount) public rewardDistributed;

    /// @notice Address of the core staker
    IStaker public immutable staker;
    /// @notice Address of wpls token
    IWPLS public immutable wpls;

    constructor(address stakerAddress, address wplsAddress) {
        staker = IStaker(stakerAddress); //mainnet-address: 0x0000000000000000000000000000000000000000
        wpls = IWPLS(wplsAddress); //mainnet-address: 0x0000000000000000000000000000000000000000
    }

    /// @notice Add a new Reward Token
    /// @param rewardAsset New Reward Token address
    function addRewardToken(address rewardAsset) external onlyOwner {
        (bool exists, ) = checkRewardToken(rewardAsset);
        require(!exists, "ALREADY EXISTS");

        rewardToken.push(rewardAsset);
    }

    /// @notice Remove an existing Reward Token
    /// @param rewardAsset Reward Token Address to remove
    function removeRewardToken(address rewardAsset) external onlyOwner {
        (bool exists, uint index) = checkRewardToken(rewardAsset);
        require(exists, "DOESN'T EXIST");

        rewardToken[index] = rewardToken[rewardToken.length - 1];
        rewardToken.pop();
    }

    function checkRewardToken(
        address rewardAsset
    ) public view returns (bool exists, uint index) {
        while (index < rewardToken.length) {
            if (rewardToken[index] == rewardAsset) {
                exists = true;
                break;
            }

            unchecked {
                ++index;
            }
        }
    }

    /// @notice Distributes all accumulated fees in supported tokens to core staker and respective split shareholders
    function distributeFees() external {
        for (uint r = 0; r < rewardToken.length; r++) {
            IERC20 token = IERC20(rewardToken[r]);
            uint balance = token.balanceOf(address(this));

            if (balance > 0) {
                token.approve(address(staker), balance);
                staker.distributeReward(address(token), balance);
                rewardDistributed[address(token)] += balance;
                emit RewardDistribution(address(token), balance);
            }
        }
    }

    /// @notice Fallback function to handle received Ether with no data
    fallback() external payable {
        wpls.deposit{value: msg.value}();
    }

    /// @notice Receive function called when Ether is sent to contract with no data
    receive() external payable {
        wpls.deposit{value: msg.value}();
    }
}