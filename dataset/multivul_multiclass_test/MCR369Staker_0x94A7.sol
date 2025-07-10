// SPDX-License-Identifier: AGPL-3.0-only
pragma solidity ^0.8.18;

/// @notice Safe ETH and ERC20 transfer library that gracefully handles missing return values.
/// @author Solmate (https://github.com/transmissions11/solmate/blob/main/src/utils/SafeTransferLib.sol)
/// @dev Use with caution! Some functions in this library knowingly create dirty bits at the destination of the free memory pointer.
/// @dev Note that none of the functions in this library check that a token has code at all! That responsibility is delegated to the caller.
library SafeTransferLib {
    /*//////////////////////////////////////////////////////////////
                             ETH OPERATIONS
    //////////////////////////////////////////////////////////////*/

    function safeTransferETH(address to, uint256 amount) internal {
        bool success;

        /// @solidity memory-safe-assembly
        assembly {
            // Transfer the ETH and store if it succeeded or not.
            success := call(gas(), to, amount, 0, 0, 0, 0)
        }

        require(success, "ETH_TRANSFER_FAILED");
    }

    /*//////////////////////////////////////////////////////////////
                            ERC20 OPERATIONS
    //////////////////////////////////////////////////////////////*/

    function safeTransferFrom(
        ERC20 token,
        address from,
        address to,
        uint256 amount
    ) internal {
        bool success;

        /// @solidity memory-safe-assembly
        assembly {
            // Get a pointer to some free memory.
            let freeMemoryPointer := mload(0x40)

            // Write the abi-encoded calldata into memory, beginning with the function selector.
            mstore(
                freeMemoryPointer,
                0x23b872dd00000000000000000000000000000000000000000000000000000000
            )
            mstore(
                add(freeMemoryPointer, 4),
                and(from, 0xffffffffffffffffffffffffffffffffffffffff)
            ) // Append and mask the "from" argument.
            mstore(
                add(freeMemoryPointer, 36),
                and(to, 0xffffffffffffffffffffffffffffffffffffffff)
            ) // Append and mask the "to" argument.
            mstore(add(freeMemoryPointer, 68), amount) // Append the "amount" argument. Masking not required as it's a full 32 byte type.

            success := and(
                // Set success to whether the call reverted, if not we check it either
                // returned exactly 1 (can't just be non-zero data), or had no return data.
                or(
                    and(eq(mload(0), 1), gt(returndatasize(), 31)),
                    iszero(returndatasize())
                ),
                // We use 100 because the length of our calldata totals up like so: 4 + 32 * 3.
                // We use 0 and 32 to copy up to 32 bytes of return data into the scratch space.
                // Counterintuitively, this call must be positioned second to the or() call in the
                // surrounding and() call or else returndatasize() will be zero during the computation.
                call(gas(), token, 0, freeMemoryPointer, 100, 0, 32)
            )
        }

        require(success, "TRANSFER_FROM_FAILED");
    }

    function safeTransfer(ERC20 token, address to, uint256 amount) internal {
        bool success;

        /// @solidity memory-safe-assembly
        assembly {
            // Get a pointer to some free memory.
            let freeMemoryPointer := mload(0x40)

            // Write the abi-encoded calldata into memory, beginning with the function selector.
            mstore(
                freeMemoryPointer,
                0xa9059cbb00000000000000000000000000000000000000000000000000000000
            )
            mstore(
                add(freeMemoryPointer, 4),
                and(to, 0xffffffffffffffffffffffffffffffffffffffff)
            ) // Append and mask the "to" argument.
            mstore(add(freeMemoryPointer, 36), amount) // Append the "amount" argument. Masking not required as it's a full 32 byte type.

            success := and(
                // Set success to whether the call reverted, if not we check it either
                // returned exactly 1 (can't just be non-zero data), or had no return data.
                or(
                    and(eq(mload(0), 1), gt(returndatasize(), 31)),
                    iszero(returndatasize())
                ),
                // We use 68 because the length of our calldata totals up like so: 4 + 32 * 2.
                // We use 0 and 32 to copy up to 32 bytes of return data into the scratch space.
                // Counterintuitively, this call must be positioned second to the or() call in the
                // surrounding and() call or else returndatasize() will be zero during the computation.
                call(gas(), token, 0, freeMemoryPointer, 68, 0, 32)
            )
        }

        require(success, "TRANSFER_FAILED");
    }

    function safeApprove(ERC20 token, address to, uint256 amount) internal {
        bool success;

        /// @solidity memory-safe-assembly
        assembly {
            // Get a pointer to some free memory.
            let freeMemoryPointer := mload(0x40)

            // Write the abi-encoded calldata into memory, beginning with the function selector.
            mstore(
                freeMemoryPointer,
                0x095ea7b300000000000000000000000000000000000000000000000000000000
            )
            mstore(
                add(freeMemoryPointer, 4),
                and(to, 0xffffffffffffffffffffffffffffffffffffffff)
            ) // Append and mask the "to" argument.
            mstore(add(freeMemoryPointer, 36), amount) // Append the "amount" argument. Masking not required as it's a full 32 byte type.

            success := and(
                // Set success to whether the call reverted, if not we check it either
                // returned exactly 1 (can't just be non-zero data), or had no return data.
                or(
                    and(eq(mload(0), 1), gt(returndatasize(), 31)),
                    iszero(returndatasize())
                ),
                // We use 68 because the length of our calldata totals up like so: 4 + 32 * 2.
                // We use 0 and 32 to copy up to 32 bytes of return data into the scratch space.
                // Counterintuitively, this call must be positioned second to the or() call in the
                // surrounding and() call or else returndatasize() will be zero during the computation.
                call(gas(), token, 0, freeMemoryPointer, 68, 0, 32)
            )
        }

        require(success, "APPROVE_FAILED");
    }
}

/// @notice Modern and gas efficient ERC20 + EIP-2612 implementation.
/// @author Solmate (https://github.com/transmissions11/solmate/blob/main/src/tokens/ERC20.sol)
/// @author Modified from Uniswap (https://github.com/Uniswap/uniswap-v2-core/blob/master/contracts/UniswapV2ERC20.sol)
/// @dev Do not manually set balances without updating totalSupply, as the sum of all user balances must not exceed it.
abstract contract ERC20 {
    /*//////////////////////////////////////////////////////////////
                                 EVENTS
    //////////////////////////////////////////////////////////////*/

    event Transfer(address indexed from, address indexed to, uint256 amount);

    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 amount
    );

    /*//////////////////////////////////////////////////////////////
                            METADATA STORAGE
    //////////////////////////////////////////////////////////////*/

    string public name;

    string public symbol;

    uint8 public immutable decimals;

    /*//////////////////////////////////////////////////////////////
                              ERC20 STORAGE
    //////////////////////////////////////////////////////////////*/

    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;

    mapping(address => mapping(address => uint256)) public allowance;

    /*//////////////////////////////////////////////////////////////
                            EIP-2612 STORAGE
    //////////////////////////////////////////////////////////////*/

    uint256 internal immutable INITIAL_CHAIN_ID;

    bytes32 internal immutable INITIAL_DOMAIN_SEPARATOR;

    mapping(address => uint256) public nonces;

    /*//////////////////////////////////////////////////////////////
                               CONSTRUCTOR
    //////////////////////////////////////////////////////////////*/

    constructor(string memory _name, string memory _symbol, uint8 _decimals) {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;

        INITIAL_CHAIN_ID = block.chainid;
        INITIAL_DOMAIN_SEPARATOR = computeDomainSeparator();
    }

    /*//////////////////////////////////////////////////////////////
                               ERC20 LOGIC
    //////////////////////////////////////////////////////////////*/

    function approve(
        address spender,
        uint256 amount
    ) public virtual returns (bool) {
        allowance[msg.sender][spender] = amount;

        emit Approval(msg.sender, spender, amount);

        return true;
    }

    function transfer(
        address to,
        uint256 amount
    ) public virtual returns (bool) {
        balanceOf[msg.sender] -= amount;

        // Cannot overflow because the sum of all user
        // balances can't exceed the max uint256 value.
        unchecked {
            balanceOf[to] += amount;
        }

        emit Transfer(msg.sender, to, amount);

        return true;
    }

    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) public virtual returns (bool) {
        uint256 allowed = allowance[from][msg.sender]; // Saves gas for limited approvals.

        if (allowed != type(uint256).max)
            allowance[from][msg.sender] = allowed - amount;

        balanceOf[from] -= amount;

        // Cannot overflow because the sum of all user
        // balances can't exceed the max uint256 value.
        unchecked {
            balanceOf[to] += amount;
        }

        emit Transfer(from, to, amount);

        return true;
    }

    /*//////////////////////////////////////////////////////////////
                             EIP-2612 LOGIC
    //////////////////////////////////////////////////////////////*/

    function permit(
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) public virtual {
        require(deadline >= block.timestamp, "PERMIT_DEADLINE_EXPIRED");

        // Unchecked because the only math done is incrementing
        // the owner's nonce which cannot realistically overflow.
        unchecked {
            address recoveredAddress = ecrecover(
                keccak256(
                    abi.encodePacked(
                        "\x19\x01",
                        DOMAIN_SEPARATOR(),
                        keccak256(
                            abi.encode(
                                keccak256(
                                    "Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)"
                                ),
                                owner,
                                spender,
                                value,
                                nonces[owner]++,
                                deadline
                            )
                        )
                    )
                ),
                v,
                r,
                s
            );

            require(
                recoveredAddress != address(0) && recoveredAddress == owner,
                "INVALID_SIGNER"
            );

            allowance[recoveredAddress][spender] = value;
        }

        emit Approval(owner, spender, value);
    }

    function DOMAIN_SEPARATOR() public view virtual returns (bytes32) {
        return
            block.chainid == INITIAL_CHAIN_ID
                ? INITIAL_DOMAIN_SEPARATOR
                : computeDomainSeparator();
    }

    function computeDomainSeparator() internal view virtual returns (bytes32) {
        return
            keccak256(
                abi.encode(
                    keccak256(
                        "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
                    ),
                    keccak256(bytes(name)),
                    keccak256("1"),
                    block.chainid,
                    address(this)
                )
            );
    }

    /*//////////////////////////////////////////////////////////////
                        INTERNAL MINT/BURN LOGIC
    //////////////////////////////////////////////////////////////*/

    function _mint(address to, uint256 amount) internal virtual {
        totalSupply += amount;

        // Cannot overflow because the sum of all user
        // balances can't exceed the max uint256 value.
        unchecked {
            balanceOf[to] += amount;
        }

        emit Transfer(address(0), to, amount);
    }

    function _burn(address from, uint256 amount) internal virtual {
        balanceOf[from] -= amount;

        // Cannot underflow because a user's balance
        // will never be larger than the total supply.
        unchecked {
            totalSupply -= amount;
        }

        emit Transfer(from, address(0), amount);
    }
}

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

/// @notice Gas optimized reentrancy protection for smart contracts.
/// @author Solmate (https://github.com/transmissions11/solmate/blob/main/src/utils/ReentrancyGuard.sol)
/// @author Modified from OpenZeppelin (https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/security/ReentrancyGuard.sol)
abstract contract ReentrancyGuard {
    uint256 private locked = 1;

    modifier nonReentrant() virtual {
        require(locked == 1, "REENTRANCY");

        locked = 2;

        _;

        locked = 1;
    }
}

contract MCR369Staker is Owned(msg.sender), ReentrancyGuard {
    event Stake(address indexed staker, uint value);
    event Unstake(address indexed staker, uint value);
    event RewardTokenAdded(address indexed rewardAsset);
    event RewardDistribution(address indexed rewardAsset, uint rewardAmount);
    event RewardClaimed(
        address indexed provider,
        address rewardAsset,
        uint rewardAmount
    );

    address public immutable mcr369;
    address[] internal rewardTokens;
    mapping(address rewardAsset => bool whitelisted) public isRewardToken;

    mapping(address staker => uint balance) public stakeBalance;
    mapping(address staker => uint blockNumber) public stakeBlock;
    uint public totalMCR369Staked;

    // REWARD TOKEN ACCOUNTING
    uint internal constant _BASE = 1e21;
    mapping(address asset => uint amount) internal _rewardDistributed; // Supply of tokens distributed as rewards
    mapping(address asset => uint index) internal _rewardIndex; // Index of tokens pools for rewarding
    mapping(address staker => mapping(address asset => uint amount))
        internal _rewardIndexOf; // Index of reward balance of a staker
    mapping(address staker => mapping(address asset => uint amount))
        internal _claimable; // Claimable reward balance of a staker
    mapping(address staker => mapping(address asset => uint amount))
        internal _earned; // Claimed reward balance of a staker

    /// @notice Contract constructor
    /// @param _mcr369 mcr369 address
    constructor(address _mcr369) {
        mcr369 = _mcr369;
    }

    /*//////////////////////////////////////////////////////////////
                            ADD REWARD TOKEN
    //////////////////////////////////////////////////////////////*/

    /// @notice Only owner can add new Reward Token
    /// @param rewardAsset Address of new Reward Token
    function addRewardToken(address rewardAsset) external onlyOwner {
        require(!isRewardToken[rewardAsset], "INVALID_REWARD_ASSET");

        rewardTokens.push(rewardAsset);
        isRewardToken[rewardAsset] = true;

        emit RewardTokenAdded(rewardAsset);
    }

    /*//////////////////////////////////////////////////////////////
                            STAKING FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /// @notice Stake `value` tokens
    /// @param value Amount to stake
    function stake(uint value) external nonReentrant {
        require(value != 0, "CAN'T_STAKE_ZERO"); // dev: need non-zero value
        SafeTransferLib.safeTransferFrom(
            ERC20(mcr369),
            msg.sender,
            address(this),
            value
        );

        _updateAllRewards(msg.sender);

        totalMCR369Staked += value;
        stakeBalance[msg.sender] += value;
        stakeBlock[msg.sender] = block.number;

        emit Stake(msg.sender, value);
    }

    /// @notice Unstake `value` tokens
    /// @param value Amount to be unstaked
    function unstake(uint value) external nonReentrant {
        require(block.number > stakeBlock[msg.sender] + 10, "WAIT_10_BLOCKS");

        _updateAllRewards(msg.sender);

        totalMCR369Staked -= value;
        stakeBalance[msg.sender] -= value;

        SafeTransferLib.safeTransfer(ERC20(mcr369), msg.sender, value);

        emit Unstake(msg.sender, value);
    }

    /*//////////////////////////////////////////////////////////////
                        DISCRETE STAKING REWARDS
    //////////////////////////////////////////////////////////////*/

    /// @notice Distributes rewards to staking pool
    /// @param rewardAsset Asset being distributed to staking pool
    /// @param rewardAmount Amount of asset being distributed
    function distributeReward(
        address rewardAsset,
        uint rewardAmount
    ) external nonReentrant {
        require(isRewardToken[rewardAsset], "INVALID_REWARD_ASSET");

        SafeTransferLib.safeTransferFrom(
            ERC20(rewardAsset),
            msg.sender,
            address(this),
            rewardAmount
        );

        _rewardDistributed[rewardAsset] += rewardAmount;
        _rewardIndex[rewardAsset] += (rewardAmount * _BASE) / totalMCR369Staked;

        emit RewardDistribution(rewardAsset, rewardAmount);
    }

    /// @notice Shows historic earning of a staker per asset
    /// @param staker Staker address
    /// @param rewardAsset Asset rewarded in
    function earned(
        address staker,
        address rewardAsset
    ) external view returns (uint) {
        return _earned[staker][rewardAsset];
    }

    /// @notice Calculates latest rewards accumulated
    /// @param staker Staker address
    /// @param rewardAsset Asset rewarded in
    function _calculateRewards(
        address staker,
        address rewardAsset
    ) internal view returns (uint) {
        return
            (stakeBalance[staker] *
                (_rewardIndex[rewardAsset] -
                    _rewardIndexOf[staker][rewardAsset])) / _BASE;
    }

    /// @notice Shows rewards available to claim
    /// @param staker Staker address
    /// @param rewardAsset Asset rewarded in
    function claimable(
        address staker,
        address rewardAsset
    ) public view returns (uint) {
        return
            _claimable[staker][rewardAsset] +
            _calculateRewards(staker, rewardAsset);
    }

    /// @notice Updates rewards in a particular asset
    /// @param staker Staker address
    /// @param rewardAsset Asset rewarded in
    function _updateRewards(address staker, address rewardAsset) internal {
        _claimable[staker][rewardAsset] += _calculateRewards(
            staker,
            rewardAsset
        );
        _rewardIndexOf[staker][rewardAsset] = _rewardIndex[rewardAsset];
    }

    /// @notice Updates rewards in all assets
    /// @param staker Staker address
    function _updateAllRewards(address staker) internal {
        uint len = rewardTokens.length;
        for (uint i = 0; i < len; ) {
            _updateRewards(staker, rewardTokens[i]);
            unchecked {
                ++i;
            }
        }
    }

    /// @notice Claim rewards in a particular asset
    /// @param staker Staker Address
    /// @param rewardAsset Asset rewarded in
    function _claimReward(address staker, address rewardAsset) internal {
        _updateRewards(staker, rewardAsset);

        uint rewardAmount = _claimable[staker][rewardAsset];
        if (rewardAmount != 0) {
            _claimable[staker][rewardAsset] = 0;
            _earned[staker][rewardAsset] += rewardAmount;
            SafeTransferLib.safeTransfer(
                ERC20(rewardAsset),
                staker,
                rewardAmount
            );
        }

        emit RewardClaimed(staker, rewardAsset, rewardAmount);
    }

    /// @notice Claim rewards in a particular asset
    /// @param rewardAsset Asset rewarded in
    function claimReward(address rewardAsset) external nonReentrant {
        _claimReward(msg.sender, rewardAsset);
    }

    /// @notice Collects all rewards
    function claimAllRewards() external nonReentrant {
        uint len = rewardTokens.length;
        for (uint i = 0; i < len; ) {
            _claimReward(msg.sender, rewardTokens[i]);
            unchecked {
                ++i;
            }
        }
    }

    /// @notice All rewards distributed in a particular asset
    /// @param rewardAsset Asset rewarded to pool
    function totalRewardsDistributed(
        address rewardAsset
    ) external view returns (uint) {
        return _rewardDistributed[rewardAsset];
    }

    /*//////////////////////////////////////////////////////////////
                                 HELPER
    //////////////////////////////////////////////////////////////*/

    function rewardTokenCount() public view returns (uint) {
        return rewardTokens.length;
    }

    function rewardTokenList() public view returns (address[] memory) {
        return rewardTokens;
    }
}