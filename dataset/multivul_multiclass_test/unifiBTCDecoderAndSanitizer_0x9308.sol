// SPDX-License-Identifier: MIT
pragma solidity =0.8.21;

// src/interfaces/DecoderCustomTypes.sol

contract DecoderCustomTypes {
    // ========================================= BALANCER =========================================
    struct JoinPoolRequest {
        address[] assets;
        uint256[] maxAmountsIn;
        bytes userData;
        bool fromInternalBalance;
    }

    struct ExitPoolRequest {
        address[] assets;
        uint256[] minAmountsOut;
        bytes userData;
        bool toInternalBalance;
    }

    enum SwapKind {
        GIVEN_IN,
        GIVEN_OUT
    }

    struct SingleSwap {
        bytes32 poolId;
        SwapKind kind;
        address assetIn;
        address assetOut;
        uint256 amount;
        bytes userData;
    }

    struct FundManagement {
        address sender;
        bool fromInternalBalance;
        address recipient;
        bool toInternalBalance;
    }

    // ========================================= VELODROME =========================================
    struct VelodromeMintParams {
        address token0;
        address token1;
        int24 tickSpacing;
        int24 tickLower;
        int24 tickUpper;
        uint256 amount0Desired;
        uint256 amount1Desired;
        uint256 amount0Min;
        uint256 amount1Min;
        address recipient;
        uint256 deadline;
        uint160 sqrtPriceX96;
    }

    // ========================================= UNISWAP V3 =========================================

    struct MintParams {
        address token0;
        address token1;
        uint24 fee;
        int24 tickLower;
        int24 tickUpper;
        uint256 amount0Desired;
        uint256 amount1Desired;
        uint256 amount0Min;
        uint256 amount1Min;
        address recipient;
        uint256 deadline;
    }

    struct IncreaseLiquidityParams {
        uint256 tokenId;
        uint256 amount0Desired;
        uint256 amount1Desired;
        uint256 amount0Min;
        uint256 amount1Min;
        uint256 deadline;
    }

    struct DecreaseLiquidityParams {
        uint256 tokenId;
        uint128 liquidity;
        uint256 amount0Min;
        uint256 amount1Min;
        uint256 deadline;
    }

    struct CollectParams {
        uint256 tokenId;
        address recipient;
        uint128 amount0Max;
        uint128 amount1Max;
    }

    struct ExactInputParams {
        bytes path;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
    }

    // ========================================= MORPHO BLUE =========================================

    struct MarketParams {
        address loanToken;
        address collateralToken;
        address oracle;
        address irm;
        uint256 lltv;
    }

    // ========================================= 1INCH =========================================

    struct SwapDescription {
        address srcToken;
        address dstToken;
        address payable srcReceiver;
        address payable dstReceiver;
        uint256 amount;
        uint256 minReturnAmount;
        uint256 flags;
    }

    // ========================================= KITTENSWAP =========================================
    struct route {
        address from;
        address to;
        bool stable;
    }

    // ========================================= PENDLE =========================================
    struct TokenInput {
        // TOKEN DATA
        address tokenIn;
        uint256 netTokenIn;
        address tokenMintSy;
        // AGGREGATOR DATA
        address pendleSwap;
        SwapData swapData;
    }

    struct TokenOutput {
        // TOKEN DATA
        address tokenOut;
        uint256 minTokenOut;
        address tokenRedeemSy;
        // AGGREGATOR DATA
        address pendleSwap;
        SwapData swapData;
    }

    struct ApproxParams {
        uint256 guessMin;
        uint256 guessMax;
        uint256 guessOffchain; // pass 0 in to skip this variable
        uint256 maxIteration; // every iteration, the diff between guessMin and guessMax will be divided by 2
        uint256 eps; // the max eps between the returned result & the correct result, base 1e18. Normally this number
            // will be set
            // to 1e15 (1e18/1000 = 0.1%)
    }

    struct SwapData {
        SwapType swapType;
        address extRouter;
        bytes extCalldata;
        bool needScale;
    }

    enum OrderType {
        SY_FOR_PT,
        PT_FOR_SY,
        SY_FOR_YT,
        YT_FOR_SY
    }

    struct Order {
        uint256 salt;
        uint256 expiry;
        uint256 nonce;
        OrderType orderType;
        address token;
        address YT;
        address maker;
        address receiver;
        uint256 makingAmount;
        uint256 lnImpliedRate;
        uint256 failSafeRate;
        bytes permit;
    }

    struct FillOrderParams {
        Order order;
        bytes signature;
        uint256 makingAmount;
    }

    struct LimitOrderData {
        address limitRouter;
        uint256 epsSkipMarket; // only used for swap operations, will be ignored otherwise
        FillOrderParams[] normalFills;
        FillOrderParams[] flashFills;
        bytes optData;
    }

    enum SwapType {
        NONE,
        KYBERSWAP,
        ONE_INCH,
        // ETH_WETH not used in Aggregator
        ETH_WETH
    }

    // ========================================= NUCLEUS =========================================

    struct AtomicRequestUCP {
        uint64 deadline; // Timestamp when request expires
        uint96 atomicPrice; // User's limit price in want asset decimals
        uint96 offerAmount; // Amount of offer asset to sell
        address recipient; // Address to receive want assets
    }

    // ========================================= SUPERBRIDGE =========================================
    /// @notice Struct representing a withdrawal transaction.
    /// @custom:field nonce    Nonce of the withdrawal transaction
    /// @custom:field sender   Address of the sender of the transaction.
    /// @custom:field target   Address of the recipient of the transaction.
    /// @custom:field value    Value to send to the recipient.
    /// @custom:field gasLimit Gas limit of the transaction.
    /// @custom:field data     Data of the transaction.
    struct WithdrawalTransaction {
        uint256 nonce;
        address sender;
        address target;
        uint256 value;
        uint256 gasLimit;
        bytes data;
    }

    /// @notice Struct representing the elements that are hashed together to generate an output root
    ///         which itself represents a snapshot of the L2 state.
    /// @custom:field version                  Version of the output root.
    /// @custom:field stateRoot                Root of the state trie at the block of this output.
    /// @custom:field messagePasserStorageRoot Root of the message passer storage trie.
    /// @custom:field latestBlockhash          Hash of the block this output was generated from.
    struct OutputRootProof {
        bytes32 version;
        bytes32 stateRoot;
        bytes32 messagePasserStorageRoot;
        bytes32 latestBlockhash;
    }

    // ========================================= EIGEN LAYER =========================================

    struct QueuedWithdrawalParams {
        // Array of strategies that the QueuedWithdrawal contains
        address[] strategies;
        // Array containing the amount of shares in each Strategy in the `strategies` array
        uint256[] shares;
        // The address of the withdrawer
        address withdrawer;
    }

    struct Withdrawal {
        // The address that originated the Withdrawal
        address staker;
        // The address that the staker was delegated to at the time that the Withdrawal was created
        address delegatedTo;
        // The address that can complete the Withdrawal + will receive funds when completing the withdrawal
        address withdrawer;
        // Nonce used to guarantee that otherwise identical withdrawals have unique hashes
        uint256 nonce;
        // Block number when the Withdrawal was created
        uint32 startBlock;
        // Array of strategies that the Withdrawal contains
        address[] strategies;
        // Array containing the amount of shares in each Strategy in the `strategies` array
        uint256[] shares;
    }

    // ========================================= Sentiment =========================================

    /// @title Operation
    /// @notice Operation type definitions that can be applied to a position
    /// @dev Every operation except NewPosition requires that the caller must be an authz caller or owner
    enum Operation {
        NewPosition, // create2 a new position with a given type, no auth needed
        // the following operations require msg.sender to be authorized
        Exec, // execute arbitrary calldata on a position
        Deposit, // Add collateral to a given position
        Transfer, // transfer assets from the position to a external address
        Approve, // allow a spender to transfer assets from a position
        Repay, // decrease position debt
        Borrow, // increase position debt
        AddToken, // upsert collateral asset to position storage
        RemoveToken // remove collateral asset from position storage

    }

    /// @title Action
    /// @notice Generic data struct to create a common data container for all operation types
    /// @dev target and data are interpreted in different ways based on the operation type
    struct Action {
        // operation type
        Operation op;
        // dynamic bytes data, interpreted differently across operation types
        bytes data;
    }
}

// src/interfaces/RawDataDecoderAndSanitizerInterfaces.sol

// Swell
interface INonFungiblePositionManager {
    struct Position {
        // the nonce for permits
        uint96 nonce;
        // the address that is approved for spending this token
        address operator;
        // the ID of the pool with which this token is connected
        uint80 poolId;
        // the tick range of the position
        int24 tickLower;
        int24 tickUpper;
        // the liquidity of the position
        uint128 liquidity;
        // the fee growth of the aggregate position as of the last action on the individual position
        uint256 feeGrowthInside0LastX128;
        uint256 feeGrowthInside1LastX128;
        // how many uncollected tokens are owed to the position, as of the last computation
        uint128 tokensOwed0;
        uint128 tokensOwed1;
    }

    function ownerOf(uint256 tokenId) external view returns (address);
    function positions(uint256 tokenId)
        external
        view
        returns (
            uint96 nonce,
            address operator,
            address token0,
            address token1,
            uint24 fee,
            int24 tickLower,
            int24 tickUpper,
            uint128 liquidity,
            uint256 feeGrowthInside0LastX128,
            uint256 feeGrowthInside1LastX128,
            uint128 tokensOwed0,
            uint128 tokensOwed1
        );
}

// src/base/DecodersAndSanitizers/BaseDecoderAndSanitizer.sol

// solhint-disable-next-line no-unused-import

contract BaseDecoderAndSanitizer {
    //============================== IMMUTABLES ===============================

    /**
     * @notice The BoringVault contract address.
     */
    address internal immutable boringVault;

    error BaseDecoderAndSanitizer__FunctionNotImplemented(bytes _calldata);

    constructor(address _boringVault) {
        boringVault = _boringVault;
    }

    // @desc The spender address to approve
    // @tag spender:address
    function approve(address spender, uint256) external pure returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(spender);
    }

    function acceptOwnership() external pure returns (bytes memory addressesFound) {
        // Nothing to decode
    }

    // @desc The new owner address
    // @tag newOwner:address
    function transferOwnership(address newOwner) external pure returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(newOwner);
    }

    fallback() external {
        revert BaseDecoderAndSanitizer__FunctionNotImplemented(msg.data);
    }
}

// src/base/DecodersAndSanitizers/Protocols/BalancerV2DecoderAndSanitizer.sol

abstract contract BalancerV2DecoderAndSanitizer is BaseDecoderAndSanitizer {
    //============================== ERRORS ===============================

    error BalancerV2DecoderAndSanitizer__SingleSwapUserDataLengthNonZero();
    error BalancerV2DecoderAndSanitizer__InternalBalancesNotSupported();

    //============================== BALANCER V2 ===============================

    // @desc Flash loan from the Balancer V2 protocol
    // @tag recipient:address:the address of the recipient
    // @tag tokens:bytes:packed bytes of every token in each address[] tokens input.
    function flashLoan(
        address recipient,
        address[] calldata tokens,
        uint256[] calldata,
        bytes calldata
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(recipient);
        for (uint256 i; i < tokens.length; ++i) {
            addressesFound = abi.encodePacked(addressesFound, tokens[i]);
        }
    }

    // @desc Swap tokens in the Balancer V2 protocol. Reverts if userData is not empty or if internal balances are used
    // @tag pool:address:the address of the pool
    // @tag assetIn:address:the address of the input asset
    // @tag assetOut:address:the address of the output asset
    // @tag sender:address:the address of the sender
    // @tag recipient:address:the address of the recipient
    function swap(
        DecoderCustomTypes.SingleSwap calldata singleSwap,
        DecoderCustomTypes.FundManagement calldata funds,
        uint256,
        uint256
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        // Sanitize raw data
        if (singleSwap.userData.length > 0) revert BalancerV2DecoderAndSanitizer__SingleSwapUserDataLengthNonZero();
        if (funds.fromInternalBalance) revert BalancerV2DecoderAndSanitizer__InternalBalancesNotSupported();
        if (funds.toInternalBalance) revert BalancerV2DecoderAndSanitizer__InternalBalancesNotSupported();

        // Return addresses found
        addressesFound = abi.encodePacked(
            _getPoolAddressFromPoolId(singleSwap.poolId),
            singleSwap.assetIn,
            singleSwap.assetOut,
            funds.sender,
            funds.recipient
        );
    }

    // @desc Join a pool in the Balancer V2 protocol, will revert if internal balances are used
    // @tag pool:address:the address of the pool
    // @tag sender:address:the address of the sender
    // @tag recipient:address:the address of the recipient
    // @tag assets:bytes:packed bytes of every address in the req.assets input.
    function joinPool(
        bytes32 poolId,
        address sender,
        address recipient,
        DecoderCustomTypes.JoinPoolRequest calldata req
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        // Sanitize raw data
        if (req.fromInternalBalance) revert BalancerV2DecoderAndSanitizer__InternalBalancesNotSupported();
        // Return addresses found
        addressesFound = abi.encodePacked(_getPoolAddressFromPoolId(poolId), sender, recipient);
        uint256 assetsLength = req.assets.length;
        for (uint256 i; i < assetsLength; ++i) {
            addressesFound = abi.encodePacked(addressesFound, req.assets[i]);
        }
    }

    // @desc Exit a pool in the Balancer V2 protocol, will revert if internal balances are used
    // @tag pool:address:the address of the pool
    // @tag sender:address:the address of the sender
    // @tag recipient:address:the address of the recipient
    // @tag assets:bytes:packed bytes of every address in the req.assets input.
    function exitPool(
        bytes32 poolId,
        address sender,
        address recipient,
        DecoderCustomTypes.ExitPoolRequest calldata req
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        // Sanitize raw data
        if (req.toInternalBalance) revert BalancerV2DecoderAndSanitizer__InternalBalancesNotSupported();
        // Return addresses found
        addressesFound = abi.encodePacked(_getPoolAddressFromPoolId(poolId), sender, recipient);
        uint256 assetsLength = req.assets.length;
        for (uint256 i; i < assetsLength; ++i) {
            addressesFound = abi.encodePacked(addressesFound, req.assets[i]);
        }
    }

    // @desc Deposit into the Balancer V2 protocol
    // @tag recipient:address:the address of the recipient
    function deposit(uint256, address recipient) external pure virtual returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(recipient);
    }

    // @desc Withdraw from the Balancer V2 protocol to msg.sender
    function withdraw(uint256) external pure virtual returns (bytes memory addressesFound) {
        // No addresses in data
        return addressesFound;
    }

    function mint(address gauge) external pure virtual returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(gauge);
    }

    // ========================================= INTERNAL HELPER FUNCTIONS =========================================

    /**
     * @notice Internal helper function that converts poolIds to pool addresses.
     */
    function _getPoolAddressFromPoolId(bytes32 poolId) internal pure returns (address) {
        return address(uint160(uint256(poolId >> 96)));
    }
}

// src/base/DecodersAndSanitizers/Protocols/CurveDecoderAndSanitizer.sol

abstract contract CurveDecoderAndSanitizer is BaseDecoderAndSanitizer {
    //============================== CURVE ===============================

    // @desc exchange on curve
    function exchange(int128, int128, uint256, uint256) external pure virtual returns (bytes memory addressesFound) {
        // Nothing to sanitize or return
        return addressesFound;
    }

    // @desc add liquidity on curve
    function add_liquidity(uint256[] calldata, uint256) external pure virtual returns (bytes memory addressesFound) {
        // Nothing to sanitize or return
        return addressesFound;
    }

    // @desc remove liquidity on curve
    // @tag user:address:the address of the user receiving rewards
    function remove_liquidity(
        uint256,
        uint256[] calldata
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        // Nothing to sanitize or return
        return addressesFound;
    }

    function deposit(uint256, address receiver) external view virtual returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(receiver);
    }

    function withdraw(uint256) external pure virtual returns (bytes memory addressesFound) {
        // Nothing to sanitize or return
        return addressesFound;
    }

    // @desc claim rewards on curve
    // @tag user:address:the address of the user receiving rewards
    function claim_rewards(address _addr) external pure virtual returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(_addr);
    }
}

// src/base/DecodersAndSanitizers/Protocols/ERC4626DecoderAndSanitizer.sol

abstract contract ERC4626DecoderAndSanitizer is BaseDecoderAndSanitizer {
    //============================== ERC4626 ===============================

    // @desc deposit into the ERC4626 vault
    // @tag receiver:address:the address of the receiver of the vault tokens
    function deposit(uint256, address receiver) external view virtual returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(receiver);
    }

    // @desc mint tokens from the ERC4626 vault
    // @tag receiver:address:the address of the receiver of the vault tokens
    function mint(uint256, address receiver) external pure virtual returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(receiver);
    }

    // @desc withdraw tokens from the ERC4626 vault
    // @tag receiver:address:the address of the receiver of the vault tokens
    // @tag owner:address:the address of the owner of the vault tokens
    function withdraw(
        uint256,
        address receiver,
        address owner
    )
        external
        view
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(receiver, owner);
    }

    // @desc redeem tokens from the ERC4626 vault
    // @tag receiver:address:the address of the receiver of the vault tokens
    // @tag owner:address:the address of the owner of the vault tokens
    function redeem(
        uint256,
        address receiver,
        address owner
    )
        external
        view
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(receiver, owner);
    }
}

// src/base/DecodersAndSanitizers/Protocols/MasterChefV3DecoderAndSanitizer.sol

abstract contract MasterChefV3DecoderAndSanitizer is BaseDecoderAndSanitizer {
    // @desc harvest rewards from staked LP positions
    // @tag to:address:receiver of the harvest tokens
    function harvest(uint256, address _to) external pure virtual returns (bytes memory addressesFound) {
        return abi.encodePacked(_to);
    }

    // @desc withdraw staked LP positions
    // @tag to:address:receiver of the withdrawn LP NFT
    function withdraw(uint256, address _to) external pure virtual returns (bytes memory addressesFound) {
        return abi.encodePacked(_to);
    }
}

// src/base/DecodersAndSanitizers/Protocols/NativeWrapperDecoderAndSanitizer.sol

abstract contract NativeWrapperDecoderAndSanitizer is BaseDecoderAndSanitizer {
    //============================== ETHERFI ===============================

    // @desc deposit native token for wrapped token
    function deposit() external pure virtual returns (bytes memory addressesFound) {
        // Nothing to sanitize or return
        return addressesFound;
    }

    // @desc withdraw wrapped token for native token
    function withdraw(uint256) external pure virtual returns (bytes memory addressesFound) {
        // Nothing to sanitize or return
        return addressesFound;
    }
}

// src/base/DecodersAndSanitizers/Protocols/PendleRouterDecoderAndSanitizer.sol

abstract contract PendleRouterDecoderAndSanitizer is BaseDecoderAndSanitizer {
    //============================== ERRORS ===============================

    error PendleRouterDecoderAndSanitizer__AggregatorSwapsNotPermitted();
    error PendleRouterDecoderAndSanitizer__LimitOrderSwapsNotPermitted();

    //============================== PENDLEROUTER ===============================

    // @desc Function to mint Pendle Sy using some token, will revert if using aggregator swaps
    // @tag user:address:The user to mint to
    // @tag sy:address:The sy token to mint
    // @tag input:address:The input token to mint from
    function mintSyFromToken(
        address user,
        address sy,
        uint256,
        DecoderCustomTypes.TokenInput calldata input
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        if (
            input.swapData.swapType != DecoderCustomTypes.SwapType.NONE || input.swapData.extRouter != address(0)
                || input.pendleSwap != address(0) || input.tokenIn != input.tokenMintSy
        ) revert PendleRouterDecoderAndSanitizer__AggregatorSwapsNotPermitted();

        addressesFound = abi.encodePacked(user, sy, input.tokenIn);
    }

    // @desc Function to mint Pendle Py using the Sy
    // @tag user:address:The user to mint to
    // @tag yt:address:The yt token to mint
    function mintPyFromSy(
        address user,
        address yt,
        uint256,
        uint256
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(user, yt);
    }

    // @desc Function to swap exact Pendle Pt for Pendle Yt
    // @tag user:address:The user to swap from
    // @tag market:address:The pendle market address
    function swapExactPtForYt(
        address user,
        address market,
        uint256,
        uint256,
        DecoderCustomTypes.ApproxParams calldata
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(user, market);
    }

    // @desc Function to withdraw from Pendle PT tokens, does not support limit orders or aggregator swaps.
    // @param receiver:address
    // @param market:address
    // @param tokenOut:address
    function swapExactPtForToken(
        address receiver,
        address market,
        uint256 minPtOut,
        DecoderCustomTypes.TokenOutput calldata output,
        DecoderCustomTypes.LimitOrderData calldata limit
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        if (limit.limitRouter != address(0)) {
            revert PendleRouterDecoderAndSanitizer__LimitOrderSwapsNotPermitted();
        }

        if (
            output.swapData.swapType != DecoderCustomTypes.SwapType.NONE || output.swapData.extRouter != address(0)
                || output.pendleSwap != address(0) || output.tokenOut != output.tokenRedeemSy
        ) revert PendleRouterDecoderAndSanitizer__AggregatorSwapsNotPermitted();

        addressesFound = abi.encodePacked(receiver, market, output.tokenOut);
    }

    // @desc Function to swap exact Pendle Yt for Pendle Pt
    // @tag user:address:The user to swap from
    // @tag market:address:The pendle market address
    function swapExactYtForPt(
        address user,
        address market,
        uint256,
        uint256,
        DecoderCustomTypes.ApproxParams calldata
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(user, market);
    }

    // @desc Function to add Pendle liquidity with Sy and Pt
    // @tag user:address:The user to add liquidity from
    // @tag market:address:The pendle market address
    function addLiquidityDualSyAndPt(
        address user,
        address market,
        uint256,
        uint256,
        uint256
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(user, market);
    }

    // @desc Function to remove Pendle liquidity to Sy and Pt
    // @tag user:address:The user to remove liquidity from
    // @tag market:address:The pendle market address
    function removeLiquidityDualSyAndPt(
        address user,
        address market,
        uint256,
        uint256,
        uint256
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(user, market);
    }

    // @desc Function to redeem Pendle Py to Sy
    // @tag user:address:The user to redeem from
    // @tag yt:address:The yt token to redeem
    function redeemPyToSy(
        address user,
        address yt,
        uint256,
        uint256
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(user, yt);
    }

    // @desc Function to redeem Pendle Sy to some token, will revert if using aggregator swaps
    // @tag user:address:The user to redeem from
    // @tag sy:address:The sy token to redeem
    // @tag output:address:The token to redeem to
    function redeemSyToToken(
        address user,
        address sy,
        uint256,
        DecoderCustomTypes.TokenOutput calldata output
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        if (
            output.swapData.swapType != DecoderCustomTypes.SwapType.NONE || output.swapData.extRouter != address(0)
                || output.pendleSwap != address(0) || output.tokenOut != output.tokenRedeemSy
        ) revert PendleRouterDecoderAndSanitizer__AggregatorSwapsNotPermitted();

        addressesFound = abi.encodePacked(user, sy, output.tokenOut);
    }

    // @desc Function to swap exact token for Pendle Pt, will revert if using aggregator swaps or limit orders
    // @tag receiver:address:The receiver of the Pendle Pt
    // @tag market:address:The pendle market address
    // @tag input:address:The token to swap from
    function swapExactTokenForPt(
        address receiver,
        address market,
        uint256 minPtOut,
        DecoderCustomTypes.ApproxParams calldata guessPtOut,
        DecoderCustomTypes.TokenInput calldata input,
        DecoderCustomTypes.LimitOrderData calldata limit
    )
        external
        pure
        virtual
        returns (bytes memory addressFound)
    {
        if (
            input.swapData.swapType != DecoderCustomTypes.SwapType.NONE || input.swapData.extRouter != address(0)
                || input.pendleSwap != address(0) || input.tokenIn != input.tokenMintSy
        ) revert PendleRouterDecoderAndSanitizer__AggregatorSwapsNotPermitted();

        if (limit.limitRouter != address(0)) {
            revert PendleRouterDecoderAndSanitizer__LimitOrderSwapsNotPermitted();
        }

        addressFound = abi.encodePacked(receiver, market, input.tokenIn);
    }

    // @desc function to claim PENDLE token rewards and interest from LPing
    // @tag packedArgs:bytes:packed all sys,yts, and markets in order
    function redeemDueInterestAndRewards(
        address user,
        address[] calldata sys,
        address[] calldata yts,
        address[] calldata markets
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(user);
        uint256 sysLength = sys.length;
        for (uint256 i; i < sysLength; ++i) {
            addressesFound = abi.encodePacked(addressesFound, sys[i]);
        }
        uint256 ytsLength = yts.length;
        for (uint256 i; i < ytsLength; ++i) {
            addressesFound = abi.encodePacked(addressesFound, yts[i]);
        }
        uint256 marketsLength = markets.length;
        for (uint256 i; i < marketsLength; ++i) {
            addressesFound = abi.encodePacked(addressesFound, markets[i]);
        }
    }

    // @desc function to add liquidity with a single token and keep the yt, will revert if using aggregator swaps
    // @tag receiver:address:The receiver of the Pendle Yt and lp
    // @tag market:address:The pendle market address
    // @tag input:address:The token to add liquidity from
    function addLiquiditySingleTokenKeepYt(
        address receiver,
        address market,
        uint256 minLpOut,
        uint256 minYtOut,
        DecoderCustomTypes.TokenInput calldata input
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        if (
            input.swapData.swapType != DecoderCustomTypes.SwapType.NONE || input.swapData.extRouter != address(0)
                || input.pendleSwap != address(0) || input.tokenIn != input.tokenMintSy
        ) revert PendleRouterDecoderAndSanitizer__AggregatorSwapsNotPermitted();

        addressesFound = abi.encodePacked(receiver, market, input.tokenIn);
    }

    // @desc Function to add liquidity with a single token, does not keep the yt, will revert if using aggregator swaps
    // or limit orders
    // @tag receiver:address:The receiver of the Pendle Yt and lp
    // @tag market:address:The pendle market address
    // @tag input:address:The token to add liquidity from
    function addLiquiditySingleToken(
        address receiver,
        address market,
        uint256 minLpOut,
        DecoderCustomTypes.ApproxParams calldata guessPtReceivedFromSy,
        DecoderCustomTypes.TokenInput calldata input,
        DecoderCustomTypes.LimitOrderData calldata limit
    )
        external
        pure
        returns (bytes memory addressesFound)
    {
        if (
            input.swapData.swapType != DecoderCustomTypes.SwapType.NONE || input.swapData.extRouter != address(0)
                || input.pendleSwap != address(0) || input.tokenIn != input.tokenMintSy
        ) revert PendleRouterDecoderAndSanitizer__AggregatorSwapsNotPermitted();

        if (limit.limitRouter != address(0)) {
            revert PendleRouterDecoderAndSanitizer__LimitOrderSwapsNotPermitted();
        }

        addressesFound = abi.encodePacked(receiver, market, input.tokenIn);
    }

    // @desc Function to remove liquidity into a single token, will revert if using aggregator swaps or limit orders
    // @tag receiver:address:The receiver of the token to remove liquidity into
    // @tag market:address:The pendle market address
    // @tag output:address:The token to receive after removing liquidity
    function removeLiquiditySingleToken(
        address receiver,
        address market,
        uint256 netLpToRemove,
        DecoderCustomTypes.TokenOutput calldata output,
        DecoderCustomTypes.LimitOrderData calldata limit
    )
        external
        pure
        virtual
        returns (bytes memory addressFound)
    {
        if (
            output.swapData.swapType != DecoderCustomTypes.SwapType.NONE || output.swapData.extRouter != address(0)
                || output.pendleSwap != address(0) || output.tokenOut != output.tokenRedeemSy
        ) revert PendleRouterDecoderAndSanitizer__AggregatorSwapsNotPermitted();

        if (limit.limitRouter != address(0)) {
            revert PendleRouterDecoderAndSanitizer__LimitOrderSwapsNotPermitted();
        }

        addressFound = abi.encodePacked(receiver, market, output.tokenOut);
    }

    function exitPostExpToToken(
        address receiver,
        address market,
        uint256 netPtIn,
        uint256 netLpIn,
        DecoderCustomTypes.TokenOutput calldata output
    )
        external
        pure
        returns (bytes memory addressFound)
    {
        if (
            output.swapData.swapType != DecoderCustomTypes.SwapType.NONE || output.swapData.extRouter != address(0)
                || output.pendleSwap != address(0)
        ) revert PendleRouterDecoderAndSanitizer__AggregatorSwapsNotPermitted();

        addressFound = abi.encodePacked(receiver, market, output.tokenOut, output.tokenRedeemSy);
    }
}

// src/base/DecodersAndSanitizers/Protocols/UniswapV3DecoderAndSanitizer.sol

abstract contract UniswapV3DecoderAndSanitizer is BaseDecoderAndSanitizer {
    //============================== ERRORS ===============================

    error UniswapV3DecoderAndSanitizer__BadPathFormat();
    error UniswapV3DecoderAndSanitizer__BadTokenId();

    //============================== IMMUTABLES ===============================

    /**
     * @notice The networks uniswapV3 nonfungible position manager.
     */
    INonFungiblePositionManager internal immutable uniswapV3NonFungiblePositionManager;

    constructor(address _uniswapV3NonFungiblePositionManager) {
        uniswapV3NonFungiblePositionManager = INonFungiblePositionManager(_uniswapV3NonFungiblePositionManager);
    }

    //============================== UNISWAP V3 ===============================

    // @desc Uniswap V3 function to swap by exactInput, will revert if the path is not valid
    // @tag packedArgs:bytes:The path ADDRESSES ONLY of the swap followed by the recipient
    function exactInput(DecoderCustomTypes.ExactInputParams calldata params)
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        // Nothing to sanitize
        // Return addresses found
        // Determine how many addresses are in params.path.
        uint256 chunkSize = 23; // 3 bytes for uint24 fee, and 20 bytes for address token
        uint256 pathLength = params.path.length;
        if (pathLength % chunkSize != 20) revert UniswapV3DecoderAndSanitizer__BadPathFormat();
        uint256 pathAddressLength = 1 + (pathLength / chunkSize);
        uint256 pathIndex;
        for (uint256 i; i < pathAddressLength; ++i) {
            addressesFound = abi.encodePacked(addressesFound, params.path[pathIndex:pathIndex + 20]);
            pathIndex += chunkSize;
        }
        addressesFound = abi.encodePacked(addressesFound, params.recipient);
    }

    // @desc Uniswap V3 function to mint LP
    // @tag token0:address:The first token in the pair
    // @tag token1:address:The second token in the pair
    // @tag recipient:address:The recipient of the LP token
    function mint(DecoderCustomTypes.MintParams calldata params)
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        // Nothing to sanitize
        // Return addresses found
        addressesFound = abi.encodePacked(params.token0, params.token1, params.recipient);
    }

    // @desc Uniswap V3 function to increase liquidity, will revert if the tokenId is not owned by the boring vault
    // @tag operator:address:The operator/owner of the liquidity
    // @tag token0:address:The first token in the pair
    // @tag token1:address:The second token in the pair
    function increaseLiquidity(DecoderCustomTypes.IncreaseLiquidityParams calldata params)
        external
        view
        virtual
        returns (bytes memory addressesFound)
    {
        // Sanitize raw data
        if (uniswapV3NonFungiblePositionManager.ownerOf(params.tokenId) != boringVault) {
            revert UniswapV3DecoderAndSanitizer__BadTokenId();
        }
        // Extract addresses from uniswapV3NonFungiblePositionManager.positions(params.tokenId).
        (, address operator, address token0, address token1,,,,,,,,) =
            uniswapV3NonFungiblePositionManager.positions(params.tokenId);
        addressesFound = abi.encodePacked(operator, token0, token1);
    }

    // @desc Uniswap V3 function to decrease liquidity, will revert if the tokenId is not owned by the boring vault
    function decreaseLiquidity(DecoderCustomTypes.DecreaseLiquidityParams calldata params)
        external
        view
        virtual
        returns (bytes memory addressesFound)
    {
        // Sanitize raw data
        // NOTE ownerOf check is done in PositionManager contract as well, but it is added here
        // just for completeness.
        if (uniswapV3NonFungiblePositionManager.ownerOf(params.tokenId) != boringVault) {
            revert UniswapV3DecoderAndSanitizer__BadTokenId();
        }

        // No addresses in data
        return addressesFound;
    }

    // @desc Uniswap V3 function to collect fees, will revert if the tokenId is not owned by the boring vault
    // @tag recipient:address:The recipient of the fees
    function collect(DecoderCustomTypes.CollectParams calldata params)
        external
        view
        virtual
        returns (bytes memory addressesFound)
    {
        // Sanitize raw data
        // NOTE ownerOf check is done in PositionManager contract as well, but it is added here
        // just for completeness.
        if (uniswapV3NonFungiblePositionManager.ownerOf(params.tokenId) != boringVault) {
            revert UniswapV3DecoderAndSanitizer__BadTokenId();
        }

        // Return addresses found
        addressesFound = abi.encodePacked(params.recipient);
    }

    // @desc Uniswap V3 function to burn empty LP NFTs, will revert if the tokenId is not owned by the boring vault
    function burn(uint256 tokenId) external view virtual returns (bytes memory addressesFound) {
        // Sanitize raw data
        // NOTE ownerOf check is done in PositionManager contract as well, but it is added here
        // just for completeness.
        if (uniswapV3NonFungiblePositionManager.ownerOf(tokenId) != boringVault) {
            revert UniswapV3DecoderAndSanitizer__BadTokenId();
        }
    }

    // @desc Uniswap V3 function to safeTransferFrom ERC721s
    // @tag to:address:The recipient of the ERC721
    function safeTransferFrom(
        address,
        address to,
        uint256
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(to);
    }
}

// src/base/DecodersAndSanitizers/unifiBTCDecoderAndSanitizer.sol

contract unifiBTCDecoderAndSanitizer is
    NativeWrapperDecoderAndSanitizer,
    UniswapV3DecoderAndSanitizer,
    MasterChefV3DecoderAndSanitizer,
    PendleRouterDecoderAndSanitizer,
    ERC4626DecoderAndSanitizer,
    CurveDecoderAndSanitizer,
    BalancerV2DecoderAndSanitizer
{
    constructor(
        address _boringVault,
        address _uniswapV3NonFungiblePositionManager
    )
        UniswapV3DecoderAndSanitizer(_uniswapV3NonFungiblePositionManager)
        BaseDecoderAndSanitizer(_boringVault)
    { }

    /**
     * @notice BalancerV2, NativeWrapper, Curve all specify a `withdraw(uint256)`,
     *         all cases are handled the same way.
     */
    function withdraw(uint256)
        external
        pure
        override(BalancerV2DecoderAndSanitizer, CurveDecoderAndSanitizer, NativeWrapperDecoderAndSanitizer)
        returns (bytes memory addressesFound)
    {
        // Nothing to sanitize or return
        return addressesFound;
    }

    /**
     * @notice Curve, BalancerV2, and ERC4626 all specify a `deposit(uint256, address receiver)`,
     *         all cases are handled the same way.
     */
    function deposit(
        uint256,
        address receiver
    )
        external
        pure
        override(CurveDecoderAndSanitizer, BalancerV2DecoderAndSanitizer, ERC4626DecoderAndSanitizer)
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(receiver);
    }
}