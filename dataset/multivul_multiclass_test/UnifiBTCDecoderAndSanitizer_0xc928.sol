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

    constructor(address _boringVault) {
        boringVault = _boringVault;
    }

    function approve(address spender, uint256) external pure returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(spender);
    }
}

// src/base/DecodersAndSanitizers/Protocols/BalancerV2DecoderAndSanitizer.sol

abstract contract BalancerV2DecoderAndSanitizer is BaseDecoderAndSanitizer {
    //============================== ERRORS ===============================

    error BalancerV2DecoderAndSanitizer__SingleSwapUserDataLengthNonZero();
    error BalancerV2DecoderAndSanitizer__InternalBalancesNotSupported();

    //============================== BALANCER V2 ===============================

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

    function deposit(uint256, address recipient) external pure virtual returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(recipient);
    }

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

    function exchange(int128, int128, uint256, uint256) external pure virtual returns (bytes memory addressesFound) {
        // Nothing to sanitize or return
        return addressesFound;
    }

    function add_liquidity(uint256[] calldata, uint256) external pure virtual returns (bytes memory addressesFound) {
        // Nothing to sanitize or return
        return addressesFound;
    }

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

    function deposit(uint256, address receiver) external pure virtual returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(receiver);
    }

    function withdraw(uint256) external pure virtual returns (bytes memory addressesFound) {
        // Nothing to sanitize or return
        return addressesFound;
    }

    function claim_rewards(address _addr) external pure virtual returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(_addr);
    }
}

// src/base/DecodersAndSanitizers/Protocols/ERC4626DecoderAndSanitizer.sol

abstract contract ERC4626DecoderAndSanitizer is BaseDecoderAndSanitizer {
    //============================== ERC4626 ===============================

    function deposit(uint256, address receiver) external pure virtual returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(receiver);
    }

    function mint(uint256, address receiver) external pure virtual returns (bytes memory addressesFound) {
        addressesFound = abi.encodePacked(receiver);
    }

    function withdraw(
        uint256,
        address receiver,
        address owner
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(receiver, owner);
    }

    function redeem(
        uint256,
        address receiver,
        address owner
    )
        external
        pure
        virtual
        returns (bytes memory addressesFound)
    {
        addressesFound = abi.encodePacked(receiver, owner);
    }
}

// src/base/DecodersAndSanitizers/Protocols/MasterChefV3DecoderAndSanitizer.sol

abstract contract MasterChefV3DecoderAndSanitizer is BaseDecoderAndSanitizer {
    function harvest(uint256, address _to) external pure virtual returns (bytes memory addressesFound) {
        return abi.encodePacked(_to);
    }

    function withdraw(uint256, address _to) external pure virtual returns (bytes memory addressesFound) {
        return abi.encodePacked(_to);
    }
}

// src/base/DecodersAndSanitizers/Protocols/NativeWrapperDecoderAndSanitizer.sol

abstract contract NativeWrapperDecoderAndSanitizer is BaseDecoderAndSanitizer {
    //============================== ETHERFI ===============================

    function deposit() external pure virtual returns (bytes memory addressesFound) {
        // Nothing to sanitize or return
        return addressesFound;
    }

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

        addressesFound =
            abi.encodePacked(user, sy, input.tokenIn, input.tokenMintSy, input.pendleSwap, input.swapData.extRouter);
    }

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

        addressesFound = abi.encodePacked(
            user, sy, output.tokenOut, output.tokenRedeemSy, output.pendleSwap, output.swapData.extRouter
        );
    }

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

        addressesFound = abi.encodePacked(receiver, market);
    }

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

        addressFound = abi.encodePacked(receiver, market);
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

contract UnifiBTCDecoderAndSanitizer is
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