// SPDX-License-Identifier: MIT
pragma solidity =0.8.24;










interface IFluidVaultResolver {

    struct Tokens {
        address token0;
        address token1;
    }

    struct ConstantViews {
        address liquidity;
        address factory;
        address operateImplementation;
        address adminImplementation;
        address secondaryImplementation;
        address deployer; // address which deploys oracle
        address supply; // either liquidity layer or DEX protocol
        address borrow; // either liquidity layer or DEX protocol
        Tokens supplyToken; // if smart collateral then address of token0 & token1 else just supply token address at token0 and token1 as empty
        Tokens borrowToken; // if smart debt then address of token0 & token1 else just borrow token address at token0 and token1 as empty
        uint256 vaultId;
        uint256 vaultType;
        bytes32 supplyExchangePriceSlot; // if smart collateral then slot is from DEX protocol else from liquidity layer
        bytes32 borrowExchangePriceSlot; // if smart debt then slot is from DEX protocol else from liquidity layer
        bytes32 userSupplySlot; // if smart collateral then slot is from DEX protocol else from liquidity layer
        bytes32 userBorrowSlot; // if smart debt then slot is from DEX protocol else from liquidity layer
    }

    struct Configs {
        // can be supplyRate instead if Vault Type is smart col. in that case if 1st bit == 1 then positive else negative
        uint16 supplyRateMagnifier;
        // can be borrowRate instead if Vault Type is smart debt. in that case if 1st bit == 1 then positive else negative
        uint16 borrowRateMagnifier;
        uint16 collateralFactor;
        uint16 liquidationThreshold;
        uint16 liquidationMaxLimit;
        uint16 withdrawalGap;
        uint16 liquidationPenalty;
        uint16 borrowFee;
        address oracle;
        // Oracle price is always debt per col, i.e. amount of debt for 1 col.
        // In case of Dex this price can be used to resolve shares values w.r.t. token0 or token1:
        // - T2: debt token per 1 col share
        // - T3: debt shares per 1 col token
        // - T4: debt shares per 1 col share
        uint oraclePriceOperate;
        uint oraclePriceLiquidate;
        address rebalancer;
        uint lastUpdateTimestamp;
    }

    struct ExchangePricesAndRates {
        uint lastStoredLiquiditySupplyExchangePrice; // 0 in case of smart col
        uint lastStoredLiquidityBorrowExchangePrice; // 0 in case of smart debt
        uint lastStoredVaultSupplyExchangePrice;
        uint lastStoredVaultBorrowExchangePrice;
        uint liquiditySupplyExchangePrice; // set to 1e12 in case of smart col
        uint liquidityBorrowExchangePrice; // set to 1e12 in case of smart debt
        uint vaultSupplyExchangePrice;
        uint vaultBorrowExchangePrice;
        uint supplyRateLiquidity; // set to 0 in case of smart col. Must get per token through DexEntireData
        uint borrowRateLiquidity; // set to 0 in case of smart debt. Must get per token through DexEntireData
        // supplyRateVault or borrowRateVault:
        // - when normal col / debt: rate at liquidity + diff rewards or fee through magnifier (rewardsOrFeeRate below)
        // - when smart col / debt: rewards or fee rate at the vault itself. always == rewardsOrFeeRate below.
        // to get the full rates for vault when smart col / debt, combine with data from DexResolver:
        // - rateAtLiquidity for token0 or token1 (DexResolver)
        // - the rewards or fee rate at the vault (VaultResolver)
        // - the Dex APR (currently off-chain compiled through tracking swap events at the DEX)
        int supplyRateVault; // can be negative in case of smart col (meaning pay to supply)
        int borrowRateVault; // can be negative in case of smart debt (meaning get paid to borrow)
        // rewardsOrFeeRateSupply: rewards or fee rate in percent 1e2 precision (1% = 100, 100% = 10000).
        // positive rewards, negative fee.
        // for smart col vaults: supplyRateVault == supplyRateLiquidity.
        // for normal col vaults: relative percent to supplyRateLiquidity, e.g.:
        // when rewards: supplyRateLiquidity = 4%, rewardsOrFeeRateSupply = 20%, supplyRateVault = 4.8%.
        // when fee: supplyRateLiquidity = 4%, rewardsOrFeeRateSupply = -30%, supplyRateVault = 2.8%.
        int rewardsOrFeeRateSupply;
        // rewardsOrFeeRateBorrow: rewards or fee rate in percent 1e2 precision (1% = 100, 100% = 10000).
        // negative rewards, positive fee.
        // for smart debt vaults: borrowRateVault == borrowRateLiquidity.
        // for normal debt vaults: relative percent to borrowRateLiquidity, e.g.:
        // when rewards: borrowRateLiquidity = 4%, rewardsOrFeeRateBorrow = -20%, borrowRateVault = 3.2%.
        // when fee: borrowRateLiquidity = 4%, rewardsOrFeeRateBorrow = 30%, borrowRateVault = 5.2%.
        int rewardsOrFeeRateBorrow;
    }

    struct TotalSupplyAndBorrow {
        uint totalSupplyVault;
        uint totalBorrowVault;
        uint totalSupplyLiquidityOrDex;
        uint totalBorrowLiquidityOrDex;
        uint absorbedSupply;
        uint absorbedBorrow;
    }

    struct LimitsAndAvailability {
        // in case of DEX: withdrawable / borrowable amount of vault at DEX, BUT there could be that DEX can not withdraw
        // that much at Liquidity! So for DEX this must be combined with returned data in DexResolver.
        uint withdrawLimit;
        uint withdrawableUntilLimit;
        uint withdrawable;
        uint borrowLimit;
        uint borrowableUntilLimit; // borrowable amount until any borrow limit (incl. max utilization limit)
        uint borrowable; // actual currently borrowable amount (borrow limit - already borrowed) & considering balance, max utilization
        uint borrowLimitUtilization; // borrow limit for `maxUtilization` config at Liquidity
        uint minimumBorrowing;
    }

    struct CurrentBranchState {
        uint status; // if 0 then not liquidated, if 1 then liquidated, if 2 then merged, if 3 then closed
        int minimaTick;
        uint debtFactor;
        uint partials;
        uint debtLiquidity;
        uint baseBranchId;
        int baseBranchMinima;
    }

    struct VaultState {
        uint totalPositions;
        int topTick;
        uint currentBranch;
        uint totalBranch;
        uint totalBorrow;
        uint totalSupply;
        CurrentBranchState currentBranchState;
    }

    struct UserSupplyData {
        bool modeWithInterest; // true if mode = with interest, false = without interest
        uint256 supply; // user supply amount
        // the withdrawal limit (e.g. if 10% is the limit, and 100M is supplied, it would be 90M)
        uint256 withdrawalLimit;
        uint256 lastUpdateTimestamp;
        uint256 expandPercent; // withdrawal limit expand percent in 1e2
        uint256 expandDuration; // withdrawal limit expand duration in seconds
        uint256 baseWithdrawalLimit;
        // the current actual max withdrawable amount (e.g. if 10% is the limit, and 100M is supplied, it would be 10M)
        uint256 withdrawableUntilLimit;
        uint256 withdrawable; // actual currently withdrawable amount (supply - withdrawal Limit) & considering balance
    }

    // amounts are always in normal (for withInterest already multiplied with exchange price)
    struct UserBorrowData {
        bool modeWithInterest; // true if mode = with interest, false = without interest
        uint256 borrow; // user borrow amount
        uint256 borrowLimit;
        uint256 lastUpdateTimestamp;
        uint256 expandPercent;
        uint256 expandDuration;
        uint256 baseBorrowLimit;
        uint256 maxBorrowLimit;
        uint256 borrowableUntilLimit; // borrowable amount until any borrow limit (incl. max utilization limit)
        uint256 borrowable; // actual currently borrowable amount (borrow limit - already borrowed) & considering balance, max utilization
        uint256 borrowLimitUtilization; // borrow limit for `maxUtilization`
    }

    struct VaultEntireData {
        address vault;
        bool isSmartCol; // true if col token is a Fluid Dex
        bool isSmartDebt; // true if debt token is a Fluid Dex
        ConstantViews constantVariables;
        Configs configs;
        ExchangePricesAndRates exchangePricesAndRates;
        TotalSupplyAndBorrow totalSupplyAndBorrow;
        LimitsAndAvailability limitsAndAvailability;
        VaultState vaultState;
        // liquidity related data such as supply amount, limits, expansion etc.
        // only set if not smart col!
        UserSupplyData liquidityUserSupplyData;
        // liquidity related data such as borrow amount, limits, expansion etc.
        // only set if not smart debt!
        UserBorrowData liquidityUserBorrowData;
    }

    struct UserPosition {
        uint nftId;
        address owner;
        bool isLiquidated;
        bool isSupplyPosition; // if true that means borrowing is 0
        int tick;
        uint tickId;
        uint beforeSupply;
        uint beforeBorrow;
        uint beforeDustBorrow;
        uint supply;
        uint borrow;
        uint dustBorrow;
    }

    /// @notice Retrieves the position data for a given NFT ID and the corresponding vault data.
    /// @param nftId_ The NFT ID for which to retrieve the position data.
    /// @return userPosition_ The UserPosition structure containing the position data.
    /// @return vaultData_ The VaultEntireData structure containing the vault data.
    function positionByNftId(
        uint nftId_
    ) external view returns (UserPosition memory userPosition_, VaultEntireData memory vaultData_);

    /// @notice Returns an array of NFT IDs for all positions of a given user.
    /// @param user_ The address of the user for whom to fetch positions.
    /// @return nftIds_ An array of NFT IDs representing the user's positions.
    function positionsNftIdOfUser(address user_) external view returns (uint[] memory nftIds_);

    /// @notice Get the addresses of all the vaults.
    /// @return vaults_ The addresses of all the vaults.
    function getAllVaultsAddresses() external view returns (address[] memory vaults_);

    function getVaultId(address vault_) external view returns (uint id_);

    function getVaultAddress(uint vaultId_) external view returns (address vault_);

    function getVaultEntireData(address vault_) external view returns (VaultEntireData memory vaultData_);

    function vaultByNftId(uint nftId_) external view returns (address vault_);
}








library FluidDexModel {

    /// @param collAmount0 Amount of collateral 0 to deposit.
    /// @param collAmount1 Amount of collateral 1 to deposit.
    /// @param minCollShares Min amount of collateral shares to mint.
    struct SupplyVariableData {
        uint256 collAmount0;
        uint256 collAmount1;
        uint256 minCollShares;
    }

    /// @param collAmount0 Amount of collateral 0 to withdraw.
    /// @param collAmount1 Amount of collateral 1 to withdraw.
    /// @param maxCollShares Max amount of collateral shares to burn. Can be empty for max withdrawal (see minCollToWithdraw)
    /// @param minCollToWithdraw Minimum amount of collateral to withdraw in one token. Only used for max withdrawal, when:
    /// 1. variableData.collAmount0 == type(uint256).max -> all collateral will be withdrawn in coll token0.
    ///    Any existing amount of token1 will be converted to token0 on fluid.
    /// 2. variableData.collAmount1 == type(uint256).max -> all collateral will be withdrawn in coll token1.
    ///    Any existing amount of token0 will be converted to token1 on fluid.
    struct WithdrawVariableData {
        uint256 collAmount0;
        uint256 collAmount1;
        uint256 maxCollShares;
        uint256 minCollToWithdraw;
    }

    /// @param debtAmount0 Amount of debt token 0 to borrow.
    /// @param debtAmount1 Amount of debt token 1 to borrow.
    /// @param maxDebtShares Max amount of debt shares to mint.
    struct BorrowVariableData {
        uint256 debtAmount0;
        uint256 debtAmount1;
        uint256 maxDebtShares;
    }

    /// @param debtAmount0 Amount of debt token 0 to payback.
    /// @param debtAmount1 Amount of debt token 1 to payback.
    /// @param minDebtShares Min amount of debt shares to burn. Can be empty for max payback (see maxAmountToPull)
    /// @param maxAmountToPull Maximum amount of debt token to pull from the user. Only used for max payback when:
    /// 1. variableData.debtAmount0 == type(uint256).max -> all debt will be paid back in debt token0.
    ///    Any existing amount of debt token1 will be converted to debt token0 on fluid.
    /// 2. variableData.debtAmount1 == type(uint256).max -> all debt will be paid back in debt token1.
    ///    Any existing amount of debt token0 will be converted to debt token1 on fluid.
    struct PaybackVariableData {
        uint256 debtAmount0;
        uint256 debtAmount1;
        uint256 minDebtShares;
        uint256 maxAmountToPull;
    }

    /// @notice Data struct for supplying liquidity to a Fluid DEX
    /// @param vault Address of the vault
    /// @param vaultType Type of the vault. For supply, it will be T2 or T4
    /// @param nftId NFT id of the position
    /// @param from Address to pull the tokens from
    /// @param variableData Data for supplying liquidity with variable amounts
    struct SupplyDexData {
        address vault;
        uint256 vaultType;
        uint256 nftId;
        address from;
        SupplyVariableData variableData;
    }

    /// @notice Data struct for withdrawing liquidity from a Fluid DEX
    /// @param vault Address of the vault
    /// @param vaultType Type of the vault. For withdraw, it will be T2 or T4
    /// @param nftId NFT id of the position
    /// @param to Address to send the tokens to
    /// @param variableData Data for withdrawing liquidity with variable amounts
    /// @param wrapWithdrawnEth Whether to wrap withdrawn ETH into WETH
    struct WithdrawDexData {
        address vault;
        uint256 vaultType;
        uint256 nftId;
        address to;
        WithdrawVariableData variableData;
        bool wrapWithdrawnEth;
    }

    /// @notice Data struct for borrowing tokens from a Fluid DEX
    /// @param vault Address of the vault
    /// @param vaultType Type of the vault. For borrow, it will be T3 or T4
    /// @param nftId NFT id of the position
    /// @param to Address to send the borrowed tokens to
    /// @param variableData Data for borrowing tokens with variable amounts
    /// @param wrapBorrowedEth Whether to wrap borrowed ETH into WETH
    struct BorrowDexData {
        address vault;
        uint256 vaultType;
        uint256 nftId;
        address to;
        BorrowVariableData variableData;
        bool wrapBorrowedEth;
    }

    /// @notice Data struct for paying back borrowed tokens to a Fluid DEX
    /// @param vault Address of the vault
    /// @param vaultType Type of the vault. For payback, it will be T3 or T4
    /// @param nftId NFT id of the position
    /// @param from Address to pull the tokens from
    /// @param variableData Data for paying back borrowed tokens with variable amounts
    /// @param position User position data fetched from Fluid Vault Resolver
    struct PaybackDexData {
        address vault;
        uint256 vaultType;
        uint256 nftId;
        address from;
        PaybackVariableData variableData;
        IFluidVaultResolver.UserPosition position;
    }
}






contract MainnetFluidAddresses {
    address internal constant FLUID_VAULT_RESOLVER = 0x814c8C7ceb1411B364c2940c4b9380e739e06686;
    address internal constant FLUID_DEX_RESOLVER = 0x71783F64719899319B56BdA4F27E1219d9AF9a3d;
    address internal constant FLUID_LENDING_RESOLVER = 0xC215485C572365AE87f908ad35233EC2572A3BEC;
    address internal constant F_WETH_TOKEN_ADDR = 0x90551c1795392094FE6D29B758EcCD233cFAa260;
    address internal constant ETH_ADDR = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE;
    address internal constant FLUID_MERKLE_DISTRIBUTOR = 0x7060FE0Dd3E31be01EFAc6B28C8D38018fD163B0;
}








contract FluidHelper is MainnetFluidAddresses {
}









contract FluidRatioHelper is FluidHelper {
    uint256 internal constant PRICE_SCALER = 1e27;
    uint256 internal constant WAD = 1e18;

    /// @notice Gets ratio for a fluid position
    /// @param _nftId nft id of the fluid position
    /// @return ratio Ratio of the position
    function getRatio(uint256 _nftId) public view returns (uint256 ratio) {
        (
            IFluidVaultResolver.UserPosition memory userPosition,
            IFluidVaultResolver.VaultEntireData memory vaultData
        ) = IFluidVaultResolver(FLUID_VAULT_RESOLVER).positionByNftId(_nftId);

        if (userPosition.borrow == 0 || userPosition.supply == 0) return uint256(0);

        uint256 collAmountInDebtToken = (userPosition.supply * vaultData.configs.oraclePriceOperate) / PRICE_SCALER;
        
        ratio = collAmountInDebtToken * WAD / userPosition.borrow;
    }
}







interface IFluidDexT1 {
    error FluidDexError(uint256 errorId);

    /// @notice used to simulate swap to find the output amount
    error FluidDexSwapResult(uint256 amountOut);

    error FluidDexPerfectLiquidityOutput(uint256 token0Amt, uint token1Amt);

    error FluidDexSingleTokenOutput(uint256 tokenAmt);

    error FluidDexLiquidityOutput(uint256 shares);

    error FluidDexPricesAndExchangeRates(PricesAndExchangePrice pex_);

    /// @notice returns the dex id
    function DEX_ID() external view returns (uint256);

    /// @notice reads uint256 data `result_` from storage at a bytes32 storage `slot_` key.
    function readFromStorage(bytes32 slot_) external view returns (uint256 result_);

    struct Implementations {
        address shift;
        address admin;
        address colOperations;
        address debtOperations;
        address perfectOperationsAndOracle;
    }

    struct ConstantViews {
        uint256 dexId;
        address liquidity;
        address factory;
        Implementations implementations;
        address deployerContract;
        address token0;
        address token1;
        bytes32 supplyToken0Slot;
        bytes32 borrowToken0Slot;
        bytes32 supplyToken1Slot;
        bytes32 borrowToken1Slot;
        bytes32 exchangePriceToken0Slot;
        bytes32 exchangePriceToken1Slot;
        uint256 oracleMapping;
    }

    struct ConstantViews2 {
        uint token0NumeratorPrecision;
        uint token0DenominatorPrecision;
        uint token1NumeratorPrecision;
        uint token1DenominatorPrecision;
    }

    struct PricesAndExchangePrice {
        uint lastStoredPrice; // last stored price in 1e27 decimals
        uint centerPrice; // last stored price in 1e27 decimals
        uint upperRange; // price at upper range in 1e27 decimals
        uint lowerRange; // price at lower range in 1e27 decimals
        uint geometricMean; // geometric mean of upper range & lower range in 1e27 decimals
        uint supplyToken0ExchangePrice;
        uint borrowToken0ExchangePrice;
        uint supplyToken1ExchangePrice;
        uint borrowToken1ExchangePrice;
    }

    struct CollateralReserves {
        uint token0RealReserves;
        uint token1RealReserves;
        uint token0ImaginaryReserves;
        uint token1ImaginaryReserves;
    }

    struct DebtReserves {
        uint token0Debt;
        uint token1Debt;
        uint token0RealReserves;
        uint token1RealReserves;
        uint token0ImaginaryReserves;
        uint token1ImaginaryReserves;
    }

    function getCollateralReserves(
        uint geometricMean_,
        uint upperRange_,
        uint lowerRange_,
        uint token0SupplyExchangePrice_,
        uint token1SupplyExchangePrice_
    ) external view returns (CollateralReserves memory c_);

    function getDebtReserves(
        uint geometricMean_,
        uint upperRange_,
        uint lowerRange_,
        uint token0BorrowExchangePrice_,
        uint token1BorrowExchangePrice_
    ) external view returns (DebtReserves memory d_);

    // reverts with FluidDexPricesAndExchangeRates(pex_);
    function getPricesAndExchangePrices() external;

    function constantsView() external view returns (ConstantViews memory constantsView_);

    function constantsView2() external view returns (ConstantViews2 memory constantsView2_);

    struct Oracle {
        uint twap1by0; // TWAP price
        uint lowestPrice1by0; // lowest price point
        uint highestPrice1by0; // highest price point
        uint twap0by1; // TWAP price
        uint lowestPrice0by1; // lowest price point
        uint highestPrice0by1; // highest price point
    }

    /// @dev This function allows users to swap a specific amount of input tokens for output tokens
    /// @param swap0to1_ Direction of swap. If true, swaps token0 for token1; if false, swaps token1 for token0
    /// @param amountIn_ The exact amount of input tokens to swap
    /// @param amountOutMin_ The minimum amount of output tokens the user is willing to accept
    /// @param to_ Recipient of swapped tokens. If to_ == address(0) then out tokens will be sent to msg.sender. If to_ == ADDRESS_DEAD then function will revert with amountOut_
    /// @return amountOut_ The amount of output tokens received from the swap
    function swapIn(
        bool swap0to1_,
        uint256 amountIn_,
        uint256 amountOutMin_,
        address to_
    ) external payable returns (uint256 amountOut_);

    /// @dev Swap tokens with perfect amount out
    /// @param swap0to1_ Direction of swap. If true, swaps token0 for token1; if false, swaps token1 for token0
    /// @param amountOut_ The exact amount of tokens to receive after swap
    /// @param amountInMax_ Maximum amount of tokens to swap in
    /// @param to_ Recipient of swapped tokens. If to_ == address(0) then out tokens will be sent to msg.sender. If to_ == ADDRESS_DEAD then function will revert with amountIn_
    /// @return amountIn_ The amount of input tokens used for the swap
    function swapOut(
        bool swap0to1_,
        uint256 amountOut_,
        uint256 amountInMax_,
        address to_
    ) external payable returns (uint256 amountIn_);

    /// @dev Deposit tokens in equal proportion to the current pool ratio
    /// @param shares_ The number of shares to mint
    /// @param maxToken0Deposit_ Maximum amount of token0 to deposit
    /// @param maxToken1Deposit_ Maximum amount of token1 to deposit
    /// @param estimate_ If true, function will revert with estimated deposit amounts without executing the deposit
    /// @return token0Amt_ Amount of token0 deposited
    /// @return token1Amt_ Amount of token1 deposited
    function depositPerfect(
        uint shares_,
        uint maxToken0Deposit_,
        uint maxToken1Deposit_,
        bool estimate_
    ) external payable returns (uint token0Amt_, uint token1Amt_);

    /// @dev This function allows users to withdraw a perfect amount of collateral liquidity
    /// @param shares_ The number of shares to withdraw
    /// @param minToken0Withdraw_ The minimum amount of token0 the user is willing to accept
    /// @param minToken1Withdraw_ The minimum amount of token1 the user is willing to accept
    /// @param to_ Recipient of swapped tokens. If to_ == address(0) then out tokens will be sent to msg.sender. If to_ == ADDRESS_DEAD then function will revert with token0Amt_ & token1Amt_
    /// @return token0Amt_ The amount of token0 withdrawn
    /// @return token1Amt_ The amount of token1 withdrawn
    function withdrawPerfect(
        uint shares_,
        uint minToken0Withdraw_,
        uint minToken1Withdraw_,
        address to_
    ) external returns (uint token0Amt_, uint token1Amt_);

    /// @dev This function allows users to borrow tokens in equal proportion to the current debt pool ratio
    /// @param shares_ The number of shares to borrow
    /// @param minToken0Borrow_ Minimum amount of token0 to borrow
    /// @param minToken1Borrow_ Minimum amount of token1 to borrow
    /// @param to_ Recipient of swapped tokens. If to_ == address(0) then out tokens will be sent to msg.sender. If to_ == ADDRESS_DEAD then function will revert with token0Amt_ & token1Amt_
    /// @return token0Amt_ Amount of token0 borrowed
    /// @return token1Amt_ Amount of token1 borrowed
    function borrowPerfect(
        uint shares_,
        uint minToken0Borrow_,
        uint minToken1Borrow_,
        address to_
    ) external returns (uint token0Amt_, uint token1Amt_);

    /// @dev This function allows users to pay back borrowed tokens in equal proportion to the current debt pool ratio
    /// @param shares_ The number of shares to pay back
    /// @param maxToken0Payback_ Maximum amount of token0 to pay back
    /// @param maxToken1Payback_ Maximum amount of token1 to pay back
    /// @param estimate_ If true, function will revert with estimated payback amounts without executing the payback
    /// @return token0Amt_ Amount of token0 paid back
    /// @return token1Amt_ Amount of token1 paid back
    function paybackPerfect(
        uint shares_,
        uint maxToken0Payback_,
        uint maxToken1Payback_,
        bool estimate_
    ) external payable returns (uint token0Amt_, uint token1Amt_);

    /// @dev This function allows users to deposit tokens in any proportion into the col pool
    /// @param token0Amt_ The amount of token0 to deposit
    /// @param token1Amt_ The amount of token1 to deposit
    /// @param minSharesAmt_ The minimum amount of shares the user expects to receive
    /// @param estimate_ If true, function will revert with estimated shares without executing the deposit
    /// @return shares_ The amount of shares minted for the deposit
    function deposit(
        uint token0Amt_,
        uint token1Amt_,
        uint minSharesAmt_,
        bool estimate_
    ) external payable returns (uint shares_);

    /// @dev This function allows users to withdraw tokens in any proportion from the col pool
    /// @param token0Amt_ The amount of token0 to withdraw
    /// @param token1Amt_ The amount of token1 to withdraw
    /// @param maxSharesAmt_ The maximum number of shares the user is willing to burn
    /// @param to_ Recipient of swapped tokens. If to_ == address(0) then out tokens will be sent to msg.sender. If to_ == ADDRESS_DEAD then function will revert with shares_
    /// @return shares_ The number of shares burned for the withdrawal
    function withdraw(
        uint token0Amt_,
        uint token1Amt_,
        uint maxSharesAmt_,
        address to_
    ) external returns (uint shares_);

    /// @dev This function allows users to borrow tokens in any proportion from the debt pool
    /// @param token0Amt_ The amount of token0 to borrow
    /// @param token1Amt_ The amount of token1 to borrow
    /// @param maxSharesAmt_ The maximum amount of shares the user is willing to receive
    /// @param to_ Recipient of swapped tokens. If to_ == address(0) then out tokens will be sent to msg.sender. If to_ == ADDRESS_DEAD then function will revert with shares_
    /// @return shares_ The amount of borrow shares minted to represent the borrowed amount
    function borrow(
        uint token0Amt_,
        uint token1Amt_,
        uint maxSharesAmt_,
        address to_
    ) external returns (uint shares_);

    /// @dev This function allows users to payback tokens in any proportion to the debt pool
    /// @param token0Amt_ The amount of token0 to payback
    /// @param token1Amt_ The amount of token1 to payback
    /// @param minSharesAmt_ The minimum amount of shares the user expects to burn
    /// @param estimate_ If true, function will revert with estimated shares without executing the payback
    /// @return shares_ The amount of borrow shares burned for the payback
    function payback(
        uint token0Amt_,
        uint token1Amt_,
        uint minSharesAmt_,
        bool estimate_
    ) external payable returns (uint shares_);

    /// @dev This function allows users to withdraw their collateral with perfect shares in one token
    /// @param shares_ The number of shares to burn for withdrawal
    /// @param minToken0_ The minimum amount of token0 the user expects to receive (set to 0 if withdrawing in token1)
    /// @param minToken1_ The minimum amount of token1 the user expects to receive (set to 0 if withdrawing in token0)
    /// @param to_ Recipient of swapped tokens. If to_ == address(0) then out tokens will be sent to msg.sender. If to_ == ADDRESS_DEAD then function will revert with withdrawAmt_
    /// @return withdrawAmt_ The amount of tokens withdrawn in the chosen token
    function withdrawPerfectInOneToken(
        uint shares_,
        uint minToken0_,
        uint minToken1_,
        address to_
    ) external returns (
        uint withdrawAmt_
    );

    /// @dev This function allows users to payback their debt with perfect shares in one token
    /// @param shares_ The number of shares to burn for payback
    /// @param maxToken0_ The maximum amount of token0 the user is willing to pay (set to 0 if paying back in token1)
    /// @param maxToken1_ The maximum amount of token1 the user is willing to pay (set to 0 if paying back in token0)
    /// @param estimate_ If true, the function will revert with the estimated payback amount without executing the payback
    /// @return paybackAmt_ The amount of tokens paid back in the chosen token
    function paybackPerfectInOneToken(
        uint shares_,
        uint maxToken0_,
        uint maxToken1_,
        bool estimate_
    ) external payable returns (
        uint paybackAmt_
    );

    /// @dev the oracle assumes last set price of pool till the next swap happens.
    /// There's a possibility that during that time some interest is generated hence the last stored price is not the 100% correct price for the whole duration
    /// but the difference due to interest will be super low so this difference is ignored
    /// For example 2 swaps happened 10min (600 seconds) apart and 1 token has 10% higher interest than other.
    /// then that token will accrue about 10% * 600 / secondsInAYear = ~0.0002%
    /// @param secondsAgos_ array of seconds ago for which TWAP is needed. If user sends [10, 30, 60] then twaps_ will return [10-0, 30-10, 60-30]
    /// @return twaps_ twap price, lowest price (aka minima) & highest price (aka maxima) between secondsAgo checkpoints
    /// @return currentPrice_ price of pool after the most recent swap
    function oraclePrice(
        uint[] memory secondsAgos_
    ) external view returns (
        Oracle[] memory twaps_,
        uint currentPrice_
    );
}







interface IFluidLiquidityResolverStructs {

    /// @notice struct to set borrow rate data for version 1
    struct RateDataV1Params {
        ///
        /// @param token for rate data
        address token;
        ///
        /// @param kink in borrow rate. in 1e2: 100% = 10_000; 1% = 100
        /// utilization below kink usually means slow increase in rate, once utilization is above kink borrow rate increases fast
        uint256 kink;
        ///
        /// @param rateAtUtilizationZero desired borrow rate when utilization is zero. in 1e2: 100% = 10_000; 1% = 100
        /// i.e. constant minimum borrow rate
        /// e.g. at utilization = 0.01% rate could still be at least 4% (rateAtUtilizationZero would be 400 then)
        uint256 rateAtUtilizationZero;
        ///
        /// @param rateAtUtilizationKink borrow rate when utilization is at kink. in 1e2: 100% = 10_000; 1% = 100
        /// e.g. when rate should be 7% at kink then rateAtUtilizationKink would be 700
        uint256 rateAtUtilizationKink;
        ///
        /// @param rateAtUtilizationMax borrow rate when utilization is maximum at 100%. in 1e2: 100% = 10_000; 1% = 100
        /// e.g. when rate should be 125% at 100% then rateAtUtilizationMax would be 12_500
        uint256 rateAtUtilizationMax;
    }

    /// @notice struct to set borrow rate data for version 2
    struct RateDataV2Params {
        ///
        /// @param token for rate data
        address token;
        ///
        /// @param kink1 first kink in borrow rate. in 1e2: 100% = 10_000; 1% = 100
        /// utilization below kink 1 usually means slow increase in rate, once utilization is above kink 1 borrow rate increases faster
        uint256 kink1;
        ///
        /// @param kink2 second kink in borrow rate. in 1e2: 100% = 10_000; 1% = 100
        /// utilization below kink 2 usually means slow / medium increase in rate, once utilization is above kink 2 borrow rate increases fast
        uint256 kink2;
        ///
        /// @param rateAtUtilizationZero desired borrow rate when utilization is zero. in 1e2: 100% = 10_000; 1% = 100
        /// i.e. constant minimum borrow rate
        /// e.g. at utilization = 0.01% rate could still be at least 4% (rateAtUtilizationZero would be 400 then)
        uint256 rateAtUtilizationZero;
        ///
        /// @param rateAtUtilizationKink1 desired borrow rate when utilization is at first kink. in 1e2: 100% = 10_000; 1% = 100
        /// e.g. when rate should be 7% at first kink then rateAtUtilizationKink would be 700
        uint256 rateAtUtilizationKink1;
        ///
        /// @param rateAtUtilizationKink2 desired borrow rate when utilization is at second kink. in 1e2: 100% = 10_000; 1% = 100
        /// e.g. when rate should be 7% at second kink then rateAtUtilizationKink would be 1_200
        uint256 rateAtUtilizationKink2;
        ///
        /// @param rateAtUtilizationMax desired borrow rate when utilization is maximum at 100%. in 1e2: 100% = 10_000; 1% = 100
        /// e.g. when rate should be 125% at 100% then rateAtUtilizationMax would be 12_500
        uint256 rateAtUtilizationMax;
    }

    struct RateData {
        uint256 version;
        RateDataV1Params rateDataV1;
        RateDataV2Params rateDataV2;
    }

    struct OverallTokenData {
        uint256 borrowRate;
        uint256 supplyRate;
        uint256 fee; // revenue fee
        uint256 lastStoredUtilization;
        uint256 storageUpdateThreshold;
        uint256 lastUpdateTimestamp;
        uint256 supplyExchangePrice;
        uint256 borrowExchangePrice;
        uint256 supplyRawInterest;
        uint256 supplyInterestFree;
        uint256 borrowRawInterest;
        uint256 borrowInterestFree;
        uint256 totalSupply;
        uint256 totalBorrow;
        uint256 revenue;
        uint256 maxUtilization; // maximum allowed utilization
        RateData rateData;
    }

    // amounts are always in normal (for withInterest already multiplied with exchange price)
    struct UserSupplyData {
        bool modeWithInterest; // true if mode = with interest, false = without interest
        uint256 supply; // user supply amount
        // the withdrawal limit (e.g. if 10% is the limit, and 100M is supplied, it would be 90M)
        uint256 withdrawalLimit;
        uint256 lastUpdateTimestamp;
        uint256 expandPercent; // withdrawal limit expand percent in 1e2
        uint256 expandDuration; // withdrawal limit expand duration in seconds
        uint256 baseWithdrawalLimit;
        // the current actual max withdrawable amount (e.g. if 10% is the limit, and 100M is supplied, it would be 10M)
        uint256 withdrawableUntilLimit;
        uint256 withdrawable; // actual currently withdrawable amount (supply - withdrawal Limit) & considering balance
    }

    // amounts are always in normal (for withInterest already multiplied with exchange price)
    struct UserBorrowData {
        bool modeWithInterest; // true if mode = with interest, false = without interest
        uint256 borrow; // user borrow amount
        uint256 borrowLimit;
        uint256 lastUpdateTimestamp;
        uint256 expandPercent;
        uint256 expandDuration;
        uint256 baseBorrowLimit;
        uint256 maxBorrowLimit;
        uint256 borrowableUntilLimit; // borrowable amount until any borrow limit (incl. max utilization limit)
        uint256 borrowable; // actual currently borrowable amount (borrow limit - already borrowed) & considering balance, max utilization
        uint256 borrowLimitUtilization; // borrow limit for `maxUtilization`
    }       
}








interface IFluidDexResolver {

    struct DexState {
        uint256 lastToLastStoredPrice;
        uint256 lastStoredPrice; // price of pool after the most recent swap
        uint256 centerPrice;
        uint256 lastUpdateTimestamp;
        uint256 lastPricesTimeDiff;
        uint256 oracleCheckPoint;
        uint256 oracleMapping;
        uint256 totalSupplyShares;
        uint256 totalBorrowShares;
        bool isSwapAndArbitragePaused; // if true, only perfect functions will be usable
        ShiftChanges shifts;
        // below values have to be combined with Oracle price data at the VaultResolver
        uint256 token0PerSupplyShare; // token0 amount per 1e18 supply shares
        uint256 token1PerSupplyShare; // token1 amount per 1e18 supply shares
        uint256 token0PerBorrowShare; // token0 amount per 1e18 borrow shares
        uint256 token1PerBorrowShare; // token1 amount per 1e18 borrow shares
    }

    struct ShiftData {
        uint256 oldUpper;
        uint256 oldLower;
        uint256 duration;
        uint256 startTimestamp;
        uint256 oldTime; // only for thresholdShift
    }

    struct CenterPriceShift {
        uint256 shiftPercentage;
        uint256 duration;
        uint256 startTimestamp;
    }

    struct ShiftChanges {
        bool isRangeChangeActive;
        bool isThresholdChangeActive;
        bool isCenterPriceShiftActive;
        ShiftData rangeShift;
        ShiftData thresholdShift;
        CenterPriceShift centerPriceShift;
    }

    struct Configs {
        bool isSmartCollateralEnabled;
        bool isSmartDebtEnabled;
        uint256 fee;
        uint256 revenueCut;
        uint256 upperRange;
        uint256 lowerRange;
        uint256 upperShiftThreshold;
        uint256 lowerShiftThreshold;
        uint256 shiftingTime;
        address centerPriceAddress;
        address hookAddress;
        uint256 maxCenterPrice;
        uint256 minCenterPrice;
        uint256 utilizationLimitToken0;
        uint256 utilizationLimitToken1;
        uint256 maxSupplyShares;
        uint256 maxBorrowShares;
    }

    // @dev note there might be other things that act as effective limits which are not fully considered here.
    // e.g. such as maximum 5% oracle shift in one swap, withdraws & borrowing together affecting each other,
    // shares being below max supply / borrow shares etc.
    struct SwapLimitsAndAvailability {
        // liquidity total amounts
        uint liquiditySupplyToken0;
        uint liquiditySupplyToken1;
        uint liquidityBorrowToken0;
        uint liquidityBorrowToken1;
        // liquidity limits
        uint liquidityWithdrawableToken0;
        uint liquidityWithdrawableToken1;
        uint liquidityBorrowableToken0;
        uint liquidityBorrowableToken1;
        // utilization limits based on config at Dex. (e.g. liquiditySupplyToken0 * Configs.utilizationLimitToken0 / 1e3)
        uint utilizationLimitToken0;
        uint utilizationLimitToken1;
        // swappable amounts until utilization limit.
        // In a swap that does both withdraw and borrow, the effective amounts might be less because withdraw / borrow affect each other
        // (both increase utilization).
        uint withdrawableUntilUtilizationLimitToken0; // x = totalSupply - totalBorrow / maxUtilizationPercentage
        uint withdrawableUntilUtilizationLimitToken1;
        uint borrowableUntilUtilizationLimitToken0; // x = maxUtilizationPercentage * totalSupply - totalBorrow.
        uint borrowableUntilUtilizationLimitToken1;
        // additional liquidity related data such as supply amount, limits, expansion etc.
        IFluidLiquidityResolverStructs.UserSupplyData liquidityUserSupplyDataToken0;
        IFluidLiquidityResolverStructs.UserSupplyData liquidityUserSupplyDataToken1;
        // additional liquidity related data such as borrow amount, limits, expansion etc.
        IFluidLiquidityResolverStructs.UserBorrowData liquidityUserBorrowDataToken0;
        IFluidLiquidityResolverStructs.UserBorrowData liquidityUserBorrowDataToken1;
        // liquidity token related data
        IFluidLiquidityResolverStructs.OverallTokenData liquidityTokenData0;
        IFluidLiquidityResolverStructs.OverallTokenData liquidityTokenData1;
    }

    struct DexEntireData {
        address dex;
        IFluidDexT1.ConstantViews constantViews;
        IFluidDexT1.ConstantViews2 constantViews2;
        Configs configs;
        IFluidDexT1.PricesAndExchangePrice pex;
        IFluidDexT1.CollateralReserves colReserves;
        IFluidDexT1.DebtReserves debtReserves;
        DexState dexState;
        SwapLimitsAndAvailability limitsAndAvailability;
    }

    // amounts are always in normal (for withInterest already multiplied with exchange price)
    struct UserSupplyData {
        bool isAllowed;
        uint256 supply; // user supply amount/shares
        // the withdrawal limit (e.g. if 10% is the limit, and 100M is supplied, it would be 90M)
        uint256 withdrawalLimit;
        uint256 lastUpdateTimestamp;
        uint256 expandPercent; // withdrawal limit expand percent in 1e2
        uint256 expandDuration; // withdrawal limit expand duration in seconds
        uint256 baseWithdrawalLimit;
        // the current actual max withdrawable amount (e.g. if 10% is the limit, and 100M is supplied, it would be 10M)
        uint256 withdrawableUntilLimit;
        uint256 withdrawable; // actual currently withdrawable amount (supply - withdrawal Limit) & considering balance
        // liquidity related data such as supply amount, limits, expansion etc.
        IFluidLiquidityResolverStructs.UserSupplyData liquidityUserSupplyDataToken0;
        IFluidLiquidityResolverStructs.UserSupplyData liquidityUserSupplyDataToken1;
        // liquidity token related data
        IFluidLiquidityResolverStructs.OverallTokenData liquidityTokenData0;
        IFluidLiquidityResolverStructs.OverallTokenData liquidityTokenData1;
    }

    // amounts are always in normal (for withInterest already multiplied with exchange price)
    struct UserBorrowData {
        bool isAllowed;
        uint256 borrow; // user borrow amount/shares
        uint256 borrowLimit;
        uint256 lastUpdateTimestamp;
        uint256 expandPercent;
        uint256 expandDuration;
        uint256 baseBorrowLimit;
        uint256 maxBorrowLimit;
        uint256 borrowableUntilLimit; // borrowable amount until any borrow limit (incl. max utilization limit)
        uint256 borrowable; // actual currently borrowable amount (borrow limit - already borrowed) & considering balance, max utilization
        // liquidity related data such as borrow amount, limits, expansion etc.
        IFluidLiquidityResolverStructs.UserBorrowData liquidityUserBorrowDataToken0;
        IFluidLiquidityResolverStructs.UserBorrowData liquidityUserBorrowDataToken1;
        // liquidity token related data
        IFluidLiquidityResolverStructs.OverallTokenData liquidityTokenData0;
        IFluidLiquidityResolverStructs.OverallTokenData liquidityTokenData1;
    }    

    /// @notice Get the entire data for a DEX
    /// @param dex_ The address of the DEX
    /// @return data_ A struct containing all the data for the DEX
    /// @dev expected to be called via callStatic
    function getDexEntireData(address dex_) external returns (DexEntireData memory data_);

    /// @notice Get the state of a DEX
    /// @param dex_ The address of the DEX
    /// @return state_ A struct containing the state of the DEX
    /// @dev expected to be called via callStatic
    function getDexState(address dex_) external returns (DexState memory state_);

    /// @dev Estimate deposit of tokens
    /// @param dex_ The address of the DEX contract
    /// @param token0Amt_ Amount of token0 to deposit
    /// @param token1Amt_ Amount of token1 to deposit
    /// @param minSharesAmt_ Minimum amount of shares to receive
    /// @return shares_ Estimated amount of shares to be minted
    function estimateDeposit(
        address dex_,
        uint token0Amt_,
        uint token1Amt_,
        uint minSharesAmt_
    ) external payable returns (uint shares_);

    /// @dev Estimate withdrawal of tokens
    /// @param dex_ The address of the DEX contract
    /// @param token0Amt_ Amount of token0 to withdraw
    /// @param token1Amt_ Amount of token1 to withdraw
    /// @param maxSharesAmt_ Maximum amount of shares to burn
    /// @return shares_ Estimated amount of shares to be burned
    function estimateWithdraw(
        address dex_,
        uint token0Amt_,
        uint token1Amt_,
        uint maxSharesAmt_
    ) external returns (uint shares_);

    /// @dev Estimate withdrawal of a perfect amount of collateral liquidity in one token
    /// @param dex_ The address of the DEX contract
    /// @param shares_ The number of shares to withdraw
    /// @param minToken0_ The minimum amount of token0 the user is willing to accept
    /// @param minToken1_ The minimum amount of token1 the user is willing to accept
    /// @return withdrawAmt_ Estimated amount of tokens to be withdrawn
    function estimateWithdrawPerfectInOneToken(
        address dex_,
        uint shares_,
        uint minToken0_,
        uint minToken1_
    ) external returns (uint withdrawAmt_);

    /// @dev Estimate borrowing of tokens
    /// @param dex_ The address of the DEX contract
    /// @param token0Amt_ Amount of token0 to borrow
    /// @param token1Amt_ Amount of token1 to borrow
    /// @param maxSharesAmt_ Maximum amount of shares to mint
    /// @return shares_ Estimated amount of shares to be minted
    function estimateBorrow(
        address dex_,
        uint token0Amt_,
        uint token1Amt_,
        uint maxSharesAmt_
    ) external returns (uint shares_);

    /// @dev Estimate paying back of borrowed tokens
    /// @param dex_ The address of the DEX contract
    /// @param token0Amt_ Amount of token0 to pay back
    /// @param token1Amt_ Amount of token1 to pay back
    /// @param minSharesAmt_ Minimum amount of shares to burn
    /// @return shares_ Estimated amount of shares to be burned
    function estimatePayback(
        address dex_,
        uint token0Amt_,
        uint token1Amt_,
        uint minSharesAmt_
    ) external payable returns (uint shares_);

    /// @dev Estimate paying back of a perfect amount of borrowed tokens in one token
    /// @param dex_ The address of the DEX contract
    /// @param shares_ The number of shares to pay back
    /// @param maxToken0_ Maximum amount of token0 to pay back
    /// @param maxToken1_ Maximum amount of token1 to pay back
    /// @return paybackAmt_ Estimated amount of tokens to be paid back
    function estimatePaybackPerfectInOneToken(
        address dex_,
        uint shares_,
        uint maxToken0_,
        uint maxToken1_
    ) external payable returns (uint paybackAmt_);
}








library FluidVaultTypes {

    error InvalidVaultType(uint256 vaultType);

    uint256 internal constant T1_VAULT_TYPE = 1e4; // 1_coll:1_debt
    uint256 internal constant T2_VAULT_TYPE = 2e4; // 2_coll:1_debt (smart coll)
    uint256 internal constant T3_VAULT_TYPE = 3e4; // 1_coll:2_debt (smart debt)
    uint256 internal constant T4_VAULT_TYPE = 4e4; // 2_coll:2_debt (smart coll, smart debt)

    function requireLiquidityCollateral(uint256 _vaultType) internal pure {
        if (_vaultType != T1_VAULT_TYPE && _vaultType != T3_VAULT_TYPE) {
            revert InvalidVaultType(_vaultType);
        }
    }

    function requireLiquidityDebt(uint256 _vaultType) internal pure {
        if (_vaultType != T1_VAULT_TYPE && _vaultType != T2_VAULT_TYPE) {
            revert InvalidVaultType(_vaultType);
        }
    }

    function requireSmartCollateral(uint256 _vaultType) internal pure {
        if (_vaultType != T2_VAULT_TYPE && _vaultType != T4_VAULT_TYPE) {
            revert InvalidVaultType(_vaultType);
        }
    }

    function requireSmartDebt(uint256 _vaultType) internal pure {
        if (_vaultType != T3_VAULT_TYPE && _vaultType != T4_VAULT_TYPE) {
            revert InvalidVaultType(_vaultType);
        }
    }

    function requireDexVault(uint256 _vaultType) internal pure {
        if (
            _vaultType != T2_VAULT_TYPE &&
            _vaultType != T3_VAULT_TYPE &&
            _vaultType != T4_VAULT_TYPE
        ) {
            revert InvalidVaultType(_vaultType);
        }
    }

    function isT1Vault(uint256 _vaultType) internal pure returns (bool) {
        return _vaultType == T1_VAULT_TYPE;
    }

    function isT2Vault(uint256 _vaultType) internal pure returns (bool) {
        return _vaultType == T2_VAULT_TYPE;
    }

    function isT3Vault(uint256 _vaultType) internal pure returns (bool) {
        return _vaultType == T3_VAULT_TYPE;
    }

    function isT4Vault(uint256 _vaultType) internal pure returns (bool) {
        return _vaultType == T4_VAULT_TYPE;
    }
}







interface IDexSmartCollOracle {
    
    /// @notice Returns number of quote tokens per 1e18 shares
    function dexSmartColSharesRates() external view returns (uint256 operate_, uint256 liquidate_);

    /// @dev Returns the configuration data of the DexSmartColOracle.
    ///
    /// @return dexPool_ The address of the Dex pool.
    /// @return reservesPegBufferPercent_ The percentage of the reserves peg buffer.
    /// @return liquidity_ The address of the liquidity contract.
    /// @return token0NumeratorPrecision_ The precision of the numerator for token0.
    /// @return token0DenominatorPrecision_ The precision of the denominator for token0.
    /// @return token1NumeratorPrecision_ The precision of the numerator for token1.
    /// @return token1DenominatorPrecision_ The precision of the denominator for token1.
    /// @return reservesConversionOracle_ The address of the reserves conversion oracle.
    /// @return reservesConversionInvert_ A boolean indicating if reserves conversion should be inverted.
    /// @return quoteInToken0_ A boolean indicating if the quote is in token0.
    function dexSmartColOracleData()
        external
        view
        returns (
            address dexPool_,
            uint256 reservesPegBufferPercent_,
            address liquidity_,
            uint256 token0NumeratorPrecision_,
            uint256 token0DenominatorPrecision_,
            uint256 token1NumeratorPrecision_,
            uint256 token1DenominatorPrecision_,
            address reservesConversionOracle_,
            bool reservesConversionInvert_,
            bool quoteInToken0_
        );

    /// @dev USED FOR NEWER DEPLOYMENTS
    /// @notice Returns the base configuration data of the FluidDexOracle.
    ///
    /// @return dexPool_ The address of the Dex pool.
    /// @return quoteInToken0_ A boolean indicating if the quote is in token0.
    /// @return liquidity_ The address of liquidity layer.
    /// @return resultMultiplier_ The result multiplier.
    /// @return resultDivisor_ The result divisor.
    function dexOracleData()
        external
        view
        returns (
            address dexPool_,
            bool quoteInToken0_,
            address liquidity_,
            uint256 resultMultiplier_,
            uint256 resultDivisor_
        );
}







interface IDexSmartDebtOracle {
    
    /// @notice Returns number of quote tokens per 1e18 shares
    function dexSmartDebtSharesRates() external view returns (uint256 operate_, uint256 liquidate_);
    
    /// @dev Returns the configuration data of the DexSmartDebtOracle.
    ///
    /// @return dexPool_ The address of the Dex pool.
    /// @return reservesPegBufferPercent_ The percentage of the reserves peg buffer.
    /// @return liquidity_ The address of the liquidity contract.
    /// @return token0NumeratorPrecision_ The precision of the numerator for token0.
    /// @return token0DenominatorPrecision_ The precision of the denominator for token0.
    /// @return token1NumeratorPrecision_ The precision of the numerator for token1.
    /// @return token1DenominatorPrecision_ The precision of the denominator for token1.
    /// @return reservesConversionOracle_ The address of the reserves conversion oracle.
    /// @return reservesConversionInvert_ A boolean indicating if reserves conversion should be inverted.
    /// @return quoteInToken0_ A boolean indicating if the quote is in token0.
    function dexSmartDebtOracleData()
        external
        view
        returns (
            address dexPool_,
            uint256 reservesPegBufferPercent_,
            address liquidity_,
            uint256 token0NumeratorPrecision_,
            uint256 token0DenominatorPrecision_,
            uint256 token1NumeratorPrecision_,
            uint256 token1DenominatorPrecision_,
            address reservesConversionOracle_,
            bool reservesConversionInvert_,
            bool quoteInToken0_
        );

    /// @dev USED FOR NEWER DEPLOYMENTS
    /// @notice Returns the base configuration data of the FluidDexOracle.
    ///
    /// @return dexPool_ The address of the Dex pool.
    /// @return quoteInToken0_ A boolean indicating if the quote is in token0.
    /// @return liquidity_ The address of liquidity layer.
    /// @return resultMultiplier_ The result multiplier.
    /// @return resultDivisor_ The result divisor.
    function dexOracleData()
        external
        view
        returns (
            address dexPool_,
            bool quoteInToken0_,
            address liquidity_,
            uint256 resultMultiplier_,
            uint256 resultDivisor_
        );

    /// @notice Returns Col/Debt Oracle data. Used in T4 vaults
    function getDexColDebtOracleData() external view returns (address colDebtOracle_, bool colDebtInvert_);
}







interface IFluidLendingResolver {

    // amounts are always in normal (for withInterest already multiplied with exchange price)
    struct UserSupplyData {
        bool modeWithInterest; // true if mode = with interest, false = without interest
        uint256 supply; // user supply amount
        // the withdrawal limit (e.g. if 10% is the limit, and 100M is supplied, it would be 90M)
        uint256 withdrawalLimit;
        uint256 lastUpdateTimestamp;
        uint256 expandPercent; // withdrawal limit expand percent in 1e2
        uint256 expandDuration; // withdrawal limit expand duration in seconds
        uint256 baseWithdrawalLimit;
        // the current actual max withdrawable amount (e.g. if 10% is the limit, and 100M is supplied, it would be 10M)
        uint256 withdrawableUntilLimit;
        uint256 withdrawable; // actual currently withdrawable amount (supply - withdrawal Limit) & considering balance
    }

    struct FTokenDetails {
        address tokenAddress;
        bool eip2612Deposits;
        bool isNativeUnderlying;
        string name;
        string symbol;
        uint256 decimals;
        address asset;
        uint256 totalAssets;
        uint256 totalSupply;
        uint256 convertToShares;
        uint256 convertToAssets;
        // additional yield from rewards, if active
        uint256 rewardsRate;
        // yield at Liquidity
        uint256 supplyRate;
        // difference between fToken assets & actual deposit at Liquidity. (supplyAtLiquidity - totalAssets).
        // if negative, rewards must be funded to guarantee withdrawal is possible for all users. This happens
        // by executing rebalance().
        int256 rebalanceDifference;
        // liquidity related data such as supply amount, limits, expansion etc.
        UserSupplyData liquidityUserSupplyData;
    }

    struct UserPosition {
        uint256 fTokenShares;
        uint256 underlyingAssets;
        uint256 underlyingBalance;
        uint256 allowance;
    }

    struct FTokenDetailsUserPosition {
        FTokenDetails fTokenDetails;
        UserPosition userPosition;
    }

    /// @notice returns the lending factory address
    function LENDING_FACTORY() external view returns (address);

    /// @notice returns the liquidity resolver address
    function LIQUIDITY_RESOLVER() external view returns (address);

    /// @notice returns all fToken types at the `LENDING_FACTORY`
    function getAllFTokenTypes() external view returns (string[] memory);

    /// @notice returns all created fTokens at the `LENDING_FACTORY`
    function getAllFTokens() external view returns (address[] memory);

    /// @notice reads if a certain `auth_` address is an allowed auth or not. Owner is auth by default.
    function isLendingFactoryAuth(address auth_) external view returns (bool);

    /// @notice reads if a certain `deployer_` address is an allowed deployer or not. Owner is deployer by default.
    function isLendingFactoryDeployer(address deployer_) external view returns (bool);

    /// @notice computes deterministic token address for `asset_` for a lending protocol
    /// @param  asset_      address of the asset
    /// @param  fTokenType_         type of fToken:
    /// - if underlying asset supports EIP-2612, the fToken should be type `EIP2612Deposits`
    /// - otherwise it should use `Permit2Deposits`
    /// - if it's the native token, it should use `NativeUnderlying`
    /// - could be more types available, check `fTokenTypes()`
    /// @return token_      detemrinistic address of the computed token
    function computeFToken(address asset_, string calldata fTokenType_) external view returns (address);

    /// @notice gets all public details for a certain `fToken_`, such as
    /// fToken type, name, symbol, decimals, underlying asset, total amounts, convertTo values, rewards.
    /// Note it also returns whether the fToken supports deposits / mints via EIP-2612, but it is not a 100% guarantee!
    /// To make sure, check for the underlying if it supports EIP-2612 manually.
    /// @param  fToken_     the fToken to get the details for
    /// @return fTokenDetails_  retrieved FTokenDetails struct
    function getFTokenDetails(address fToken_) external view returns (FTokenDetails memory fTokenDetails_);

    /// @notice returns config, rewards and exchange prices data of an fToken.
    /// @param  fToken_ the fToken to get the data for
    /// @return liquidity_ address of the Liquidity contract.
    /// @return lendingFactory_ address of the Lending factory contract.
    /// @return lendingRewardsRateModel_ address of the rewards rate model contract. changeable by LendingFactory auths.
    /// @return permit2_ address of the Permit2 contract used for deposits / mint with signature
    /// @return rebalancer_ address of the rebalancer allowed to execute `rebalance()`
    /// @return rewardsActive_ true if rewards are currently active
    /// @return liquidityBalance_ current Liquidity supply balance of `address(this)` for the underyling asset
    /// @return liquidityExchangePrice_ (updated) exchange price for the underlying assset in the liquidity protocol (without rewards)
    /// @return tokenExchangePrice_ (updated) exchange price between fToken and the underlying assset (with rewards)
    function getFTokenInternalData(
        address fToken_
    )
        external
        view
        returns (
            address liquidity_,
            address lendingFactory_,
            address lendingRewardsRateModel_,
            address permit2_,
            address rebalancer_,
            bool rewardsActive_,
            uint256 liquidityBalance_,
            uint256 liquidityExchangePrice_,
            uint256 tokenExchangePrice_
        );

    /// @notice gets all public details for all itokens, such as
    /// fToken type, name, symbol, decimals, underlying asset, total amounts, convertTo values, rewards
    function getFTokensEntireData() external view returns (FTokenDetails[] memory);

    /// @notice gets all public details for all itokens, such as
    /// fToken type, name, symbol, decimals, underlying asset, total amounts, convertTo values, rewards
    /// and user position for each token
    function getUserPositions(address user_) external view returns (FTokenDetailsUserPosition[] memory);

    /// @notice gets rewards related data: the `rewardsRateModel_` contract and the current `rewardsRate_` for the `fToken_`
    function getFTokenRewards(
        address fToken_
    ) external view returns (address rewardsRateModel_, uint256 rewardsRate_);

    /// @notice gets rewards rate model config constants
    function getFTokenRewardsRateModelConfig(
        address fToken_
    )
        external
        view
        returns (
            uint256 duration_,
            uint256 startTime_,
            uint256 endTime_,
            uint256 startTvl_,
            uint256 maxRate_,
            uint256 rewardAmount_,
            address initiator_
        );

    /// @notice gets a `user_` position for an `fToken_`.
    /// @return userPosition user position struct
    function getUserPosition(
        address fToken_,
        address user_
    ) external view returns (UserPosition memory userPosition);

    /// @notice gets `fToken_` preview amounts for `assets_` or `shares_`.
    /// @return previewDeposit_ preview for deposit of `assets_`
    /// @return previewMint_ preview for mint of `shares_`
    /// @return previewWithdraw_ preview for withdraw of `assets_`
    /// @return previewRedeem_ preview for redeem of `shares_`
    function getPreviews(
        address fToken_,
        uint256 assets_,
        uint256 shares_
    )
        external
        view
        returns (uint256 previewDeposit_, uint256 previewMint_, uint256 previewWithdraw_, uint256 previewRedeem_);
}







interface IFluidVault {

    /// @notice emitted when an operate() method is executed that changes collateral (`colAmt_`) / debt (debtAmt_`)
    /// amount for a `user_` position with `nftId_`. Receiver of any funds is the address `to_`.
    event LogOperate(address user_, uint256 nftId_, int256 colAmt_, int256 debtAmt_, address to_);

    struct Tokens {
        address token0;
        address token1;
    }

    struct ConstantViews {
        address liquidity;
        address factory;
        address operateImplementation;
        address adminImplementation;
        address secondaryImplementation;
        address deployer; // address which deploys oracle
        address supply; // either liquidity layer or DEX protocol
        address borrow; // either liquidity layer or DEX protocol
        Tokens supplyToken; // if smart collateral then address of token0 & token1 else just supply token address at token0 and token1 as empty
        Tokens borrowToken; // if smart debt then address of token0 & token1 else just borrow token address at token0 and token1 as empty
        uint256 vaultId;
        uint256 vaultType;
        bytes32 supplyExchangePriceSlot; // if smart collateral then slot is from DEX protocol else from liquidity layer
        bytes32 borrowExchangePriceSlot; // if smart debt then slot is from DEX protocol else from liquidity layer
        bytes32 userSupplySlot; // if smart collateral then slot is from DEX protocol else from liquidity layer
        bytes32 userBorrowSlot; // if smart debt then slot is from DEX protocol else from liquidity layer
    }

    // @notice returns all Vault constants
    function constantsView() external view returns (ConstantViews memory constantsView_);
}







interface IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint256 digits);
    function totalSupply() external view returns (uint256 supply);

    function balanceOf(address _owner) external view returns (uint256 balance);

    function transfer(address _to, uint256 _value) external returns (bool success);

    function transferFrom(
        address _from,
        address _to,
        uint256 _value
    ) external returns (bool success);

    function approve(address _spender, uint256 _value) external returns (bool success);

    function allowance(address _owner, address _spender) external view returns (uint256 remaining);

    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}

















contract FluidView is FluidRatioHelper {
    using FluidVaultTypes for uint256;


    /**
     *
     *                         DATA SPECIFICATION
     *
     */
    /// @notice User position data
    struct UserPosition {
        uint256 nftId; // unique id of the position
        address owner; // owner of the position
        bool isLiquidated; // true if the position is liquidated
        bool isSupplyPosition; // true if the position is a supply position, means no debt.
        uint256 supply; // amount of supply tokens. For smart collateral vaults, this will be the amount of coll shares.
        uint256 borrow; // amount of borrow tokens. For smart debt vaults, this will be the amount of debt shares.
        uint256 ratio; // ratio of the position in 1e18
        int256 tick; // in which tick the position is. Used to calculate the ratio and borrow amount.
        uint256 tickId; // tick id of the position
    }

    /// @notice Data for the supply dex pool used in T2 and T4 vaults
    struct DexSupplyData {
        address dexPool; // address of the dex pool
        uint256 dexId; // id of the dex pool
        uint256 fee; // fee of the dex pool
        uint256 lastStoredPrice; // last stored price of the dex pool
        uint256 centerPrice; // center price of the dex pool
        uint256 token0Utilization; // token0 utilization
        uint256 token1Utilization; // token1 utilization
        // ONLY FOR SUPPLY
        uint256 totalSupplyShares; // total supply shares, in 1e18
        uint256 maxSupplyShares; // max supply shares, in 1e18
        uint256 token0Supplied; // token0 supplied, in token0 decimals
        uint256 token1Supplied; // token1 supplied, in token1 decimals
        uint256 sharesWithdrawable; // shares withdrawable, in 1e18
        uint256 token0Withdrawable; // token0 withdrawable, in token0 decimals
        uint256 token1Withdrawable; // token1 withdrawable, in token1 decimals
        uint256 token0PerSupplyShare; // token0 amount per 1e18 supply shares
        uint256 token1PerSupplyShare; // token1 amount per 1e18 supply shares
        uint256 token0SupplyRate; // token0 supply rate. E.g 320 = 3.2% APR
        uint256 token1SupplyRate; // token1 supply rate. E.g 320 = 3.2% APR
        address quoteToken; // quote token used in dex oracle. Either token0 or token1
        uint256 quoteTokensPerShare; // quote tokens per 1e18 shares (all reserves are converted to quote token).
        uint256 supplyToken0Reserves; // supply token0 reserves inside dex
        uint256 supplyToken1Reserves; // supply token1 reserves inside dex
    }

    /// @notice Data for the borrow dex pool used in T3 and T4 vaults
    struct DexBorrowData {
        address dexPool; // address of the dex pool
        uint256 dexId; // id of the dex pool
        uint256 fee; // fee of the dex pool
        uint256 lastStoredPrice; // last stored price of the dex pool
        uint256 centerPrice; // center price of the dex pool
        uint256 token0Utilization; // token0 utilization
        uint256 token1Utilization; // token1 utilization
        // ONLY FOR BORROW
        uint256 totalBorrowShares; // total borrow shares, in 1e18
        uint256 maxBorrowShares; // max borrow shares, in 1e18
        uint256 token0Borrowed; // token0 borrowed, in token0 decimals
        uint256 token1Borrowed; // token1 borrowed, in token1 decimals
        uint256 sharesBorrowable; // shares borrowable in 1e18
        uint256 token0Borrowable; // token0 borrowable in token0 decimals
        uint256 token1Borrowable; // token1 borrowable in token1 decimals
        uint256 token0PerBorrowShare; // token0 amount per 1e18 borrow shares
        uint256 token1PerBorrowShare; // token1 amount per 1e18 borrow shares
        uint256 token0BorrowRate; // token0 borrow rate. E.g 320 = 3.2% APR
        uint256 token1BorrowRate; // token1 borrow rate. E.g 320 = 3.2% APR
        address quoteToken; // quote token used in dex oracle. Either token0 or token1
        uint256 quoteTokensPerShare; // quote tokens per 1e18 shares (all reserves are converted to quote token).
        uint256 borrowToken0Reserves; // borrow token0 reserves inside dex
        uint256 borrowToken1Reserves; // borrow token1 reserves inside dex
    }

    /// @notice Full vault data including dex data.
    /// @dev This data is obtained by combining calls to FluidVaultResolver and FluidDexResolver.
    struct VaultData {
        address vault; // address of the vault
        uint256 vaultId; // unique id of the vault
        uint256 vaultType; // 10000 = Vault(1 coll / 1 debt), 20000 = 2/1, 30000 = 1/2, 40000 = 2/2
        bool isSmartColl; // smart collateral vaults have 2 tokens as collateral
        bool isSmartDebt; // smart debt vaults have 2 tokens as debt
        address supplyToken0; // always present
        address supplyToken1; // only used for smart collateral vaults
        address borrowToken0; // always present
        address borrowToken1; // only used for smart debt vaults
        uint256 supplyToken0Decimals; // decimals of the collateral token 0
        uint256 supplyToken1Decimals; // decimals of the collateral token 1. 0 if not present
        uint256 borrowToken0Decimals; // decimals of the debt token 0
        uint256 borrowToken1Decimals; // decimals of the debt token 1. 0 if not present
        uint16 collateralFactor; // e.g 8500 = 85%
        uint16 liquidationThreshold; // e.g 9000 = 90%
        uint16 liquidationMaxLimit;  // LML is the threshold above which 100% of your position gets liquidated instantly
        uint16 withdrawalGap; // Safety non-withdrawable amount to guarantee liquidations. E.g 500 = 5%
        uint16 liquidationPenalty; // e.g 100 = 1%, 500 = 5%
        uint16 borrowFee; // if there is any additional fee for borrowing.
        address oracle; // address of the oracle
        uint256 oraclePriceOperate; // price of the oracle (Called during operations)
        uint256 oraclePriceLiquidate; // price of the oracle (If liquidation requires different price)
        uint256 vaultSupplyExchangePrice; // vault supply exchange price.
        uint256 vaultBorrowExchangePrice; // vault borrow exchange price.
        int256 supplyRateVault; // supply rate of the vault
        int256 borrowRateVault; // borrow rate of the vault
        int256 rewardsOrFeeRateSupply; // rewards or fee rate for supply
        int256 rewardsOrFeeRateBorrow; // rewards or fee rate for borrow
        uint256 totalPositions; // Total positions in the vault
        uint256 totalSupplyVault; // Total supplied assets to the vault
        uint256 totalBorrowVault; // Total borrowed assets from the vault
        uint256 withdrawalLimit; // The limit until where you can withdraw. If 0, all users can withdraw
        uint256 withdrawableUntilLimit; // The amount that can be withdrawn until the limit
        uint256 withdrawable; // min(supply - withdrawalGap - currentLimit, availableBalance)
        uint256 baseWithdrawalLimit; // The minimum limit for a vault's withdraw. The further expansion happens on this base
        uint256 withdrawExpandPercent; // The rate at which limits would increase or decrease over the given duration. E.g 2500 = 25%
        uint256 withdrawExpandDuration; // The time for which the limits expand at the given rate (in seconds)
        uint256 borrowLimit; // The limit until where user can borrow
        uint256 borrowableUntilLimit; // borrowable amount until any borrow limit (incl. max utilization limit)
        uint256 borrowable; // min(currentLimit - borrow, borrowableMaxUtilization - borrow, availableBalance)
        uint256 borrowLimitUtilization;  // Total borrow limit for the maximum allowed utilization
        uint256 maxBorrowLimit; // The maximum limit for a vault above which it is not possible to borrow
        uint256 borrowExpandPercent; // The rate at which limits would increase or decrease over the given duration. E.g 2500 = 25%
        uint256 borrowExpandDuration; // The time for which the limits expand at the given rate (in seconds)
        uint256 baseBorrowLimit; // The minimum limit for a vault's borrow. The further expansion happens on this base
        uint256 minimumBorrowing; // The minimum amount that can be borrowed from the vault
        DexSupplyData dexSupplyData; // Dex pool supply data. Used only for T2 and T4 vaults
        DexBorrowData dexBorrowData; // Dex pool borrow data. Used only for T3 and T4 vaults
    }

    /// @notice Helper struct to group nftId, vaultId and vault address
    struct NftWithVault {
        uint256 nftId; // unique id of the position
        uint256 vaultId; // unique id of the vault
        address vaultAddr; // address of the vault
    }

    /// @notice User earn position data
    struct UserEarnPosition {
        uint256 fTokenShares; // amount of fToken shares
        uint256 underlyingAssets; // amount of underlying assets
        uint256 underlyingBalance; // amount of underlying assets in the user's wallet
        uint256 allowance; // amount of allowance for the user to spend
    }

    /// @notice FToken data for a specific fToken address used for 'earn' positions
    struct FTokenData {
        address tokenAddress; // address of the fToken
        bool isNativeUnderlying; // true if the underlying asset is native to the chain
        string name; // name of the fToken
        string symbol; // symbol of the fToken
        uint256 decimals; // decimals of the fToken
        address asset; // address of the underlying asset
        uint256 totalAssets; // total amount of underlying assets
        uint256 totalSupply; // total amount of fToken shares
        uint256 convertToShares; // convert amount of underlying assets to fToken shares
        uint256 convertToAssets; // convert amount of fToken shares to underlying assets
        uint256 rewardsRate; // additional yield from rewards, if active
        uint256 supplyRate; // yield at Liquidity
        uint256 withdrawable; // actual currently withdrawable amount (supply - withdrawal Limit) & considering balance
        bool modeWithInterest; // true if mode = with interest, false = without interest
        uint256 expandPercent; // withdrawal limit expand percent in 1e2
        uint256 expandDuration; // withdrawal limit expand duration in seconds
    }

    /**
     *
     *                         EXTERNAL FUNCTIONS
     *
     */
    /// @notice Get all user positions with vault data.
    /// @dev This should be called with static call.
    function getUserPositions(address _user) 
        external returns (UserPosition[] memory positions, VaultData[] memory vaults) 
    {
        uint256[] memory nftIds = IFluidVaultResolver(FLUID_VAULT_RESOLVER).positionsNftIdOfUser(_user);

        positions = new UserPosition[](nftIds.length);
        vaults = new VaultData[](nftIds.length);

        for (uint256 i = 0; i < nftIds.length; i++) {
            (positions[i], vaults[i]) = getPositionByNftId(nftIds[i]);
        }

        return (positions, vaults);
    }

    /// @notice Get all nftIds for a specific user
    function getUserNftIds(address _user) external view returns (uint256[] memory) {
        return IFluidVaultResolver(FLUID_VAULT_RESOLVER).positionsNftIdOfUser(_user);
    }

    /// @notice Get all nftIds with vaultIds for a specific user
    function getUserNftIdsWithVaultIds(address _user) external view returns (NftWithVault[] memory retVal) {
        uint256[] memory nftIds = IFluidVaultResolver(FLUID_VAULT_RESOLVER).positionsNftIdOfUser(_user);
        retVal = new NftWithVault[](nftIds.length);

        for (uint256 i = 0; i < nftIds.length; i++) {
            address vaultByNft = IFluidVaultResolver(FLUID_VAULT_RESOLVER).vaultByNftId(nftIds[i]);
            uint256 vaultId = IFluidVaultResolver(FLUID_VAULT_RESOLVER).getVaultId(vaultByNft);

            retVal[i] = NftWithVault({
                nftId: nftIds[i],
                vaultId: vaultId,
                vaultAddr: vaultByNft
            });
        }
    }

    /// @notice Get position data with vault and dex data for a specific nftId
    /// @dev This should be called with static call.
    function getPositionByNftId(uint256 _nftId) public returns (UserPosition memory position, VaultData memory vault) {
        (
            IFluidVaultResolver.UserPosition memory userPosition,
            IFluidVaultResolver.VaultEntireData memory vaultData
        ) = IFluidVaultResolver(FLUID_VAULT_RESOLVER).positionByNftId(_nftId);

        position = UserPosition({
            nftId: userPosition.nftId,
            owner: userPosition.owner,
            isLiquidated: userPosition.isLiquidated,
            isSupplyPosition: userPosition.isSupplyPosition,
            supply: userPosition.supply,
            borrow: userPosition.borrow,
            ratio: getRatio(userPosition.nftId),
            tick: userPosition.tick,
            tickId: userPosition.tickId
        });

        vault = getVaultData(vaultData.vault);
    }

    /// @notice Get vault data for a specific vault address. This also includes dex data.
    /// @dev This should be called with static call.
    function getVaultData(address _vault) public returns (VaultData memory vaultData) {
        IFluidVaultResolver.VaultEntireData memory data = 
            IFluidVaultResolver(FLUID_VAULT_RESOLVER).getVaultEntireData(_vault);

        address supplyToken0 = data.constantVariables.supplyToken.token0;
        address supplyToken1 = data.constantVariables.supplyToken.token1;
        address borrowToken0 = data.constantVariables.borrowToken.token0;
        address borrowToken1 = data.constantVariables.borrowToken.token1;

        DexSupplyData memory dexSupplyData;
        DexBorrowData memory dexBorrowData;

        vaultData = VaultData({
            vault: _vault,
            vaultId: data.constantVariables.vaultId,
            vaultType: data.constantVariables.vaultType,
            isSmartColl: data.isSmartCol,
            isSmartDebt: data.isSmartDebt,
            
            supplyToken0: supplyToken0,
            supplyToken1: supplyToken1,
            borrowToken0: borrowToken0,
            borrowToken1: borrowToken1,

            supplyToken0Decimals: supplyToken0 != ETH_ADDR ? IERC20(supplyToken0).decimals() : 18,
            supplyToken1Decimals: supplyToken1 != address(0) ? (supplyToken1 != ETH_ADDR ? IERC20(supplyToken1).decimals() : 18) : 0,
            borrowToken0Decimals: borrowToken0 != ETH_ADDR ? IERC20(borrowToken0).decimals(): 18,
            borrowToken1Decimals: borrowToken1 != address(0) ? (borrowToken1 != ETH_ADDR ? IERC20(borrowToken1).decimals() : 18) : 0,

            collateralFactor: data.configs.collateralFactor,
            liquidationThreshold: data.configs.liquidationThreshold,
            liquidationMaxLimit: data.configs.liquidationMaxLimit,
            withdrawalGap: data.configs.withdrawalGap,
            liquidationPenalty: data.configs.liquidationPenalty,
            borrowFee: data.configs.borrowFee,
            oracle: data.configs.oracle,
            oraclePriceOperate: data.configs.oraclePriceOperate,
            oraclePriceLiquidate: data.configs.oraclePriceLiquidate,

            vaultSupplyExchangePrice: data.exchangePricesAndRates.vaultSupplyExchangePrice,
            vaultBorrowExchangePrice: data.exchangePricesAndRates.vaultBorrowExchangePrice,
            supplyRateVault: data.exchangePricesAndRates.supplyRateVault,
            borrowRateVault: data.exchangePricesAndRates.borrowRateVault,
            rewardsOrFeeRateSupply: data.exchangePricesAndRates.rewardsOrFeeRateSupply,
            rewardsOrFeeRateBorrow: data.exchangePricesAndRates.rewardsOrFeeRateBorrow,

            totalPositions: data.vaultState.totalPositions,

            totalSupplyVault: data.totalSupplyAndBorrow.totalSupplyVault,
            totalBorrowVault: data.totalSupplyAndBorrow.totalBorrowVault,

            withdrawalLimit: data.liquidityUserSupplyData.withdrawalLimit,
            withdrawableUntilLimit: data.liquidityUserSupplyData.withdrawableUntilLimit,
            withdrawable: data.liquidityUserSupplyData.withdrawable,
            baseWithdrawalLimit: data.liquidityUserSupplyData.baseWithdrawalLimit,
            withdrawExpandPercent: data.liquidityUserSupplyData.expandPercent,
            withdrawExpandDuration: data.liquidityUserSupplyData.expandDuration,

            borrowLimit: data.liquidityUserBorrowData.borrowLimit,
            borrowableUntilLimit: data.liquidityUserBorrowData.borrowableUntilLimit,
            borrowable: data.liquidityUserBorrowData.borrowable,
            borrowLimitUtilization: data.liquidityUserBorrowData.borrowLimitUtilization,
            maxBorrowLimit: data.liquidityUserBorrowData.maxBorrowLimit,
            borrowExpandPercent: data.liquidityUserBorrowData.expandPercent,
            borrowExpandDuration: data.liquidityUserBorrowData.expandDuration,
            baseBorrowLimit: data.liquidityUserBorrowData.baseBorrowLimit,

            minimumBorrowing: data.limitsAndAvailability.minimumBorrowing,

            dexSupplyData: dexSupplyData,
            dexBorrowData: dexBorrowData
        });

        // smart coll
        if (vaultData.vaultType.isT2Vault()) {
            IFluidDexResolver.DexEntireData memory dexData =
                IFluidDexResolver(FLUID_DEX_RESOLVER).getDexEntireData(data.constantVariables.supply);
            vaultData.dexSupplyData = _fillDexSupplyData(dexData, vaultData.oracle, vaultData.withdrawable);
        }

        // smart debt
        if (vaultData.vaultType.isT3Vault()) {
            IFluidDexResolver.DexEntireData memory dexData =
                IFluidDexResolver(FLUID_DEX_RESOLVER).getDexEntireData(data.constantVariables.borrow);
            vaultData.dexBorrowData = _fillDexBorrowData(dexData, vaultData.oracle, vaultData.borrowable);
        }

        // smart coll and smart debt
        if (vaultData.vaultType.isT4Vault()) {
            IFluidDexResolver.DexEntireData memory dexData =
                IFluidDexResolver(FLUID_DEX_RESOLVER).getDexEntireData(data.constantVariables.supply);

            vaultData.dexSupplyData = _fillDexSupplyData(
                dexData,
                _getSmartCollateralDexOracle(vaultData.oracle),
                vaultData.withdrawable
            );

            // if it's a same dex, no need to fetch again
            if (data.constantVariables.borrow == data.constantVariables.supply) {
                vaultData.dexBorrowData = _fillDexBorrowData(dexData, vaultData.oracle, vaultData.borrowable);
            } else {
                dexData = IFluidDexResolver(FLUID_DEX_RESOLVER).getDexEntireData(data.constantVariables.borrow);
                vaultData.dexBorrowData = _fillDexBorrowData(dexData, vaultData.oracle, vaultData.borrowable);
            }

            // In the case of T4 vaults, quoteTokensPerShare is actually returned as shareTokensPerQuote, so we invert it here.
            vaultData.dexBorrowData.quoteTokensPerShare = 1e54 / vaultData.dexBorrowData.quoteTokensPerShare;
        }
    }

    /// @notice Get current share rates for supply and borrow dex inside the vault
    /// @dev This should be called with static call.
    /// @dev Function will revert for T1 vaults and is expected to be called only for dex vaults
    /// @param _vault Address of the vault
    /// @return token0PerSupplyShare - filed for T2 and T4 vaults
    /// @return token1PerSupplyShare - filed for T2 and T4 vaults
    /// @return token0PerBorrowShare - filed for T3 and T4 vaults
    /// @return token1PerBorrowShare - filed fro T3 and T4 vaults
    function getDexShareRates(
        address _vault
    ) external returns (
        uint256 token0PerSupplyShare,
        uint256 token1PerSupplyShare,
        uint256 token0PerBorrowShare,
        uint256 token1PerBorrowShare
    ) {
        // Reverts for T1 vaults
        IFluidVault.ConstantViews memory vaultData = IFluidVault(_vault).constantsView();

        if (vaultData.vaultType.isT2Vault()) {
            IFluidDexResolver.DexState memory dexData = IFluidDexResolver(FLUID_DEX_RESOLVER).getDexState(vaultData.supply);
            token0PerSupplyShare = dexData.token0PerSupplyShare;
            token1PerSupplyShare = dexData.token1PerSupplyShare;
        }

        if (vaultData.vaultType.isT3Vault()) {
            IFluidDexResolver.DexState memory dexData = IFluidDexResolver(FLUID_DEX_RESOLVER).getDexState(vaultData.borrow);
            token0PerBorrowShare = dexData.token0PerBorrowShare;
            token1PerBorrowShare = dexData.token1PerBorrowShare;
        }

        if (vaultData.vaultType.isT4Vault()) {
            IFluidDexResolver.DexState memory dexData = IFluidDexResolver(FLUID_DEX_RESOLVER).getDexState(vaultData.supply);
            token0PerSupplyShare = dexData.token0PerSupplyShare;
            token1PerSupplyShare = dexData.token1PerSupplyShare;

            if (vaultData.borrow == vaultData.supply) {
                token0PerBorrowShare = dexData.token0PerBorrowShare;
                token1PerBorrowShare = dexData.token1PerBorrowShare;
            } else {
                dexData = IFluidDexResolver(FLUID_DEX_RESOLVER).getDexState(vaultData.borrow);
                token0PerBorrowShare = dexData.token0PerBorrowShare;
                token1PerBorrowShare = dexData.token1PerBorrowShare;
            }
        }
    }

    /// @notice Estimate how much shares will be received for a variable dex deposit for T2 and T4 vaults
    /// @dev This should be called with static call.
    /// @param _vault Address of the vault
    /// @param _token0Amount Amount of token0 to deposit
    /// @param _token1Amount Amount of token1 to deposit
    /// @param _minSharesAmount Minimum amount of shares to receive
    ///         This value can be set to low value like 1 to just check for minted shares.
    ///         However, it can also be used to check if transaction will revert when sending this amount of shares (Slippage check)
    /// @return shares Amount of shares received
    function estimateDeposit(
        address _vault,
        uint256 _token0Amount,
        uint256 _token1Amount,
        uint256 _minSharesAmount
    ) external payable returns (uint256 shares) {
        IFluidVault.ConstantViews memory constants = IFluidVault(_vault).constantsView();

        shares = IFluidDexResolver(FLUID_DEX_RESOLVER).estimateDeposit(
            constants.supply,
            _token0Amount,
            _token1Amount,
            _minSharesAmount
        );
    }

    /// @notice Estimate how much deposit shares will be burned for a variable dex withdraw for T2 and T4 vaults
    /// @dev This should be called with static call.
    /// @param _vault Address of the vault
    /// @param _token0Amount Amount of token0 to withdraw
    /// @param _token1Amount Amount of token1 to withdraw
    /// @param _maxSharesAmount Maximum amount of shares to withdraw
    ///        This value can be set to high value like type(int256).max to just check for burned shares.
    ///        However, it can also be used to check if transaction will revert when sending this amount of shares (Slippage check)
    /// @return shares Amount of shares burned
    function estimateWithdraw(
        address _vault,
        uint256 _token0Amount,
        uint256 _token1Amount,
        uint256 _maxSharesAmount
    ) external returns (uint256 shares) {
        IFluidVault.ConstantViews memory constants = IFluidVault(_vault).constantsView();

        shares = IFluidDexResolver(FLUID_DEX_RESOLVER).estimateWithdraw(
            constants.supply,
            _token0Amount,
            _token1Amount,
            _maxSharesAmount
        );
    }

    /// @notice Estimate how much debt shares will be received for a variable dex borrow for T3 and T4 vaults
    /// @dev This should be called with static call.
    /// @param _vault Address of the vault
    /// @param _token0Amount Amount of token0 to borrow
    /// @param _token1Amount Amount of token1 to borrow
    /// @param _maxSharesAmount Maximum amount of shares to borrow
    ///        This value can be set to high value like type(int256).max to just check for minted shares.
    ///        However, it can also be used to check if transaction will revert when sending this amount of shares (Slippage check)
    /// @return shares Amount of shares received
    function estimateBorrow(
        address _vault,
        uint256 _token0Amount,
        uint256 _token1Amount,
        uint256 _maxSharesAmount
    ) external returns (uint256 shares) {
        IFluidVault.ConstantViews memory constants = IFluidVault(_vault).constantsView();

        shares = IFluidDexResolver(FLUID_DEX_RESOLVER).estimateBorrow(  
            constants.borrow,
            _token0Amount,
            _token1Amount,
            _maxSharesAmount
        );
    }

    /// @notice Estimate how much debt shares will be burned for a variable dex payback for T3 and T4 vaults
    /// @dev This should be called with static call.
    /// @param _vault Address of the vault
    /// @param _token0Amount Amount of token0 to payback
    /// @param _token1Amount Amount of token1 to payback
    /// @param _minSharesAmount Minimum amount of shares to payback
    ///         This value can be set to low value like 1 to just check for burned shares.
    ///         However, it can also be used to check if transaction will revert when sending this amount of shares (Slippage check)
    /// @return shares Amount of shares burned
    function estimatePayback(
        address _vault,
        uint256 _token0Amount,
        uint256 _token1Amount,
        uint256 _minSharesAmount
    ) external returns (uint256 shares) {
        IFluidVault.ConstantViews memory constants = IFluidVault(_vault).constantsView();

        shares = IFluidDexResolver(FLUID_DEX_RESOLVER).estimatePayback(
            constants.borrow,
            _token0Amount,
            _token1Amount,
            _minSharesAmount
        );
    }

    /// @notice Estimate how much collateral is worth in terms of one token for a given nft of dex position.
    /// @notice This function can be used to estimate max collateral withdrawal in one token.
    /// @dev This should be called with static call.
    /// @dev Only first non zero value will be used, and other will be capped to 0.
    /// @param _nftId Nft id of the dex position
    /// @param _minToken0AmountToAccept Minimum amount of token0 to accept. If 0, withdrawal is calculated in token1.
    ///         This value can be set to low value like 1 to just check for withdrawable collateral.
    /// @param _minToken1AmountToAccept Minimum amount of token1 to accept. If 0, withdrawal is calculated in token0.
    ///         This value can be set to low value like 1 to just check for withdrawable collateral.
    /// @return collateral Amount of collateral in one token
    function estimateDexPositionCollateralInOneToken(
        uint256 _nftId,
        uint256 _minToken0AmountToAccept,
        uint256 _minToken1AmountToAccept
    ) external returns (uint256 collateral) {
        require(_minToken0AmountToAccept > 0 || _minToken1AmountToAccept > 0);

        // Make sure only one token is specified
        if (_minToken0AmountToAccept > 0) {
            _minToken1AmountToAccept = 0;
        }

        // Make sure only one token is specified
        if (_minToken1AmountToAccept > 0) {
            _minToken0AmountToAccept = 0;
        }

        (
            IFluidVaultResolver.UserPosition memory userPosition,
            IFluidVaultResolver.VaultEntireData memory vaultData
        ) = IFluidVaultResolver(FLUID_VAULT_RESOLVER).positionByNftId(_nftId);

        collateral = IFluidDexResolver(FLUID_DEX_RESOLVER).estimateWithdrawPerfectInOneToken(
            vaultData.constantVariables.supply,
            userPosition.supply,
            _minToken0AmountToAccept,
            _minToken1AmountToAccept
        );
    }

    /// @notice Estimate how much debt is worth in terms of one token for a given nft of dex position.
    /// @notice This function can be used to estimate max debt payback in one token.
    /// @dev This should be called with static call.
    /// @dev Only first non zero value will be used, and other will be capped to 0.
    /// @param _nftId Nft id of the dex position
    /// @param _maxToken0AmountToPayback Maximum amount of token0 to payback. If 0, payback is calculated in token1.
    ///         This value can be set to high value like type(int256).max to just check for full debt payback.
    /// @param _maxToken1AmountToPayback Maximum amount of token1 to payback. If 0, payback is calculated in token0.
    ///         This value can be set to high value like type(int256).max to just check for full debt payback.
    /// @return debt Amount of debt in one token
    function estimateDexPositionDebtInOneToken(
        uint256 _nftId,
        uint256 _maxToken0AmountToPayback,
        uint256 _maxToken1AmountToPayback
    ) external returns (uint256 debt) {
        require(_maxToken0AmountToPayback > 0 || _maxToken1AmountToPayback > 0);

        // Make sure only one token is specified
        if (_maxToken0AmountToPayback > 0) {
            _maxToken1AmountToPayback = 0;
        }

        // Make sure only one token is specified
        if (_maxToken1AmountToPayback > 0) {
            _maxToken0AmountToPayback = 0;
        }

        (
            IFluidVaultResolver.UserPosition memory userPosition,
            IFluidVaultResolver.VaultEntireData memory vaultData
        ) = IFluidVaultResolver(FLUID_VAULT_RESOLVER).positionByNftId(_nftId);

        debt = IFluidDexResolver(FLUID_DEX_RESOLVER).estimatePaybackPerfectInOneToken(
            vaultData.constantVariables.borrow,
            userPosition.borrow,
            _maxToken0AmountToPayback,
            _maxToken1AmountToPayback
        );
    }

    /*//////////////////////////////////////////////////////////////
                        FLUID EARN - F TOKENs UTILS
    //////////////////////////////////////////////////////////////*/
    /// @notice Get all fTokens addresses
    function getAllFTokens() external view returns (address[] memory) {
        return IFluidLendingResolver(FLUID_LENDING_RESOLVER).getAllFTokens();
    }

    /// @notice Get fToken data for a specific fToken address
    function getFTokenData(address _fToken) public view returns (FTokenData memory fTokenData) {
        
        IFluidLendingResolver.FTokenDetails memory details;

        // Fluid Lending Resolver checks if the fToken's underlying asset supports EIP-2612.
        // For WETH, this triggers the fallback function, which attempts a deposit.
        // This panics because of write protection and consumes all gas, leaving only 1/64th for the caller (EIP-150).
        // To lower the gas cost, we cap the gas limit at 9M, ensuring ~140k gas remains for fetching fWETH details
        // and enough gas is left for further operations within the same block.
        // For arbitrum, we don't need to cap as WETH will have EIP-2612 support.
        if (_fToken == F_WETH_TOKEN_ADDR && block.chainid != 42161) {
            details = IFluidLendingResolver(FLUID_LENDING_RESOLVER).getFTokenDetails{ gas: 9_000_000 }(_fToken);
        } else {
            details = IFluidLendingResolver(FLUID_LENDING_RESOLVER).getFTokenDetails(_fToken);
        }
        
        fTokenData = _filterFTokenData(details);
    }

    /// @notice Get fToken data for all fTokens
    function getAllFTokensData() public view returns (FTokenData[] memory) {
        address[] memory fTokens = IFluidLendingResolver(FLUID_LENDING_RESOLVER).getAllFTokens();
        FTokenData[] memory fTokenData = new FTokenData[](fTokens.length);

        for (uint256 i = 0; i < fTokens.length; i++) {
            fTokenData[i] = getFTokenData(fTokens[i]);
        } 

        return fTokenData;
    }

    /// @notice Get user position for a specific fToken address
    function getUserEarnPosition(address _fToken, address _user) public view returns (UserEarnPosition memory) {
        IFluidLendingResolver.UserPosition memory data = 
            IFluidLendingResolver(FLUID_LENDING_RESOLVER).getUserPosition(_fToken, _user);

        return UserEarnPosition({
            fTokenShares: data.fTokenShares,
            underlyingAssets: data.underlyingAssets,
            underlyingBalance: data.underlyingBalance,
            allowance: data.allowance
        });
    }

    /// @notice Get user position for a specific fToken address
    function getUserEarnPositionWithFToken(
        address _fToken,
        address _user
    ) public view returns (UserEarnPosition memory userPosition, FTokenData memory fTokenData) {
        IFluidLendingResolver.UserPosition memory userData = 
            IFluidLendingResolver(FLUID_LENDING_RESOLVER).getUserPosition(_fToken, _user);

        userPosition = UserEarnPosition({
            fTokenShares: userData.fTokenShares,
            underlyingAssets: userData.underlyingAssets,
            underlyingBalance: userData.underlyingBalance,
            allowance: userData.allowance
        });

        fTokenData = getFTokenData(_fToken);
    }

    /// @notice Get user positions for all fTokens
    function getAllUserEarnPositionsWithFTokens(address _user)
        external
        view
        returns (UserEarnPosition[] memory userPositions, FTokenData[] memory fTokensData)
    {
        fTokensData = getAllFTokensData();

        userPositions = new UserEarnPosition[](fTokensData.length);

        for (uint256 i = 0; i < fTokensData.length; i++) {
            userPositions[i] = getUserEarnPosition(fTokensData[i].tokenAddress, _user);
        }
    }

    /**
     *
     *                         INTERNAL FUNCTIONS
     *
     */
    /// @notice Helper function used for T4 vaults to determine which oracle has to be used for smart collateral DEX.
    function _getSmartCollateralDexOracle(address _vaultOracle) internal view returns (address smartCollOracle) {
        /// @dev Some T4 vaults use main oracles that contain both dexSmartDebtSharesRates and dexSmartCollSharesRates.
        /// But some use only the debt oracle as main and link the collateral oracle with a call to getDexColDebtOracleData.
        try IDexSmartCollOracle(_vaultOracle).dexSmartColSharesRates() returns (
            uint256, uint256
        ) {
            return _vaultOracle;
        } catch {
            (smartCollOracle, ) = IDexSmartDebtOracle(_vaultOracle).getDexColDebtOracleData();
        }
    }

    /// @notice Helper function to adapt dex data to DexSupplyData
    function _fillDexSupplyData(
        IFluidDexResolver.DexEntireData memory _dexData,
        address _oracle,
        uint256 _sharesWithdrawable
    ) internal view returns (DexSupplyData memory dexSupplyData) {
        address quoteToken = _isQuoteInToken0ForSmartCollOracle(_oracle)
            ? _dexData.constantViews.token0
            : _dexData.constantViews.token1;

        (uint256 quoteTokensPerShare, ) = IDexSmartCollOracle(_oracle).dexSmartColSharesRates();

        dexSupplyData = DexSupplyData({
            dexPool: _dexData.dex,
            dexId: _dexData.constantViews.dexId,
            fee: _dexData.configs.fee,
            lastStoredPrice: _dexData.dexState.lastStoredPrice,
            centerPrice: _dexData.dexState.centerPrice,
            token0Utilization: _dexData.limitsAndAvailability.liquidityTokenData0.lastStoredUtilization,
            token1Utilization: _dexData.limitsAndAvailability.liquidityTokenData1.lastStoredUtilization,
            totalSupplyShares: _dexData.dexState.totalSupplyShares,
            maxSupplyShares: _dexData.configs.maxSupplyShares,
            token0Supplied: _dexData.dexState.totalSupplyShares * _dexData.dexState.token0PerSupplyShare / 1e18,
            token1Supplied: _dexData.dexState.totalSupplyShares * _dexData.dexState.token1PerSupplyShare / 1e18,
            sharesWithdrawable: _sharesWithdrawable,
            token0Withdrawable: _sharesWithdrawable * _dexData.dexState.token0PerSupplyShare / 1e18,
            token1Withdrawable: _sharesWithdrawable * _dexData.dexState.token1PerSupplyShare / 1e18,
            token0PerSupplyShare: _dexData.dexState.token0PerSupplyShare,
            token1PerSupplyShare: _dexData.dexState.token1PerSupplyShare,
            token0SupplyRate: _dexData.limitsAndAvailability.liquidityTokenData0.supplyRate,
            token1SupplyRate: _dexData.limitsAndAvailability.liquidityTokenData1.supplyRate,
            quoteToken: quoteToken,
            quoteTokensPerShare: quoteTokensPerShare,
            supplyToken0Reserves: _dexData.colReserves.token0RealReserves,
            supplyToken1Reserves: _dexData.colReserves.token1RealReserves
        });
    }

    /// @notice Helper function to adapt dex data to DexBorrowData
    function _fillDexBorrowData(
        IFluidDexResolver.DexEntireData memory _dexData,
        address _oracle,
        uint256 _sharesBorrowable
    ) internal view returns (DexBorrowData memory dexBorrowData) {
        address quoteToken = _isQuoteInToken0ForSmartDebtOracle(_oracle)
            ? _dexData.constantViews.token0
            : _dexData.constantViews.token1;

        (uint256 quoteTokensPerShare, ) = IDexSmartDebtOracle(_oracle).dexSmartDebtSharesRates();

        dexBorrowData = DexBorrowData({
            dexPool: _dexData.dex,
            dexId: _dexData.constantViews.dexId,
            fee: _dexData.configs.fee,
            lastStoredPrice: _dexData.dexState.lastStoredPrice,
            centerPrice: _dexData.dexState.centerPrice,
            token0Utilization: _dexData.limitsAndAvailability.liquidityTokenData0.lastStoredUtilization,
            token1Utilization: _dexData.limitsAndAvailability.liquidityTokenData1.lastStoredUtilization,
            totalBorrowShares: _dexData.dexState.totalBorrowShares,
            maxBorrowShares: _dexData.configs.maxBorrowShares,
            token0Borrowed: _dexData.dexState.totalBorrowShares * _dexData.dexState.token0PerBorrowShare / 1e18,
            token1Borrowed: _dexData.dexState.totalBorrowShares * _dexData.dexState.token1PerBorrowShare / 1e18,
            sharesBorrowable: _sharesBorrowable,
            token0Borrowable: _sharesBorrowable * _dexData.dexState.token0PerBorrowShare / 1e18,
            token1Borrowable: _sharesBorrowable * _dexData.dexState.token1PerBorrowShare / 1e18,
            token0PerBorrowShare: _dexData.dexState.token0PerBorrowShare,
            token1PerBorrowShare: _dexData.dexState.token1PerBorrowShare,
            token0BorrowRate: _dexData.limitsAndAvailability.liquidityTokenData0.borrowRate,
            token1BorrowRate: _dexData.limitsAndAvailability.liquidityTokenData1.borrowRate,
            quoteToken: quoteToken,
            quoteTokensPerShare: quoteTokensPerShare,
            borrowToken0Reserves: _dexData.debtReserves.token0RealReserves,
            borrowToken1Reserves: _dexData.debtReserves.token1RealReserves
        });
    }

    /// @notice Helper function to get information whether the quote token is token0 or token1 in smart collateral dex oracle
    function _isQuoteInToken0ForSmartCollOracle(
        address _oracle
    ) internal view returns (bool quoteInToken0) {
        // Try to call the newer function signature first
        try IDexSmartCollOracle(_oracle).dexOracleData() returns (
            address, bool _quoteInToken0, address, uint256, uint256
        ) {
            return _quoteInToken0;
        } catch {
            // If the newer function fails, try the older function signature
            (,,,,,,,,, quoteInToken0) = IDexSmartCollOracle(_oracle).dexSmartColOracleData();
        }
    }

    /// @notice Helper function to get information whether the quote token is token0 or token1 in smart debt dex oracle
    function _isQuoteInToken0ForSmartDebtOracle(
        address _oracle
    ) internal view returns (bool quoteInToken0) {
        // Try to call the newer function signature first
        try IDexSmartDebtOracle(_oracle).dexOracleData() returns (
            address, bool _quoteInToken0, address, uint256, uint256
        ) {
            return _quoteInToken0;
        } catch {
            // If the newer function fails, try the older function signature
            (,,,,,,,,, quoteInToken0) = IDexSmartDebtOracle(_oracle).dexSmartDebtOracleData();
        }
    }

    /// @notice Helper function to filter FTokenDetails to FTokenData
    function _filterFTokenData(
        IFluidLendingResolver.FTokenDetails memory _details
    ) internal pure returns (FTokenData memory fTokenData) {
        fTokenData = FTokenData({
            tokenAddress: _details.tokenAddress,
            isNativeUnderlying: _details.isNativeUnderlying,
            name: _details.name,
            symbol: _details.symbol,
            decimals: _details.decimals,
            asset: _details.asset,
            totalAssets: _details.totalAssets,
            totalSupply: _details.totalSupply,
            convertToShares: _details.convertToShares,
            convertToAssets: _details.convertToAssets,
            rewardsRate: _details.rewardsRate,
            supplyRate: _details.supplyRate,
            withdrawable: _details.liquidityUserSupplyData.withdrawable,
            modeWithInterest: _details.liquidityUserSupplyData.modeWithInterest,
            expandPercent: _details.liquidityUserSupplyData.expandPercent,
            expandDuration: _details.liquidityUserSupplyData.expandDuration
        });
    }
}
