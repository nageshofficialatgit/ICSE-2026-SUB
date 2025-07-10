// SPDX-License-Identifier: MIT
pragma solidity =0.8.24;










contract MainnetAaveV3Addresses {
    address internal constant AAVE_REWARDS_CONTROLLER_ADDRESS = 0x8164Cc65827dcFe994AB23944CBC90e0aa80bFcb;
    address internal constant DEFAULT_AAVE_MARKET = 0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e;
    address internal constant AAVE_ORACLE_V3 = 0x54586bE62E3c3580375aE3723C145253060Ca0C2;
    address internal constant STAKED_GHO_TOKEN = 0x1a88Df1cFe15Af22B3c4c783D4e6F7F9e0C1885d;
    address internal constant AAVE_GOV_TOKEN = 0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9;
}






interface IAaveProtocolDataProvider {
    /**
     * @notice Returns the user data in a reserve
     * @param asset The address of the underlying asset of the reserve
     * @param user The address of the user
     * @return currentATokenBalance The current AToken balance of the user
     * @return currentStableDebt The current stable debt of the user
     * @return currentVariableDebt The current variable debt of the user
     * @return principalStableDebt The principal stable debt of the user
     * @return scaledVariableDebt The scaled variable debt of the user
     * @return stableBorrowRate The stable borrow rate of the user
     * @return liquidityRate The liquidity rate of the reserve
     * @return stableRateLastUpdated The timestamp of the last update of the user stable rate
     * @return usageAsCollateralEnabled True if the user is using the asset as collateral, false
     *         otherwise
     **/
    function getUserReserveData(
        address asset,
        address user
    )
        external
        view
        returns (
            uint256 currentATokenBalance,
            uint256 currentStableDebt,
            uint256 currentVariableDebt,
            uint256 principalStableDebt,
            uint256 scaledVariableDebt,
            uint256 stableBorrowRate,
            uint256 liquidityRate,
            uint40 stableRateLastUpdated,
            bool usageAsCollateralEnabled
        );

    function getSiloedBorrowing(address asset) external view returns (bool);

    /**
     * @notice Returns if the pool is paused
     * @param asset The address of the underlying asset of the reserve
     * @return isPaused True if the pool is paused, false otherwise
     **/
    function getPaused(address asset) external view returns (bool isPaused);

    /**
     * @notice Returns the configuration data of the reserve
     * @dev Not returning borrow and supply caps for compatibility, nor pause flag
     * @param asset The address of the underlying asset of the reserve
     * @return decimals The number of decimals of the reserve
     * @return ltv The ltv of the reserve
     * @return liquidationThreshold The liquidationThreshold of the reserve
     * @return liquidationBonus The liquidationBonus of the reserve
     * @return reserveFactor The reserveFactor of the reserve
     * @return usageAsCollateralEnabled True if the usage as collateral is enabled, false otherwise
     * @return borrowingEnabled True if borrowing is enabled, false otherwise
     * @return stableBorrowRateEnabled True if stable rate borrowing is enabled, false otherwise
     * @return isActive True if it is active, false otherwise
     * @return isFrozen True if it is frozen, false otherwise
     */
    function getReserveConfigurationData(
        address asset
    )
        external
        view
        returns (
            uint256 decimals,
            uint256 ltv,
            uint256 liquidationThreshold,
            uint256 liquidationBonus,
            uint256 reserveFactor,
            bool usageAsCollateralEnabled,
            bool borrowingEnabled,
            bool stableBorrowRateEnabled,
            bool isActive,
            bool isFrozen
        );

    /**
     * @notice Returns the reserve data
     * @param asset The address of the underlying asset of the reserve
     * @return unbacked The amount of unbacked tokens
     * @return accruedToTreasuryScaled The scaled amount of tokens accrued to treasury that is to be minted
     * @return totalAToken The total supply of the aToken
     * @return totalStableDebt The total stable debt of the reserve
     * @return totalVariableDebt The total variable debt of the reserve
     * @return liquidityRate The liquidity rate of the reserve
     * @return variableBorrowRate The variable borrow rate of the reserve
     * @return stableBorrowRate The stable borrow rate of the reserve
     * @return averageStableBorrowRate The average stable borrow rate of the reserve
     * @return liquidityIndex The liquidity index of the reserve
     * @return variableBorrowIndex The variable borrow index of the reserve
     * @return lastUpdateTimestamp The timestamp of the last update of the reserve
     */
    function getReserveData(
        address asset
    )
        external
        view
        returns (
            uint256 unbacked,
            uint256 accruedToTreasuryScaled,
            uint256 totalAToken,
            uint256 totalStableDebt,
            uint256 totalVariableDebt,
            uint256 liquidityRate,
            uint256 variableBorrowRate,
            uint256 stableBorrowRate,
            uint256 averageStableBorrowRate,
            uint256 liquidityIndex,
            uint256 variableBorrowIndex,
            uint40 lastUpdateTimestamp
        );
}






library DataTypes {
  struct ReserveData {
    //stores the reserve configuration
    ReserveConfigurationMap configuration;
    //the liquidity index. Expressed in ray
    uint128 liquidityIndex;
    //the current supply rate. Expressed in ray
    uint128 currentLiquidityRate;
    //variable borrow index. Expressed in ray
    uint128 variableBorrowIndex;
    // DEPRECATED on v3.2.0
    uint128 currentVariableBorrowRate;
    //the current stable borrow rate. Expressed in ray
    uint128 currentStableBorrowRate;
    //timestamp of last update
    uint40 lastUpdateTimestamp;
    //the id of the reserve. Represents the position in the list of the active reserves
    uint16 id;
    //aToken address
    address aTokenAddress;
    // DEPRECATED on v3.2.0
    address stableDebtTokenAddress;
    //variableDebtToken address
    address variableDebtTokenAddress;
    //address of the interest rate strategy
    address interestRateStrategyAddress;
    //the current treasury balance, scaled
    uint128 accruedToTreasury;
    //the outstanding unbacked aTokens minted through the bridging feature
    uint128 unbacked;
    //the outstanding debt borrowed against this asset in isolation mode
    uint128 isolationModeTotalDebt;
  }
  struct ReserveDataExtended {
    //stores the reserve configuration
    ReserveConfigurationMap configuration;
    //the liquidity index. Expressed in ray
    uint128 liquidityIndex;
    //the current supply rate. Expressed in ray
    uint128 currentLiquidityRate;
    //variable borrow index. Expressed in ray
    uint128 variableBorrowIndex;
    //the current variable borrow rate. Expressed in ray
    uint128 currentVariableBorrowRate;
    //the current stable borrow rate. Expressed in ray
    uint128 currentStableBorrowRate;
    //timestamp of last update
    uint40 lastUpdateTimestamp;
    //the id of the reserve. Represents the position in the list of the active reserves
    uint16 id;
    //timestamp in the future, until when liquidations are not allowed on the reserve
    uint40 liquidationGracePeriodUntil;
    //aToken address
    address aTokenAddress;
    //stableDebtToken address
    address stableDebtTokenAddress;
    //variableDebtToken address
    address variableDebtTokenAddress;
    //address of the interest rate strategy
    address interestRateStrategyAddress;
    //the current treasury balance, scaled
    uint128 accruedToTreasury;
    //the outstanding unbacked aTokens minted through the bridging feature
    uint128 unbacked;
    //the outstanding debt borrowed against this asset in isolation mode
    uint128 isolationModeTotalDebt;
    //the amount of underlying accounted for by the protocol
    uint128 virtualUnderlyingBalance;
  }

  struct ReserveConfigurationMap {
    //bit 0-15: LTV
    //bit 16-31: Liq. threshold
    //bit 32-47: Liq. bonus
    //bit 48-55: Decimals
    //bit 56: reserve is active
    //bit 57: reserve is frozen
    //bit 58: borrowing is enabled
    //bit 59: stable rate borrowing enabled
    //bit 60: asset is paused
    //bit 61: borrowing in isolation mode is enabled
    //bit 62-63: reserved
    //bit 64-79: reserve factor
    //bit 80-115 borrow cap in whole tokens, borrowCap == 0 => no cap
    //bit 116-151 supply cap in whole tokens, supplyCap == 0 => no cap
    //bit 152-167 liquidation protocol fee
    //bit 168-175 eMode category
    //bit 176-211 unbacked mint cap in whole tokens, unbackedMintCap == 0 => minting disabled
    //bit 212-251 debt ceiling for isolation mode with (ReserveConfiguration::DEBT_CEILING_DECIMALS) decimals
    //bit 252: virtual accounting is enabled for the reserve
    //bit 253-255 unused

    uint256 data;
  }

  struct UserConfigurationMap {
    /**
     * @dev Bitmap of the users collaterals and borrows. It is divided in pairs of bits, one pair per asset.
     * The first bit indicates if an asset is used as collateral by the user, the second whether an
     * asset is borrowed by the user.
     */
    uint256 data;
  }

  struct EModeCategory {
    // each eMode category has a custom ltv and liquidation threshold
    uint16 ltv;
    uint16 liquidationThreshold;
    uint16 liquidationBonus;
    // each eMode category may or may not have a custom oracle to override the individual assets price oracles
    address priceSource;
    string label;
  }

  // DEPRECATED: kept for backwards compatibility, might be removed in a future version
  struct EModeCategoryLegacy {
     // each eMode category has a custom ltv and liquidation threshold
     uint16 ltv;
     uint16 liquidationThreshold;
     uint16 liquidationBonus;
    // DEPRECATED
     address priceSource;
     string label;
   }

   struct CollateralConfig {
    uint16 ltv;
    uint16 liquidationThreshold;
    uint16 liquidationBonus;
  }

  struct EModeCategoryBaseConfiguration {
    uint16 ltv;
    uint16 liquidationThreshold;
    uint16 liquidationBonus;
    string label;
  }

  struct EModeCategoryNew {
    // each eMode category has a custom ltv and liquidation threshold
    uint16 ltv;
    uint16 liquidationThreshold;
    uint16 liquidationBonus;
    uint128 collateralBitmap;
    string label;
    uint128 borrowableBitmap;
  }

  enum InterestRateMode {
    NONE,
    _DEPRECATED,
    VARIABLE
  }

  struct CalculateUserAccountDataParams {
    UserConfigurationMap userConfig;
    uint256 reservesCount;
    address user;
    address oracle;
    uint8 userEModeCategory;
  }

  struct CalculateInterestRatesParams {
    uint256 unbacked;
    uint256 liquidityAdded;
    uint256 liquidityTaken;
    uint256 totalDebt;
    uint256 reserveFactor;
    address reserve;
    bool usingVirtualBalance;
    uint256 virtualUnderlyingBalance;
  }
}






interface IPoolAddressesProvider {
  /**
   * @dev Emitted when the market identifier is updated.
   * @param oldMarketId The old id of the market
   * @param newMarketId The new id of the market
   */
  event MarketIdSet(string indexed oldMarketId, string indexed newMarketId);

  /**
   * @dev Emitted when the pool is updated.
   * @param oldAddress The old address of the Pool
   * @param newAddress The new address of the Pool
   */
  event PoolUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the pool configurator is updated.
   * @param oldAddress The old address of the PoolConfigurator
   * @param newAddress The new address of the PoolConfigurator
   */
  event PoolConfiguratorUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the price oracle is updated.
   * @param oldAddress The old address of the PriceOracle
   * @param newAddress The new address of the PriceOracle
   */
  event PriceOracleUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the ACL manager is updated.
   * @param oldAddress The old address of the ACLManager
   * @param newAddress The new address of the ACLManager
   */
  event ACLManagerUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the ACL admin is updated.
   * @param oldAddress The old address of the ACLAdmin
   * @param newAddress The new address of the ACLAdmin
   */
  event ACLAdminUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the price oracle sentinel is updated.
   * @param oldAddress The old address of the PriceOracleSentinel
   * @param newAddress The new address of the PriceOracleSentinel
   */
  event PriceOracleSentinelUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the pool data provider is updated.
   * @param oldAddress The old address of the PoolDataProvider
   * @param newAddress The new address of the PoolDataProvider
   */
  event PoolDataProviderUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when a new proxy is created.
   * @param id The identifier of the proxy
   * @param proxyAddress The address of the created proxy contract
   * @param implementationAddress The address of the implementation contract
   */
  event ProxyCreated(
    bytes32 indexed id,
    address indexed proxyAddress,
    address indexed implementationAddress
  );

  /**
   * @dev Emitted when a new non-proxied contract address is registered.
   * @param id The identifier of the contract
   * @param oldAddress The address of the old contract
   * @param newAddress The address of the new contract
   */
  event AddressSet(bytes32 indexed id, address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the implementation of the proxy registered with id is updated
   * @param id The identifier of the contract
   * @param proxyAddress The address of the proxy contract
   * @param oldImplementationAddress The address of the old implementation contract
   * @param newImplementationAddress The address of the new implementation contract
   */
  event AddressSetAsProxy(
    bytes32 indexed id,
    address indexed proxyAddress,
    address oldImplementationAddress,
    address indexed newImplementationAddress
  );

  /**
   * @notice Returns the id of the Aave market to which this contract points to.
   * @return The market id
   **/
  function getMarketId() external view returns (string memory);

  /**
   * @notice Associates an id with a specific PoolAddressesProvider.
   * @dev This can be used to create an onchain registry of PoolAddressesProviders to
   * identify and validate multiple Aave markets.
   * @param newMarketId The market id
   */
  function setMarketId(string calldata newMarketId) external;

  /**
   * @notice Returns an address by its identifier.
   * @dev The returned address might be an EOA or a contract, potentially proxied
   * @dev It returns ZERO if there is no registered address with the given id
   * @param id The id
   * @return The address of the registered for the specified id
   */
  function getAddress(bytes32 id) external view returns (address);

  /**
   * @notice General function to update the implementation of a proxy registered with
   * certain `id`. If there is no proxy registered, it will instantiate one and
   * set as implementation the `newImplementationAddress`.
   * @dev IMPORTANT Use this function carefully, only for ids that don't have an explicit
   * setter function, in order to avoid unexpected consequences
   * @param id The id
   * @param newImplementationAddress The address of the new implementation
   */
  function setAddressAsProxy(bytes32 id, address newImplementationAddress) external;

  /**
   * @notice Sets an address for an id replacing the address saved in the addresses map.
   * @dev IMPORTANT Use this function carefully, as it will do a hard replacement
   * @param id The id
   * @param newAddress The address to set
   */
  function setAddress(bytes32 id, address newAddress) external;

  /**
   * @notice Returns the address of the Pool proxy.
   * @return The Pool proxy address
   **/
  function getPool() external view returns (address);

  /**
   * @notice Updates the implementation of the Pool, or creates a proxy
   * setting the new `pool` implementation when the function is called for the first time.
   * @param newPoolImpl The new Pool implementation
   **/
  function setPoolImpl(address newPoolImpl) external;

  /**
   * @notice Returns the address of the PoolConfigurator proxy.
   * @return The PoolConfigurator proxy address
   **/
  function getPoolConfigurator() external view returns (address);

  /**
   * @notice Updates the implementation of the PoolConfigurator, or creates a proxy
   * setting the new `PoolConfigurator` implementation when the function is called for the first time.
   * @param newPoolConfiguratorImpl The new PoolConfigurator implementation
   **/
  function setPoolConfiguratorImpl(address newPoolConfiguratorImpl) external;

  /**
   * @notice Returns the address of the price oracle.
   * @return The address of the PriceOracle
   */
  function getPriceOracle() external view returns (address);

  /**
   * @notice Updates the address of the price oracle.
   * @param newPriceOracle The address of the new PriceOracle
   */
  function setPriceOracle(address newPriceOracle) external;

  /**
   * @notice Returns the address of the ACL manager.
   * @return The address of the ACLManager
   */
  function getACLManager() external view returns (address);

  /**
   * @notice Updates the address of the ACL manager.
   * @param newAclManager The address of the new ACLManager
   **/
  function setACLManager(address newAclManager) external;

  /**
   * @notice Returns the address of the ACL admin.
   * @return The address of the ACL admin
   */
  function getACLAdmin() external view returns (address);

  /**
   * @notice Updates the address of the ACL admin.
   * @param newAclAdmin The address of the new ACL admin
   */
  function setACLAdmin(address newAclAdmin) external;

  /**
   * @notice Returns the address of the price oracle sentinel.
   * @return The address of the PriceOracleSentinel
   */
  function getPriceOracleSentinel() external view returns (address);

  /**
   * @notice Updates the address of the price oracle sentinel.
   * @param newPriceOracleSentinel The address of the new PriceOracleSentinel
   **/
  function setPriceOracleSentinel(address newPriceOracleSentinel) external;

  /**
   * @notice Returns the address of the data provider.
   * @return The address of the DataProvider
   */
  function getPoolDataProvider() external view returns (address);

  /**
   * @notice Updates the address of the data provider.
   * @param newDataProvider The address of the new DataProvider
   **/
  function setPoolDataProvider(address newDataProvider) external;
}







interface IPoolV3 {
  /**
   * @dev Emitted on mintUnbacked()
   * @param reserve The address of the underlying asset of the reserve
   * @param user The address initiating the supply
   * @param onBehalfOf The beneficiary of the supplied assets, receiving the aTokens
   * @param amount The amount of supplied assets
   * @param referralCode The referral code used
   **/
  event MintUnbacked(
    address indexed reserve,
    address user,
    address indexed onBehalfOf,
    uint256 amount,
    uint16 indexed referralCode
  );

  /**
   * @dev Emitted on backUnbacked()
   * @param reserve The address of the underlying asset of the reserve
   * @param backer The address paying for the backing
   * @param amount The amount added as backing
   * @param fee The amount paid in fees
   **/
  event BackUnbacked(address indexed reserve, address indexed backer, uint256 amount, uint256 fee);

  /**
   * @dev Emitted on supply()
   * @param reserve The address of the underlying asset of the reserve
   * @param user The address initiating the supply
   * @param onBehalfOf The beneficiary of the supply, receiving the aTokens
   * @param amount The amount supplied
   * @param referralCode The referral code used
   **/
  event Supply(
    address indexed reserve,
    address user,
    address indexed onBehalfOf,
    uint256 amount,
    uint16 indexed referralCode
  );

  /**
   * @dev Emitted on withdraw()
   * @param reserve The address of the underlying asset being withdrawn
   * @param user The address initiating the withdrawal, owner of aTokens
   * @param to The address that will receive the underlying
   * @param amount The amount to be withdrawn
   **/
  event Withdraw(address indexed reserve, address indexed user, address indexed to, uint256 amount);

  /**
   * @dev Emitted on borrow() and flashLoan() when debt needs to be opened
   * @param reserve The address of the underlying asset being borrowed
   * @param user The address of the user initiating the borrow(), receiving the funds on borrow() or just
   * initiator of the transaction on flashLoan()
   * @param onBehalfOf The address that will be getting the debt
   * @param amount The amount borrowed out
   * @param interestRateMode The rate mode: 1 is deprecated, 2 for Variable
   * @param borrowRate The numeric rate at which the user has borrowed, expressed in ray
   * @param referralCode The referral code used
   **/
  event Borrow(
    address indexed reserve,
    address user,
    address indexed onBehalfOf,
    uint256 amount,
    DataTypes.InterestRateMode interestRateMode,
    uint256 borrowRate,
    uint16 indexed referralCode
  );

  /**
   * @dev Emitted on repay()
   * @param reserve The address of the underlying asset of the reserve
   * @param user The beneficiary of the repayment, getting his debt reduced
   * @param repayer The address of the user initiating the repay(), providing the funds
   * @param amount The amount repaid
   * @param useATokens True if the repayment is done using aTokens, `false` if done with underlying asset directly
   **/
  event Repay(
    address indexed reserve,
    address indexed user,
    address indexed repayer,
    uint256 amount,
    bool useATokens
  );

  /**
   * @dev Emitted on swapBorrowRateMode()
   * @param reserve The address of the underlying asset of the reserve
   * @param user The address of the user swapping his rate mode
   * @param interestRateMode The current interest rate mode of the position being swapped: 1 for Stable, 2 for Variable
   **/
  event SwapBorrowRateMode(
    address indexed reserve,
    address indexed user,
    DataTypes.InterestRateMode interestRateMode
  );

  /**
   * @dev Emitted on borrow(), repay() and liquidationCall() when using isolated assets
   * @param asset The address of the underlying asset of the reserve
   * @param totalDebt The total isolation mode debt for the reserve
   */
  event IsolationModeTotalDebtUpdated(address indexed asset, uint256 totalDebt);

  /**
   * @dev Emitted when the user selects a certain asset category for eMode
   * @param user The address of the user
   * @param categoryId The category id
   **/
  event UserEModeSet(address indexed user, uint8 categoryId);

  /**
   * @dev Emitted on setUserUseReserveAsCollateral()
   * @param reserve The address of the underlying asset of the reserve
   * @param user The address of the user enabling the usage as collateral
   **/
  event ReserveUsedAsCollateralEnabled(address indexed reserve, address indexed user);

  /**
   * @dev Emitted on setUserUseReserveAsCollateral()
   * @param reserve The address of the underlying asset of the reserve
   * @param user The address of the user enabling the usage as collateral
   **/
  event ReserveUsedAsCollateralDisabled(address indexed reserve, address indexed user);

  /**
   * @dev Emitted on rebalanceStableBorrowRate()
   * @param reserve The address of the underlying asset of the reserve
   * @param user The address of the user for which the rebalance has been executed
   **/
  event RebalanceStableBorrowRate(address indexed reserve, address indexed user);

  /**
   * @dev Emitted on flashLoan()
   * @param target The address of the flash loan receiver contract
   * @param initiator The address initiating the flash loan
   * @param asset The address of the asset being flash borrowed
   * @param amount The amount flash borrowed
   * @param interestRateMode The flashloan mode: 0 for regular flashloan, 1 is deprecated, 2 for Variable debt
   * @param premium The fee flash borrowed
   * @param referralCode The referral code used
   **/
  event FlashLoan(
    address indexed target,
    address initiator,
    address indexed asset,
    uint256 amount,
    DataTypes.InterestRateMode interestRateMode,
    uint256 premium,
    uint16 indexed referralCode
  );

  /**
   * @dev Emitted when a borrower is liquidated.
   * @param collateralAsset The address of the underlying asset used as collateral, to receive as result of the liquidation
   * @param debtAsset The address of the underlying borrowed asset to be repaid with the liquidation
   * @param user The address of the borrower getting liquidated
   * @param debtToCover The debt amount of borrowed `asset` the liquidator wants to cover
   * @param liquidatedCollateralAmount The amount of collateral received by the liquidator
   * @param liquidator The address of the liquidator
   * @param receiveAToken True if the liquidators wants to receive the collateral aTokens, `false` if he wants
   * to receive the underlying collateral asset directly
   **/
  event LiquidationCall(
    address indexed collateralAsset,
    address indexed debtAsset,
    address indexed user,
    uint256 debtToCover,
    uint256 liquidatedCollateralAmount,
    address liquidator,
    bool receiveAToken
  );

  /**
   * @dev Emitted when the state of a reserve is updated.
   * @param reserve The address of the underlying asset of the reserve
   * @param liquidityRate The next liquidity rate
   * @param stableBorrowRate The next stable borrow rate
   * @param variableBorrowRate The next variable borrow rate
   * @param liquidityIndex The next liquidity index
   * @param variableBorrowIndex The next variable borrow index
   **/
  event ReserveDataUpdated(
    address indexed reserve,
    uint256 liquidityRate,
    uint256 stableBorrowRate,
    uint256 variableBorrowRate,
    uint256 liquidityIndex,
    uint256 variableBorrowIndex
  );

  /**
   * @dev Emitted when the protocol treasury receives minted aTokens from the accrued interest.
   * @param reserve The address of the reserve
   * @param amountMinted The amount minted to the treasury
   **/
  event MintedToTreasury(address indexed reserve, uint256 amountMinted);

  /**
   * @dev Mints an `amount` of aTokens to the `onBehalfOf`
   * @param asset The address of the underlying asset to mint
   * @param amount The amount to mint
   * @param onBehalfOf The address that will receive the aTokens
   * @param referralCode Code used to register the integrator originating the operation, for potential rewards.
   *   0 if the action is executed directly by the user, without any middle-man
   **/
  function mintUnbacked(
    address asset,
    uint256 amount,
    address onBehalfOf,
    uint16 referralCode
  ) external;

  /**
   * @dev Back the current unbacked underlying with `amount` and pay `fee`.
   * @param asset The address of the underlying asset to back
   * @param amount The amount to back
   * @param fee The amount paid in fees
   **/
  function backUnbacked(
    address asset,
    uint256 amount,
    uint256 fee
  ) external;

  /**
   * @notice Supplies an `amount` of underlying asset into the reserve, receiving in return overlying aTokens.
   * - E.g. User supplies 100 USDC and gets in return 100 aUSDC
   * @param asset The address of the underlying asset to supply
   * @param amount The amount to be supplied
   * @param onBehalfOf The address that will receive the aTokens, same as msg.sender if the user
   *   wants to receive them on his own wallet, or a different address if the beneficiary of aTokens
   *   is a different wallet
   * @param referralCode Code used to register the integrator originating the operation, for potential rewards.
   *   0 if the action is executed directly by the user, without any middle-man
   **/
  function supply(
    address asset,
    uint256 amount,
    address onBehalfOf,
    uint16 referralCode
  ) external;

  /**
   * @notice Supply with transfer approval of asset to be supplied done via permit function
   * see: https://eips.ethereum.org/EIPS/eip-2612 and https://eips.ethereum.org/EIPS/eip-713
   * @param asset The address of the underlying asset to supply
   * @param amount The amount to be supplied
   * @param onBehalfOf The address that will receive the aTokens, same as msg.sender if the user
   *   wants to receive them on his own wallet, or a different address if the beneficiary of aTokens
   *   is a different wallet
   * @param deadline The deadline timestamp that the permit is valid
   * @param referralCode Code used to register the integrator originating the operation, for potential rewards.
   *   0 if the action is executed directly by the user, without any middle-man
   * @param permitV The V parameter of ERC712 permit sig
   * @param permitR The R parameter of ERC712 permit sig
   * @param permitS The S parameter of ERC712 permit sig
   **/
  function supplyWithPermit(
    address asset,
    uint256 amount,
    address onBehalfOf,
    uint16 referralCode,
    uint256 deadline,
    uint8 permitV,
    bytes32 permitR,
    bytes32 permitS
  ) external;

  /**
   * @notice Withdraws an `amount` of underlying asset from the reserve, burning the equivalent aTokens owned
   * E.g. User has 100 aUSDC, calls withdraw() and receives 100 USDC, burning the 100 aUSDC
   * @param asset The address of the underlying asset to withdraw
   * @param amount The underlying amount to be withdrawn
   *   - Send the value type(uint256).max in order to withdraw the whole aToken balance
   * @param to The address that will receive the underlying, same as msg.sender if the user
   *   wants to receive it on his own wallet, or a different address if the beneficiary is a
   *   different wallet
   * @return The final amount withdrawn
   **/
  function withdraw(
    address asset,
    uint256 amount,
    address to
  ) external returns (uint256);

  /**
   * @notice Allows users to borrow a specific `amount` of the reserve underlying asset, provided that the borrower
   * already supplied enough collateral, or he was given enough allowance by a credit delegator on the
   * corresponding debt token (StableDebtToken or VariableDebtToken)
   * - E.g. User borrows 100 USDC passing as `onBehalfOf` his own address, receiving the 100 USDC in his wallet
   *   and 100 stable/variable debt tokens, depending on the `interestRateMode`
   * @param asset The address of the underlying asset to borrow
   * @param amount The amount to be borrowed
   * @param interestRateMode The interest rate mode at which the user wants to borrow: 1 is deprecated, 2 for Variable
   * @param referralCode The code used to register the integrator originating the operation, for potential rewards.
   *   0 if the action is executed directly by the user, without any middle-man
   * @param onBehalfOf The address of the user who will receive the debt. Should be the address of the borrower itself
   * calling the function if he wants to borrow against his own collateral, or the address of the credit delegator
   * if he has been given credit delegation allowance
   **/
  function borrow(
    address asset,
    uint256 amount,
    uint256 interestRateMode,
    uint16 referralCode,
    address onBehalfOf
  ) external;

  /**
   * @notice Repays a borrowed `amount` on a specific reserve, burning the equivalent debt tokens owned
   * - E.g. User repays 100 USDC, burning 100 variable/stable debt tokens of the `onBehalfOf` address
   * @param asset The address of the borrowed underlying asset previously borrowed
   * @param amount The amount to repay
   * - Send the value type(uint256).max in order to repay the whole debt for `asset` on the specific `debtMode`
   * @param interestRateMode The interest rate mode at of the debt the user wants to repay: 1 is deprecated, 2 for Variable
   * @param onBehalfOf The address of the user who will get his debt reduced/removed. Should be the address of the
   * user calling the function if he wants to reduce/remove his own debt, or the address of any other
   * other borrower whose debt should be removed
   * @return The final amount repaid
   **/
  function repay(
    address asset,
    uint256 amount,
    uint256 interestRateMode,
    address onBehalfOf
  ) external returns (uint256);

  /**
   * @notice Repay with transfer approval of asset to be repaid done via permit function
   * see: https://eips.ethereum.org/EIPS/eip-2612 and https://eips.ethereum.org/EIPS/eip-713
   * @param asset The address of the borrowed underlying asset previously borrowed
   * @param amount The amount to repay
   * - Send the value type(uint256).max in order to repay the whole debt for `asset` on the specific `debtMode`
   * @param interestRateMode The interest rate mode at of the debt the user wants to repay: 1 is deprecated, 2 for Variable
   * @param onBehalfOf Address of the user who will get his debt reduced/removed. Should be the address of the
   * user calling the function if he wants to reduce/remove his own debt, or the address of any other
   * other borrower whose debt should be removed
   * @param deadline The deadline timestamp that the permit is valid
   * @param permitV The V parameter of ERC712 permit sig
   * @param permitR The R parameter of ERC712 permit sig
   * @param permitS The S parameter of ERC712 permit sig
   * @return The final amount repaid
   **/
  function repayWithPermit(
    address asset,
    uint256 amount,
    uint256 interestRateMode,
    address onBehalfOf,
    uint256 deadline,
    uint8 permitV,
    bytes32 permitR,
    bytes32 permitS
  ) external returns (uint256);

  /**
   * @notice Repays a borrowed `amount` on a specific reserve using the reserve aTokens, burning the
   * equivalent debt tokens
   * - E.g. User repays 100 USDC using 100 aUSDC, burning 100 variable/stable debt tokens
   * @dev  Passing uint256.max as amount will clean up any residual aToken dust balance, if the user aToken
   * balance is not enough to cover the whole debt
   * @param asset The address of the borrowed underlying asset previously borrowed
   * @param amount The amount to repay
   * - Send the value type(uint256).max in order to repay the whole debt for `asset` on the specific `debtMode`
   * @param interestRateMode The interest rate mode at of the debt the user wants to repay: 1 is deprecated, 2 for Variable
   * @return The final amount repaid
   **/
  function repayWithATokens(
    address asset,
    uint256 amount,
    uint256 interestRateMode
  ) external returns (uint256);

  /**
   * @notice Allows a borrower to swap his debt between stable and variable mode, or vice versa
   * @param asset The address of the underlying asset borrowed
   * @param interestRateMode The current interest rate mode of the position being swapped: 1 for Stable, 2 for Variable
   **/
  function swapBorrowRateMode(address asset, uint256 interestRateMode) external;

  /**
   * @notice Rebalances the stable interest rate of a user to the current stable rate defined on the reserve.
   * - Users can be rebalanced if the following conditions are satisfied:
   *     1. Usage ratio is above 95%
   *     2. the current supply APY is below REBALANCE_UP_THRESHOLD * maxVariableBorrowRate, which means that too
   *        much has been borrowed at a stable rate and suppliers are not earning enough
   * @param asset The address of the underlying asset borrowed
   * @param user The address of the user to be rebalanced
   **/
  function rebalanceStableBorrowRate(address asset, address user) external;

  /**
   * @notice Allows suppliers to enable/disable a specific supplied asset as collateral
   * @param asset The address of the underlying asset supplied
   * @param useAsCollateral True if the user wants to use the supply as collateral, false otherwise
   **/
  function setUserUseReserveAsCollateral(address asset, bool useAsCollateral) external;

  /**
   * @notice Function to liquidate a non-healthy position collateral-wise, with Health Factor below 1
   * - The caller (liquidator) covers `debtToCover` amount of debt of the user getting liquidated, and receives
   *   a proportionally amount of the `collateralAsset` plus a bonus to cover market risk
   * @param collateralAsset The address of the underlying asset used as collateral, to receive as result of the liquidation
   * @param debtAsset The address of the underlying borrowed asset to be repaid with the liquidation
   * @param user The address of the borrower getting liquidated
   * @param debtToCover The debt amount of borrowed `asset` the liquidator wants to cover
   * @param receiveAToken True if the liquidators wants to receive the collateral aTokens, `false` if he wants
   * to receive the underlying collateral asset directly
   **/
  function liquidationCall(
    address collateralAsset,
    address debtAsset,
    address user,
    uint256 debtToCover,
    bool receiveAToken
  ) external;

  /**
   * @notice Allows smartcontracts to access the liquidity of the pool within one transaction,
   * as long as the amount taken plus a fee is returned.
   * @dev IMPORTANT There are security concerns for developers of flashloan receiver contracts that must be kept
   * into consideration. For further details please visit https://developers.aave.com
   * @param receiverAddress The address of the contract receiving the funds, implementing IFlashLoanReceiver interface
   * @param assets The addresses of the assets being flash-borrowed
   * @param amounts The amounts of the assets being flash-borrowed
   * @param interestRateModes Types of the debt to open if the flash loan is not returned:
   *   0 -> Don't open any debt, just revert if funds can't be transferred from the receiver
   *   1 -> Open debt at stable rate for the value of the amount flash-borrowed to the `onBehalfOf` address
   *   2 -> Open debt at variable rate for the value of the amount flash-borrowed to the `onBehalfOf` address
   * @param onBehalfOf The address  that will receive the debt in the case of using on `modes` 1 or 2
   * @param params Variadic packed params to pass to the receiver as extra information
   * @param referralCode The code used to register the integrator originating the operation, for potential rewards.
   *   0 if the action is executed directly by the user, without any middle-man
   **/
  function flashLoan(
    address receiverAddress,
    address[] calldata assets,
    uint256[] calldata amounts,
    uint256[] calldata interestRateModes,
    address onBehalfOf,
    bytes calldata params,
    uint16 referralCode
  ) external;

  /**
   * @notice Allows smartcontracts to access the liquidity of the pool within one transaction,
   * as long as the amount taken plus a fee is returned.
   * @dev IMPORTANT There are security concerns for developers of flashloan receiver contracts that must be kept
   * into consideration. For further details please visit https://developers.aave.com
   * @param receiverAddress The address of the contract receiving the funds, implementing IFlashLoanSimpleReceiver interface
   * @param asset The address of the asset being flash-borrowed
   * @param amount The amount of the asset being flash-borrowed
   * @param params Variadic packed params to pass to the receiver as extra information
   * @param referralCode The code used to register the integrator originating the operation, for potential rewards.
   *   0 if the action is executed directly by the user, without any middle-man
   **/
  function flashLoanSimple(
    address receiverAddress,
    address asset,
    uint256 amount,
    bytes calldata params,
    uint16 referralCode
  ) external;

  /**
   * @notice Returns the user account data across all the reserves
   * @param user The address of the user
   * @return totalCollateralBase The total collateral of the user in the base currency used by the price feed
   * @return totalDebtBase The total debt of the user in the base currency used by the price feed
   * @return availableBorrowsBase The borrowing power left of the user in the base currency used by the price feed
   * @return currentLiquidationThreshold The liquidation threshold of the user
   * @return ltv The loan to value of The user
   * @return healthFactor The current health factor of the user
   **/
  function getUserAccountData(address user)
    external
    view
    returns (
      uint256 totalCollateralBase,
      uint256 totalDebtBase,
      uint256 availableBorrowsBase,
      uint256 currentLiquidationThreshold,
      uint256 ltv,
      uint256 healthFactor
    );

  /**
   * @notice Initializes a reserve, activating it, assigning an aToken and debt tokens and an
   * interest rate strategy
   * @dev Only callable by the PoolConfigurator contract
   * @param asset The address of the underlying asset of the reserve
   * @param aTokenAddress The address of the aToken that will be assigned to the reserve
   * @param stableDebtAddress The address of the StableDebtToken that will be assigned to the reserve
   * @param variableDebtAddress The address of the VariableDebtToken that will be assigned to the reserve
   * @param interestRateStrategyAddress The address of the interest rate strategy contract
   **/
  function initReserve(
    address asset,
    address aTokenAddress,
    address stableDebtAddress,
    address variableDebtAddress,
    address interestRateStrategyAddress
  ) external;

  /**
   * @notice Drop a reserve
   * @dev Only callable by the PoolConfigurator contract
   * @param asset The address of the underlying asset of the reserve
   **/
  function dropReserve(address asset) external;

  /**
   * @notice Updates the address of the interest rate strategy contract
   * @dev Only callable by the PoolConfigurator contract
   * @param asset The address of the underlying asset of the reserve
   * @param rateStrategyAddress The address of the interest rate strategy contract
   **/
  function setReserveInterestRateStrategyAddress(address asset, address rateStrategyAddress)
    external;

  /**
   * @notice Sets the configuration bitmap of the reserve as a whole
   * @dev Only callable by the PoolConfigurator contract
   * @param asset The address of the underlying asset of the reserve
   * @param configuration The new configuration bitmap
   **/
  function setConfiguration(address asset, DataTypes.ReserveConfigurationMap calldata configuration)
    external;

  /**
   * @notice Returns the configuration of the reserve
   * @param asset The address of the underlying asset of the reserve
   * @return The configuration of the reserve
   **/
  function getConfiguration(address asset)
    external
    view
    returns (DataTypes.ReserveConfigurationMap memory);

  /**
   * @notice Returns the configuration of the user across all the reserves
   * @param user The user address
   * @return The configuration of the user
   **/
  function getUserConfiguration(address user)
    external
    view
    returns (DataTypes.UserConfigurationMap memory);

  /**
   * @notice Returns the normalized income normalized income of the reserve
   * @param asset The address of the underlying asset of the reserve
   * @return The reserve's normalized income
   */
  function getReserveNormalizedIncome(address asset) external view returns (uint256);

  /**
   * @notice Returns the normalized variable debt per unit of asset
   * @param asset The address of the underlying asset of the reserve
   * @return The reserve normalized variable debt
   */
  function getReserveNormalizedVariableDebt(address asset) external view returns (uint256);

  /**
   * @notice Returns the state and configuration of the reserve
   * @param asset The address of the underlying asset of the reserve
   * @return The state and configuration data of the reserve
   **/
  function getReserveData(address asset) external view returns (DataTypes.ReserveData memory);

  /**
   * @notice Returns the state and configuration of the reserve
   * @param asset The address of the underlying asset of the reserve
   * @return The state and configuration data of the reserve
   **/
  function getReserveDataExtended(address asset) external view returns (DataTypes.ReserveDataExtended memory);

  /**
   * @notice Validates and finalizes an aToken transfer
   * @dev Only callable by the overlying aToken of the `asset`
   * @param asset The address of the underlying asset of the aToken
   * @param from The user from which the aTokens are transferred
   * @param to The user receiving the aTokens
   * @param amount The amount being transferred/withdrawn
   * @param balanceFromBefore The aToken balance of the `from` user before the transfer
   * @param balanceToBefore The aToken balance of the `to` user before the transfer
   */
  function finalizeTransfer(
    address asset,
    address from,
    address to,
    uint256 amount,
    uint256 balanceFromBefore,
    uint256 balanceToBefore
  ) external;

  /**
   * @notice Returns the list of the initialized reserves
   * @dev It does not include dropped reserves
   * @return The addresses of the reserves
   **/
  function getReservesList() external view returns (address[] memory);

  /**
   * @notice Returns the PoolAddressesProvider connected to this contract
   * @return The address of the PoolAddressesProvider
   **/
  function ADDRESSES_PROVIDER() external view returns (IPoolAddressesProvider);

  /**
   * @notice Updates the protocol fee on the bridging
   * @param bridgeProtocolFee The part of the premium sent to the protocol treasury
   */
  function updateBridgeProtocolFee(uint256 bridgeProtocolFee) external;

  /**
   * @notice Updates flash loan premiums. Flash loan premium consists of two parts:
   * - A part is sent to aToken holders as extra, one time accumulated interest
   * - A part is collected by the protocol treasury
   * @dev The total premium is calculated on the total borrowed amount
   * @dev The premium to protocol is calculated on the total premium, being a percentage of `flashLoanPremiumTotal`
   * @dev Only callable by the PoolConfigurator contract
   * @param flashLoanPremiumTotal The total premium, expressed in bps
   * @param flashLoanPremiumToProtocol The part of the premium sent to the protocol treasury, expressed in bps
   */
  function updateFlashloanPremiums(
    uint128 flashLoanPremiumTotal,
    uint128 flashLoanPremiumToProtocol
  ) external;

  /**
   * @notice Configures a new category for the eMode.
   * @dev In eMode, the protocol allows very high borrowing power to borrow assets of the same category.
   * The category 0 is reserved as it's the default for volatile assets
   * @param id The id of the category
   * @param config The configuration of the category
   */
  function configureEModeCategory(uint8 id, DataTypes.EModeCategory memory config) external;

  /**
   * @notice Returns the data of an eMode category
   * @param id The id of the category
   * @return The configuration data of the category
   */
  function getEModeCategoryData(uint8 id) external view returns (DataTypes.EModeCategoryLegacy memory);

  /**
   * @notice Allows a user to use the protocol in eMode
   * @param categoryId The id of the category
   */
  function setUserEMode(uint8 categoryId) external;

  /**
   * @notice Returns the eMode the user is using
   * @param user The address of the user
   * @return The eMode id
   */
  function getUserEMode(address user) external view returns (uint256);

  /**
   * @notice Resets the isolation mode total debt of the given asset to zero
   * @dev It requires the given asset has zero debt ceiling
   * @param asset The address of the underlying asset to reset the isolationModeTotalDebt
   */
  function resetIsolationModeTotalDebt(address asset) external;

  /**
   * @notice Returns the percentage of available liquidity that can be borrowed at once at stable rate
   * @return The percentage of available liquidity to borrow, expressed in bps
   */
  function MAX_STABLE_RATE_BORROW_SIZE_PERCENT() external view returns (uint256);

  /**
   * @notice Returns the total fee on flash loans
   * @return The total fee on flashloans
   */
  function FLASHLOAN_PREMIUM_TOTAL() external view returns (uint128);

  /**
   * @notice Returns the part of the bridge fees sent to protocol
   * @return The bridge fee sent to the protocol treasury
   */
  function BRIDGE_PROTOCOL_FEE() external view returns (uint256);

  /**
   * @notice Returns the part of the flashloan fees sent to protocol
   * @return The flashloan fee sent to the protocol treasury
   */
  function FLASHLOAN_PREMIUM_TO_PROTOCOL() external view returns (uint128);

  /**
   * @notice Returns the maximum number of reserves supported to be listed in this Pool
   * @return The maximum number of reserves supported
   */
  function MAX_NUMBER_RESERVES() external view returns (uint16);

  /**
   * @notice Mints the assets accrued through the reserve factor to the treasury in the form of aTokens
   * @param assets The list of reserves for which the minting needs to be executed
   **/
  function mintToTreasury(address[] calldata assets) external;

  /**
   * @notice Rescue and transfer tokens locked in this contract
   * @param token The address of the token
   * @param to The address of the recipient
   * @param amount The amount of token to transfer
   */
  function rescueTokens(
    address token,
    address to,
    uint256 amount
  ) external;

  /**
   * @notice Supplies an `amount` of underlying asset into the reserve, receiving in return overlying aTokens.
   * - E.g. User supplies 100 USDC and gets in return 100 aUSDC
   * @dev Deprecated: Use the `supply` function instead
   * @param asset The address of the underlying asset to supply
   * @param amount The amount to be supplied
   * @param onBehalfOf The address that will receive the aTokens, same as msg.sender if the user
   *   wants to receive them on his own wallet, or a different address if the beneficiary of aTokens
   *   is a different wallet
   * @param referralCode Code used to register the integrator originating the operation, for potential rewards.
   *   0 if the action is executed directly by the user, without any middle-man
   **/
  function deposit(
    address asset,
    uint256 amount,
    address onBehalfOf,
    uint16 referralCode
  ) external;

  /**
   * @notice Returns the address of the underlying asset of a reserve by the reserve id as stored in the DataTypes.ReserveData struct
   * @param id The id of the reserve as stored in the DataTypes.ReserveData struct
   * @return The address of the reserve associated with id
   **/
  function getReserveAddressById(uint16 id) external view returns (address);

  
    /**
   * @notice Returns the label of an eMode category
   * @param id The id of the category
   * @return The label of the category
   */
  function getEModeCategoryLabel(uint8 id) external view returns (string memory);

  /**
   * @notice Returns the collateral config of an eMode category
   * @param id The id of the category
   * @return The ltv,lt,lb of the category
   */
  function getEModeCategoryCollateralConfig(
    uint8 id
  ) external view returns (DataTypes.CollateralConfig memory);

  /**
   * @notice Returns the collateralBitmap of an eMode category
   * @param id The id of the category
   * @return The collateralBitmap of the category
   */
  function getEModeCategoryCollateralBitmap(uint8 id) external view returns (uint128);

  /**
   * @notice Returns the borrowableBitmap of an eMode category
   * @param id The id of the category
   * @return The borrowableBitmap of the category
   */
  function getEModeCategoryBorrowableBitmap(uint8 id) external view returns (uint128);

}







interface IL2PoolV3 is IPoolV3{
  /**
   * @notice Calldata efficient wrapper of the supply function on behalf of the caller
   * @param args Arguments for the supply function packed in one bytes32
   *    96 bits       16 bits         128 bits      16 bits
   * | 0-padding | referralCode | shortenedAmount | assetId |
   * @dev the shortenedAmount is cast to 256 bits at decode time, if type(uint128).max the value will be expanded to
   * type(uint256).max
   * @dev assetId is the index of the asset in the reservesList.
   */
  function supply(bytes32 args) external;

  /**
   * @notice Calldata efficient wrapper of the supplyWithPermit function on behalf of the caller
   * @param args Arguments for the supply function packed in one bytes32
   *    56 bits    8 bits         32 bits           16 bits         128 bits      16 bits
   * | 0-padding | permitV | shortenedDeadline | referralCode | shortenedAmount | assetId |
   * @dev the shortenedAmount is cast to 256 bits at decode time, if type(uint128).max the value will be expanded to
   * type(uint256).max
   * @dev assetId is the index of the asset in the reservesList.
   * @param r The R parameter of ERC712 permit sig
   * @param s The S parameter of ERC712 permit sig
   */
  function supplyWithPermit(
    bytes32 args,
    bytes32 r,
    bytes32 s
  ) external;

  /**
   * @notice Calldata efficient wrapper of the withdraw function, withdrawing to the caller
   * @param args Arguments for the withdraw function packed in one bytes32
   *    112 bits       128 bits      16 bits
   * | 0-padding | shortenedAmount | assetId |
   * @dev the shortenedAmount is cast to 256 bits at decode time, if type(uint128).max the value will be expanded to
   * type(uint256).max
   * @dev assetId is the index of the asset in the reservesList.
   */
  function withdraw(bytes32 args) external;

  /**
   * @notice Calldata efficient wrapper of the borrow function, borrowing on behalf of the caller
   * @param args Arguments for the borrow function packed in one bytes32
   *    88 bits       16 bits             8 bits                 128 bits       16 bits
   * | 0-padding | referralCode | shortenedInterestRateMode | shortenedAmount | assetId |
   * @dev the shortenedAmount is cast to 256 bits at decode time, if type(uint128).max the value will be expanded to
   * type(uint256).max
   * @dev assetId is the index of the asset in the reservesList.
   */
  function borrow(bytes32 args) external;

  /**
   * @notice Calldata efficient wrapper of the repay function, repaying on behalf of the caller
   * @param args Arguments for the repay function packed in one bytes32
   *    104 bits             8 bits               128 bits       16 bits
   * | 0-padding | shortenedInterestRateMode | shortenedAmount | assetId |
   * @dev the shortenedAmount is cast to 256 bits at decode time, if type(uint128).max the value will be expanded to
   * type(uint256).max
   * @dev assetId is the index of the asset in the reservesList.
   * @return The final amount repaid
   */
  function repay(bytes32 args) external returns (uint256);

  /**
   * @notice Calldata efficient wrapper of the repayWithPermit function, repaying on behalf of the caller
   * @param args Arguments for the repayWithPermit function packed in one bytes32
   *    64 bits    8 bits        32 bits                   8 bits               128 bits       16 bits
   * | 0-padding | permitV | shortenedDeadline | shortenedInterestRateMode | shortenedAmount | assetId |
   * @dev the shortenedAmount is cast to 256 bits at decode time, if type(uint128).max the value will be expanded to
   * type(uint256).max
   * @dev assetId is the index of the asset in the reservesList.
   * @param r The R parameter of ERC712 permit sig
   * @param s The S parameter of ERC712 permit sig
   * @return The final amount repaid
   */
  function repayWithPermit(
    bytes32 args,
    bytes32 r,
    bytes32 s
  ) external returns (uint256);

  /**
   * @notice Calldata efficient wrapper of the repayWithATokens function
   * @param args Arguments for the repayWithATokens function packed in one bytes32
   *    104 bits             8 bits               128 bits       16 bits
   * | 0-padding | shortenedInterestRateMode | shortenedAmount | assetId |
   * @dev the shortenedAmount is cast to 256 bits at decode time, if type(uint128).max the value will be expanded to
   * type(uint256).max
   * @dev assetId is the index of the asset in the reservesList.
   * @return The final amount repaid
   */
  function repayWithATokens(bytes32 args) external returns (uint256);

  /**
   * @notice Calldata efficient wrapper of the swapBorrowRateMode function
   * @param args Arguments for the swapBorrowRateMode function packed in one bytes32
   *    232 bits            8 bits             16 bits
   * | 0-padding | shortenedInterestRateMode | assetId |
   * @dev assetId is the index of the asset in the reservesList.
   */
  function swapBorrowRateMode(bytes32 args) external;

  /**
   * @notice Calldata efficient wrapper of the rebalanceStableBorrowRate function
   * @param args Arguments for the rebalanceStableBorrowRate function packed in one bytes32
   *    80 bits      160 bits     16 bits
   * | 0-padding | user address | assetId |
   * @dev assetId is the index of the asset in the reservesList.
   */
  function rebalanceStableBorrowRate(bytes32 args) external;

  /**
   * @notice Calldata efficient wrapper of the setUserUseReserveAsCollateral function
   * @param args Arguments for the setUserUseReserveAsCollateral function packed in one bytes32
   *    239 bits         1 bit       16 bits
   * | 0-padding | useAsCollateral | assetId |
   * @dev assetId is the index of the asset in the reservesList.
   */
  function setUserUseReserveAsCollateral(bytes32 args) external;

  /**
   * @notice Calldata efficient wrapper of the liquidationCall function
   * @param args1 part of the arguments for the liquidationCall function packed in one bytes32
   *    64 bits      160 bits       16 bits         16 bits
   * | 0-padding | user address | debtAssetId | collateralAssetId |
   * @param args2 part of the arguments for the liquidationCall function packed in one bytes32
   *    127 bits       1 bit             128 bits
   * | 0-padding | receiveAToken | shortenedDebtToCover |
   * @dev the shortenedDebtToCover is cast to 256 bits at decode time,
   * if type(uint128).max the value will be expanded to type(uint256).max
   */
  function liquidationCall(bytes32 args1, bytes32 args2) external;
}










contract AaveV3Helper is MainnetAaveV3Addresses {
    
    uint16 public constant AAVE_REFERRAL_CODE = 64;

    /// @notice Returns the lending pool contract of the specified market
    function getLendingPool(address _market) internal virtual view returns (IL2PoolV3) {
        return IL2PoolV3(IPoolAddressesProvider(_market).getPool());
    }

    /// @notice Fetch the data provider for the specified market
    function getDataProvider(address _market) internal virtual view returns (IAaveProtocolDataProvider) {
        return
            IAaveProtocolDataProvider(
                IPoolAddressesProvider(_market).getPoolDataProvider()
            );
    }

    function boolToBytes(bool x) internal virtual pure returns (bytes1 r) {
       return x ? bytes1(0x01) : bytes1(0x00);
    }

    function bytesToBool(bytes1 x) internal virtual pure returns (bool r) {
        return x != bytes1(0x00);
    }
    
    function getWholeDebt(address _market, address _tokenAddr, uint _borrowType, address _debtOwner) internal virtual view returns (uint256 debt) {
        uint256 STABLE_ID = 1;
        uint256 VARIABLE_ID = 2;

        IAaveProtocolDataProvider dataProvider = getDataProvider(_market);
        (, uint256 borrowsStable, uint256 borrowsVariable, , , , , , ) =
            dataProvider.getUserReserveData(_tokenAddr, _debtOwner);

        if (_borrowType == STABLE_ID) {
            debt = borrowsStable;
        } else if (_borrowType == VARIABLE_ID) {
            debt = borrowsVariable;
        }
    }
}







contract MainnetCompV3Addresses {
    address internal constant COMET_REWARDS_ADDR = 0x1B0e765F6224C21223AeA2af16c1C46E38885a40;
    address internal constant COMP_ETH_COMET = 0xA17581A9E3356d9A858b789D68B4d866e593aE94;
}







abstract contract IComet {

    struct AssetInfo {
        uint8 offset;
        address asset;
        address priceFeed;
        uint64 scale;
        uint64 borrowCollateralFactor;
        uint64 liquidateCollateralFactor;
        uint64 liquidationFactor;
        uint128 supplyCap;
    }

    struct TotalsCollateral {
        uint128 totalSupplyAsset;
        uint128 _reserved;
    }

      struct UserCollateral {
        uint128 balance;
        uint128 _reserved;
    }


    struct UserBasic {
        int104 principal;
        uint64 baseTrackingIndex;
        uint64 baseTrackingAccrued;
        uint16 assetsIn;
        uint8 _reserved;
    }

    struct TotalsBasic {
        uint64 baseSupplyIndex;
        uint64 baseBorrowIndex;
        uint64 trackingSupplyIndex;
        uint64 trackingBorrowIndex;
        uint104 totalSupplyBase;
        uint104 totalBorrowBase;
        uint40 lastAccrualTime;
        uint8 pauseFlags;
    }

    function totalsBasic() public virtual view returns (TotalsBasic memory);

    function totalsCollateral(address) public virtual returns (TotalsCollateral memory);
    
    function supply(address asset, uint amount) virtual external;
    function supplyTo(address dst, address asset, uint amount) virtual external;
    function supplyFrom(address from, address dst, address asset, uint amount) virtual external;

    function transfer(address dst, uint amount) virtual external returns (bool);
    function transferFrom(address src, address dst, uint amount) virtual external returns (bool);

    function transferAsset(address dst, address asset, uint amount) virtual external;
    function transferAssetFrom(address src, address dst, address asset, uint amount) virtual external;

    function withdraw(address asset, uint amount) virtual external;
    function withdrawTo(address to, address asset, uint amount) virtual external;
    function withdrawFrom(address src, address to, address asset, uint amount) virtual external;

    function accrueAccount(address account) virtual external;
    function getSupplyRate(uint utilization) virtual public view returns (uint64);
    function getBorrowRate(uint utilization) virtual public view returns (uint64);
    function getUtilization() virtual public view returns (uint);

    function governor() virtual external view returns (address);
    function baseToken() virtual external view returns (address);
    function baseTokenPriceFeed() virtual external view returns (address);

    function balanceOf(address account) virtual public view returns (uint256);
    function collateralBalanceOf(address account, address asset) virtual external view returns (uint128);
    function borrowBalanceOf(address account) virtual public view returns (uint256);
    function totalSupply() virtual external view returns (uint256);

    function numAssets() virtual public view returns (uint8);
    function getAssetInfo(uint8 i) virtual public view returns (AssetInfo memory);
    function getAssetInfoByAddress(address asset) virtual public view returns (AssetInfo memory);
    function getPrice(address priceFeed) virtual public view returns (uint256);

    function allow(address manager, bool isAllowed) virtual external;
    function allowance(address owner, address spender) virtual external view returns (uint256);

    function isSupplyPaused() virtual external view returns (bool);
    function isTransferPaused() virtual external view returns (bool);
    function isWithdrawPaused() virtual external view returns (bool);
    function isAbsorbPaused() virtual external view returns (bool);
    function baseIndexScale() virtual external pure returns (uint64);

    function userBasic(address) virtual external view returns (UserBasic memory);
    function userCollateral(address, address) virtual external view returns (UserCollateral memory);
    function priceScale() virtual external pure returns (uint64);
    function factorScale() virtual external pure returns (uint64);

    function baseBorrowMin() virtual external pure returns (uint256);
    function baseTrackingBorrowSpeed() virtual external pure returns (uint256);
    function baseTrackingSupplySpeed() virtual external pure returns (uint256);

}








contract CompV3Helper is MainnetCompV3Addresses {
    
}







contract LSVUtilMainnetAddresses {
    address internal constant RETH_ADDRESS = 0xae78736Cd615f374D3085123A210448E74Fc6393;
    address internal constant CBETH_ADDRESS = 0xBe9895146f7AF43049ca1c1AE358B0541Ea49704;
    address internal constant WSTETH_ADDRESS = 0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0;
    address internal constant WEETH_ADDRESS = 0xCd5fE23C85820F7B72D0926FC9b05b43E359b7ee;
    address internal constant EZETH_ADDRESS = 0xbf5495Efe5DB9ce00f80364C8B423567e58d2110;
    address internal constant RENZO_MANAGER = 0x74a09653A083691711cF8215a6ab074BB4e99ef5;
    address internal constant LSV_PROFIT_TRACKER_ADDRESS = 0xa5941F267995553dA9369F04Dc02f9411EEb2F12;
}






contract DSMath {
    function add(uint256 x, uint256 y) internal pure returns (uint256 z) {
        z = x + y;
    }

    function sub(uint256 x, uint256 y) internal pure returns (uint256 z) {
        z = x - y;
    }

    function mul(uint256 x, uint256 y) internal pure returns (uint256 z) {
        z = x * y;
    }

    function div(uint256 x, uint256 y) internal pure returns (uint256 z) {
        return x / y;
    }

    function min(uint256 x, uint256 y) internal pure returns (uint256 z) {
        return x <= y ? x : y;
    }

    function max(uint256 x, uint256 y) internal pure returns (uint256 z) {
        return x >= y ? x : y;
    }

    function imin(int256 x, int256 y) internal pure returns (int256 z) {
        return x <= y ? x : y;
    }

    function imax(int256 x, int256 y) internal pure returns (int256 z) {
        return x >= y ? x : y;
    }

    uint256 constant WAD = 10**18;
    uint256 constant RAY = 10**27;

    function wmul(uint256 x, uint256 y) internal pure returns (uint256 z) {
        z = add(mul(x, y), WAD / 2) / WAD;
    }

    function rmul(uint256 x, uint256 y) internal pure returns (uint256 z) {
        z = add(mul(x, y), RAY / 2) / RAY;
    }

    function wdiv(uint256 x, uint256 y) internal pure returns (uint256 z) {
        z = add(mul(x, WAD), y / 2) / y;
    }

    function rdiv(uint256 x, uint256 y) internal pure returns (uint256 z) {
        z = add(mul(x, RAY), y / 2) / y;
    }

    // This famous algorithm is called "exponentiation by squaring"
    // and calculates x^n with x as fixed-point and n as regular unsigned.
    //
    // It's O(log n), instead of O(n) for naive repeated multiplication.
    //
    // These facts are why it works:
    //
    //  If n is even, then x^n = (x^2)^(n/2).
    //  If n is odd,  then x^n = x * x^(n-1),
    //   and applying the equation for even x gives
    //    x^n = x * (x^2)^((n-1) / 2).
    //
    //  Also, EVM division is flooring and
    //    floor[(n-1) / 2] = floor[n / 2].
    //
    function rpow(uint256 x, uint256 n) internal pure returns (uint256 z) {
        z = n % 2 != 0 ? x : RAY;

        for (n /= 2; n != 0; n /= 2) {
            x = rmul(x, x);

            if (n % 2 != 0) {
                z = rmul(z, x);
            }
        }
    }
}







interface IWeEth {
    function wrap(uint256 _eETHAmount) external returns (uint256);
	function unwrap(uint256 _weETHAmount) external returns (uint256);

    /// @notice Fetches the amount of weEth respective to the amount of eEth sent in
    function getWeETHByeETH(uint256 _eETHAmount) external view returns (uint256);

    /// @notice Fetches the amount of eEth respective to the amount of weEth sent in
    function getEETHByWeETH(uint256 _weETHAmount) external view returns (uint256);

    /// @notice Fetches the exchange rate of eETH for 1 weETH
    /// @return The amount of eETH for 1 weETH
    function getRate() external view returns (uint256);
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







interface ICBETH {
    function exchangeRate() external view returns (uint256 wethAmountPerCBETH);
}







interface IRETH {
    function getEthValue(uint256 rethAmount) external view returns (uint256 wethAmount);
    function getRethValue(uint256 wethAmount) external view returns (uint256 rethAmount);
}







interface IWstETH {
    function getStETHByWstETH(uint256 wstethAmount) external view returns(uint256 wethAmount);
    function getWstETHByStETH(uint256 wethAmount) external view returns(uint256 wstethAmount);
}






interface IRenzoOracle {
    function calculateMintAmount(
        uint256 _currentValueInProtocol,
        uint256 _newValueAdded,
        uint256 _existingEzETHSupply
    ) external pure returns (uint256);

    function calculateRedeemAmount(
        uint256 _ezETHBeingBurned,
        uint256 _existingEzETHSupply,
        uint256 _currentValueInProtocol
    ) external pure returns (uint256);
}






interface IRestakeManager {
    function depositETH() external payable;
    function renzoOracle() external view returns (address);
    function calculateTVLs() external view returns (uint256[][] memory, uint256[] memory, uint256);
}









contract LSVProfitTracker{

    mapping(uint256 => mapping(address => int256)) public unrealisedProfit;

    function supply(uint256 _protocol, uint256 _amount) public {
        unrealisedProfit[_protocol][msg.sender] -= downCastUintToInt(_amount); 
    }

    function borrow(uint256 _protocol, uint256 _amount) public {
        unrealisedProfit[_protocol][msg.sender] += downCastUintToInt(_amount); 
    }
    
    function payback(uint256 _protocol, uint256 _amount) public {
        unrealisedProfit[_protocol][msg.sender] -= downCastUintToInt(_amount); 
    }

    function withdraw(uint256 _protocol, uint256 _amount,  bool _isClosingVault) public returns (uint256 realisedProfit){
        unrealisedProfit[_protocol][msg.sender] += downCastUintToInt(_amount);
        
        if (unrealisedProfit[_protocol][msg.sender] > 0){
            realisedProfit = uint256(unrealisedProfit[_protocol][msg.sender]);
            unrealisedProfit[_protocol][msg.sender] = 0;
        } else if (_isClosingVault) {
            unrealisedProfit[_protocol][msg.sender] = 0;
        }

        return realisedProfit;
    }

    function downCastUintToInt(uint256 uintAmount) internal pure returns(int256 amount){
        require(uintAmount <= uint256(type(int256).max));
        return int256(uintAmount);
    }
}
















contract LSVUtilHelper is DSMath, LSVUtilMainnetAddresses{
    
    function getAmountInETHFromLST(address lstAddress, uint256 lstAmount) public view returns (uint256 ethAmount){
        if (lstAmount == 0) return 0;

        if (lstAddress == RETH_ADDRESS){
            return IRETH(RETH_ADDRESS).getEthValue(lstAmount);
        }
        if (lstAddress == CBETH_ADDRESS){
            uint256 rate = ICBETH(CBETH_ADDRESS).exchangeRate();
            return wmul(lstAmount, rate);
        }
        if (lstAddress == WSTETH_ADDRESS){
            return IWstETH(WSTETH_ADDRESS).getStETHByWstETH(lstAmount);
        }
        if (lstAddress == WEETH_ADDRESS) {
            return IWeEth(WEETH_ADDRESS).getEETHByWeETH(lstAmount);
        }
        if (lstAddress == EZETH_ADDRESS) {
            return getRenzoRate(lstAmount, false);
        }

        return lstAmount;
    }

    function getAmountInLSTFromETH(address lstAddress, uint256 ethAmount) public view returns (uint256 lstAmount) {
        if (ethAmount == 0) return 0;

        if (lstAddress == RETH_ADDRESS){
            return IRETH(RETH_ADDRESS).getRethValue(ethAmount);
        }
        if (lstAddress == CBETH_ADDRESS){
            uint256 rate = ICBETH(CBETH_ADDRESS).exchangeRate();
            return wdiv(ethAmount, rate);
        }
        if (lstAddress == WSTETH_ADDRESS){
            return IWstETH(WSTETH_ADDRESS).getWstETHByStETH(ethAmount);
        }
        if (lstAddress == WEETH_ADDRESS) {
            return IWeEth(WEETH_ADDRESS).getWeETHByeETH(ethAmount);
        }
        if (lstAddress == EZETH_ADDRESS) {
            return getRenzoRate(ethAmount, true);
        }
        
        return ethAmount;
    }

    function getRenzoRate(uint256 _amount, bool _convertFromEthToLst) public view returns (uint256 rate) {
        IRestakeManager manager = IRestakeManager(RENZO_MANAGER);
        IRenzoOracle oracle = IRenzoOracle(manager.renzoOracle());
        (, , uint256 totalTVL) = manager.calculateTVLs();
        uint256 ezEthTotalSupply = IERC20(EZETH_ADDRESS).totalSupply();

        rate = _convertFromEthToLst ?
            oracle.calculateMintAmount(totalTVL, _amount, ezEthTotalSupply) :
            oracle.calculateRedeemAmount(_amount, ezEthTotalSupply, totalTVL);
    }
}






contract MainnetMorphoBlueAddresses {
    address public constant MORPHO_BLUE_ADDRESS = 0xBBBBBbbBBb9cC5e90e3b3Af64bdAF62C37EEFFCb;
    address public constant MORPHO_TOKEN_WRAPPER = 0x9D03bb2092270648d7480049d0E58d2FcF0E5123;
    address public constant LEGACY_MORPHO_TOKEN = 0x9994E35Db50125E0DF82e4c2dde62496CE330999;
    address public constant PUBLIC_ALLOCATOR = 0xfd32fA2ca22c76dD6E550706Ad913FC6CE91c75D;
}






type Id is bytes32;

struct MarketParams {
    address loanToken;
    address collateralToken;
    address oracle;
    address irm;
    uint256 lltv;
}



struct MorphoBluePosition {
    uint256 supplyShares;
    uint128 borrowShares;
    uint128 collateral;
}





struct Market {
    uint128 totalSupplyAssets;
    uint128 totalSupplyShares;
    uint128 totalBorrowAssets;
    uint128 totalBorrowShares;
    uint128 lastUpdate;
    uint128 fee;
}

struct Authorization {
    address authorizer;
    address authorized;
    bool isAuthorized;
    uint256 nonce;
    uint256 deadline;
}

struct Signature {
    uint8 v;
    bytes32 r;
    bytes32 s;
}



interface IMorphoBase {
    /// @notice The EIP-712 domain separator.
    /// @dev Warning: Every EIP-712 signed message based on this domain separator can be reused on another chain sharing
    /// the same chain id because the domain separator would be the same.
    function DOMAIN_SEPARATOR() external view returns (bytes32);

    /// @notice The owner of the contract.
    /// @dev It has the power to change the owner.
    /// @dev It has the power to set fees on markets and set the fee recipient.
    /// @dev It has the power to enable but not disable IRMs and LLTVs.
    function owner() external view returns (address);

    /// @notice The fee recipient of all markets.
    /// @dev The recipient receives the fees of a given market through a supply position on that market.
    function feeRecipient() external view returns (address);

    /// @notice Whether the `irm` is enabled.
    function isIrmEnabled(address irm) external view returns (bool);

    /// @notice Whether the `lltv` is enabled.
    function isLltvEnabled(uint256 lltv) external view returns (bool);

    /// @notice Whether `authorized` is authorized to modify `authorizer`'s positions.
    /// @dev Anyone is authorized to modify their own positions, regardless of this variable.
    function isAuthorized(address authorizer, address authorized) external view returns (bool);

    /// @notice The `authorizer`'s current nonce. Used to prevent replay attacks with EIP-712 signatures.
    function nonce(address authorizer) external view returns (uint256);

    /// @notice Sets `newOwner` as `owner` of the contract.
    /// @dev Warning: No two-step transfer ownership.
    /// @dev Warning: The owner can be set to the zero address.
    function setOwner(address newOwner) external;

    /// @notice Enables `irm` as a possible IRM for market creation.
    /// @dev Warning: It is not possible to disable an IRM.
    function enableIrm(address irm) external;

    /// @notice Enables `lltv` as a possible LLTV for market creation.
    /// @dev Warning: It is not possible to disable a LLTV.
    function enableLltv(uint256 lltv) external;

    /// @notice Sets the `newFee` for the given market `marketParams`.
    /// @dev Warning: The recipient can be the zero address.
    function setFee(MarketParams memory marketParams, uint256 newFee) external;

    /// @notice Sets `newFeeRecipient` as `feeRecipient` of the fee.
    /// @dev Warning: If the fee recipient is set to the zero address, fees will accrue there and will be lost.
    /// @dev Modifying the fee recipient will allow the new recipient to claim any pending fees not yet accrued. To
    /// ensure that the current recipient receives all due fees, accrue interest manually prior to making any changes.
    function setFeeRecipient(address newFeeRecipient) external;

    /// @notice Creates the market `marketParams`.
    /// @dev Here is the list of assumptions on the market's dependencies (tokens, IRM and oracle) that guarantees
    /// Morpho behaves as expected:
    /// - The token should be ERC-20 compliant, except that it can omit return values on `transfer` and `transferFrom`.
    /// - The token balance of Morpho should only decrease on `transfer` and `transferFrom`. In particular, tokens with
    /// burn functions are not supported.
    /// - The token should not re-enter Morpho on `transfer` nor `transferFrom`.
    /// - The token balance of the sender (resp. receiver) should decrease (resp. increase) by exactly the given amount
    /// on `transfer` and `transferFrom`. In particular, tokens with fees on transfer are not supported.
    /// - The IRM should not re-enter Morpho.
    /// - The oracle should return a price with the correct scaling.
    /// @dev Here is a list of properties on the market's dependencies that could break Morpho's liveness properties
    /// (funds could get stuck):
    /// - The token can revert on `transfer` and `transferFrom` for a reason other than an approval or balance issue.
    /// - A very high amount of assets (~1e35) supplied or borrowed can make the computation of `toSharesUp` and
    /// `toSharesDown` overflow.
    /// - The IRM can revert on `borrowRate`.
    /// - A very high borrow rate returned by the IRM can make the computation of `interest` in `_accrueInterest`
    /// overflow.
    /// - The oracle can revert on `price`. Note that this can be used to prevent `borrow`, `withdrawCollateral` and
    /// `liquidate` from being used under certain market conditions.
    /// - A very high price returned by the oracle can make the computation of `maxBorrow` in `_isHealthy` overflow, or
    /// the computation of `assetsRepaid` in `liquidate` overflow.
    /// @dev The borrow share price of a market with less than 1e4 assets borrowed can be decreased by manipulations, to
    /// the point where `totalBorrowShares` is very large and borrowing overflows.
    function createMarket(MarketParams memory marketParams) external;

    /// @notice Supplies `assets` or `shares` on behalf of `onBehalf`, optionally calling back the caller's
    /// `onMorphoSupply` function with the given `data`.
    /// @dev Either `assets` or `shares` should be zero. Most usecases should rely on `assets` as an input so the caller
    /// is guaranteed to have `assets` tokens pulled from their balance, but the possibility to mint a specific amount
    /// of shares is given for full compatibility and precision.
    /// @dev If the supply of a market gets depleted, the supply share price instantly resets to
    /// `VIRTUAL_ASSETS`:`VIRTUAL_SHARES`.
    /// @dev Supplying a large amount can revert for overflow.
    /// @param marketParams The market to supply assets to.
    /// @param assets The amount of assets to supply.
    /// @param shares The amount of shares to mint.
    /// @param onBehalf The address that will own the increased supply position.
    /// @param data Arbitrary data to pass to the `onMorphoSupply` callback. Pass empty data if not needed.
    /// @return assetsSupplied The amount of assets supplied.
    /// @return sharesSupplied The amount of shares minted.
    function supply(
        MarketParams memory marketParams,
        uint256 assets,
        uint256 shares,
        address onBehalf,
        bytes memory data
    ) external returns (uint256 assetsSupplied, uint256 sharesSupplied);

    /// @notice Withdraws `assets` or `shares` on behalf of `onBehalf` to `receiver`.
    /// @dev Either `assets` or `shares` should be zero. To withdraw max, pass the `shares`'s balance of `onBehalf`.
    /// @dev `msg.sender` must be authorized to manage `onBehalf`'s positions.
    /// @dev Withdrawing an amount corresponding to more shares than supplied will revert for underflow.
    /// @dev It is advised to use the `shares` input when withdrawing the full position to avoid reverts due to
    /// conversion roundings between shares and assets.
    /// @param marketParams The market to withdraw assets from.
    /// @param assets The amount of assets to withdraw.
    /// @param shares The amount of shares to burn.
    /// @param onBehalf The address of the owner of the supply position.
    /// @param receiver The address that will receive the withdrawn assets.
    /// @return assetsWithdrawn The amount of assets withdrawn.
    /// @return sharesWithdrawn The amount of shares burned.
    function withdraw(
        MarketParams memory marketParams,
        uint256 assets,
        uint256 shares,
        address onBehalf,
        address receiver
    ) external returns (uint256 assetsWithdrawn, uint256 sharesWithdrawn);

    /// @notice Borrows `assets` or `shares` on behalf of `onBehalf` to `receiver`.
    /// @dev Either `assets` or `shares` should be zero. Most usecases should rely on `assets` as an input so the caller
    /// is guaranteed to borrow `assets` of tokens, but the possibility to mint a specific amount of shares is given for
    /// full compatibility and precision.
    /// @dev If the borrow of a market gets depleted, the borrow share price instantly resets to
    /// `VIRTUAL_ASSETS`:`VIRTUAL_SHARES`.
    /// @dev `msg.sender` must be authorized to manage `onBehalf`'s positions.
    /// @dev Borrowing a large amount can revert for overflow.
    /// @param marketParams The market to borrow assets from.
    /// @param assets The amount of assets to borrow.
    /// @param shares The amount of shares to mint.
    /// @param onBehalf The address that will own the increased borrow position.
    /// @param receiver The address that will receive the borrowed assets.
    /// @return assetsBorrowed The amount of assets borrowed.
    /// @return sharesBorrowed The amount of shares minted.
    function borrow(
        MarketParams memory marketParams,
        uint256 assets,
        uint256 shares,
        address onBehalf,
        address receiver
    ) external returns (uint256 assetsBorrowed, uint256 sharesBorrowed);

    /// @notice Repays `assets` or `shares` on behalf of `onBehalf`, optionally calling back the caller's
    /// `onMorphoReplay` function with the given `data`.
    /// @dev Either `assets` or `shares` should be zero. To repay max, pass the `shares`'s balance of `onBehalf`.
    /// @dev Repaying an amount corresponding to more shares than borrowed will revert for underflow.
    /// @dev It is advised to use the `shares` input when repaying the full position to avoid reverts due to conversion
    /// roundings between shares and assets.
    /// @param marketParams The market to repay assets to.
    /// @param assets The amount of assets to repay.
    /// @param shares The amount of shares to burn.
    /// @param onBehalf The address of the owner of the debt position.
    /// @param data Arbitrary data to pass to the `onMorphoRepay` callback. Pass empty data if not needed.
    /// @return assetsRepaid The amount of assets repaid.
    /// @return sharesRepaid The amount of shares burned.
    function repay(
        MarketParams memory marketParams,
        uint256 assets,
        uint256 shares,
        address onBehalf,
        bytes memory data
    ) external returns (uint256 assetsRepaid, uint256 sharesRepaid);

    /// @notice Supplies `assets` of collateral on behalf of `onBehalf`, optionally calling back the caller's
    /// `onMorphoSupplyCollateral` function with the given `data`.
    /// @dev Interest are not accrued since it's not required and it saves gas.
    /// @dev Supplying a large amount can revert for overflow.
    /// @param marketParams The market to supply collateral to.
    /// @param assets The amount of collateral to supply.
    /// @param onBehalf The address that will own the increased collateral position.
    /// @param data Arbitrary data to pass to the `onMorphoSupplyCollateral` callback. Pass empty data if not needed.
    function supplyCollateral(MarketParams memory marketParams, uint256 assets, address onBehalf, bytes memory data)
        external;

    /// @notice Withdraws `assets` of collateral on behalf of `onBehalf` to `receiver`.
    /// @dev `msg.sender` must be authorized to manage `onBehalf`'s positions.
    /// @dev Withdrawing an amount corresponding to more collateral than supplied will revert for underflow.
    /// @param marketParams The market to withdraw collateral from.
    /// @param assets The amount of collateral to withdraw.
    /// @param onBehalf The address of the owner of the collateral position.
    /// @param receiver The address that will receive the collateral assets.
    function withdrawCollateral(MarketParams memory marketParams, uint256 assets, address onBehalf, address receiver)
        external;

    /// @notice Liquidates the given `repaidShares` of debt asset or seize the given `seizedAssets` of collateral on the
    /// given market `marketParams` of the given `borrower`'s position, optionally calling back the caller's
    /// `onMorphoLiquidate` function with the given `data`.
    /// @dev Either `seizedAssets` or `repaidShares` should be zero.
    /// @dev Seizing more than the collateral balance will underflow and revert without any error message.
    /// @dev Repaying more than the borrow balance will underflow and revert without any error message.
    /// @param marketParams The market of the position.
    /// @param borrower The owner of the position.
    /// @param seizedAssets The amount of collateral to seize.
    /// @param repaidShares The amount of shares to repay.
    /// @param data Arbitrary data to pass to the `onMorphoLiquidate` callback. Pass empty data if not needed.
    /// @return The amount of assets seized.
    /// @return The amount of assets repaid.
    function liquidate(
        MarketParams memory marketParams,
        address borrower,
        uint256 seizedAssets,
        uint256 repaidShares,
        bytes memory data
    ) external returns (uint256, uint256);

    /// @notice Executes a flash loan.
    /// @dev Flash loans have access to the whole balance of the contract (the liquidity and deposited collateral of all
    /// markets combined, plus donations).
    /// @dev Warning: Not ERC-3156 compliant but compatibility is easily reached:
    /// - `flashFee` is zero.
    /// - `maxFlashLoan` is the token's balance of this contract.
    /// - The receiver of `assets` is the caller.
    /// @param token The token to flash loan.
    /// @param assets The amount of assets to flash loan.
    /// @param data Arbitrary data to pass to the `onMorphoFlashLoan` callback.
    function flashLoan(address token, uint256 assets, bytes calldata data) external;

    /// @notice Sets the authorization for `authorized` to manage `msg.sender`'s positions.
    /// @param authorized The authorized address.
    /// @param newIsAuthorized The new authorization status.
    function setAuthorization(address authorized, bool newIsAuthorized) external;

    /// @notice Sets the authorization for `authorization.authorized` to manage `authorization.authorizer`'s positions.
    /// @dev Warning: Reverts if the signature has already been submitted.
    /// @dev The signature is malleable, but it has no impact on the security here.
    /// @dev The nonce is passed as argument to be able to revert with a different error message.
    /// @param authorization The `Authorization` struct.
    /// @param signature The signature.
    function setAuthorizationWithSig(Authorization calldata authorization, Signature calldata signature) external;

    /// @notice Accrues interest for the given market `marketParams`.
    function accrueInterest(MarketParams memory marketParams) external;

    /// @notice Returns the data stored on the different `slots`.
    function extSloads(bytes32[] memory slots) external view returns (bytes32[] memory);
}



interface IMorphoStaticTyping is IMorphoBase {
    /// @notice The state of the position of `user` on the market corresponding to `id`.
    /// @dev Warning: For `feeRecipient`, `supplyShares` does not contain the accrued shares since the last interest
    /// accrual.
    function position(Id id, address user)
        external
        view
        returns (uint256 supplyShares, uint128 borrowShares, uint128 collateral);

    /// @notice The state of the market corresponding to `id`.
    /// @dev Warning: `totalSupplyAssets` does not contain the accrued interest since the last interest accrual.
    /// @dev Warning: `totalBorrowAssets` does not contain the accrued interest since the last interest accrual.
    /// @dev Warning: `totalSupplyShares` does not contain the accrued shares by `feeRecipient` since the last interest
    /// accrual.
    function market(Id id)
        external
        view
        returns (
            uint128 totalSupplyAssets,
            uint128 totalSupplyShares,
            uint128 totalBorrowAssets,
            uint128 totalBorrowShares,
            uint128 lastUpdate,
            uint128 fee
        );

    /// @notice The market params corresponding to `id`.
    /// @dev This mapping is not used in Morpho. It is there to enable reducing the cost associated to calldata on layer
    /// 2s by creating a wrapper contract with functions that take `id` as input instead of `marketParams`.
    function idToMarketParams(Id id)
        external
        view
        returns (address loanToken, address collateralToken, address oracle, address irm, uint256 lltv);
}





interface IMorphoBlue is IMorphoBase {
    /// @notice The state of the position of `user` on the market corresponding to `id`.
    /// @dev Warning: For `feeRecipient`, `p.supplyShares` does not contain the accrued shares since the last interest
    /// accrual.
    function position(Id id, address user) external view returns (MorphoBluePosition memory p);

    /// @notice The state of the market corresponding to `id`.
    /// @dev Warning: `m.totalSupplyAssets` does not contain the accrued interest since the last interest accrual.
    /// @dev Warning: `m.totalBorrowAssets` does not contain the accrued interest since the last interest accrual.
    /// @dev Warning: `m.totalSupplyShares` does not contain the accrued shares by `feeRecipient` since the last
    /// interest accrual.
    function market(Id id) external view returns (Market memory m);

    /// @notice The market params corresponding to `id`.
    /// @dev This mapping is not used in Morpho. It is there to enable reducing the cost associated to calldata on layer
    /// 2s by creating a wrapper contract with functions that take `id` as input instead of `marketParams`.
    function idToMarketParams(Id id) external view returns (MarketParams memory);
}










interface IIrm {
    /// @notice Returns the borrow rate of the market `marketParams`.
    /// @dev Assumes that `market` corresponds to `marketParams`.
    function borrowRate(MarketParams memory marketParams, Market memory market) external returns (uint256);

    /// @notice Returns the borrow rate of the market `marketParams` without modifying any storage.
    /// @dev Assumes that `market` corresponds to `marketParams`.
    function borrowRateView(MarketParams memory marketParams, Market memory market) external view returns (uint256);
}












library MorphoStorageLib {
    /* SLOTS */

    uint256 internal constant OWNER_SLOT = 0;
    uint256 internal constant FEE_RECIPIENT_SLOT = 1;
    uint256 internal constant POSITION_SLOT = 2;
    uint256 internal constant MARKET_SLOT = 3;
    uint256 internal constant IS_IRM_ENABLED_SLOT = 4;
    uint256 internal constant IS_LLTV_ENABLED_SLOT = 5;
    uint256 internal constant IS_AUTHORIZED_SLOT = 6;
    uint256 internal constant NONCE_SLOT = 7;
    uint256 internal constant ID_TO_MARKET_PARAMS_SLOT = 8;

    /* SLOT OFFSETS */

    uint256 internal constant LOAN_TOKEN_OFFSET = 0;
    uint256 internal constant COLLATERAL_TOKEN_OFFSET = 1;
    uint256 internal constant ORACLE_OFFSET = 2;
    uint256 internal constant IRM_OFFSET = 3;
    uint256 internal constant LLTV_OFFSET = 4;

    uint256 internal constant SUPPLY_SHARES_OFFSET = 0;
    uint256 internal constant BORROW_SHARES_AND_COLLATERAL_OFFSET = 1;

    uint256 internal constant TOTAL_SUPPLY_ASSETS_AND_SHARES_OFFSET = 0;
    uint256 internal constant TOTAL_BORROW_ASSETS_AND_SHARES_OFFSET = 1;
    uint256 internal constant LAST_UPDATE_AND_FEE_OFFSET = 2;

    /* GETTERS */

    function ownerSlot() internal pure returns (bytes32) {
        return bytes32(OWNER_SLOT);
    }

    function feeRecipientSlot() internal pure returns (bytes32) {
        return bytes32(FEE_RECIPIENT_SLOT);
    }

    function positionSupplySharesSlot(Id id, address user) internal pure returns (bytes32) {
        return bytes32(
            uint256(keccak256(abi.encode(user, keccak256(abi.encode(id, POSITION_SLOT))))) + SUPPLY_SHARES_OFFSET
        );
    }

    function positionBorrowSharesAndCollateralSlot(Id id, address user) internal pure returns (bytes32) {
        return bytes32(
            uint256(keccak256(abi.encode(user, keccak256(abi.encode(id, POSITION_SLOT)))))
                + BORROW_SHARES_AND_COLLATERAL_OFFSET
        );
    }

    function marketTotalSupplyAssetsAndSharesSlot(Id id) internal pure returns (bytes32) {
        return bytes32(uint256(keccak256(abi.encode(id, MARKET_SLOT))) + TOTAL_SUPPLY_ASSETS_AND_SHARES_OFFSET);
    }

    function marketTotalBorrowAssetsAndSharesSlot(Id id) internal pure returns (bytes32) {
        return bytes32(uint256(keccak256(abi.encode(id, MARKET_SLOT))) + TOTAL_BORROW_ASSETS_AND_SHARES_OFFSET);
    }

    function marketLastUpdateAndFeeSlot(Id id) internal pure returns (bytes32) {
        return bytes32(uint256(keccak256(abi.encode(id, MARKET_SLOT))) + LAST_UPDATE_AND_FEE_OFFSET);
    }

    function isIrmEnabledSlot(address irm) internal pure returns (bytes32) {
        return keccak256(abi.encode(irm, IS_IRM_ENABLED_SLOT));
    }

    function isLltvEnabledSlot(uint256 lltv) internal pure returns (bytes32) {
        return keccak256(abi.encode(lltv, IS_LLTV_ENABLED_SLOT));
    }

    function isAuthorizedSlot(address authorizer, address authorizee) internal pure returns (bytes32) {
        return keccak256(abi.encode(authorizee, keccak256(abi.encode(authorizer, IS_AUTHORIZED_SLOT))));
    }

    function nonceSlot(address authorizer) internal pure returns (bytes32) {
        return keccak256(abi.encode(authorizer, NONCE_SLOT));
    }

    function idToLoanTokenSlot(Id id) internal pure returns (bytes32) {
        return bytes32(uint256(keccak256(abi.encode(id, ID_TO_MARKET_PARAMS_SLOT))) + LOAN_TOKEN_OFFSET);
    }

    function idToCollateralTokenSlot(Id id) internal pure returns (bytes32) {
        return bytes32(uint256(keccak256(abi.encode(id, ID_TO_MARKET_PARAMS_SLOT))) + COLLATERAL_TOKEN_OFFSET);
    }

    function idToOracleSlot(Id id) internal pure returns (bytes32) {
        return bytes32(uint256(keccak256(abi.encode(id, ID_TO_MARKET_PARAMS_SLOT))) + ORACLE_OFFSET);
    }

    function idToIrmSlot(Id id) internal pure returns (bytes32) {
        return bytes32(uint256(keccak256(abi.encode(id, ID_TO_MARKET_PARAMS_SLOT))) + IRM_OFFSET);
    }

    function idToLltvSlot(Id id) internal pure returns (bytes32) {
        return bytes32(uint256(keccak256(abi.encode(id, ID_TO_MARKET_PARAMS_SLOT))) + LLTV_OFFSET);
    }
}






library MorphoLib {
    function supplyShares(IMorphoBlue morpho, Id id, address user) internal view returns (uint256) {
        bytes32[] memory slot = _array(MorphoStorageLib.positionSupplySharesSlot(id, user));
        return uint256(morpho.extSloads(slot)[0]);
    }

    function borrowShares(IMorphoBlue morpho, Id id, address user) internal view returns (uint256) {
        bytes32[] memory slot = _array(MorphoStorageLib.positionBorrowSharesAndCollateralSlot(id, user));
        return uint128(uint256(morpho.extSloads(slot)[0]));
    }

    function collateral(IMorphoBlue morpho, Id id, address user) internal view returns (uint256) {
        bytes32[] memory slot = _array(MorphoStorageLib.positionBorrowSharesAndCollateralSlot(id, user));
        return uint256(morpho.extSloads(slot)[0] >> 128);
    }

    function totalSupplyAssets(IMorphoBlue morpho, Id id) internal view returns (uint256) {
        bytes32[] memory slot = _array(MorphoStorageLib.marketTotalSupplyAssetsAndSharesSlot(id));
        return uint128(uint256(morpho.extSloads(slot)[0]));
    }

    function totalSupplyShares(IMorphoBlue morpho, Id id) internal view returns (uint256) {
        bytes32[] memory slot = _array(MorphoStorageLib.marketTotalSupplyAssetsAndSharesSlot(id));
        return uint256(morpho.extSloads(slot)[0] >> 128);
    }

    function totalBorrowAssets(IMorphoBlue morpho, Id id) internal view returns (uint256) {
        bytes32[] memory slot = _array(MorphoStorageLib.marketTotalBorrowAssetsAndSharesSlot(id));
        return uint128(uint256(morpho.extSloads(slot)[0]));
    }

    function totalBorrowShares(IMorphoBlue morpho, Id id) internal view returns (uint256) {
        bytes32[] memory slot = _array(MorphoStorageLib.marketTotalBorrowAssetsAndSharesSlot(id));
        return uint256(morpho.extSloads(slot)[0] >> 128);
    }

    function lastUpdate(IMorphoBlue morpho, Id id) internal view returns (uint128) {
        bytes32[] memory slot = _array(MorphoStorageLib.marketLastUpdateAndFeeSlot(id));
        return uint128(uint256(morpho.extSloads(slot)[0]));
    }

    function fee(IMorphoBlue morpho, Id id) internal view returns (uint128) {
        bytes32[] memory slot = _array(MorphoStorageLib.marketLastUpdateAndFeeSlot(id));
        return uint128(uint256(morpho.extSloads(slot)[0] >> 128));
    }

    function _array(bytes32 x) private pure returns (bytes32[] memory) {
        bytes32[] memory res = new bytes32[](1);
        res[0] = x;
        return res;
    }
}

uint256 constant WAD = 1e18;





library MathLib {
    /// @dev Returns (`x` * `y`) / `WAD` rounded down.
    function wMulDown(uint256 x, uint256 y) internal pure returns (uint256) {
        return mulDivDown(x, y, WAD);
    }

    /// @dev Returns (`x` * `WAD`) / `y` rounded down.
    function wDivDown(uint256 x, uint256 y) internal pure returns (uint256) {
        return mulDivDown(x, WAD, y);
    }

    /// @dev Returns (`x` * `WAD`) / `y` rounded up.
    function wDivUp(uint256 x, uint256 y) internal pure returns (uint256) {
        return mulDivUp(x, WAD, y);
    }

    /// @dev Returns (`x` * `y`) / `d` rounded down.
    function mulDivDown(uint256 x, uint256 y, uint256 d) internal pure returns (uint256) {
        return (x * y) / d;
    }

    /// @dev Returns (`x` * `y`) / `d` rounded up.
    function mulDivUp(uint256 x, uint256 y, uint256 d) internal pure returns (uint256) {
        return (x * y + (d - 1)) / d;
    }

    /// @dev Returns the sum of the first three non-zero terms of a Taylor expansion of e^(nx) - 1, to approximate a
    /// continuous compound interest rate.
    function wTaylorCompounded(uint256 x, uint256 n) internal pure returns (uint256) {
        uint256 firstTerm = x * n;
        uint256 secondTerm = mulDivDown(firstTerm, firstTerm, 2 * WAD);
        uint256 thirdTerm = mulDivDown(secondTerm, firstTerm, 3 * WAD);

        return firstTerm + secondTerm + thirdTerm;
    }
}





library ErrorsLib {
    /// @notice Thrown when the caller is not the owner.
    string internal constant NOT_OWNER = "not owner";

    /// @notice Thrown when the LLTV to enable exceeds the maximum LLTV.
    string internal constant MAX_LLTV_EXCEEDED = "max LLTV exceeded";

    /// @notice Thrown when the fee to set exceeds the maximum fee.
    string internal constant MAX_FEE_EXCEEDED = "max fee exceeded";

    /// @notice Thrown when the value is already set.
    string internal constant ALREADY_SET = "already set";

    /// @notice Thrown when the IRM is not enabled at market creation.
    string internal constant IRM_NOT_ENABLED = "IRM not enabled";

    /// @notice Thrown when the LLTV is not enabled at market creation.
    string internal constant LLTV_NOT_ENABLED = "LLTV not enabled";

    /// @notice Thrown when the market is already created.
    string internal constant MARKET_ALREADY_CREATED = "market already created";

    /// @notice Thrown when a token to transfer doesn't have code.
    string internal constant NO_CODE = "no code";

    /// @notice Thrown when the market is not created.
    string internal constant MARKET_NOT_CREATED = "market not created";

    /// @notice Thrown when not exactly one of the input amount is zero.
    string internal constant INCONSISTENT_INPUT = "inconsistent input";

    /// @notice Thrown when zero assets is passed as input.
    string internal constant ZERO_ASSETS = "zero assets";

    /// @notice Thrown when a zero address is passed as input.
    string internal constant ZERO_ADDRESS = "zero address";

    /// @notice Thrown when the caller is not authorized to conduct an action.
    string internal constant UNAUTHORIZED = "unauthorized";

    /// @notice Thrown when the collateral is insufficient to `borrow` or `withdrawCollateral`.
    string internal constant INSUFFICIENT_COLLATERAL = "insufficient collateral";

    /// @notice Thrown when the liquidity is insufficient to `withdraw` or `borrow`.
    string internal constant INSUFFICIENT_LIQUIDITY = "insufficient liquidity";

    /// @notice Thrown when the position to liquidate is healthy.
    string internal constant HEALTHY_POSITION = "position is healthy";

    /// @notice Thrown when the authorization signature is invalid.
    string internal constant INVALID_SIGNATURE = "invalid signature";

    /// @notice Thrown when the authorization signature is expired.
    string internal constant SIGNATURE_EXPIRED = "signature expired";

    /// @notice Thrown when the nonce is invalid.
    string internal constant INVALID_NONCE = "invalid nonce";

    /// @notice Thrown when a token transfer reverted.
    string internal constant TRANSFER_REVERTED = "transfer reverted";

    /// @notice Thrown when a token transfer returned false.
    string internal constant TRANSFER_RETURNED_FALSE = "transfer returned false";

    /// @notice Thrown when a token transferFrom reverted.
    string internal constant TRANSFER_FROM_REVERTED = "transferFrom reverted";

    /// @notice Thrown when a token transferFrom returned false
    string internal constant TRANSFER_FROM_RETURNED_FALSE = "transferFrom returned false";

    /// @notice Thrown when the maximum uint128 is exceeded.
    string internal constant MAX_UINT128_EXCEEDED = "max uint128 exceeded";
}






library UtilsLib {
    /// @dev Returns true if there is exactly one zero among `x` and `y`.
    function exactlyOneZero(uint256 x, uint256 y) internal pure returns (bool z) {
        assembly {
            z := xor(iszero(x), iszero(y))
        }
    }

    /// @dev Returns the min of `x` and `y`.
    function min(uint256 x, uint256 y) internal pure returns (uint256 z) {
        assembly {
            z := xor(x, mul(xor(x, y), lt(y, x)))
        }
    }

    /// @dev Returns `x` safely cast to uint128.
    function toUint128(uint256 x) internal pure returns (uint128) {
        require(x <= type(uint128).max, ErrorsLib.MAX_UINT128_EXCEEDED);
        return uint128(x);
    }

    /// @dev Returns max(0, x - y).
    function zeroFloorSub(uint256 x, uint256 y) internal pure returns (uint256 z) {
        assembly {
            z := mul(gt(x, y), sub(x, y))
        }
    }
}





library MarketParamsLib {
    /// @notice The length of the data used to compute the id of a market.
    /// @dev The length is 5 * 32 because `MarketParams` has 5 variables of 32 bytes each.
    uint256 internal constant MARKET_PARAMS_BYTES_LENGTH = 5 * 32;

    /// @notice Returns the id of the market `marketParams`.
    function id(MarketParams memory marketParams) internal pure returns (Id marketParamsId) {
        assembly {
            marketParamsId := keccak256(marketParams, MARKET_PARAMS_BYTES_LENGTH)
        }
    }
}







library SharesMathLib {
    using MathLib for uint256;

    /// @dev The number of virtual shares has been chosen low enough to prevent overflows, and high enough to ensure
    /// high precision computations.
    /// @dev Virtual shares can never be redeemed for the assets they are entitled to, but it is assumed the share price
    /// stays low enough not to inflate these assets to a significant value.
    /// @dev Warning: The assets to which virtual borrow shares are entitled behave like unrealizable bad debt.
    uint256 internal constant VIRTUAL_SHARES = 1e6;

    /// @dev A number of virtual assets of 1 enforces a conversion rate between shares and assets when a market is
    /// empty.
    uint256 internal constant VIRTUAL_ASSETS = 1;

    /// @dev Calculates the value of `assets` quoted in shares, rounding down.
    function toSharesDown(uint256 assets, uint256 totalAssets, uint256 totalShares) internal pure returns (uint256) {
        return assets.mulDivDown(totalShares + VIRTUAL_SHARES, totalAssets + VIRTUAL_ASSETS);
    }

    /// @dev Calculates the value of `shares` quoted in assets, rounding down.
    function toAssetsDown(uint256 shares, uint256 totalAssets, uint256 totalShares) internal pure returns (uint256) {
        return shares.mulDivDown(totalAssets + VIRTUAL_ASSETS, totalShares + VIRTUAL_SHARES);
    }

    /// @dev Calculates the value of `assets` quoted in shares, rounding up.
    function toSharesUp(uint256 assets, uint256 totalAssets, uint256 totalShares) internal pure returns (uint256) {
        return assets.mulDivUp(totalShares + VIRTUAL_SHARES, totalAssets + VIRTUAL_ASSETS);
    }

    /// @dev Calculates the value of `shares` quoted in assets, rounding up.
    function toAssetsUp(uint256 shares, uint256 totalAssets, uint256 totalShares) internal pure returns (uint256) {
        return shares.mulDivUp(totalAssets + VIRTUAL_ASSETS, totalShares + VIRTUAL_SHARES);
    }
}








library MorphoBalancesLib {
    using MathLib for uint256;
    using MathLib for uint128;
    using UtilsLib for uint256;
    using MorphoLib for IMorphoBlue;
    using SharesMathLib for uint256;
    using MarketParamsLib for MarketParams;

    /// @notice Returns the expected market balances of a market after having accrued interest.
    /// @return The expected total supply assets.
    /// @return The expected total supply shares.
    /// @return The expected total borrow assets.
    /// @return The expected total borrow shares.
    function expectedMarketBalances(IMorphoBlue morpho, MarketParams memory marketParams)
        internal
        view
        returns (uint256, uint256, uint256, uint256)
    {
        Id id = marketParams.id();
        Market memory market = morpho.market(id);

        uint256 elapsed = block.timestamp - market.lastUpdate;

        // Skipped if elapsed == 0 or totalBorrowAssets == 0 because interest would be null, or if irm == address(0).
        if (elapsed != 0 && market.totalBorrowAssets != 0 && marketParams.irm != address(0)) {
            uint256 borrowRate = IIrm(marketParams.irm).borrowRateView(marketParams, market);
            uint256 interest = market.totalBorrowAssets.wMulDown(borrowRate.wTaylorCompounded(elapsed));
            market.totalBorrowAssets += interest.toUint128();
            market.totalSupplyAssets += interest.toUint128();

            if (market.fee != 0) {
                uint256 feeAmount = interest.wMulDown(market.fee);
                // The fee amount is subtracted from the total supply in this calculation to compensate for the fact
                // that total supply is already updated.
                uint256 feeShares =
                    feeAmount.toSharesDown(market.totalSupplyAssets - feeAmount, market.totalSupplyShares);
                market.totalSupplyShares += feeShares.toUint128();
            }
        }

        return (market.totalSupplyAssets, market.totalSupplyShares, market.totalBorrowAssets, market.totalBorrowShares);
    }

    /// @notice Returns the expected total supply assets of a market after having accrued interest.
    function expectedTotalSupplyAssets(IMorphoBlue morpho, MarketParams memory marketParams)
        internal
        view
        returns (uint256 totalSupplyAssets)
    {
        (totalSupplyAssets,,,) = expectedMarketBalances(morpho, marketParams);
    }

    /// @notice Returns the expected total borrow assets of a market after having accrued interest.
    function expectedTotalBorrowAssets(IMorphoBlue morpho, MarketParams memory marketParams)
        internal
        view
        returns (uint256 totalBorrowAssets)
    {
        (,, totalBorrowAssets,) = expectedMarketBalances(morpho, marketParams);
    }

    /// @notice Returns the expected total supply shares of a market after having accrued interest.
    function expectedTotalSupplyShares(IMorphoBlue morpho, MarketParams memory marketParams)
        internal
        view
        returns (uint256 totalSupplyShares)
    {
        (, totalSupplyShares,,) = expectedMarketBalances(morpho, marketParams);
    }

    /// @notice Returns the expected supply assets balance of `user` on a market after having accrued interest.
    /// @dev Warning: Wrong for `feeRecipient` because their supply shares increase is not taken into account.
    /// @dev Warning: Withdrawing using the expected supply assets can lead to a revert due to conversion roundings from
    /// assets to shares.
    function expectedSupplyAssets(IMorphoBlue morpho, MarketParams memory marketParams, address user)
        internal
        view
        returns (uint256)
    {
        Id id = marketParams.id();
        uint256 supplyShares = morpho.supplyShares(id, user);
        (uint256 totalSupplyAssets, uint256 totalSupplyShares,,) = expectedMarketBalances(morpho, marketParams);

        return supplyShares.toAssetsDown(totalSupplyAssets, totalSupplyShares);
    }

    /// @notice Returns the expected borrow assets balance of `user` on a market after having accrued interest.
    /// @dev Warning: The expected balance is rounded up, so it may be greater than the market's expected total borrow
    /// assets.
    function expectedBorrowAssets(IMorphoBlue morpho, MarketParams memory marketParams, address user)
        internal
        view
        returns (uint256)
    {
        Id id = marketParams.id();
        uint256 borrowShares = morpho.borrowShares(id, user);
        (,, uint256 totalBorrowAssets, uint256 totalBorrowShares) = expectedMarketBalances(morpho, marketParams);

        return borrowShares.toAssetsUp(totalBorrowAssets, totalBorrowShares);
    }
}











interface IOracle {
    /// @notice Returns the price of 1 asset of collateral token quoted in 1 asset of loan token, scaled by 1e36.
    /// @dev It corresponds to the price of 10**(collateral token decimals) assets of collateral token quoted in
    /// 10**(loan token decimals) assets of loan token with `36 + loan token decimals - collateral token decimals`
    /// decimals of precision.
    function price() external view returns (uint256);
}









contract MorphoBlueHelper is MainnetMorphoBlueAddresses {
    IMorphoBlue public constant morphoBlue = IMorphoBlue(MORPHO_BLUE_ADDRESS);
    
    uint256 internal constant MARKET_PARAMS_BYTES_LENGTH = 5 * 32;
    uint256 internal constant ORACLE_PRICE_DECIMALS = 36;
    uint256 internal constant VIRTUAL_SHARES = 1e6;
    uint256 internal constant VIRTUAL_ASSETS = 1;

    /// @dev this changes state and uses up more gas
    /// @dev if you call other morpho blue state changing function in same transaction then its _accrueInterest will return early and save gas
    /// Function fetches debt in assets so that we know exactly how many tokens we need to repay the whole debt
    function getCurrentDebt(MarketParams memory marketParams, address owner) internal returns (uint256 currentDebtInAssets, uint256 borrowShares){
        morphoBlue.accrueInterest(marketParams);
        Id marketId = MarketParamsLib.id(marketParams);
        borrowShares = MorphoLib.borrowShares(morphoBlue, marketId, owner);
        currentDebtInAssets = SharesMathLib.toAssetsUp(
            borrowShares, 
            MorphoLib.totalBorrowAssets(morphoBlue, marketId),
            MorphoLib.totalBorrowShares(morphoBlue, marketId)
        );
    }
    /// Function reads supply shares for a given user from MorphoBlue state
    function getSupplyShares(MarketParams memory marketParams, address owner) internal view returns (uint256 supplyShares){
        Id marketId = MarketParamsLib.id(marketParams);
        supplyShares = MorphoLib.supplyShares(morphoBlue, marketId, owner);
    }

    function getRatioUsingParams(MarketParams memory marketParams, address owner) public returns (uint256 ratio){
        Id marketId = MarketParamsLib.id(marketParams);
        ratio = getRatio(marketId, marketParams, owner);
    }

    function getRatioUsingId(Id marketId, address owner) public returns (uint256 ratio){
        MarketParams memory marketParams = morphoBlue.idToMarketParams(marketId);
        ratio = getRatio(marketId, marketParams, owner);
    }

    function getRatio(Id marketId, MarketParams memory marketParams, address owner) public returns (uint256 ratio){
        uint256 oraclePrice = IOracle(marketParams.oracle).price();
        morphoBlue.accrueInterest(marketParams);

        Market memory market = morphoBlue.market(marketId);
        MorphoBluePosition memory position = morphoBlue.position(marketId, owner);
        
        uint256 collateral = position.collateral;
        if (collateral == 0) return 0;
        uint256 debt = SharesMathLib.toAssetsUp(position.borrowShares, market.totalBorrowAssets, market.totalBorrowShares);
        if (debt == 0) return 0;
        ratio = collateral * oraclePrice / debt / 1e18;
    }
}






contract MainnetMorphoAaveV3Addresses {
    address public constant MORPHO_TOKEN_ADDR = 0x9994E35Db50125E0DF82e4c2dde62496CE330999;
    address public constant MORPHO_MARKET_STORAGE = 0x56EE33811a6C8c1Fd443E53685ed1605E43b3971;
}







library BucketDLL {
    /// STRUCTS ///

    struct Account {
        address prev;
        address next;
    }

    struct List {
        mapping(address => Account) accounts;
    }

    /// INTERNAL ///

    /// @notice Returns the address at the head of the `_list`.
    /// @param _list The list from which to get the head.
    /// @return The address of the head.
    function getHead(List storage _list) internal view returns (address) {
        return _list.accounts[address(0)].next;
    }

    /// @notice Returns the address at the tail of the `_list`.
    /// @param _list The list from which to get the tail.
    /// @return The address of the tail.
    function getTail(List storage _list) internal view returns (address) {
        return _list.accounts[address(0)].prev;
    }

    /// @notice Returns the next id address from the current `_id`.
    /// @param _list The list to search in.
    /// @param _id The address of the current account.
    /// @return The address of the next account.
    function getNext(List storage _list, address _id) internal view returns (address) {
        return _list.accounts[_id].next;
    }

    /// @notice Returns the previous id address from the current `_id`.
    /// @param _list The list to search in.
    /// @param _id The address of the current account.
    /// @return The address of the previous account.
    function getPrev(List storage _list, address _id) internal view returns (address) {
        return _list.accounts[_id].prev;
    }

    /// @notice Removes an account of the `_list`.
    /// @dev This function should not be called with `_id` equal to address 0.
    /// @param _list The list to search in.
    /// @param _id The address of the account.
    /// @return Whether the bucket is empty after removal.
    function remove(List storage _list, address _id) internal returns (bool) {
        Account memory account = _list.accounts[_id];
        address prev = account.prev;
        address next = account.next;

        _list.accounts[prev].next = next;
        _list.accounts[next].prev = prev;

        delete _list.accounts[_id];

        return (prev == address(0) && next == address(0));
    }

    /// @notice Inserts an account in the `_list`.
    /// @dev This function should not be called with `_id` equal to address 0.
    /// @param _list The list to search in.
    /// @param _id The address of the account.
    /// @param _head Tells whether to insert at the head or at the tail of the list.
    /// @return Whether the bucket was empty before insertion.
    function insert(
        List storage _list,
        address _id,
        bool _head
    ) internal returns (bool) {
        if (_head) {
            address head = _list.accounts[address(0)].next;
            _list.accounts[address(0)].next = _id;
            _list.accounts[head].prev = _id;
            _list.accounts[_id].next = head;
            return head == address(0);
        } else {
            address tail = _list.accounts[address(0)].prev;
            _list.accounts[address(0)].prev = _id;
            _list.accounts[tail].next = _id;
            _list.accounts[_id].prev = tail;
            return tail == address(0);
        }
    }
}


library LogarithmicBuckets {
    using BucketDLL for BucketDLL.List;

    struct Buckets {
        mapping(uint256 => BucketDLL.List) buckets;
        mapping(address => uint256) valueOf;
        uint256 bucketsMask;
    }

    /// ERRORS ///

    /// @notice Thrown when the address is zero at insertion.
    error ZeroAddress();

    /// @notice Thrown when 0 value is inserted.
    error ZeroValue();

    /// INTERNAL ///

    /// @notice Updates an account in the `_buckets`.
    /// @param _buckets The buckets to update.
    /// @param _id The address of the account.
    /// @param _newValue The new value of the account.
    /// @param _head Indicates whether to insert the new values at the head or at the tail of the buckets list.
    function update(
        Buckets storage _buckets,
        address _id,
        uint256 _newValue,
        bool _head
    ) internal {
        if (_id == address(0)) revert ZeroAddress();
        uint256 value = _buckets.valueOf[_id];
        _buckets.valueOf[_id] = _newValue;

        if (value == 0) {
            if (_newValue == 0) revert ZeroValue();
            _insert(_buckets, _id, computeBucket(_newValue), _head);
            return;
        }

        uint256 currentBucket = computeBucket(value);
        if (_newValue == 0) {
            _remove(_buckets, _id, currentBucket);
            return;
        }

        uint256 newBucket = computeBucket(_newValue);
        if (newBucket != currentBucket) {
            _remove(_buckets, _id, currentBucket);
            _insert(_buckets, _id, newBucket, _head);
        }
    }

    /// @notice Returns the address in `_buckets` that is a candidate for matching the value `_value`.
    /// @param _buckets The buckets to get the head.
    /// @param _value The value to match.
    /// @return The address of the head.
    function getMatch(Buckets storage _buckets, uint256 _value) internal view returns (address) {
        uint256 bucketsMask = _buckets.bucketsMask;
        if (bucketsMask == 0) return address(0);
        uint256 lowerMask = setLowerBits(_value);

        uint256 next = nextBucket(lowerMask, bucketsMask);

        if (next != 0) return _buckets.buckets[next].getHead();

        uint256 prev = prevBucket(lowerMask, bucketsMask);

        return _buckets.buckets[prev].getHead();
    }

    /// PRIVATE ///

    /// @notice Removes an account in the `_buckets`.
    /// @dev Does not update the value.
    /// @param _buckets The buckets to modify.
    /// @param _id The address of the account to remove.
    /// @param _bucket The mask of the bucket where to remove.
    function _remove(
        Buckets storage _buckets,
        address _id,
        uint256 _bucket
    ) private {
        if (_buckets.buckets[_bucket].remove(_id)) _buckets.bucketsMask &= ~_bucket;
    }

    /// @notice Inserts an account in the `_buckets`.
    /// @dev Expects that `_id` != 0.
    /// @dev Does not update the value.
    /// @param _buckets The buckets to modify.
    /// @param _id The address of the account to update.
    /// @param _bucket The mask of the bucket where to insert.
    /// @param _head Whether to insert at the head or at the tail of the list.
    function _insert(
        Buckets storage _buckets,
        address _id,
        uint256 _bucket,
        bool _head
    ) private {
        if (_buckets.buckets[_bucket].insert(_id, _head)) _buckets.bucketsMask |= _bucket;
    }

    /// PURE HELPERS ///

    /// @notice Returns the bucket in which the given value would fall.
    function computeBucket(uint256 _value) internal pure returns (uint256) {
        uint256 lowerMask = setLowerBits(_value);
        return lowerMask ^ (lowerMask >> 1);
    }

    /// @notice Sets all the bits lower than (or equal to) the highest bit in the input.
    /// @dev This is the same as rounding the input the nearest upper value of the form `2 ** n - 1`.
    function setLowerBits(uint256 x) internal pure returns (uint256 y) {
        assembly {
            x := or(x, shr(1, x))
            x := or(x, shr(2, x))
            x := or(x, shr(4, x))
            x := or(x, shr(8, x))
            x := or(x, shr(16, x))
            x := or(x, shr(32, x))
            x := or(x, shr(64, x))
            y := or(x, shr(128, x))
        }
    }

    /// @notice Returns the following bucket which contains greater values.
    /// @dev The bucket returned is the lowest that is in `bucketsMask` and not in `lowerMask`.
    function nextBucket(uint256 lowerMask, uint256 bucketsMask)
        internal
        pure
        returns (uint256 bucket)
    {
        assembly {
            let higherBucketsMask := and(not(lowerMask), bucketsMask)
            bucket := and(higherBucketsMask, add(not(higherBucketsMask), 1))
        }
    }

    /// @notice Returns the preceding bucket which contains smaller values.
    /// @dev The bucket returned is the highest that is in both `bucketsMask` and `lowerMask`.
    function prevBucket(uint256 lowerMask, uint256 bucketsMask) internal pure returns (uint256) {
        uint256 lowerBucketsMask = setLowerBits(lowerMask & bucketsMask);
        return lowerBucketsMask ^ (lowerBucketsMask >> 1);
    }
}


library DataTypesAaveV3 {
  struct ReserveData {
    //stores the reserve configuration
    ReserveConfigurationMap configuration;
    //the liquidity index. Expressed in ray
    uint128 liquidityIndex;
    //the current supply rate. Expressed in ray
    uint128 currentLiquidityRate;
    //variable borrow index. Expressed in ray
    uint128 variableBorrowIndex;
    //the current variable borrow rate. Expressed in ray
    uint128 currentVariableBorrowRate;
    //the current stable borrow rate. Expressed in ray
    uint128 currentStableBorrowRate;
    //timestamp of last update
    uint40 lastUpdateTimestamp;
    //the id of the reserve. Represents the position in the list of the active reserves
    uint16 id;
    //aToken address
    address aTokenAddress;
    //stableDebtToken address
    address stableDebtTokenAddress;
    //variableDebtToken address
    address variableDebtTokenAddress;
    //address of the interest rate strategy
    address interestRateStrategyAddress;
    //the current treasury balance, scaled
    uint128 accruedToTreasury;
    //the outstanding unbacked aTokens minted through the bridging feature
    uint128 unbacked;
    //the outstanding debt borrowed against this asset in isolation mode
    uint128 isolationModeTotalDebt;
  }

  struct ReserveConfigurationMap {
    //bit 0-15: LTV
    //bit 16-31: Liq. threshold
    //bit 32-47: Liq. bonus
    //bit 48-55: Decimals
    //bit 56: reserve is active
    //bit 57: reserve is frozen
    //bit 58: borrowing is enabled
    //bit 59: stable rate borrowing enabled
    //bit 60: asset is paused
    //bit 61: borrowing in isolation mode is enabled
    //bit 62-63: reserved
    //bit 64-79: reserve factor
    //bit 80-115 borrow cap in whole tokens, borrowCap == 0 => no cap
    //bit 116-151 supply cap in whole tokens, supplyCap == 0 => no cap
    //bit 152-167 liquidation protocol fee
    //bit 168-175 eMode category
    //bit 176-211 unbacked mint cap in whole tokens, unbackedMintCap == 0 => minting disabled
    //bit 212-251 debt ceiling for isolation mode with (ReserveConfiguration::DEBT_CEILING_DECIMALS) decimals
    //bit 252-255 unused

    uint256 data;
  }

  struct UserConfigurationMap {
    /**
     * @dev Bitmap of the users collaterals and borrows. It is divided in pairs of bits, one pair per asset.
     * The first bit indicates if an asset is used as collateral by the user, the second whether an
     * asset is borrowed by the user.
     */
    uint256 data;
  }

  struct EModeCategory {
    // each eMode category has a custom ltv and liquidation threshold
    uint16 ltv;
    uint16 liquidationThreshold;
    uint16 liquidationBonus;
    // each eMode category may or may not have a custom oracle to override the individual assets price oracles
    address priceSource;
    string label;
  }

  enum InterestRateMode {
    NONE,
    STABLE,
    VARIABLE
  }

  struct ReserveCache {
    uint256 currScaledVariableDebt;
    uint256 nextScaledVariableDebt;
    uint256 currPrincipalStableDebt;
    uint256 currAvgStableBorrowRate;
    uint256 currTotalStableDebt;
    uint256 nextAvgStableBorrowRate;
    uint256 nextTotalStableDebt;
    uint256 currLiquidityIndex;
    uint256 nextLiquidityIndex;
    uint256 currVariableBorrowIndex;
    uint256 nextVariableBorrowIndex;
    uint256 currLiquidityRate;
    uint256 currVariableBorrowRate;
    uint256 reserveFactor;
    ReserveConfigurationMap reserveConfiguration;
    address aTokenAddress;
    address stableDebtTokenAddress;
    address variableDebtTokenAddress;
    uint40 reserveLastUpdateTimestamp;
    uint40 stableDebtLastUpdateTimestamp;
  }

  struct ExecuteLiquidationCallParams {
    uint256 reservesCount;
    uint256 debtToCover;
    address collateralAsset;
    address debtAsset;
    address user;
    bool receiveAToken;
    address priceOracle;
    uint8 userEModeCategory;
    address priceOracleSentinel;
  }

  struct ExecuteSupplyParams {
    address asset;
    uint256 amount;
    address onBehalfOf;
    uint16 referralCode;
  }

  struct ExecuteBorrowParams {
    address asset;
    address user;
    address onBehalfOf;
    uint256 amount;
    InterestRateMode interestRateMode;
    uint16 referralCode;
    bool releaseUnderlying;
    uint256 maxStableRateBorrowSizePercent;
    uint256 reservesCount;
    address oracle;
    uint8 userEModeCategory;
    address priceOracleSentinel;
  }

  struct ExecuteRepayParams {
    address asset;
    uint256 amount;
    InterestRateMode interestRateMode;
    address onBehalfOf;
    bool useATokens;
  }

  struct ExecuteWithdrawParams {
    address asset;
    uint256 amount;
    address to;
    uint256 reservesCount;
    address oracle;
    uint8 userEModeCategory;
  }

  struct ExecuteSetUserEModeParams {
    uint256 reservesCount;
    address oracle;
    uint8 categoryId;
  }

  struct FinalizeTransferParams {
    address asset;
    address from;
    address to;
    uint256 amount;
    uint256 balanceFromBefore;
    uint256 balanceToBefore;
    uint256 reservesCount;
    address oracle;
    uint8 fromEModeCategory;
  }

  struct FlashloanParams {
    address receiverAddress;
    address[] assets;
    uint256[] amounts;
    uint256[] interestRateModes;
    address onBehalfOf;
    bytes params;
    uint16 referralCode;
    uint256 flashLoanPremiumToProtocol;
    uint256 flashLoanPremiumTotal;
    uint256 maxStableRateBorrowSizePercent;
    uint256 reservesCount;
    address addressesProvider;
    uint8 userEModeCategory;
    bool isAuthorizedFlashBorrower;
  }

  struct FlashloanSimpleParams {
    address receiverAddress;
    address asset;
    uint256 amount;
    bytes params;
    uint16 referralCode;
    uint256 flashLoanPremiumToProtocol;
    uint256 flashLoanPremiumTotal;
  }

  struct FlashLoanRepaymentParams {
    uint256 amount;
    uint256 totalPremium;
    uint256 flashLoanPremiumToProtocol;
    address asset;
    address receiverAddress;
    uint16 referralCode;
  }

  struct CalculateUserAccountDataParams {
    UserConfigurationMap userConfig;
    uint256 reservesCount;
    address user;
    address oracle;
    uint8 userEModeCategory;
  }

  struct ValidateBorrowParams {
    ReserveCache reserveCache;
    UserConfigurationMap userConfig;
    address asset;
    address userAddress;
    uint256 amount;
    InterestRateMode interestRateMode;
    uint256 maxStableLoanPercent;
    uint256 reservesCount;
    address oracle;
    uint8 userEModeCategory;
    address priceOracleSentinel;
    bool isolationModeActive;
    address isolationModeCollateralAddress;
    uint256 isolationModeDebtCeiling;
  }

  struct ValidateLiquidationCallParams {
    ReserveCache debtReserveCache;
    uint256 totalDebt;
    uint256 healthFactor;
    address priceOracleSentinel;
  }

  struct CalculateInterestRatesParams {
    uint256 unbacked;
    uint256 liquidityAdded;
    uint256 liquidityTaken;
    uint256 totalStableDebt;
    uint256 totalVariableDebt;
    uint256 averageStableBorrowRate;
    uint256 reserveFactor;
    address reserve;
    address aToken;
  }

  struct InitReserveParams {
    address asset;
    address aTokenAddress;
    address stableDebtAddress;
    address variableDebtAddress;
    address interestRateStrategyAddress;
    uint16 reservesCount;
    uint16 maxNumberReserves;
  }
}


interface IPriceOracleGetter {
  /**
   * @notice Returns the base currency address
   * @dev Address 0x0 is reserved for USD as base currency.
   * @return Returns the base currency address.
   */
  function BASE_CURRENCY() external view returns (address);

  /**
   * @notice Returns the base currency unit
   * @dev 1 ether for ETH, 1e8 for USD.
   * @return Returns the base currency unit.
   */
  function BASE_CURRENCY_UNIT() external view returns (uint256);

  /**
   * @notice Returns the asset price in the base currency
   * @param asset The address of the asset
   * @return The price of the asset
   */
  function getAssetPrice(address asset) external view returns (uint256);
}

interface IAavePoolAddressesProvider {
  /**
   * @dev Emitted when the market identifier is updated.
   * @param oldMarketId The old id of the market
   * @param newMarketId The new id of the market
   */
  event MarketIdSet(string indexed oldMarketId, string indexed newMarketId);

  /**
   * @dev Emitted when the pool is updated.
   * @param oldAddress The old address of the Pool
   * @param newAddress The new address of the Pool
   */
  event PoolUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the pool configurator is updated.
   * @param oldAddress The old address of the PoolConfigurator
   * @param newAddress The new address of the PoolConfigurator
   */
  event PoolConfiguratorUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the price oracle is updated.
   * @param oldAddress The old address of the PriceOracle
   * @param newAddress The new address of the PriceOracle
   */
  event PriceOracleUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the ACL manager is updated.
   * @param oldAddress The old address of the ACLManager
   * @param newAddress The new address of the ACLManager
   */
  event ACLManagerUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the ACL admin is updated.
   * @param oldAddress The old address of the ACLAdmin
   * @param newAddress The new address of the ACLAdmin
   */
  event ACLAdminUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the price oracle sentinel is updated.
   * @param oldAddress The old address of the PriceOracleSentinel
   * @param newAddress The new address of the PriceOracleSentinel
   */
  event PriceOracleSentinelUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the pool data provider is updated.
   * @param oldAddress The old address of the PoolDataProvider
   * @param newAddress The new address of the PoolDataProvider
   */
  event PoolDataProviderUpdated(address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when a new proxy is created.
   * @param id The identifier of the proxy
   * @param proxyAddress The address of the created proxy contract
   * @param implementationAddress The address of the implementation contract
   */
  event ProxyCreated(
    bytes32 indexed id,
    address indexed proxyAddress,
    address indexed implementationAddress
  );

  /**
   * @dev Emitted when a new non-proxied contract address is registered.
   * @param id The identifier of the contract
   * @param oldAddress The address of the old contract
   * @param newAddress The address of the new contract
   */
  event AddressSet(bytes32 indexed id, address indexed oldAddress, address indexed newAddress);

  /**
   * @dev Emitted when the implementation of the proxy registered with id is updated
   * @param id The identifier of the contract
   * @param proxyAddress The address of the proxy contract
   * @param oldImplementationAddress The address of the old implementation contract
   * @param newImplementationAddress The address of the new implementation contract
   */
  event AddressSetAsProxy(
    bytes32 indexed id,
    address indexed proxyAddress,
    address oldImplementationAddress,
    address indexed newImplementationAddress
  );

  /**
   * @notice Returns the id of the Aave market to which this contract points to.
   * @return The market id
   */
  function getMarketId() external view returns (string memory);

  /**
   * @notice Associates an id with a specific PoolAddressesProvider.
   * @dev This can be used to create an onchain registry of PoolAddressesProviders to
   * identify and validate multiple Aave markets.
   * @param newMarketId The market id
   */
  function setMarketId(string calldata newMarketId) external;

  /**
   * @notice Returns an address by its identifier.
   * @dev The returned address might be an EOA or a contract, potentially proxied
   * @dev It returns ZERO if there is no registered address with the given id
   * @param id The id
   * @return The address of the registered for the specified id
   */
  function getAddress(bytes32 id) external view returns (address);

  /**
   * @notice General function to update the implementation of a proxy registered with
   * certain `id`. If there is no proxy registered, it will instantiate one and
   * set as implementation the `newImplementationAddress`.
   * @dev IMPORTANT Use this function carefully, only for ids that don't have an explicit
   * setter function, in order to avoid unexpected consequences
   * @param id The id
   * @param newImplementationAddress The address of the new implementation
   */
  function setAddressAsProxy(bytes32 id, address newImplementationAddress) external;

  /**
   * @notice Sets an address for an id replacing the address saved in the addresses map.
   * @dev IMPORTANT Use this function carefully, as it will do a hard replacement
   * @param id The id
   * @param newAddress The address to set
   */
  function setAddress(bytes32 id, address newAddress) external;

  /**
   * @notice Returns the address of the Pool proxy.
   * @return The Pool proxy address
   */
  function getPool() external view returns (address);

  /**
   * @notice Updates the implementation of the Pool, or creates a proxy
   * setting the new `pool` implementation when the function is called for the first time.
   * @param newPoolImpl The new Pool implementation
   */
  function setPoolImpl(address newPoolImpl) external;

  /**
   * @notice Returns the address of the PoolConfigurator proxy.
   * @return The PoolConfigurator proxy address
   */
  function getPoolConfigurator() external view returns (address);

  /**
   * @notice Updates the implementation of the PoolConfigurator, or creates a proxy
   * setting the new `PoolConfigurator` implementation when the function is called for the first time.
   * @param newPoolConfiguratorImpl The new PoolConfigurator implementation
   */
  function setPoolConfiguratorImpl(address newPoolConfiguratorImpl) external;

  /**
   * @notice Returns the address of the price oracle.
   * @return The address of the PriceOracle
   */
  function getPriceOracle() external view returns (address);

  /**
   * @notice Updates the address of the price oracle.
   * @param newPriceOracle The address of the new PriceOracle
   */
  function setPriceOracle(address newPriceOracle) external;

  /**
   * @notice Returns the address of the ACL manager.
   * @return The address of the ACLManager
   */
  function getACLManager() external view returns (address);

  /**
   * @notice Updates the address of the ACL manager.
   * @param newAclManager The address of the new ACLManager
   */
  function setACLManager(address newAclManager) external;

  /**
   * @notice Returns the address of the ACL admin.
   * @return The address of the ACL admin
   */
  function getACLAdmin() external view returns (address);

  /**
   * @notice Updates the address of the ACL admin.
   * @param newAclAdmin The address of the new ACL admin
   */
  function setACLAdmin(address newAclAdmin) external;

  /**
   * @notice Returns the address of the price oracle sentinel.
   * @return The address of the PriceOracleSentinel
   */
  function getPriceOracleSentinel() external view returns (address);

  /**
   * @notice Updates the address of the price oracle sentinel.
   * @param newPriceOracleSentinel The address of the new PriceOracleSentinel
   */
  function setPriceOracleSentinel(address newPriceOracleSentinel) external;

  /**
   * @notice Returns the address of the data provider.
   * @return The address of the DataProvider
   */
  function getPoolDataProvider() external view returns (address);

  /**
   * @notice Updates the address of the data provider.
   * @param newDataProvider The address of the new DataProvider
   */
  function setPoolDataProvider(address newDataProvider) external;
}

interface IAaveOracle is IPriceOracleGetter {
  /**
   * @dev Emitted after the base currency is set
   * @param baseCurrency The base currency of used for price quotes
   * @param baseCurrencyUnit The unit of the base currency
   */
  event BaseCurrencySet(address indexed baseCurrency, uint256 baseCurrencyUnit);

  /**
   * @dev Emitted after the price source of an asset is updated
   * @param asset The address of the asset
   * @param source The price source of the asset
   */
  event AssetSourceUpdated(address indexed asset, address indexed source);

  /**
   * @dev Emitted after the address of fallback oracle is updated
   * @param fallbackOracle The address of the fallback oracle
   */
  event FallbackOracleUpdated(address indexed fallbackOracle);

  /**
   * @notice Returns the PoolAddressesProvider
   * @return The address of the PoolAddressesProvider contract
   */
  function ADDRESSES_PROVIDER() external view returns (IAavePoolAddressesProvider);

  /**
   * @notice Sets or replaces price sources of assets
   * @param assets The addresses of the assets
   * @param sources The addresses of the price sources
   */
  function setAssetSources(address[] calldata assets, address[] calldata sources) external;

  /**
   * @notice Sets the fallback oracle
   * @param fallbackOracle The address of the fallback oracle
   */
  function setFallbackOracle(address fallbackOracle) external;

  /**
   * @notice Returns a list of prices from a list of assets addresses
   * @param assets The list of assets addresses
   * @return The prices of the given assets
   */
  function getAssetsPrices(address[] calldata assets) external view returns (uint256[] memory);

  /**
   * @notice Returns the address of the source for an asset address
   * @param asset The address of the asset
   * @return The address of the source
   */
  function getSourceOfAsset(address asset) external view returns (address);

  /**
   * @notice Returns the address of the fallback oracle
   * @return The address of the fallback oracle
   */
  function getFallbackOracle() external view returns (address);
}





library Types {
    /* ENUMS */

    enum Position {
        POOL_SUPPLIER,
        P2P_SUPPLIER,
        POOL_BORROWER,
        P2P_BORROWER
    }

    /* NESTED STRUCTS */

    struct MarketSideDelta {
        uint256 scaledDelta; // In pool unit.
        uint256 scaledP2PTotal; // In peer-to-peer unit.
    }

    struct Deltas {
        MarketSideDelta supply;
        MarketSideDelta borrow;
    }

    struct MarketSideIndexes {
        uint128 poolIndex;
        uint128 p2pIndex;
    }

    struct Indexes {
        MarketSideIndexes supply;
        MarketSideIndexes borrow;
    }

    struct PauseStatuses {
        bool isP2PDisabled;
        bool isSupplyPaused;
        bool isSupplyCollateralPaused;
        bool isBorrowPaused;
        bool isWithdrawPaused;
        bool isWithdrawCollateralPaused;
        bool isRepayPaused;
        bool isLiquidateCollateralPaused;
        bool isLiquidateBorrowPaused;
        bool isDeprecated;
    }

    /* STORAGE STRUCTS */

    // This market struct is able to be passed into memory.
    struct Market {
        // SLOT 0-1
        Indexes indexes;
        // SLOT 2-5
        Deltas deltas; // 1024 bits
        // SLOT 6
        address underlying; // 160 bits
        PauseStatuses pauseStatuses; // 80 bits
        bool isCollateral; // 8 bits
        // SLOT 7
        address variableDebtToken; // 160 bits
        uint32 lastUpdateTimestamp; // 32 bits
        uint16 reserveFactor; // 16 bits
        uint16 p2pIndexCursor; // 16 bits
        // SLOT 8
        address aToken; // 160 bits
        // SLOT 9
        address stableDebtToken; // 160 bits
        // SLOT 10
        uint256 idleSupply; // 256 bits
    }

    // Contains storage-only dynamic arrays and mappings.
    struct MarketBalances {
        LogarithmicBuckets.Buckets poolSuppliers; // In pool unit.
        LogarithmicBuckets.Buckets p2pSuppliers; // In peer-to-peer unit.
        LogarithmicBuckets.Buckets poolBorrowers; // In pool unit.
        LogarithmicBuckets.Buckets p2pBorrowers; // In peer-to-peer unit.
        mapping(address => uint256) collateral; // In pool unit.
    }

    struct Iterations {
        uint128 repay;
        uint128 withdraw;
    }

    /* STACK AND RETURN STRUCTS */

    struct LiquidityData {
        uint256 borrowable; // The maximum debt value allowed to borrow (in base currency).
        uint256 maxDebt; // The maximum debt value allowed before being liquidatable (in base currency).
        uint256 debt; // The debt value (in base currency).
    }

    struct IndexesParams {
        MarketSideIndexes256 lastSupplyIndexes;
        MarketSideIndexes256 lastBorrowIndexes;
        uint256 poolSupplyIndex; // The current pool supply index.
        uint256 poolBorrowIndex; // The current pool borrow index.
        uint256 reserveFactor; // The reserve factor percentage (10 000 = 100%).
        uint256 p2pIndexCursor; // The peer-to-peer index cursor (10 000 = 100%).
        Deltas deltas; // The deltas and peer-to-peer amounts.
        uint256 proportionIdle; // in ray.
    }

    struct GrowthFactors {
        uint256 poolSupplyGrowthFactor; // The pool's supply index growth factor (in ray).
        uint256 p2pSupplyGrowthFactor; // Peer-to-peer supply index growth factor (in ray).
        uint256 poolBorrowGrowthFactor; // The pool's borrow index growth factor (in ray).
        uint256 p2pBorrowGrowthFactor; // Peer-to-peer borrow index growth factor (in ray).
    }

    struct MarketSideIndexes256 {
        uint256 poolIndex;
        uint256 p2pIndex;
    }

    struct Indexes256 {
        MarketSideIndexes256 supply;
        MarketSideIndexes256 borrow;
    }

    struct Signature {
        uint8 v;
        bytes32 r;
        bytes32 s;
    }

    struct MatchingEngineVars {
        address underlying;
        MarketSideIndexes256 indexes;
        uint256 amount;
        uint256 maxIterations;
        bool borrow;
        function (address, address, uint256, uint256, bool) returns (uint256, uint256) updateDS; // This function will be used to update the data-structure.
        bool demoting; // True for demote, False for promote.
        function(uint256, uint256, MarketSideIndexes256 memory, uint256)
            pure returns (uint256, uint256, uint256) step; // This function will be used to decide whether to use the algorithm for promoting or for demoting.
    }

    struct LiquidityVars {
        address user;
        IAaveOracle oracle;
        DataTypesAaveV3.EModeCategory eModeCategory;
    }

    struct PromoteVars {
        address underlying;
        uint256 amount;
        uint256 p2pIndex;
        uint256 maxIterations;
        function(address, uint256, uint256) returns (uint256, uint256) promote;
    }

    struct BorrowWithdrawVars {
        uint256 onPool;
        uint256 inP2P;
        uint256 toWithdraw;
        uint256 toBorrow;
    }

    struct SupplyRepayVars {
        uint256 onPool;
        uint256 inP2P;
        uint256 toSupply;
        uint256 toRepay;
    }

    struct LiquidateVars {
        uint256 closeFactor;
        uint256 seized;
    }

    struct AmountToSeizeVars {
        uint256 liquidationBonus;
        uint256 borrowedTokenUnit;
        uint256 collateralTokenUnit;
        uint256 borrowedPrice;
        uint256 collateralPrice;
    }
}






interface IMorphoGetters {
    function DOMAIN_SEPARATOR() external view returns (bytes32);
    function pool() external view returns (address);
    function addressesProvider() external view returns (address);
    function eModeCategoryId() external view returns (uint256);

    function market(address underlying) external view returns (Types.Market memory);
    function marketsCreated() external view returns (address[] memory);

    function scaledCollateralBalance(address underlying, address user) external view returns (uint256);
    function scaledP2PBorrowBalance(address underlying, address user) external view returns (uint256);
    function scaledP2PSupplyBalance(address underlying, address user) external view returns (uint256);
    function scaledPoolBorrowBalance(address underlying, address user) external view returns (uint256);
    function scaledPoolSupplyBalance(address underlying, address user) external view returns (uint256);

    function supplyBalance(address underlying, address user) external view returns (uint256);
    function borrowBalance(address underlying, address user) external view returns (uint256);
    function collateralBalance(address underlying, address user) external view returns (uint256);

    function userCollaterals(address user) external view returns (address[] memory);
    function userBorrows(address user) external view returns (address[] memory);

    function isManaging(address delegator, address manager) external view returns (bool);
    function userNonce(address user) external view returns (uint256);

    function defaultIterations() external view returns (Types.Iterations memory);
    function positionsManager() external view returns (address);
    function rewardsManager() external view returns (address);
    function treasuryVault() external view returns (address);

    function isClaimRewardsPaused() external view returns (bool);

    function updatedIndexes(address underlying) external view returns (Types.Indexes256 memory);
    function liquidityData(address user) external view returns (Types.LiquidityData memory);
    function getNext(address underlying, Types.Position position, address user) external view returns (address);
    function getBucketsMask(address underlying, Types.Position position) external view returns (uint256);
}

interface IMorphoSetters {
    function createMarket(address underlying, uint16 reserveFactor, uint16 p2pIndexCursor) external;
    function increaseP2PDeltas(address underlying, uint256 amount) external;
    function claimToTreasury(address[] calldata underlyings, uint256[] calldata amounts) external;

    function setPositionsManager(address positionsManager) external;
    function setRewardsManager(address rewardsManager) external;
    function setTreasuryVault(address treasuryVault) external;
    function setDefaultIterations(Types.Iterations memory defaultIterations) external;
    function setP2PIndexCursor(address underlying, uint16 p2pIndexCursor) external;
    function setReserveFactor(address underlying, uint16 newReserveFactor) external;

    function setIsClaimRewardsPaused(bool isPaused) external;
    function setIsPaused(address underlying, bool isPaused) external;
    function setIsPausedForAllMarkets(bool isPaused) external;
    function setIsSupplyPaused(address underlying, bool isPaused) external;
    function setIsSupplyCollateralPaused(address underlying, bool isPaused) external;
    function setIsBorrowPaused(address underlying, bool isPaused) external;
    function setIsRepayPaused(address underlying, bool isPaused) external;
    function setIsWithdrawPaused(address underlying, bool isPaused) external;
    function setIsWithdrawCollateralPaused(address underlying, bool isPaused) external;
    function setIsLiquidateBorrowPaused(address underlying, bool isPaused) external;
    function setIsLiquidateCollateralPaused(address underlying, bool isPaused) external;
    function setIsP2PDisabled(address underlying, bool isP2PDisabled) external;
    function setIsDeprecated(address underlying, bool isDeprecated) external;
}

interface IMorphoAaveV3 is IMorphoGetters, IMorphoSetters {
    function initialize(
        address addressesProvider,
        uint8 eModeCategoryId,
        address newPositionsManager,
        Types.Iterations memory newDefaultIterations
    ) external;

    function supply(address underlying, uint256 amount, address onBehalf, uint256 maxIterations)
        external
        returns (uint256 supplied);
    function supplyWithPermit(
        address underlying,
        uint256 amount,
        address onBehalf,
        uint256 maxIterations,
        uint256 deadline,
        Types.Signature calldata signature
    ) external returns (uint256 supplied);
    function supplyCollateral(address underlying, uint256 amount, address onBehalf)
        external
        returns (uint256 supplied);
    function supplyCollateralWithPermit(
        address underlying,
        uint256 amount,
        address onBehalf,
        uint256 deadline,
        Types.Signature calldata signature
    ) external returns (uint256 supplied);

    function borrow(address underlying, uint256 amount, address onBehalf, address receiver, uint256 maxIterations)
        external
        returns (uint256 borrowed);

    function repay(address underlying, uint256 amount, address onBehalf) external returns (uint256 repaid);
    function repayWithPermit(
        address underlying,
        uint256 amount,
        address onBehalf,
        uint256 deadline,
        Types.Signature calldata signature
    ) external returns (uint256 repaid);

    function withdraw(address underlying, uint256 amount, address onBehalf, address receiver, uint256 maxIterations)
        external
        returns (uint256 withdrawn);
    function withdrawCollateral(address underlying, uint256 amount, address onBehalf, address receiver)
        external
        returns (uint256 withdrawn);

    function approveManager(address manager, bool isAllowed) external;
    function approveManagerWithSig(
        address delegator,
        address manager,
        bool isAllowed,
        uint256 nonce,
        uint256 deadline,
        Types.Signature calldata signature
    ) external;

    function isManagedBy(address delegator, address manager) external view returns(bool);

    function liquidate(address underlyingBorrowed, address underlyingCollateral, address user, uint256 amount)
        external
        returns (uint256 repaid, uint256 seized);

    function claimRewards(address[] calldata assets, address onBehalf)
        external
        returns (address[] memory rewardTokens, uint256[] memory claimedAmounts);
}







contract MainnetAuthAddresses {
    address internal constant ADMIN_VAULT_ADDR = 0xCCf3d848e08b94478Ed8f46fFead3008faF581fD;
    address internal constant DSGUARD_FACTORY_ADDRESS = 0x5a15566417e6C1c9546523066500bDDBc53F88C7;
    address internal constant ADMIN_ADDR = 0x25eFA336886C74eA8E282ac466BdCd0199f85BB9; // USED IN ADMIN VAULT CONSTRUCTOR
    address internal constant PROXY_AUTH_ADDRESS = 0x149667b6FAe2c63D1B4317C716b0D0e4d3E2bD70;
    address internal constant MODULE_AUTH_ADDRESS = 0x7407974DDBF539e552F1d051e44573090912CC3D;
}







contract AuthHelper is MainnetAuthAddresses {
}








contract AdminVault is AuthHelper {
    address public owner;
    address public admin;

    error SenderNotAdmin();

    constructor() {
        owner = msg.sender;
        admin = ADMIN_ADDR;
    }

    /// @notice Admin is able to change owner
    /// @param _owner Address of new owner
    function changeOwner(address _owner) public {
        if (admin != msg.sender){
            revert SenderNotAdmin();
        }
        owner = _owner;
    }

    /// @notice Admin is able to set new admin
    /// @param _admin Address of multisig that becomes new admin
    function changeAdmin(address _admin) public {
        if (admin != msg.sender){
            revert SenderNotAdmin();
        }
        admin = _admin;
    }

}







library Address {
    //insufficient balance
    error InsufficientBalance(uint256 available, uint256 required);
    //unable to send value, recipient may have reverted
    error SendingValueFail();
    //insufficient balance for call
    error InsufficientBalanceForCall(uint256 available, uint256 required);
    //call to non-contract
    error NonContractCall();
    
    function isContract(address account) internal view returns (bool) {
        // According to EIP-1052, 0x0 is the value returned for not-yet created accounts
        // and 0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470 is returned
        // for accounts without code, i.e. `keccak256('')`
        bytes32 codehash;
        bytes32 accountHash = 0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470;
        // solhint-disable-next-line no-inline-assembly
        assembly {
            codehash := extcodehash(account)
        }
        return (codehash != accountHash && codehash != 0x0);
    }

    function sendValue(address payable recipient, uint256 amount) internal {
        uint256 balance = address(this).balance;
        if (balance < amount){
            revert InsufficientBalance(balance, amount);
        }

        // solhint-disable-next-line avoid-low-level-calls, avoid-call-value
        (bool success, ) = recipient.call{value: amount}("");
        if (!(success)){
            revert SendingValueFail();
        }
    }

    function functionCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionCall(target, data, "Address: low-level call failed");
    }

    function functionCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal returns (bytes memory) {
        return _functionCallWithValue(target, data, 0, errorMessage);
    }

    function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value
    ) internal returns (bytes memory) {
        return
            functionCallWithValue(target, data, value, "Address: low-level call with value failed");
    }

    function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value,
        string memory errorMessage
    ) internal returns (bytes memory) {
        uint256 balance = address(this).balance;
        if (balance < value){
            revert InsufficientBalanceForCall(balance, value);
        }
        return _functionCallWithValue(target, data, value, errorMessage);
    }

    function _functionCallWithValue(
        address target,
        bytes memory data,
        uint256 weiValue,
        string memory errorMessage
    ) private returns (bytes memory) {
        if (!(isContract(target))){
            revert NonContractCall();
        }

        // solhint-disable-next-line avoid-low-level-calls
        (bool success, bytes memory returndata) = target.call{value: weiValue}(data);
        if (success) {
            return returndata;
        } else {
            // Look for revert reason and bubble it up if present
            if (returndata.length > 0) {
                // The easiest way to bubble the revert reason is using memory via assembly

                // solhint-disable-next-line no-inline-assembly
                assembly {
                    let returndata_size := mload(returndata)
                    revert(add(32, returndata), returndata_size)
                }
            } else {
                revert(errorMessage);
            }
        }
    }
}











library SafeERC20 {
    using Address for address;

    /**
     * @dev Transfer `value` amount of `token` from the calling contract to `to`. If `token` returns no value,
     * non-reverting calls are assumed to be successful.
     */
    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transfer.selector, to, value));
    }

    /**
     * @dev Transfer `value` amount of `token` from `from` to `to`, spending the approval given by `from` to the
     * calling contract. If `token` returns no value, non-reverting calls are assumed to be successful.
     */
    function safeTransferFrom(IERC20 token, address from, address to, uint256 value) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transferFrom.selector, from, to, value));
    }

    /**
     * @dev Set the calling contract's allowance toward `spender` to `value`. If `token` returns no value,
     * non-reverting calls are assumed to be successful. Compatible with tokens that require the approval to be set to
     * 0 before setting it to a non-zero value.
     */
    function safeApprove(IERC20 token, address spender, uint256 value) internal {
        bytes memory approvalCall = abi.encodeWithSelector(token.approve.selector, spender, value);

        if (!_callOptionalReturnBool(token, approvalCall)) {
            _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, 0));
            _callOptionalReturn(token, approvalCall);
        }
    }

    /**
     * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
     * on the return value: the return value is optional (but if data is returned, it must not be false).
     * @param token The token targeted by the call.
     * @param data The call data (encoded using abi.encode or one of its variants).
     */
    function _callOptionalReturn(IERC20 token, bytes memory data) private {
        // We need to perform a low level call here, to bypass Solidity's return data size checking mechanism, since
        // we're implementing it ourselves. We use {Address-functionCall} to perform this call, which verifies that
        // the target address contains contract code and also asserts for success in the low-level call.

        bytes memory returndata = address(token).functionCall(data, "SafeERC20: low-level call failed");
        require(returndata.length == 0 || abi.decode(returndata, (bool)), "SafeERC20: ERC20 operation did not succeed");
    }

    /**
     * @dev Imitates a Solidity high-level call (i.e. a regular function call to a contract), relaxing the requirement
     * on the return value: the return value is optional (but if data is returned, it must not be false).
     * @param token The token targeted by the call.
     * @param data The call data (encoded using abi.encode or one of its variants).
     *
     * This is a variant of {_callOptionalReturn} that silents catches all reverts and returns a bool instead.
     */
    function _callOptionalReturnBool(IERC20 token, bytes memory data) private returns (bool) {
        // We need to perform a low level call here, to bypass Solidity's return data size checking mechanism, since
        // we're implementing it ourselves. We cannot use {Address-functionCall} here since this should return false
        // and not revert is the subcall reverts.

        (bool success, bytes memory returndata) = address(token).call(data);
        return success && (returndata.length == 0 || abi.decode(returndata, (bool))) && address(token).code.length > 0;
    }
}











contract AdminAuth is AuthHelper {
    using SafeERC20 for IERC20;

    AdminVault public constant adminVault = AdminVault(ADMIN_VAULT_ADDR);

    error SenderNotOwner();
    error SenderNotAdmin();

    modifier onlyOwner() {
        if (adminVault.owner() != msg.sender){
            revert SenderNotOwner();
        }
        _;
    }

    modifier onlyAdmin() {
        if (adminVault.admin() != msg.sender){
            revert SenderNotAdmin();
        }
        _;
    }

    /// @notice withdraw stuck funds
    function withdrawStuckFunds(address _token, address _receiver, uint256 _amount) public onlyOwner {
        if (_token == 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE) {
            payable(_receiver).transfer(_amount);
        } else {
            IERC20(_token).safeTransfer(_receiver, _amount);
        }
    }

    /// @notice Destroy the contract
    /// @dev Deprecated method, selfdestruct will soon just send eth
    function kill() public onlyAdmin {
        selfdestruct(payable(msg.sender));
    }
}






contract MorphoMarketStorage is AdminAuth {

    mapping (uint256 => address) public morphoAddresses;

    constructor() {
        address morpho_aaveV3_eth = 0x33333aea097c193e66081E930c33020272b33333;
        morphoAddresses[1] = morpho_aaveV3_eth;
    }

    function getMorphoAddress(uint256 _emodeId) public view returns (address) {
        return morphoAddresses[_emodeId];
    }

    function addNewMorphoAddress(uint256 _emodeId, address _morphoAddress) public onlyOwner {
        morphoAddresses[_emodeId] = _morphoAddress;
    }
}










contract MorphoAaveV3Helper is MainnetMorphoAaveV3Addresses, DSMath {

    MorphoMarketStorage internal constant morphoMarketStorage = MorphoMarketStorage(MORPHO_MARKET_STORAGE);

    error InvalidEModeId(uint256 _emodeId);

    function getMorphoAddressByEmode(uint256 _emodeId) public view returns (address) {
        address morphoAddr = morphoMarketStorage.getMorphoAddress(_emodeId);

        if (morphoAddr == address(0)) revert InvalidEModeId(_emodeId);

        return morphoAddr;
    }

    function getSafetyRatio(address _morphoAddr, address _usr) internal view returns (uint256) {
        Types.LiquidityData memory liqData = IMorphoAaveV3(_morphoAddr).liquidityData(_usr);
        if (liqData.debt == 0) return 0;
        return wdiv(liqData.borrowable, liqData.debt);
    }
}







contract MainnetSparkAddresses {
    address internal constant SPARK_REWARDS_CONTROLLER_ADDRESS = 0x4370D3b6C9588E02ce9D22e684387859c7Ff5b34;
    address internal constant DEFAULT_SPARK_MARKET = 0x02C3eA4e34C0cBd694D2adFa2c690EECbC1793eE;
    address internal constant SPARK_ORACLE_V3 = 0x8105f69D9C41644c6A0803fDA7D03Aa70996cFD9;
    address internal constant SDAI_ADDR = 0x83F20F44975D03b1b09e64809B757c47f942BEeA;
}









contract SparkHelper is MainnetSparkAddresses {
    
    uint16 public constant SPARK_REFERRAL_CODE = 0;
    
    
    /// @notice Returns the lending pool contract of the specified market
    function getLendingPool(address _market) internal virtual view returns (IL2PoolV3) {
        return IL2PoolV3(IPoolAddressesProvider(_market).getPool());
    }

    /// @notice Fetch the data provider for the specified market
    function getDataProvider(address _market) internal virtual view returns (IAaveProtocolDataProvider) {
        return
            IAaveProtocolDataProvider(
                IPoolAddressesProvider(_market).getPoolDataProvider()
            );
    }

    function boolToBytes(bool x) internal virtual pure returns (bytes1 r) {
       return x ? bytes1(0x01) : bytes1(0x00);
    }

    function bytesToBool(bytes1 x) internal virtual pure returns (bool r) {
        return x != bytes1(0x00);
    }
    
    function getWholeDebt(address _market, address _tokenAddr, uint _borrowType, address _debtOwner) internal virtual view returns (uint256 debt) {
        uint256 STABLE_ID = 1;
        uint256 VARIABLE_ID = 2;
        
        IAaveProtocolDataProvider dataProvider = getDataProvider(_market);
        (, uint256 borrowsStable, uint256 borrowsVariable, , , , , , ) =
            dataProvider.getUserReserveData(_tokenAddr, _debtOwner);

        if (_borrowType == STABLE_ID) {
            debt = borrowsStable;
        } else if (_borrowType == VARIABLE_ID) {
            debt = borrowsVariable;
        }
    }
}







contract MainnetActionsUtilAddresses {
    address internal constant DFS_REG_CONTROLLER_ADDR = 0xF8f8B3C98Cf2E63Df3041b73f80F362a4cf3A576;
    address internal constant REGISTRY_ADDR = 0x287778F121F134C66212FB16c9b53eC991D32f5b;
    address internal constant DFS_LOGGER_ADDR = 0xcE7a977Cac4a481bc84AC06b2Da0df614e621cf3;
    address internal constant SUB_STORAGE_ADDR = 0x1612fc28Ee0AB882eC99842Cde0Fc77ff0691e90;
    address internal constant PROXY_AUTH_ADDR = 0x149667b6FAe2c63D1B4317C716b0D0e4d3E2bD70;
    address internal constant LSV_PROXY_REGISTRY_ADDRESS = 0xa8a3c86c4A2DcCf350E84D2b3c46BDeBc711C16e;
    address internal constant TRANSIENT_STORAGE = 0x2F7Ef2ea5E8c97B8687CA703A0e50Aa5a49B7eb2;
}







contract ActionsUtilHelper is MainnetActionsUtilAddresses {
}








contract DFSRegistry is AdminAuth {
    error EntryAlreadyExistsError(bytes4);
    error EntryNonExistentError(bytes4);
    error EntryNotInChangeError(bytes4);
    error ChangeNotReadyError(uint256,uint256);
    error EmptyPrevAddrError(bytes4);
    error AlreadyInContractChangeError(bytes4);
    error AlreadyInWaitPeriodChangeError(bytes4);

    event AddNewContract(address,bytes4,address,uint256);
    event RevertToPreviousAddress(address,bytes4,address,address);
    event StartContractChange(address,bytes4,address,address);
    event ApproveContractChange(address,bytes4,address,address);
    event CancelContractChange(address,bytes4,address,address);
    event StartWaitPeriodChange(address,bytes4,uint256);
    event ApproveWaitPeriodChange(address,bytes4,uint256,uint256);
    event CancelWaitPeriodChange(address,bytes4,uint256,uint256);

    struct Entry {
        address contractAddr;
        uint256 waitPeriod;
        uint256 changeStartTime;
        bool inContractChange;
        bool inWaitPeriodChange;
        bool exists;
    }

    mapping(bytes4 => Entry) public entries;
    mapping(bytes4 => address) public previousAddresses;

    mapping(bytes4 => address) public pendingAddresses;
    mapping(bytes4 => uint256) public pendingWaitTimes;

    /// @notice Given an contract id returns the registered address
    /// @dev Id is keccak256 of the contract name
    /// @param _id Id of contract
    function getAddr(bytes4 _id) public view returns (address) {
        return entries[_id].contractAddr;
    }

    /// @notice Helper function to easily query if id is registered
    /// @param _id Id of contract
    function isRegistered(bytes4 _id) public view returns (bool) {
        return entries[_id].exists;
    }

    /////////////////////////// OWNER ONLY FUNCTIONS ///////////////////////////

    /// @notice Adds a new contract to the registry
    /// @param _id Id of contract
    /// @param _contractAddr Address of the contract
    /// @param _waitPeriod Amount of time to wait before a contract address can be changed
    function addNewContract(
        bytes4 _id,
        address _contractAddr,
        uint256 _waitPeriod
    ) public onlyOwner {
        if (entries[_id].exists){
            revert EntryAlreadyExistsError(_id);
        }

        entries[_id] = Entry({
            contractAddr: _contractAddr,
            waitPeriod: _waitPeriod,
            changeStartTime: 0,
            inContractChange: false,
            inWaitPeriodChange: false,
            exists: true
        });

        emit AddNewContract(msg.sender, _id, _contractAddr, _waitPeriod);
    }

    /// @notice Reverts to the previous address immediately
    /// @dev In case the new version has a fault, a quick way to fallback to the old contract
    /// @param _id Id of contract
    function revertToPreviousAddress(bytes4 _id) public onlyOwner {
        if (!(entries[_id].exists)){
            revert EntryNonExistentError(_id);
        }
        if (previousAddresses[_id] == address(0)){
            revert EmptyPrevAddrError(_id);
        }

        address currentAddr = entries[_id].contractAddr;
        entries[_id].contractAddr = previousAddresses[_id];

        emit RevertToPreviousAddress(msg.sender, _id, currentAddr, previousAddresses[_id]);
    }

    /// @notice Starts an address change for an existing entry
    /// @dev Can override a change that is currently in progress
    /// @param _id Id of contract
    /// @param _newContractAddr Address of the new contract
    function startContractChange(bytes4 _id, address _newContractAddr) public onlyOwner {
        if (!entries[_id].exists){
            revert EntryNonExistentError(_id);
        }
        if (entries[_id].inWaitPeriodChange){
            revert AlreadyInWaitPeriodChangeError(_id);
        }

        entries[_id].changeStartTime = block.timestamp; // solhint-disable-line
        entries[_id].inContractChange = true;

        pendingAddresses[_id] = _newContractAddr;

        emit StartContractChange(msg.sender, _id, entries[_id].contractAddr, _newContractAddr);
    }

    /// @notice Changes new contract address, correct time must have passed
    /// @param _id Id of contract
    function approveContractChange(bytes4 _id) public onlyOwner {
        if (!entries[_id].exists){
            revert EntryNonExistentError(_id);
        }
        if (!entries[_id].inContractChange){
            revert EntryNotInChangeError(_id);
        }
        if (block.timestamp < (entries[_id].changeStartTime + entries[_id].waitPeriod)){// solhint-disable-line
            revert ChangeNotReadyError(block.timestamp, (entries[_id].changeStartTime + entries[_id].waitPeriod));
        }

        address oldContractAddr = entries[_id].contractAddr;
        entries[_id].contractAddr = pendingAddresses[_id];
        entries[_id].inContractChange = false;
        entries[_id].changeStartTime = 0;

        pendingAddresses[_id] = address(0);
        previousAddresses[_id] = oldContractAddr;

        emit ApproveContractChange(msg.sender, _id, oldContractAddr, entries[_id].contractAddr);
    }

    /// @notice Cancel pending change
    /// @param _id Id of contract
    function cancelContractChange(bytes4 _id) public onlyOwner {
        if (!entries[_id].exists){
            revert EntryNonExistentError(_id);
        }
        if (!entries[_id].inContractChange){
            revert EntryNotInChangeError(_id);
        }

        address oldContractAddr = pendingAddresses[_id];

        pendingAddresses[_id] = address(0);
        entries[_id].inContractChange = false;
        entries[_id].changeStartTime = 0;

        emit CancelContractChange(msg.sender, _id, oldContractAddr, entries[_id].contractAddr);
    }

    /// @notice Starts the change for waitPeriod
    /// @param _id Id of contract
    /// @param _newWaitPeriod New wait time
    function startWaitPeriodChange(bytes4 _id, uint256 _newWaitPeriod) public onlyOwner {
        if (!entries[_id].exists){
            revert EntryNonExistentError(_id);
        }
        if (entries[_id].inContractChange){
            revert AlreadyInContractChangeError(_id);
        }

        pendingWaitTimes[_id] = _newWaitPeriod;

        entries[_id].changeStartTime = block.timestamp; // solhint-disable-line
        entries[_id].inWaitPeriodChange = true;

        emit StartWaitPeriodChange(msg.sender, _id, _newWaitPeriod);
    }

    /// @notice Changes new wait period, correct time must have passed
    /// @param _id Id of contract
    function approveWaitPeriodChange(bytes4 _id) public onlyOwner {
        if (!entries[_id].exists){
            revert EntryNonExistentError(_id);
        }
        if (!entries[_id].inWaitPeriodChange){
            revert EntryNotInChangeError(_id);
        }
        if (block.timestamp < (entries[_id].changeStartTime + entries[_id].waitPeriod)){ // solhint-disable-line
            revert ChangeNotReadyError(block.timestamp, (entries[_id].changeStartTime + entries[_id].waitPeriod));
        }

        uint256 oldWaitTime = entries[_id].waitPeriod;
        entries[_id].waitPeriod = pendingWaitTimes[_id];
        
        entries[_id].inWaitPeriodChange = false;
        entries[_id].changeStartTime = 0;

        pendingWaitTimes[_id] = 0;

        emit ApproveWaitPeriodChange(msg.sender, _id, oldWaitTime, entries[_id].waitPeriod);
    }

    /// @notice Cancel wait period change
    /// @param _id Id of contract
    function cancelWaitPeriodChange(bytes4 _id) public onlyOwner {
        if (!entries[_id].exists){
            revert EntryNonExistentError(_id);
        }
        if (!entries[_id].inWaitPeriodChange){
            revert EntryNotInChangeError(_id);
        }

        uint256 oldWaitPeriod = pendingWaitTimes[_id];

        pendingWaitTimes[_id] = 0;
        entries[_id].inWaitPeriodChange = false;
        entries[_id].changeStartTime = 0;

        emit CancelWaitPeriodChange(msg.sender, _id, oldWaitPeriod, entries[_id].waitPeriod);
    }
}







abstract contract DSAuthority {
    function canCall(
        address src,
        address dst,
        bytes4 sig
    ) public view virtual returns (bool);
}







contract DSAuthEvents {
    event LogSetAuthority(address indexed authority);
    event LogSetOwner(address indexed owner);
}

contract DSAuth is DSAuthEvents {
    DSAuthority public authority;
    address public owner;

    constructor() {
        owner = msg.sender;
        emit LogSetOwner(msg.sender);
    }

    function setOwner(address owner_) public auth {
        owner = owner_;
        emit LogSetOwner(owner);
    }

    function setAuthority(DSAuthority authority_) public auth {
        authority = authority_;
        emit LogSetAuthority(address(authority));
    }

    modifier auth {
        require(isAuthorized(msg.sender, msg.sig), "Not authorized");
        _;
    }

    function isAuthorized(address src, bytes4 sig) internal view returns (bool) {
        if (src == address(this)) {
            return true;
        } else if (src == owner) {
            return true;
        } else if (authority == DSAuthority(address(0))) {
            return false;
        } else {
            return authority.canCall(src, address(this), sig);
        }
    }
}







contract DSNote {
    event LogNote(
        bytes4 indexed sig,
        address indexed guy,
        bytes32 indexed foo,
        bytes32 indexed bar,
        uint256 wad,
        bytes fax
    ) anonymous;

    modifier note {
        bytes32 foo;
        bytes32 bar;

        assembly {
            foo := calldataload(4)
            bar := calldataload(36)
        }

        emit LogNote(msg.sig, msg.sender, foo, bar, msg.value, msg.data);

        _;
    }
}








abstract contract DSProxy is DSAuth, DSNote {
    DSProxyCache public cache; // global cache for contracts

    constructor(address _cacheAddr) {
        if (!(setCache(_cacheAddr))){
            require(isAuthorized(msg.sender, msg.sig), "Not authorized");
        }
    }

    // solhint-disable-next-line no-empty-blocks
    receive() external payable {}

    // use the proxy to execute calldata _data on contract _code
    function execute(bytes memory _code, bytes memory _data)
        public
        payable
        virtual
        returns (address target, bytes32 response);

    function execute(address _target, bytes memory _data)
        public
        payable
        virtual
        returns (bytes32 response);

    //set new cache
    function setCache(address _cacheAddr) public payable virtual returns (bool);
}

contract DSProxyCache {
    mapping(bytes32 => address) cache;

    function read(bytes memory _code) public view returns (address) {
        bytes32 hash = keccak256(_code);
        return cache[hash];
    }

    function write(bytes memory _code) public returns (address target) {
        assembly {
            target := create(0, add(_code, 0x20), mload(_code))
            switch iszero(extcodesize(target))
                case 1 {
                    // throw if contract failed to deploy
                    revert(0, 0)
                }
        }
        bytes32 hash = keccak256(_code);
        cache[hash] = target;
    }
}







abstract contract DSProxyFactoryInterface {
    function build(address owner) public virtual returns (DSProxy proxy);
    function build() public virtual returns (DSProxy proxy);
}







abstract contract IWETH {
    function allowance(address, address) public virtual view returns (uint256);

    function balanceOf(address) public virtual view returns (uint256);

    function approve(address, uint256) public virtual;

    function transfer(address, uint256) public virtual returns (bool);

    function transferFrom(
        address,
        address,
        uint256
    ) public virtual returns (bool);

    function deposit() public payable virtual;

    function withdraw(uint256) public virtual;
}







abstract contract IDSProxy {
    // function execute(bytes memory _code, bytes memory _data)
    //     public
    //     payable
    //     virtual
    //     returns (address, bytes32);

    function execute(address _target, bytes memory _data) public payable virtual returns (bytes32);

    function setCache(address _cacheAddr) public payable virtual returns (bool);

    function owner() public view virtual returns (address);
}







abstract contract IProxyRegistry {
    function proxies(address _owner) public virtual view returns (address);
    function build(address) public virtual returns (address);
}







contract MainnetUtilAddresses {
    address internal refillCaller = 0x8973f5e6142ed2e2F50EEE8Bb34a47C2DAa6624a;

    address internal constant FEE_RECIPIENT_ADDR = 0x39C4a92Dc506300c3Ea4c67ca4CA611102ee6F2A;
    address internal constant BOT_REGISTRY_ADDRESS = 0x637726f8b08a7ABE3aE3aCaB01A80E2d8ddeF77B;
    address internal constant UNI_V2_ROUTER = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
    address internal constant MKR_PROXY_REGISTRY = 0x4678f0a6958e4D2Bc4F1BAF7Bc52E8F3564f3fE4;
    address internal constant AAVE_MARKET = 0xB53C1a33016B2DC2fF3653530bfF1848a515c8c5;
    address internal constant AAVE_V3_MARKET = 0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e;
    address internal constant SPARK_MARKET = 0x02C3eA4e34C0cBd694D2adFa2c690EECbC1793eE;

    address internal constant DFS_PROXY_REGISTRY_ADDR = 0x29474FdaC7142f9aB7773B8e38264FA15E3805ed;

    address internal constant WETH_ADDR = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address internal constant ETH_ADDR = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE;
    address internal constant WSTETH_ADDR = 0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0;
    address internal constant STETH_ADDR = 0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84;
    address internal constant EETH_ADDR = 0x35fA164735182de50811E8e2E824cFb9B6118ac2;
    address internal constant WEETH_ADDR = 0xCd5fE23C85820F7B72D0926FC9b05b43E359b7ee; 
    address internal constant WBTC_ADDR = 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599;
    address internal constant CHAINLINK_WBTC_ADDR = 0xbBbBBBBbbBBBbbbBbbBbbbbBBbBbbbbBbBbbBBbB;
    address internal constant DAI_ADDR = 0x6B175474E89094C44Da98b954EedeAC495271d0F;

    address internal constant ETHER_FI_LIQUIDITY_POOL = 0x308861A430be4cce5502d0A12724771Fc6DaF216;

    address internal constant FEE_RECEIVER_ADMIN_ADDR = 0xA74e9791D7D66c6a14B2C571BdA0F2A1f6D64E06;

    address internal constant UNI_V3_ROUTER = 0xE592427A0AEce92De3Edee1F18E0157C05861564;
    address internal constant UNI_V3_QUOTER = 0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6;

    address internal constant FEE_RECIPIENT = 0x39C4a92Dc506300c3Ea4c67ca4CA611102ee6F2A;

    // not needed on mainnet
    address internal constant DEFAULT_BOT = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE;

    address internal constant CHAINLINK_FEED_REGISTRY = 0x47Fb2585D2C56Fe188D0E6ec628a38b74fCeeeDf;

    address internal constant TX_SAVER_FEE_RECIPIENT = 0x0eD7f3223266Ca1694F85C23aBe06E614Af3A479;

}







contract UtilHelper is MainnetUtilAddresses{
}













contract DFSProxyRegistry is AdminAuth, UtilHelper, ActionsUtilHelper {

    IProxyRegistry public mcdRegistry = IProxyRegistry(MKR_PROXY_REGISTRY);
    DFSRegistry public constant registry = DFSRegistry(REGISTRY_ADDR);

    bytes4 public DFS_PROXY_REGISTRY_CONTROLLER_ID = 0xcbbb53f2;

    mapping(address => address) public changedOwners;
    mapping(address => address[]) public additionalProxies;

    /// @notice Changes the proxy that is returned for the user
    /// @dev Used when the user changed DSProxy ownership himself
    function changeMcdOwner(address _user, address _proxy) public {
        address dfsProxyRegistryController = registry.getAddr(DFS_PROXY_REGISTRY_CONTROLLER_ID);
        require(msg.sender == dfsProxyRegistryController);
        
        if (IDSProxy(_proxy).owner() == _user) {
            changedOwners[_user] = _proxy;
        }
    }

    /// @notice Returns the proxy address associated with the user account
    /// @dev If user changed ownership of DSProxy admin can hardcode replacement
    function getMcdProxy(address _user) public view returns (address) {
        address proxyAddr = mcdRegistry.proxies(_user);

        // if check changed proxies
        if (changedOwners[_user] != address(0)) {
            return changedOwners[_user];
        }

        return proxyAddr;
    }

    function addAdditionalProxy(address _user, address _proxy) public {
        address dfsProxyRegistryController = registry.getAddr(DFS_PROXY_REGISTRY_CONTROLLER_ID);
        require(msg.sender == dfsProxyRegistryController);

        if (IDSProxy(_proxy).owner() == _user) {
            additionalProxies[_user].push(_proxy);
        }
    }
			
    function getAllProxies(address _user) public view returns (address, address[] memory) {
        return (getMcdProxy(_user), additionalProxies[_user]);
    }
}







contract MainnetProxyFactoryAddresses {
    address internal constant PROXY_FACTORY_ADDR = 0xA26e15C895EFc0616177B7c1e7270A4C7D51C997;
}







contract DSProxyFactoryHelper is MainnetProxyFactoryAddresses {
}














contract LSVProxyRegistry is AdminAuth, UtilHelper, ActionsUtilHelper, DSProxyFactoryHelper {

    /// @dev List of proxies a user owns
    mapping(address => address[]) public proxies;

    /// @dev List of prebuilt proxies the users can claim to save gas
    address[] public proxyPool;

    event NewProxy(address, address);
    event ChangedOwner(address oldOwner, address newOwner, address proxy);

    /// @notice User calls from EOA to build a new LSV registered proxy
    function addNewProxy() public returns (address) {
        address newProxy = getFromPoolOrBuild(msg.sender);
        proxies[msg.sender].push(newProxy);

        emit NewProxy(msg.sender, newProxy);

        return newProxy;
    }

    function updateRegistry(address _proxyAddr, address _oldOwner, uint256 _indexNumInOldOwnerProxiesArr) public {
        // check if msg.sender is the owner of proxy in question
        require(DSProxy(payable(_proxyAddr)).owner() == msg.sender);

        // check if oldOwner really was the owner of proxy in question
        require(proxies[_oldOwner][_indexNumInOldOwnerProxiesArr] == _proxyAddr);

        // remove proxy from oldOwners proxies
        uint256 oldOwnersProxyCount = proxies[_oldOwner].length;
        if (oldOwnersProxyCount > 1 && _indexNumInOldOwnerProxiesArr < (oldOwnersProxyCount - 1))  {
            proxies[_oldOwner][_indexNumInOldOwnerProxiesArr] = proxies[_oldOwner][oldOwnersProxyCount - 1];
        }
        proxies[_oldOwner].pop();

        // add proxy to msg.sender proxies
        proxies[msg.sender].push(_proxyAddr);
    }

    /// @notice Adds proxies to pool for users to later claim and save on gas
    function addToPool(uint256 _numNewProxies) public {
        for (uint256 i = 0; i < _numNewProxies; ++i) {
            DSProxy newProxy = DSProxyFactoryInterface(PROXY_FACTORY_ADDR).build();
            proxyPool.push(address(newProxy));
        }
    }

    /// @notice helper function to get all users proxies
    function getProxies(address _user) public view returns (address[] memory){
        address[] memory resultProxies = new address[](proxies[_user].length);
        for (uint256 i = 0; i < proxies[_user].length; i++){
            resultProxies[i] = proxies[_user][i];
        }
        return resultProxies;
    }
    
    /// @notice helper function to check how many proxies are there in the proxy pool for cheaper user onboarding
    function getProxyPoolCount() public view returns (uint256) {
        return proxyPool.length;
    }

    /// @notice Create a new DSProxy or grabs a prebuilt one
    function getFromPoolOrBuild(address _user) internal returns (address) {
        if (proxyPool.length > 0) {
            address newProxy = proxyPool[proxyPool.length - 1];
            proxyPool.pop();
            
            DSAuth(newProxy).setOwner(_user);
            return newProxy;
        } else {
            DSProxy newProxy = DSProxyFactoryInterface(PROXY_FACTORY_ADDR).build(_user);
            return address(newProxy);
        }
    }
    
}









library TokenUtils {
    using SafeERC20 for IERC20;

    address public constant WETH_ADDR = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address public constant ETH_ADDR = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE;

    /// @dev Only approves the amount if allowance is lower than amount, does not decrease allowance
    function approveToken(
        address _tokenAddr,
        address _to,
        uint256 _amount
    ) internal {
        if (_tokenAddr == ETH_ADDR) return;

        if (IERC20(_tokenAddr).allowance(address(this), _to) < _amount) {
            IERC20(_tokenAddr).safeApprove(_to, _amount);
        }
    }

    function pullTokensIfNeeded(
        address _token,
        address _from,
        uint256 _amount
    ) internal returns (uint256) {
        // handle max uint amount
        if (_amount == type(uint256).max) {
            _amount = getBalance(_token, _from);
        }

        if (_from != address(0) && _from != address(this) && _token != ETH_ADDR && _amount != 0) {
            IERC20(_token).safeTransferFrom(_from, address(this), _amount);
        }

        return _amount;
    }

    function withdrawTokens(
        address _token,
        address _to,
        uint256 _amount
    ) internal returns (uint256) {
        if (_amount == type(uint256).max) {
            _amount = getBalance(_token, address(this));
        }

        if (_to != address(0) && _to != address(this) && _amount != 0) {
            if (_token != ETH_ADDR) {
                IERC20(_token).safeTransfer(_to, _amount);
            } else {
                (bool success, ) = _to.call{value: _amount}("");
                require(success, "Eth send fail");
            }
        }

        return _amount;
    }

    function depositWeth(uint256 _amount) internal {
        IWETH(WETH_ADDR).deposit{value: _amount}();
    }

    function withdrawWeth(uint256 _amount) internal {
        IWETH(WETH_ADDR).withdraw(_amount);
    }

    function getBalance(address _tokenAddr, address _acc) internal view returns (uint256) {
        if (_tokenAddr == ETH_ADDR) {
            return _acc.balance;
        } else {
            return IERC20(_tokenAddr).balanceOf(_acc);
        }
    }

    function getTokenDecimals(address _token) internal view returns (uint256) {
        if (_token == ETH_ADDR) return 18;

        return IERC20(_token).decimals();
    }
}


























struct Position {
    uint8 protocol;
    address proxy;
    address collateralToken;
    address debtToken;
    uint256 collateral;
    uint256 debt;
}

contract LSVView is ActionsUtilHelper, UtilHelper, AaveV3Helper, MorphoAaveV3Helper, CompV3Helper, SparkHelper, MorphoBlueHelper, LSVUtilHelper {
    enum Protocol {
        AAVE_V3,
        MORPHO_AAVE_V3,
        COMPOUND_V3,
        SPARK,
        MORPHO_BLUE_WSTETH_MARKET_RATE,
        MORPHO_BLUE_WSTETH_LIDO_RATE_945,
        MORPHO_BLUE_WSTETH_LIDO_RATE_965,
        AAVE_V3_LIDO_INSTANCE,
        MORPHO_BLUE_WEETH_RATE_945,
        MORPHO_BLUE_WEETH_RATE_86,
        MORPHO_BLUE_EZETH_945,
        MORPHO_BLUE_EZETH_86
    }

    uint256 public constant NUMBER_OF_SUPPORTED_PROTOCOLS = 12;
    
    using TokenUtils for address;

    function getAllPositionForLSVUser(
        address _user,
        address[] memory _collTokens
    ) public view returns (address[] memory proxies, Position[] memory positions) {
        proxies = LSVProxyRegistry(LSV_PROXY_REGISTRY_ADDRESS).getProxies(_user);
        positions = _getPositions(_collTokens, proxies);
    }

    function getAllPositionForDFSUser(
        address _user,
        address[] memory _collTokens
    ) public view returns (address[] memory proxies, Position[] memory positions) {
        (address mcdProxy, address[] memory additionalProxies) = DFSProxyRegistry(
            DFS_PROXY_REGISTRY_ADDR
        ).getAllProxies(_user);

        if (mcdProxy == address(0)) {
            proxies = new address[](additionalProxies.length);
            for (uint256 i = 0; i < proxies.length; i++) {
                proxies[i] = additionalProxies[i];
            }
        } else {
            proxies = new address[](additionalProxies.length + 1);
            uint256 i;
            for (i; i < proxies.length - 1; i++) {
                proxies[i] = additionalProxies[i];
            }
            proxies[i] = mcdProxy;
        }
        positions = _getPositions(_collTokens, proxies);
    }

    function getAllPositionForEOA(
        address _user,
        address[] memory _collTokens
    ) public view returns (address[] memory proxies, Position[] memory positions) {
        address[] memory users = new address[](1);
        users[0] = _user;

        positions = _getPositions(_collTokens, users);

        proxies = new address[](positions.length);

        for (uint256 i = 0; i < positions.length; i++) {
            proxies[i] = positions[i].proxy;
        }
    }

    function _getPositions(address[] memory _collTokens, address[] memory _users) internal view returns (Position[] memory positions) {
        Position[] memory tempPositions = new Position[](_users.length * NUMBER_OF_SUPPORTED_PROTOCOLS);
        uint256 positionCounter;

        for (uint256 i = 0; i < _users.length; i++) {
            positionCounter = _getAaveV3Positions(_collTokens, _users[i], tempPositions, positionCounter);
            positionCounter = _getMorphoAavePositions(_collTokens, _users[i], tempPositions, positionCounter);
            positionCounter = _getCompoundV3Positions(_collTokens, _users[i], tempPositions, positionCounter);
            positionCounter = _getSparkPositions(_collTokens, _users[i], tempPositions, positionCounter);
            positionCounter = _getMorphoBluePositions(_users[i], tempPositions, positionCounter);
            positionCounter = _getAaveLidoPositions(_collTokens, _users[i], tempPositions, positionCounter);
        }
        
        positions = new Position[](positionCounter);
        for (uint256 i = 0; i < positionCounter; i++) {
            positions[i] = tempPositions[i];
        }
    }

    function _getAaveV3Positions(
        address[] memory _collTokens,
        address _user,
        Position[] memory _positions,
        uint256 _counter
    ) internal view returns (uint256 counter) {
        counter = _counter;
        IPoolV3 lendingPool = getLendingPool(DEFAULT_AAVE_MARKET);
        DataTypes.ReserveData memory wethReserveData = lendingPool.getReserveData(
            TokenUtils.WETH_ADDR
        );
        for (uint256 j = 0; j < _collTokens.length; j++) {
            DataTypes.ReserveData memory reserveData = lendingPool.getReserveData(
                _collTokens[j]
            );
            if (reserveData.aTokenAddress != address(0)) {
                uint256 collBalance = reserveData.aTokenAddress.getBalance(_user);
                if (collBalance > 0) {
                    uint256 debtBalance = wethReserveData.variableDebtTokenAddress.getBalance(_user);
                    _positions[counter++] = Position(
                        uint8(Protocol.AAVE_V3),
                        _user,
                        _collTokens[j],
                        TokenUtils.WETH_ADDR,
                        collBalance,
                        debtBalance
                    );
                    j = _collTokens.length;
                }
            }
        }
    }

    function _getMorphoAavePositions(
        address[] memory _collTokens,
        address _user,
        Position[] memory _positions,
        uint256 _counter
    ) internal view returns (uint256 counter) {
        counter = _counter;
        address morphoAddr = getMorphoAddressByEmode(1);
        for (uint256 j = 0; j < _collTokens.length; j++) {
            uint256 collBalance = IMorphoAaveV3(morphoAddr).collateralBalance(
                _collTokens[j],
                _user
            );
            if (collBalance > 0) {
                uint256 debtBalance = IMorphoAaveV3(morphoAddr).borrowBalance(
                    TokenUtils.WETH_ADDR,
                    _user
                );
                _positions[counter++] = Position(
                    uint8(Protocol.MORPHO_AAVE_V3),
                    _user,
                    _collTokens[j],
                    TokenUtils.WETH_ADDR,
                    collBalance,
                    debtBalance
                );
                j = _collTokens.length;
            }
        }
    }

    function _getCompoundV3Positions(
        address[] memory _collTokens,
        address _user,
        Position[] memory _positions,
        uint256 _counter
    ) internal view returns (uint256 counter) {
        counter = _counter;
        IComet comet = IComet(COMP_ETH_COMET);
        for (uint256 j = 0; j < _collTokens.length; j++) {
            uint256 collBalance = comet.collateralBalanceOf(_user, _collTokens[j]);
            if (collBalance > 0) {
                uint256 debtBalance = comet.borrowBalanceOf(_user);
                _positions[counter++] = Position(
                    uint8(Protocol.COMPOUND_V3),
                    _user,
                    _collTokens[j],
                    TokenUtils.WETH_ADDR,
                    collBalance,
                    debtBalance
                );
                j = _collTokens.length;
            }
        }
    }

    function _getMorphoBluePositions(address _user, Position[] memory _positions, uint256 _counter) internal view returns (uint256 counter) {
        counter = _counter;

        uint256 marketSize = 7;

        MarketParams[] memory markets = new MarketParams[](marketSize);
        markets[0] = _morphoBlueWstEthMarketRate();
        markets[1] = _morphoBlueWstEthLidoRate945();
        markets[2] = _morphoBlueWstEthLidoRate965();
        markets[3] = _morphoBlueWeEth945();
        markets[4] = _morphoBlueWeEth86();
        markets[5] = _morphoBlueEzEth945();
        markets[6] = _morphoBlueEzEth86();

        Protocol[] memory protocols = new Protocol[](marketSize);
        protocols[0] = Protocol.MORPHO_BLUE_WSTETH_MARKET_RATE;
        protocols[1] = Protocol.MORPHO_BLUE_WSTETH_LIDO_RATE_945;
        protocols[2] = Protocol.MORPHO_BLUE_WSTETH_LIDO_RATE_965;
        protocols[3] = Protocol.MORPHO_BLUE_WEETH_RATE_945;
        protocols[4] = Protocol.MORPHO_BLUE_WEETH_RATE_86;
        protocols[5] = Protocol.MORPHO_BLUE_EZETH_945;
        protocols[6] = Protocol.MORPHO_BLUE_EZETH_86;

        for (uint256 i = 0; i < marketSize; i++) {
            Position memory position = _getMorphoBlueWethPosition(markets[i], _user, protocols[i]);
            if (position.proxy != address(0)) _positions[counter++] = position;
        }
    }

    function _getMorphoBlueWethPosition(
        MarketParams memory _marketParams,
        address _user,
        Protocol _protocol
    ) internal view returns (Position memory position) {
        Id marketId = MarketParamsLib.id(_marketParams);
        if (MorphoLib.collateral(morphoBlue, marketId, _user) > 0){
            position = Position(
                uint8(_protocol),
                _user,
                _marketParams.collateralToken,
                TokenUtils.WETH_ADDR,
                MorphoLib.collateral(morphoBlue, marketId, _user),
                MorphoBalancesLib.expectedBorrowAssets(morphoBlue, _marketParams, _user)
            );
        }
    }

    function _getSparkPositions(
        address[] memory _collTokens,
        address _user,
        Position[] memory _positions,
        uint256 _counter
    ) internal view returns (uint256 counter) {
        counter = _counter;
        IPoolV3 lendingPool = getLendingPool(DEFAULT_SPARK_MARKET);
        DataTypes.ReserveData memory wethReserveData = lendingPool.getReserveData(
            TokenUtils.WETH_ADDR
        );
        for (uint256 j = 0; j < _collTokens.length; j++) {
            DataTypes.ReserveData memory reserveData = lendingPool.getReserveData(
                _collTokens[j]
            );
            if (reserveData.aTokenAddress != address(0)) {
                uint256 collBalance = reserveData.aTokenAddress.getBalance(_user);
                if (collBalance > 0) {
                    uint256 debtBalance = wethReserveData.variableDebtTokenAddress.getBalance(_user);
                    _positions[counter++] = Position(
                        uint8(Protocol.SPARK),
                        _user,
                        _collTokens[j],
                        TokenUtils.WETH_ADDR,
                        collBalance,
                        debtBalance
                    );
                    j = _collTokens.length;
                }
            }
        }
    }

    function _getAaveLidoPositions(
        address[] memory _collTokens,
        address _user, 
        Position[] memory _positions,
        uint256 _counter
    ) internal view returns (uint256 counter) {
        counter = _counter;
        IPoolV3 lendingPool = getLendingPool(0xcfBf336fe147D643B9Cb705648500e101504B16d);
        DataTypes.ReserveData memory wethReserveData = lendingPool.getReserveData(
            TokenUtils.WETH_ADDR
        );
        for (uint256 j = 0; j < _collTokens.length; j++) {
            DataTypes.ReserveData memory reserveData = lendingPool.getReserveData(
                _collTokens[j]
            );
            if (reserveData.aTokenAddress != address(0)) {
                uint256 collBalance = reserveData.aTokenAddress.getBalance(_user);
                if (collBalance > 0) {
                    uint256 debtBalance = wethReserveData.variableDebtTokenAddress.getBalance(_user);
                    _positions[counter++] = Position(
                        uint8(Protocol.AAVE_V3_LIDO_INSTANCE),
                        _user,
                        _collTokens[j],
                        TokenUtils.WETH_ADDR,
                        collBalance,
                        debtBalance
                    );
                    j = _collTokens.length;
                }
            }
        }
    }

    function getInfoForLSVPosition(uint8 _protocol, address _lsvProxy, address[] memory _collTokens) public view returns (uint256 netWorth, int256 unrealisedProfit) {
        unrealisedProfit = LSVProfitTracker(LSV_PROFIT_TRACKER_ADDRESS).unrealisedProfit(_protocol, _lsvProxy);
        (uint256 collBalance, uint256 ethDebtBalance, address collToken) = findCollAndDebtBalance(_protocol, _lsvProxy, _collTokens);
        uint256 collBalanceInETH = getAmountInETHFromLST(collToken, collBalance);
        if (collBalanceInETH >= ethDebtBalance){
            netWorth = collBalanceInETH  - ethDebtBalance;
        } else {
            return (0,0);
        }
    }
    
    function findCollAndDebtBalance(uint8 protocol, address _user, address[] memory _collTokens) public view returns (uint256, uint256, address){
        if (protocol == uint8(Protocol.AAVE_V3)) return findCollAndDebtForAaveV3Position(_user, _collTokens);
        if (protocol == uint8(Protocol.MORPHO_AAVE_V3)) return findCollAndDebtForMorphoAaveV3Position(_user, _collTokens);
        if (protocol == uint8(Protocol.COMPOUND_V3)) return findCollAndDebtForCompV3Position(_user, _collTokens);
        if (protocol == uint8(Protocol.SPARK)) return findCollAndDebtForSparkPosition(_user, _collTokens);
        if (protocol == uint8(Protocol.MORPHO_BLUE_WSTETH_MARKET_RATE)) return findCollAndDebtForMorphoBlueWstethPosition(_user, Protocol(protocol));
        if (protocol == uint8(Protocol.MORPHO_BLUE_WSTETH_LIDO_RATE_945)) return findCollAndDebtForMorphoBlueWstethPosition(_user, Protocol(protocol));
        if (protocol == uint8(Protocol.MORPHO_BLUE_WSTETH_LIDO_RATE_965)) return findCollAndDebtForMorphoBlueWstethPosition(_user, Protocol(protocol));
        if (protocol == uint8(Protocol.AAVE_V3_LIDO_INSTANCE)) return findCollAndDebtForAaveV3LidoPosition(_user, _collTokens);
        if (protocol == uint8(Protocol.MORPHO_BLUE_WEETH_RATE_945)) return findCollAndDebtForMorphoBlueWstethPosition(_user, Protocol(protocol));
        if (protocol == uint8(Protocol.MORPHO_BLUE_WEETH_RATE_86)) return findCollAndDebtForMorphoBlueWstethPosition(_user, Protocol(protocol));
        if (protocol == uint8(Protocol.MORPHO_BLUE_EZETH_945)) return findCollAndDebtForMorphoBlueWstethPosition(_user, Protocol(protocol));
        if (protocol == uint8(Protocol.MORPHO_BLUE_EZETH_86)) return findCollAndDebtForMorphoBlueWstethPosition(_user, Protocol(protocol));
        
        return (0, 0, address(0));
    }

    /// @dev we assume it only has one LST token as collateral, and only ETH as debt
    function findCollAndDebtForAaveV3Position(
        address _user,
        address[] memory _collTokens
    ) public view returns (uint256 collAmount, uint256 debtAmount, address collToken) 
        {
        IPoolV3 lendingPool = getLendingPool(DEFAULT_AAVE_MARKET);
        DataTypes.ReserveData memory wethReserveData = lendingPool.getReserveData(
            TokenUtils.WETH_ADDR
        );
        uint256 ethDebtAmount = wethReserveData.variableDebtTokenAddress.getBalance(_user);
        for (uint256 j = 0; j < _collTokens.length; j++) {
            DataTypes.ReserveData memory reserveData = lendingPool.getReserveData(
                _collTokens[j]
            );
            if (reserveData.aTokenAddress != address(0)) {
                uint256 lstCollAmount = reserveData.aTokenAddress.getBalance(_user);
                if (lstCollAmount > 0) {
                    collAmount = lstCollAmount;
                    debtAmount = ethDebtAmount;
                    collToken = _collTokens[j];
                }
            }
        }
    }

    /// @dev we assume it only has one LST token as collateral, and only ETH as debt
    function findCollAndDebtForAaveV3LidoPosition(
        address _user,
        address[] memory _collTokens
    ) public view returns (uint256 collAmount, uint256 debtAmount, address collToken) 
        {
        IPoolV3 lendingPool = getLendingPool(0xcfBf336fe147D643B9Cb705648500e101504B16d);
        DataTypes.ReserveData memory wethReserveData = lendingPool.getReserveData(
            TokenUtils.WETH_ADDR
        );
        uint256 ethDebtAmount = wethReserveData.variableDebtTokenAddress.getBalance(_user);
        for (uint256 j = 0; j < _collTokens.length; j++) {
            DataTypes.ReserveData memory reserveData = lendingPool.getReserveData(
                _collTokens[j]
            );
            if (reserveData.aTokenAddress != address(0)) {
                uint256 lstCollAmount = reserveData.aTokenAddress.getBalance(_user);
                if (lstCollAmount > 0) {
                    collAmount = lstCollAmount;
                    debtAmount = ethDebtAmount;
                    collToken = _collTokens[j];
                }
            }
        }
    }

    /// @dev we assume it only has one LST token as collateral, and only ETH as debt
    function findCollAndDebtForSparkPosition(
        address _user,
        address[] memory _collTokens
    ) public view returns (uint256 collAmount, uint256 debtAmount, address collToken) {
        IPoolV3 lendingPool = getLendingPool(DEFAULT_SPARK_MARKET);
        DataTypes.ReserveData memory wethReserveData = lendingPool.getReserveData(
            TokenUtils.WETH_ADDR
        );
        uint256 ethDebtAmount = wethReserveData.variableDebtTokenAddress.getBalance(_user);
        for (uint256 j = 0; j < _collTokens.length; j++) {
            DataTypes.ReserveData memory reserveData = lendingPool.getReserveData(
                _collTokens[j]
            );
            if (reserveData.aTokenAddress != address(0)) {
                uint256 lstCollAmount = reserveData.aTokenAddress.getBalance(_user);
                if (lstCollAmount > 0) {
                    collAmount = lstCollAmount;
                    debtAmount = ethDebtAmount;
                    collToken = _collTokens[j];
                }
            }
        }
    }

    /// @dev we assume it only has one LST token as collateral, and only ETH as debt
    function findCollAndDebtForMorphoAaveV3Position(
        address _user,
        address[] memory _collTokens
    ) public view returns (uint256 collAmount, uint256 debtAmount, address collToken) {
        address morphoAddr = getMorphoAddressByEmode(1);
        uint256 debtBalance = IMorphoAaveV3(morphoAddr).borrowBalance(
            TokenUtils.WETH_ADDR,
            _user
        );

        for (uint256 j = 0; j < _collTokens.length; j++) {
            uint256 collBalance = IMorphoAaveV3(morphoAddr).collateralBalance(
                _collTokens[j],
                _user
            );
            if (collBalance > 0) {
                collAmount = collBalance;
                debtAmount = debtBalance;
                collToken = _collTokens[j];
            }
        }
    }

    /// @dev we assume it only has one LST token as collateral, and only ETH as debt
    function findCollAndDebtForCompV3Position(
        address _user,
        address[] memory _collTokens
    ) public view returns (uint256 collAmount, uint256 debtAmount, address collToken) {
        IComet comet = IComet(COMP_ETH_COMET);

        uint256 debtBalance = comet.borrowBalanceOf(_user);
        for (uint256 j = 0; j < _collTokens.length; j++) {
            uint256 collBalance = comet.collateralBalanceOf(_user, _collTokens[j]);
            if (collBalance > 0) {
                collAmount = collBalance;
                debtAmount = debtBalance;
                collToken = _collTokens[j];
            }
        }
    }
    /// @dev we assume it only has one LST token as collateral, and only ETH as debt
    function findCollAndDebtForMorphoBlueWstethPosition(address _user, Protocol protocol) public view returns (uint256, uint256, address) {
        MarketParams memory marketParams;
        if (protocol == Protocol.MORPHO_BLUE_WSTETH_MARKET_RATE) marketParams = _morphoBlueWstEthMarketRate();
        else if (protocol == Protocol.MORPHO_BLUE_WSTETH_LIDO_RATE_945) marketParams = _morphoBlueWstEthLidoRate945();
        else if (protocol == Protocol.MORPHO_BLUE_WSTETH_LIDO_RATE_965) marketParams = _morphoBlueWstEthLidoRate965();
        else if (protocol == Protocol.MORPHO_BLUE_WEETH_RATE_945) marketParams = _morphoBlueWeEth945();
        else if (protocol == Protocol.MORPHO_BLUE_WEETH_RATE_86) marketParams = _morphoBlueWeEth86();
        else if (protocol == Protocol.MORPHO_BLUE_EZETH_945) marketParams = _morphoBlueEzEth945();
        else if (protocol == Protocol.MORPHO_BLUE_EZETH_86) marketParams = _morphoBlueEzEth86();

        Id marketId = MarketParamsLib.id(marketParams);
        uint256 collBalance = MorphoLib.collateral(morphoBlue, marketId, _user);
        uint256 debtBalance = MorphoBalancesLib.expectedBorrowAssets(morphoBlue, marketParams, _user);
        return (collBalance, debtBalance, marketParams.collateralToken);
    }

    function _morphoBlueWstEthMarketRate() internal pure returns (MarketParams memory marketParams) {
        marketParams = MarketParams({
            loanToken: 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,
            collateralToken: 0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0,
            oracle: 0x2a01EB9496094dA03c4E364Def50f5aD1280AD72,
            irm: 0x870aC11D48B15DB9a138Cf899d20F13F79Ba00BC,
            lltv: 945000000000000000
        });
    }
    function _morphoBlueWstEthLidoRate945() internal pure returns (MarketParams memory marketParams) {
        marketParams = MarketParams({
            loanToken: 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,
            collateralToken: 0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0,
            oracle: 0xbD60A6770b27E084E8617335ddE769241B0e71D8,
            irm: 0x870aC11D48B15DB9a138Cf899d20F13F79Ba00BC,
            lltv: 945000000000000000
        });
    }
    function _morphoBlueWstEthLidoRate965() internal pure returns (MarketParams memory marketParams) {
        marketParams = MarketParams({
            loanToken: 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,
            collateralToken: 0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0,
            oracle: 0xbD60A6770b27E084E8617335ddE769241B0e71D8,
            irm: 0x870aC11D48B15DB9a138Cf899d20F13F79Ba00BC,
            lltv: 965000000000000000
        });
    }
     function _morphoBlueWeEth945() internal pure returns (MarketParams memory marketParams) {
        marketParams = MarketParams({
            loanToken: 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,
            collateralToken: 0xCd5fE23C85820F7B72D0926FC9b05b43E359b7ee,
            oracle: 0xbDd2F2D473E8D63d1BFb0185B5bDB8046ca48a72,
            irm: 0x870aC11D48B15DB9a138Cf899d20F13F79Ba00BC,
            lltv: 945000000000000000
        });
    }
    function _morphoBlueWeEth86() internal pure returns (MarketParams memory marketParams) {
        marketParams = MarketParams({
            loanToken: 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,
            collateralToken: 0xCd5fE23C85820F7B72D0926FC9b05b43E359b7ee,
            oracle: 0x3fa58b74e9a8eA8768eb33c8453e9C2Ed089A40a,
            irm: 0x870aC11D48B15DB9a138Cf899d20F13F79Ba00BC,
            lltv: 860000000000000000
        });
    }
    function _morphoBlueEzEth945() internal pure returns (MarketParams memory marketParams) {
        marketParams = MarketParams({
            loanToken: 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,
            collateralToken: 0xbf5495Efe5DB9ce00f80364C8B423567e58d2110,
            oracle: 0x94f93f1eADb8a2f73C415AD4C19cB791e6D0192b,
            irm: 0x870aC11D48B15DB9a138Cf899d20F13F79Ba00BC,
            lltv: 945000000000000000
        });
    }
    function _morphoBlueEzEth86() internal pure returns (MarketParams memory marketParams) {
        marketParams = MarketParams({
            loanToken: 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,
            collateralToken: 0xbf5495Efe5DB9ce00f80364C8B423567e58d2110,
            oracle: 0x61025e2B0122ac8bE4e37365A4003d87ad888Cc3,
            irm: 0x870aC11D48B15DB9a138Cf899d20F13F79Ba00BC,
            lltv: 860000000000000000
        });
    }

    /// @notice Returns the lending pool contract of the specified market
    function getLendingPool(address _market) override(AaveV3Helper, SparkHelper) internal view returns (IL2PoolV3) {
        return IL2PoolV3(IPoolAddressesProvider(_market).getPool());
    }

    /// @notice Fetch the data provider for the specified market
    function getDataProvider(address _market) override(AaveV3Helper, SparkHelper) internal view returns (IAaveProtocolDataProvider) {
        return
            IAaveProtocolDataProvider(
                IPoolAddressesProvider(_market).getPoolDataProvider()
            );
    }

    function boolToBytes(bool x) override(AaveV3Helper, SparkHelper) internal pure returns (bytes1 r) {
       return x ? bytes1(0x01) : bytes1(0x00);
    }

    function bytesToBool(bytes1 x) override(AaveV3Helper, SparkHelper) internal pure returns (bool r) {
        return x != bytes1(0x00);
    }
    
    function getWholeDebt(address _market, address _tokenAddr, uint256 _borrowType, address _debtOwner) override(AaveV3Helper, SparkHelper) internal view returns (uint256 debt) {
        uint256 STABLE_ID = 1;
        uint256 VARIABLE_ID = 2;

        IAaveProtocolDataProvider dataProvider = getDataProvider(_market);
        (, uint256 borrowsStable, uint256 borrowsVariable, , , , , , ) =
            dataProvider.getUserReserveData(_tokenAddr, _debtOwner);

        if (_borrowType == STABLE_ID) {
            debt = borrowsStable;
        } else if (_borrowType == VARIABLE_ID) {
            debt = borrowsVariable;
        }
    }
}
