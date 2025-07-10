// SPDX-License-Identifier: MIT
pragma solidity =0.8.24;









contract MainnetFluidAddresses {
    address internal constant FLUID_VAULT_RESOLVER = 0x814c8C7ceb1411B364c2940c4b9380e739e06686;
    address internal constant FLUID_LENDING_RESOLVER = 0xC215485C572365AE87f908ad35233EC2572A3BEC;
    address internal constant F_WETH_TOKEN_ADDR = 0x90551c1795392094FE6D29B758EcCD233cFAa260;
    address internal constant ETH_ADDR = 0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE;
}









contract DFSMath {

    /// @notice Converts an unsigned 256-bit integer to a signed 256-bit integer.
    /// @dev Reverts if the input value exceeds the maximum value of int256.
    /// @param x The unsigned integer to convert.
    /// @return The signed integer representation of `x`.
    function signed256(uint256 x) internal pure returns (int256) {
        require(x <= uint256(type(int256).max));
        return int256(x);
    }
}








contract FluidHelper is DFSMath, MainnetFluidAddresses {
}







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











contract FluidRatioHelper is FluidHelper {

    uint256 internal constant T1_VAULT_TYPE = 1e4;
    uint256 internal constant T2_VAULT_TYPE = 2e4;
    uint256 internal constant T3_VAULT_TYPE = 3e4;
    uint256 internal constant T4_VAULT_TYPE = 4e4;

    uint256 internal constant ORACLE_PRICE_DECIMALS = 27;
    uint256 internal constant ETH_DECIMALS = 18;
    uint256 internal constant WAD = 1e18;

    /// @notice Gets ratio for a fluid position
    /// @param _nftId nft id of the fluid position
    function getRatio(uint256 _nftId) public view returns (uint256 ratio) {
        (
            IFluidVaultResolver.UserPosition memory userPosition,
            IFluidVaultResolver.VaultEntireData memory vaultData
        ) = IFluidVaultResolver(FLUID_VAULT_RESOLVER).positionByNftId(_nftId);

        // For now, only handle the case for T1 Vaults
        if (vaultData.constantVariables.vaultType == T1_VAULT_TYPE) {
            uint256 collAmount = userPosition.supply;
            address collToken = vaultData.constantVariables.supplyToken.token0;

            uint256 debtAmount = userPosition.borrow;
            address debtToken = vaultData.constantVariables.borrowToken.token0;

            if (debtAmount == 0) return uint256(0);

            uint256 collDec = collToken != TokenUtils.ETH_ADDR ? IERC20(collToken).decimals() : ETH_DECIMALS;
            uint256 debtDec = debtToken != TokenUtils.ETH_ADDR ? IERC20(debtToken).decimals() : ETH_DECIMALS;

            /**
            * @dev Examples:
            *
            * 1. (2.5 WBTC / 50k USDC)
            *    price = 1028534478997854690000000000000
            *    priceScaler = 10 ** (27 - 8 + 6) = 1e25
            *    collAmount = 2.5 * 1e8
            *    debtAmount = 50000 * 1e6
            *    collAmountInDebtToken = ((2.5 * 1e8 * 1028534478997854690000000000000) / 1e25) * 1e6 / 1e8 = 257133619749
            *    ratio = 257133619749 * 1e18 / (50000 * 1e6) = 5.14267239498e18 = 514.267239498 %
            *
            * 2. (3.2 weETH / 1.5 wstETH)
            *    price = 888143936867381715436793889
            *    priceScaler = 10 ** (27 - 18 + 18) = 1e27
            *    collAmount = 3.2 * 1e18
            *    debtAmount = 1.5 * 1e18
            *    collAmountInDebtToken = ((3.2 * 1e18 * 888143936867381715436793889) / 1e27) * 1e18 / 1e18 = 2.8420605979756216e18
            *    ratio = 2.8420605979756216e18 * 1e18 / (1.5 * 1e18) = 1.894707065317081e+18 = 189.47070653170812 %
            *
            * 3. (2 WBTC / 30 ETH)
            *    price =  321857633689348920539866335783307690682 / 10 ** (27 - 8 + 18) = 32.18576336893489
            *    priceScaler = 10 ** (27 - 8 + 18) = 1e37
            *    collAmount = 2 * 1e8
            *    debtAmount = 30 * 1e18
            *    collAmountInDebtToken = ((2 * 1e8 * 321857633689348920539866335783307690682) / 1e37) * 1e18 / 1e8 = 6.437152673786979e19
            *    ratio = 6.437152673786979e19 * 1e18 / (30 * 1e18) = 2.145717557928993e18 = 214.5717557928993 %
            */
            uint256 price = vaultData.configs.oraclePriceOperate;
            uint256 priceScaler = 10 ** (ORACLE_PRICE_DECIMALS - collDec + debtDec);
            uint256 collAmountInDebtToken = ((collAmount * price) / priceScaler) * (10 ** debtDec) / (10 ** collDec);

            ratio = collAmountInDebtToken * WAD / debtAmount;
        }
    }
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











contract FluidView is FluidRatioHelper {

    struct UserPosition {
        uint256 nftId;
        address owner;
        bool isLiquidated;
        bool isSupplyPosition;
        uint256 supply;
        uint256 borrow;
        uint256 ratio;
        int256 tick;
        uint256 tickId;
    }

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
        uint16 borrowFee;
        address oracle; // address of the oracle
        uint256 oraclePriceOperate;
        uint256 oraclePriceLiquidate;
        uint256 vaultSupplyExchangePrice;
        uint256 vaultBorrowExchangePrice;
        int256 supplyRateVault;
        int256 borrowRateVault;
        int256 rewardsOrFeeRateSupply;
        int256 rewardsOrFeeRateBorrow;
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
    }
    
    struct NftWithVault {
        uint256 nftId;
        uint256 vaultId;
        address vaultAddr;
    }

    struct UserEarnPosition {
        uint256 fTokenShares;
        uint256 underlyingAssets;
        uint256 underlyingBalance;
        uint256 allowance;
    }

    struct FTokenData {
        address tokenAddress;
        bool isNativeUnderlying;
        string name;
        string symbol;
        uint256 decimals;
        address asset;
        uint256 totalAssets;
        uint256 totalSupply;
        uint256 convertToShares;
        uint256 convertToAssets;
        uint256 rewardsRate; // additional yield from rewards, if active
        uint256 supplyRate; // yield at Liquidity
        uint256 withdrawable; // actual currently withdrawable amount (supply - withdrawal Limit) & considering balance
        bool modeWithInterest; // true if mode = with interest, false = without interest
        uint256 expandPercent; // withdrawal limit expand percent in 1e2
        uint256 expandDuration; // withdrawal limit expand duration in seconds
    }

    /// @notice Get all user positions with vault data
    function getUserPositions(address _user) 
        external view returns (UserPosition[] memory positions, VaultData[] memory vaults) 
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

    /// @notice Get position data with vault data for a specific nftId
    function getPositionByNftId(uint256 _nftId) public view returns (UserPosition memory position, VaultData memory vault) {
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

    /// @notice Get vault data for a specific vault address
    function getVaultData(address _vault) public view returns (VaultData memory vaultData) {
        IFluidVaultResolver.VaultEntireData memory data = 
            IFluidVaultResolver(FLUID_VAULT_RESOLVER).getVaultEntireData(_vault);

        address supplyToken0 = data.constantVariables.supplyToken.token0;
        address supplyToken1 = data.constantVariables.supplyToken.token1;
        address borrowToken0 = data.constantVariables.borrowToken.token0;
        address borrowToken1 = data.constantVariables.borrowToken.token1;

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

            minimumBorrowing: data.limitsAndAvailability.minimumBorrowing
        });
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
