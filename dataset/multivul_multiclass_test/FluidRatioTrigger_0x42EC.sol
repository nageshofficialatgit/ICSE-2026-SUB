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






abstract contract ITrigger {
    function isTriggered(bytes memory, bytes memory) public virtual returns (bool);
    function isChangeable() public virtual returns (bool);
    function changedSubData(bytes memory) public virtual returns (bytes memory);
}







contract MainnetTriggerAddresses {
    address public constant UNISWAP_V3_NONFUNGIBLE_POSITION_MANAGER = 0xC36442b4a4522E871399CD717aBDD847Ab11FE88;
    address public constant UNISWAP_V3_FACTORY = 0x1F98431c8aD98523631AE4a59f267346ea31F984;
    address public constant MCD_PRICE_VERIFIER = 0xeAa474cbFFA87Ae0F1a6f68a3aBA6C77C656F72c;
    address public constant TRANSIENT_STORAGE = 0x2F7Ef2ea5E8c97B8687CA703A0e50Aa5a49B7eb2;
}







contract TriggerHelper is MainnetTriggerAddresses {
}








contract TransientStorage {
    mapping (string => bytes32) public tempStore;

    /// @notice Set a bytes32 value by a string key
    /// @dev Anyone can add a value because it's only used per tx
    function setBytes32(string memory _key, bytes32 _value) public {
        tempStore[_key] = _value;
    }


    /// @dev Can't enforce per tx usage so caller must be careful reading the value
    function getBytes32(string memory _key) public view returns (bytes32) {
        return tempStore[_key];
    }
}












contract FluidRatioTrigger is 
    ITrigger,
    AdminAuth,
    FluidRatioHelper,
    TriggerHelper
{
    TransientStorage public constant tempStorage = TransientStorage(TRANSIENT_STORAGE);

    enum RatioState { OVER, UNDER }

    /// @param nftId nft id of the fluid position
    /// @param ratio ratio that represents the triggerable point
    /// @param state represents if we want the current state to be higher or lower than ratio param
    struct SubParams {
        uint256 nftId;
        uint256 ratio;
        uint8 state;
    }
    /// @dev checks current ratio of a fluid position and triggers if it's in a correct state
    function isTriggered(bytes memory, bytes memory _subData)
        public
        override
        returns (bool)
    {   
        SubParams memory triggerSubData = parseSubInputs(_subData);

        uint256 currRatio = getRatio(triggerSubData.nftId);
        
        if (currRatio == 0) return false;

        tempStorage.setBytes32("FLUID_RATIO", bytes32(currRatio));
        
        if (RatioState(triggerSubData.state) == RatioState.OVER) {
            if (currRatio > triggerSubData.ratio) return true;
        }

        if (RatioState(triggerSubData.state) == RatioState.UNDER) {
            if (currRatio < triggerSubData.ratio) return true;
        }

        return false;
    }

    function parseSubInputs(bytes memory _subData) public pure returns (SubParams memory params) {
        params = abi.decode(_subData, (SubParams));
    }

    function changedSubData(bytes memory _subData) public pure override  returns (bytes memory) {}
    
    function isChangeable() public pure override returns (bool) {
        return false;
    }
}
