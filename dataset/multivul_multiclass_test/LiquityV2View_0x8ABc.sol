// SPDX-License-Identifier: MIT
pragma solidity =0.8.24;









contract MainnetLiquityV2Addresses {
    address internal constant BOLD_ADDR = 0xb01dd87B29d187F3E3a4Bf6cdAebfb97F3D9aB98;
    address internal constant MULTI_TROVE_GETTER_ADDR = 0xA4a99F8332527A799AC46F616942dBD0d270fC41;
    address internal constant WETH_MARKET_ADDR = 0x38e1F07b954cFaB7239D7acab49997FBaAD96476;
    address internal constant WSTETH_MARKET_ADDR = 0x2D4ef56cb626E9a4C90c156018BA9CE269573c61;
    address internal constant RETH_MARKET_ADDR = 0x3b48169809DD827F22C9e0F2d71ff12Ea7A94a2F;
}







contract LiquityV2Helper is MainnetLiquityV2Addresses {

    // Amount of ETH to be locked in gas pool on opening troves
    uint256 constant ETH_GAS_COMPENSATION = 0.0375 ether;

    // Minimum amount of net Bold debt a trove must have
    uint256 constant MIN_DEBT = 2000e18;

    // collateral indexes for different branches (markets)
    uint256 constant WETH_COLL_INDEX = 0;
    uint256 constant WSTETH_COLL_INDEX = 1;
    uint256 constant RETH_COLL_INDEX = 2;
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







interface IAddressesRegistry {
    function CCR() external view returns (uint256);
    function SCR() external view returns (uint256);
    function MCR() external view returns (uint256);
    function LIQUIDATION_PENALTY_SP() external view returns (uint256);
    function LIQUIDATION_PENALTY_REDISTRIBUTION() external view returns (uint256);
    function WETH() external view returns (address);
    function troveNFT() external view returns (address);
    function collToken() external view returns (address);
    function boldToken() external view returns (address);
    function borrowerOperations() external view returns (address);
    function troveManager() external view returns (address);
    function stabilityPool() external view returns (address);
    function activePool() external view returns (address);
    function defaultPool() external view returns (address);
    function sortedTroves() external view returns (address);
    function collSurplusPool() external view returns (address);
    function hintHelpers() external view returns (address);
    function priceFeed() external view returns (address);
    function gasPoolAddress() external view returns (address);
}







interface IBorrowerOperations {
    function CCR() external view returns (uint256);
    function MCR() external view returns (uint256);
    function SCR() external view returns (uint256);

    function openTrove(
        address _owner,
        uint256 _ownerIndex,
        uint256 _collAmount,
        uint256 _boldAmount,
        uint256 _upperHint,
        uint256 _lowerHint,
        uint256 _annualInterestRate,
        uint256 _maxUpfrontFee,
        address _addManager,
        address _removeManager,
        address _receiver
    ) external returns (uint256);

    struct OpenTroveAndJoinInterestBatchManagerParams {
        address owner;
        uint256 ownerIndex;
        uint256 collAmount;
        uint256 boldAmount;
        uint256 upperHint;
        uint256 lowerHint;
        address interestBatchManager;
        uint256 maxUpfrontFee;
        address addManager;
        address removeManager;
        address receiver;
    }

    function openTroveAndJoinInterestBatchManager(OpenTroveAndJoinInterestBatchManagerParams calldata _params)
        external
        returns (uint256);

    function addColl(uint256 _troveId, uint256 _ETHAmount) external;

    function withdrawColl(uint256 _troveId, uint256 _amount) external;

    function withdrawBold(uint256 _troveId, uint256 _amount, uint256 _maxUpfrontFee) external;

    function repayBold(uint256 _troveId, uint256 _amount) external;

    function closeTrove(uint256 _troveId) external;

    function adjustTrove(
        uint256 _troveId,
        uint256 _collChange,
        bool _isCollIncrease,
        uint256 _debtChange,
        bool isDebtIncrease,
        uint256 _maxUpfrontFee
    ) external;

    function adjustZombieTrove(
        uint256 _troveId,
        uint256 _collChange,
        bool _isCollIncrease,
        uint256 _boldChange,
        bool _isDebtIncrease,
        uint256 _upperHint,
        uint256 _lowerHint,
        uint256 _maxUpfrontFee
    ) external;

    function adjustTroveInterestRate(
        uint256 _troveId,
        uint256 _newAnnualInterestRate,
        uint256 _upperHint,
        uint256 _lowerHint,
        uint256 _maxUpfrontFee
    ) external;

    function applyPendingDebt(uint256 _troveId, uint256 _lowerHint, uint256 _upperHint) external;

    function onLiquidateTrove(uint256 _troveId) external;

    function claimCollateral() external;

    function hasBeenShutDown() external view returns (bool);
    function shutdown() external;
    function shutdownFromOracleFailure(address _failedOracleAddr) external;

    function checkBatchManagerExists(address _batchMananger) external view returns (bool);

    // -- individual delegation --
    struct InterestIndividualDelegate {
        address account;
        uint128 minInterestRate;
        uint128 maxInterestRate;
    }

    function getInterestIndividualDelegateOf(uint256 _troveId)
        external
        view
        returns (InterestIndividualDelegate memory);
    function setInterestIndividualDelegate(
        uint256 _troveId,
        address _delegate,
        uint128 _minInterestRate,
        uint128 _maxInterestRate,
        // only needed if trove was previously in a batch:
        uint256 _newAnnualInterestRate,
        uint256 _upperHint,
        uint256 _lowerHint,
        uint256 _maxUpfrontFee
    ) external;
    function removeInterestIndividualDelegate(uint256 _troveId) external;

    // -- batches --
    struct InterestBatchManager {
        uint128 minInterestRate;
        uint128 maxInterestRate;
        uint256 minInterestRateChangePeriod;
    }

    function registerBatchManager(
        uint128 minInterestRate,
        uint128 maxInterestRate,
        uint128 currentInterestRate,
        uint128 fee,
        uint128 minInterestRateChangePeriod
    ) external;
    function lowerBatchManagementFee(uint256 _newAnnualFee) external;
    function setBatchManagerAnnualInterestRate(
        uint128 _newAnnualInterestRate,
        uint256 _upperHint,
        uint256 _lowerHint,
        uint256 _maxUpfrontFee
    ) external;
    function interestBatchManagerOf(uint256 _troveId) external view returns (address);
    function getInterestBatchManager(address _account) external view returns (InterestBatchManager memory);
    function setInterestBatchManager(
        uint256 _troveId,
        address _newBatchManager,
        uint256 _upperHint,
        uint256 _lowerHint,
        uint256 _maxUpfrontFee
    ) external;
    function removeFromBatch(
        uint256 _troveId,
        uint256 _newAnnualInterestRate,
        uint256 _upperHint,
        uint256 _lowerHint,
        uint256 _maxUpfrontFee
    ) external;
    function switchBatchManager(
        uint256 _troveId,
        uint256 _removeUpperHint,
        uint256 _removeLowerHint,
        address _newBatchManager,
        uint256 _addUpperHint,
        uint256 _addLowerHint,
        uint256 _maxUpfrontFee
    ) external;

    function getEntireSystemColl() external view returns (uint);

    function getEntireSystemDebt() external view returns (uint);
}







interface IHintHelpers {
    function getApproxHint(uint256 _collIndex, uint256 _interestRate, uint256 _numTrials, uint256 _inputRandomSeed)
        external
        view
        returns (uint256 hintId, uint256 diff, uint256 latestRandomSeed);

    function predictOpenTroveUpfrontFee(uint256 _collIndex, uint256 _borrowedAmount, uint256 _interestRate)
        external
        view
        returns (uint256);

    function predictAdjustInterestRateUpfrontFee(uint256 _collIndex, uint256 _troveId, uint256 _newInterestRate)
        external
        view
        returns (uint256);

    function forcePredictAdjustInterestRateUpfrontFee(uint256 _collIndex, uint256 _troveId, uint256 _newInterestRate)
        external
        view
        returns (uint256);

    function predictAdjustTroveUpfrontFee(uint256 _collIndex, uint256 _troveId, uint256 _debtIncrease)
        external
        view
        returns (uint256);

    function predictAdjustBatchInterestRateUpfrontFee(
        uint256 _collIndex,
        address _batchAddress,
        uint256 _newInterestRate
    ) external view returns (uint256);

    function predictJoinBatchInterestRateUpfrontFee(uint256 _collIndex, uint256 _troveId, address _batchAddress)
        external
        view
        returns (uint256);

    function predictOpenTroveAndJoinBatchUpfrontFee(uint256 _collIndex, uint256 _borrowedAmount, address _batchAddress)
        external
        view
        returns (uint256);
}







interface IMultiTroveGetter {
    struct CombinedTroveData {
        uint256 id;
        uint256 debt;
        uint256 coll;
        uint256 stake;
        uint256 annualInterestRate;
        uint256 lastDebtUpdateTime;
        uint256 lastInterestRateAdjTime;
        address interestBatchManager;
        uint256 batchDebtShares;
        uint256 batchCollShares;
        uint256 snapshotETH;
        uint256 snapshotBoldDebt;
    }

    struct DebtPerInterestRate {
        address interestBatchManager;
        uint256 interestRate;
        uint256 debt;
    }

    function getMultipleSortedTroves(uint256 _collIndex, int256 _startIdx, uint256 _count)
        external
        view
        returns (CombinedTroveData[] memory _troves);

    function getDebtPerInterestRateAscending(uint256 _collIndex, uint256 _startId, uint256 _maxIterations)
        external
        view
        returns (DebtPerInterestRate[] memory, uint256 currId);
}







interface IPriceFeed {
    function fetchPrice() external returns (uint256, bool);
    function lastGoodPrice() external view returns (uint256);
    function setAddresses(address _borrowerOperationsAddress) external;
}







interface ISortedTroves {
    // -- Mutating functions (permissioned) --
    function insert(uint256 _id, uint256 _annualInterestRate, uint256 _prevId, uint256 _nextId) external;
    function insertIntoBatch(
        uint256 _troveId,
        address _batchId,
        uint256 _annualInterestRate,
        uint256 _prevId,
        uint256 _nextId
    ) external;

    function remove(uint256 _id) external;
    function removeFromBatch(uint256 _id) external;

    function reInsert(uint256 _id, uint256 _newAnnualInterestRate, uint256 _prevId, uint256 _nextId) external;
    function reInsertBatch(address _id, uint256 _newAnnualInterestRate, uint256 _prevId, uint256 _nextId) external;

    // -- View functions --

    function contains(uint256 _id) external view returns (bool);
    function isBatchedNode(uint256 _id) external view returns (bool);
    function isEmptyBatch(address _id) external view returns (bool);

    function isEmpty() external view returns (bool);
    function getSize() external view returns (uint256);

    function getFirst() external view returns (uint256);
    function getLast() external view returns (uint256);
    function getNext(uint256 _id) external view returns (uint256);
    function getPrev(uint256 _id) external view returns (uint256);

    function validInsertPosition(uint256 _annualInterestRate, uint256 _prevId, uint256 _nextId)
        external
        view
        returns (bool);
    function findInsertPosition(uint256 _annualInterestRate, uint256 _prevId, uint256 _nextId)
        external
        view
        returns (uint256, uint256);

    // Public state variable getters
    function size() external view returns (uint256);
    function nodes(uint256 _id) external view returns (uint256 nextId, uint256 prevId, address batchId, bool exists);
    function batches(address _id) external view returns (uint256 head, uint256 tail);
}







interface IStabilityPool {
    /*  provideToSP():
    * - Calculates depositor's Coll gain
    * - Calculates the compounded deposit
    * - Increases deposit, and takes new snapshots of accumulators P and S
    * - Sends depositor's accumulated Coll gains to depositor
    */
    function provideToSP(uint256 _amount, bool _doClaim) external;

    /*  withdrawFromSP():
    * - Calculates depositor's Coll gain
    * - Calculates the compounded deposit
    * - Sends the requested BOLD withdrawal to depositor
    * - (If _amount > userDeposit, the user withdraws all of their compounded deposit)
    * - Decreases deposit by withdrawn amount and takes new snapshots of accumulators P and S
    */
    function withdrawFromSP(uint256 _amount, bool doClaim) external;

    function claimAllCollGains() external;

    /*
     * Initial checks:
     * - Caller is TroveManager
     * ---
     * Cancels out the specified debt against the Bold contained in the Stability Pool (as far as possible)
     * and transfers the Trove's collateral from ActivePool to StabilityPool.
     * Only called by liquidation functions in the TroveManager.
     */
    function offset(uint256 _debt, uint256 _coll) external;

    function deposits(address _depositor) external view returns (uint256 initialValue);
    function stashedColl(address _depositor) external view returns (uint256);

    /*
     * Returns the total amount of Coll held by the pool, accounted in an internal variable instead of `balance`,
     * to exclude edge cases like Coll received from a self-destruct.
     */
    function getCollBalance() external view returns (uint256);

    /*
     * Returns Bold held in the pool. Changes when users deposit/withdraw, and when Trove debt is offset.
     */
    function getTotalBoldDeposits() external view returns (uint256);

    function getYieldGainsOwed() external view returns (uint256);
    function getYieldGainsPending() external view returns (uint256);

    /*
     * Calculates the Coll gain earned by the deposit since its last snapshots were taken.
     */
    function getDepositorCollGain(address _depositor) external view returns (uint256);

    /*
     * Calculates the BOLD yield gain earned by the deposit since its last snapshots were taken.
     */
    function getDepositorYieldGain(address _depositor) external view returns (uint256);

    /*
     * Calculates what `getDepositorYieldGain` will be if interest is minted now.
     */
    function getDepositorYieldGainWithPending(address _depositor) external view returns (uint256);

    /*
     * Return the user's compounded deposit.
     */
    function getCompoundedBoldDeposit(address _depositor) external view returns (uint256);

    function epochToScaleToS(uint128 _epoch, uint128 _scale) external view returns (uint256);

    function epochToScaleToB(uint128 _epoch, uint128 _scale) external view returns (uint256);

    function P() external view returns (uint256);
    function currentScale() external view returns (uint128);
    function currentEpoch() external view returns (uint128);
}







interface ITroveManager {
    enum Status {
        nonExistent,
        active,
        closedByOwner,
        closedByLiquidation,
        zombie
    }

    struct LatestTroveData {
        uint256 entireDebt;
        uint256 entireColl;
        uint256 redistBoldDebtGain;
        uint256 redistCollGain;
        uint256 accruedInterest;
        uint256 recordedDebt;
        uint256 annualInterestRate;
        uint256 weightedRecordedDebt;
        uint256 accruedBatchManagementFee;
        uint256 lastInterestRateAdjTime;
    }

    struct LatestBatchData {
        uint256 entireDebtWithoutRedistribution;
        uint256 entireCollWithoutRedistribution;
        uint256 accruedInterest;
        uint256 recordedDebt;
        uint256 annualInterestRate;
        uint256 weightedRecordedDebt;
        uint256 annualManagementFee;
        uint256 accruedManagementFee;
        uint256 weightedRecordedBatchManagementFee;
        uint256 lastDebtUpdateTime;
        uint256 lastInterestRateAdjTime;
    }


    function Troves(uint256 _id)
        external
        view
        returns (
            uint256 debt,
            uint256 coll,
            uint256 stake,
            Status status,
            uint64 arrayIndex,
            uint64 lastDebtUpdateTime,
            uint64 lastInterestRateAdjTime,
            uint256 annualInterestRate,
            address interestBatchManager,
            uint256 batchDebtShares
        );

    function shutdownTime() external view returns (uint256);
    function troveNFT() external view returns (address);
    function getLatestTroveData(uint256 _troveId) external view returns (LatestTroveData memory);
    function getCurrentICR(uint256 _troveId, uint256 _price) external view returns (uint256);
    function getTroveStatus(uint256 _troveId) external view returns (Status);
    function getTroveAnnualInterestRate(uint256 _troveId) external view returns (uint256);
    function getLatestBatchData(address _batchAddress) external view returns (LatestBatchData memory);
}







interface ITroveNFT {
    function mint(address _owner, uint256 _troveId) external;
    function burn(uint256 _troveId) external;
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function tokenURI(uint256 tokenId) external view returns (string memory);
    function ownerOf(uint256 tokenId) external view returns (address);
}
















contract LiquityV2View is LiquityV2Helper {
    using TokenUtils for address;

    error InvalidMarketAddress();

    struct TroveData {
        uint256 troveId;
        address owner;
        address collToken;
        ITroveManager.Status status;
        uint256 collAmount;
        uint256 debtAmount;
        uint256 collPrice;
        uint256 TCRatio;
        uint256 annualInterestRate;
        address interestBatchManager;
        uint256 batchDebtShares;
        uint256 lastInterestRateAdjTime;
    }

    struct MarketData {
        address market;
        uint256 CCR;
        uint256 MCR;
        uint256 SCR;
        uint256 LIQUIDATION_PENALTY_SP;
        uint256 LIQUIDATION_PENALTY_REDISTRIBUTION;
        uint256 entireSystemColl;
        uint256 entireSystemDebt;
        address collToken;
        address troveNFT;
        address borrowerOperations;
        address troveManager;
        address stabilityPool;
        address sortedTroves;
        address collSurplusPool;
        address activePool;
        address hintHelpers;
        address priceFeed;
        uint256 collPrice;
        bool isShutDown;
        uint256 boldDepositInSp;
    }

    function isShutDown(address _market) public view returns (bool) {
        ITroveManager troveManager = ITroveManager(IAddressesRegistry(_market).troveManager());
        return troveManager.shutdownTime() != 0;      
    }

    function getApproxHint(
        address _market,
        uint256 _collIndex,
        uint256 _interestRate,
        uint256 _numTrials,
        uint256 _inputRandomSeed
    )
        public
        view
        returns (
            uint256 hintId,
            uint256 diff,
            uint256 latestRandomSeed
        )
    {   
        IHintHelpers hintHelpers = IHintHelpers(IAddressesRegistry(_market).hintHelpers());

        return hintHelpers.getApproxHint(
            _collIndex,
            _interestRate,
            _numTrials,
            _inputRandomSeed
        );
    }

    function findInsertPosition(
        address _market,
        uint256 _interestRate,
        uint256 _prevId,
        uint256 _nextId
    ) public view returns (uint256 prevId, uint256 nextId) {
        ISortedTroves sortedTroves = ISortedTroves(IAddressesRegistry(_market).sortedTroves());

        return sortedTroves.findInsertPosition(
            _interestRate,
            _prevId,
            _nextId
        );
    }

    function getInsertPosition(
        address _market,
        uint256 _collIndex,
        uint256 _interestRate,
        uint256 _numTrials,
        uint256 _inputRandomSeed
    ) external view returns (uint256 prevId, uint256 nextId) {
        (uint256 hintId, , ) = getApproxHint(
            _market,
            _collIndex,
            _interestRate,
            _numTrials,
            _inputRandomSeed
        );
        return findInsertPosition(_market, _interestRate, hintId, hintId);
    }

    function getTrovePosition(
        address _market,
        uint256 _collIndex,
        uint256 _troveId,
        uint256 _numTrials,
        uint256 _inputRandomSeed
    ) external view returns (uint256 prevId, uint256 nextId) {
        ITroveManager troveManager = ITroveManager(IAddressesRegistry(_market).troveManager());
        ISortedTroves sortedTroves = ISortedTroves(IAddressesRegistry(_market).sortedTroves());
        uint256 troveInterestRate = troveManager.getTroveAnnualInterestRate(_troveId);
       
        (uint256 hintId, , ) = getApproxHint(
            _market,
            _collIndex,
            troveInterestRate,
            _numTrials,
            _inputRandomSeed
        );

        (prevId, nextId) = sortedTroves.findInsertPosition(
            troveInterestRate,
            hintId,
            hintId
        );

        if (prevId == _troveId) prevId = sortedTroves.getPrev(_troveId);
        if (nextId == _troveId) nextId = sortedTroves.getNext(_troveId);
    }

    function getTroveInfo(address _market, uint256 _troveId) external returns (TroveData memory trove) {
        ITroveManager troveManager = ITroveManager(IAddressesRegistry(_market).troveManager());
        IPriceFeed priceFeed = IPriceFeed(IAddressesRegistry(_market).priceFeed());
        ITroveManager.LatestTroveData memory latestTroveData = troveManager.getLatestTroveData(_troveId);
        ITroveNFT troveNFT = ITroveNFT(IAddressesRegistry(_market).troveNFT());

        (
            , , ,
            trove.status,
            , , , ,
            trove.interestBatchManager,
            trove.batchDebtShares
        ) = troveManager.Troves(_troveId);

        trove.troveId = _troveId;
        trove.annualInterestRate = latestTroveData.annualInterestRate;
        trove.collAmount = latestTroveData.entireColl;
        trove.debtAmount = latestTroveData.entireDebt;
        (trove.collPrice, ) = priceFeed.fetchPrice();
        trove.TCRatio = troveManager.getCurrentICR(_troveId, trove.collPrice);
        trove.collToken = IAddressesRegistry(_market).collToken();
        trove.lastInterestRateAdjTime = latestTroveData.lastInterestRateAdjTime;

        try troveNFT.ownerOf(_troveId) returns (address owner) {
            trove.owner = owner;
        } catch {
            trove.owner = address(0);
        }
    }

    /// @notice Helper struct to store troves when fetching user troves
    /// @param troveId The trove ID
    /// @param ownedByUser Whether the trove is owned by the user or not
    struct ExistingTrove {
        uint256 troveId;
        bool ownedByUser;
    }

    /// @notice Get the trove IDs for a user in a give market
    /// @param _user The user address
    /// @param _market The market address
    /// @param _startIndex The start index to search for troves (inclusive)
    /// @param _endIndex The end index to search for troves (exclusive)
    /// @return troves The trove IDs for the given range
    /// @return nextFreeTroveIndex The next free trove index if exists, or -1 if no free index found in given range
    function getUserTroves(
        address _user,
        address _market,
        uint256 _startIndex,
        uint256 _endIndex
    ) external view returns (ExistingTrove[] memory troves, int256 nextFreeTroveIndex) 
    {   
        nextFreeTroveIndex = -1; 
        IAddressesRegistry market = IAddressesRegistry(_market);
        ITroveManager troveManager = ITroveManager(market.troveManager());
        ITroveNFT troveNFT = ITroveNFT(market.troveNFT());
        
        uint256 numTroves = _endIndex - _startIndex;
        troves = new ExistingTrove[](numTroves);

        for (uint256 i = _startIndex; i < _endIndex; ++i) {
            uint256 troveId = uint256(keccak256(abi.encode(_user, i)));
            ITroveManager.Status status = troveManager.getTroveStatus(troveId);
            if (status == ITroveManager.Status.active || status == ITroveManager.Status.zombie) {
                troves[i - _startIndex] = ExistingTrove({ 
                    troveId: troveId,
                    ownedByUser: troveNFT.ownerOf(troveId) == _user 
                });
            } else if (nextFreeTroveIndex == -1) {
                nextFreeTroveIndex = int256(i);
            }
        }
    }

    function getMarketData(address _market) external returns (MarketData memory data) {
        IAddressesRegistry registry = IAddressesRegistry(_market);
        address borrowerOperations = registry.borrowerOperations();
        (uint256 collPrice, ) = IPriceFeed(registry.priceFeed()).fetchPrice();
        data = MarketData({
            market: _market,
            CCR: registry.CCR(),
            MCR: registry.MCR(),
            SCR: registry.SCR(),
            LIQUIDATION_PENALTY_SP: registry.LIQUIDATION_PENALTY_SP(),
            LIQUIDATION_PENALTY_REDISTRIBUTION: registry.LIQUIDATION_PENALTY_REDISTRIBUTION(),
            entireSystemColl: IBorrowerOperations(borrowerOperations).getEntireSystemColl(),
            entireSystemDebt: IBorrowerOperations(borrowerOperations).getEntireSystemDebt(),
            collToken: registry.collToken(),
            troveNFT: registry.troveNFT(),
            borrowerOperations: borrowerOperations,
            troveManager: registry.troveManager(),
            stabilityPool: registry.stabilityPool(),
            sortedTroves: registry.sortedTroves(),
            collSurplusPool: registry.collSurplusPool(),
            activePool: registry.activePool(),
            hintHelpers: registry.hintHelpers(),
            priceFeed: registry.priceFeed(),
            collPrice: collPrice,
            isShutDown: isShutDown(_market),
            boldDepositInSp: IStabilityPool(registry.stabilityPool()).getTotalBoldDeposits()
        });
    }

    function getDepositorInfo(address _market, address _depositor) 
        external
        view
        returns (
            uint256 compoundedBOLD,
            uint256 collGain,
            uint256 boldGain
        )
    {
        IStabilityPool stabilityPool = IStabilityPool(IAddressesRegistry(_market).stabilityPool());
        compoundedBOLD = stabilityPool.getCompoundedBoldDeposit(_depositor);
        collGain = stabilityPool.getDepositorCollGain(_depositor) + stabilityPool.stashedColl(_depositor);
        boldGain = stabilityPool.getDepositorYieldGain(_depositor);
    }

    function getDebtInFront(
        address _market,
        uint256 _troveId,
        uint256 _acc,
        uint256 _iterations
    ) external view returns (uint256 next, uint256 debt) {
        ITroveManager troveManager = ITroveManager(IAddressesRegistry(_market).troveManager());
        ISortedTroves sortedTroves = ISortedTroves(IAddressesRegistry(_market).sortedTroves());

        next = _troveId;
        debt = _acc;
        for (uint256 i = 0; i < _iterations; ++i) {
            next = sortedTroves.getNext(next);
            if (next == 0) return (next, debt);
            debt += _getTroveDebt(troveManager, next);
        }
    }

    function getDebtInFrontByInterestRate(
        address _market,
        uint256 _troveId,
        uint256 _acc,
        uint256 _iterations,
        uint256 _targetIR
    ) external view returns (uint256 next, uint256 debt) {
        ITroveManager troveManager = ITroveManager(IAddressesRegistry(_market).troveManager());
        ISortedTroves sortedTroves = ISortedTroves(IAddressesRegistry(_market).sortedTroves());

        next = _troveId == 0 ? sortedTroves.getLast() : _troveId;
        debt = _acc;

        for (uint256 i = 0; i < _iterations && next != 0; ++i) {
            if (troveManager.getTroveAnnualInterestRate(next) >= _targetIR) return (0, debt);

            debt += _getTroveDebt(troveManager, next);
            next = ISortedTroves(sortedTroves).getPrev(next);
        }
    }

    function getDebtInFrontByTroveNum(address _market, uint256 _numTroves) external view returns (uint256 debt) {
        ITroveManager troveManager = ITroveManager(IAddressesRegistry(_market).troveManager());
        ISortedTroves sortedTroves = ISortedTroves(IAddressesRegistry(_market).sortedTroves());

        uint256 next = sortedTroves.getLast();

        for (uint256 i = 0; i < _numTroves; i++) {
            if (next == 0) return debt;
            debt += _getTroveDebt(troveManager, next);
            next = sortedTroves.getPrev(next);
        }
    }

    function getNumOfTrovesInFrontOfTrove(address _market, uint256 _troveId, uint256 _iterations) 
        external view returns (uint256 next, uint256 numTroves) 
    {        
        ISortedTroves sortedTroves = ISortedTroves(IAddressesRegistry(_market).sortedTroves());
        next = _troveId;
        for (uint256 i = 0; i < _iterations; i++) {
            next = sortedTroves.getNext(next);
            if (next == 0) return (next, numTroves);
            numTroves++;
        }
    }

    function predictAdjustTroveUpfrontFee(address _market, uint256 _collIndex, uint256 _troveId, uint256 _debtIncrease) 
        external view returns (uint256)
    {
        IAddressesRegistry market = IAddressesRegistry(_market);
        IHintHelpers hintHelpers = IHintHelpers(market.hintHelpers());

        return hintHelpers.predictAdjustTroveUpfrontFee(_collIndex, _troveId, _debtIncrease);
    }

    function _getTroveDebt(ITroveManager _troveManager, uint256 _troveId) internal view returns (uint256 debt) {
        (debt, , , , , , , , , ) = _troveManager.Troves(_troveId);
    }

    function getMultipleSortedTroves(address _market, int256 _startIdx, uint256 _count)
        external view returns (IMultiTroveGetter.CombinedTroveData[] memory) 
    {
        uint256 collIndex = _getCollIndexFromMarket(_market);
        ITroveManager troveManager = ITroveManager(IAddressesRegistry(_market).troveManager());

        IMultiTroveGetter.CombinedTroveData[] memory troves = 
            IMultiTroveGetter(MULTI_TROVE_GETTER_ADDR).getMultipleSortedTroves(collIndex, _startIdx, _count);

        for (uint256 i = 0; i < troves.length; i++) {
            ITroveManager.LatestTroveData memory latestTroveData = troveManager.getLatestTroveData(troves[i].id);

            troves[i].debt = latestTroveData.entireDebt;
            troves[i].coll = latestTroveData.entireColl;
            troves[i].annualInterestRate = latestTroveData.annualInterestRate;
            troves[i].lastInterestRateAdjTime = latestTroveData.lastInterestRateAdjTime;
        }

        return troves;
    }

    function getBatchManagerInfo(address _market, address _manager)
        external view returns (
            IBorrowerOperations.InterestBatchManager memory managerData,
            ITroveManager.LatestBatchData memory batchData
        )
    {
        IBorrowerOperations borrowOps = IBorrowerOperations(IAddressesRegistry(_market).borrowerOperations());
        ITroveManager troveManager = ITroveManager(IAddressesRegistry(_market).troveManager());

        managerData = borrowOps.getInterestBatchManager(_manager);
        batchData = troveManager.getLatestBatchData(_manager);
    }

    function _getCollIndexFromMarket(address _market) internal pure returns (uint256) {
        if (_market == WETH_MARKET_ADDR) return WETH_COLL_INDEX;
        if (_market == WSTETH_MARKET_ADDR) return WSTETH_COLL_INDEX;
        if (_market == RETH_MARKET_ADDR) return RETH_COLL_INDEX;

        revert InvalidMarketAddress();
    }
}
