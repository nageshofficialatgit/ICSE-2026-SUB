// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @dev Minimal ERC20 interface for basic token operations.
interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function totalSupply() external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}

/// @dev Minimal interface for Aave PoolV3 deposit/withdraw logic.
interface IPoolV3 {
    function deposit(address asset, uint256 amount, address onBehalfOf, uint16 referralCode) external;
    function withdraw(address asset, uint256 amount, address to) external returns (uint256);
}

/// @dev Chainlink aggregator interface for retrieving price data (e.g., CHF/USD).
interface IAggregatorV3 {
    function latestRoundData()
        external
        view
        returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );
}

/**
 * @title Treasury
 *
 * This contract is supported by HypeLoot.com
 * Stay connected with us:
 * - Website: https://hypeloot.com/
 * - X (formerly Twitter): https://x.com/HypeLootCom
 * - Telegram Channel: https://t.me/Hypelootcom
 * - Instagram: https://instagram.com/hypelootcom
 *
 * For platform support, please email: support[at]hypeloot.com
 * We are continuously expanding our activities.
 *
 * @notice This contract acts as a treasury that:
 *  1) Receives notifications of user deposits (via `afterDeposit`) from a whitelisted token contract.
 *  2) Forwards deposited USDC to an Aave PoolV3, receiving aEthUSDC in return.
 *  3) Allows the owner to manage and withdraw rewards in USDC, aEthUSDC, and Frankencoin (ZCHF).
 *  4) Supports rescuing stuck ETH or non-critical ERC20 tokens.
 *  5) Includes a custom reward system in Frankencoin (ZCHF) with an epoch-based claim mechanism.
 *  6) Integrates a Chainlink oracle for the CHF/USD price feed to calculate reward ratios.
 *  7) Can only be properly initialized once (for proxy usage).
 */
contract Treasury {
    //--------------------------------------------------------------------------------
    // Storage
    //--------------------------------------------------------------------------------

    // Address of the contract owner.
    address public owner;

    // Mapping of addresses that are whitelisted to call certain restricted functions (e.g., afterDeposit).
    mapping(address => bool) public whitelisted;

    // Address of the USDC token.
    address public usdc;

    // Address of the aEthUSDC token (Aave-wrapped USDC).
    address public aEthUSDC;

    // Address of the Aave PoolV3 contract.
    address public poolV3;

    // Address of the Frankencoin (ZCHF) token.
    address public frankencoin;

    // Tracks the total amount of USDC deposited by all users, used for share calculations.
    uint256 public totalDepositedUSDC;

    // Maps how much USDC each user has deposited overall.
    mapping(address => uint256) public userTotalDepositedUSDC;

    // Indicates whether the contract has been initialized (for proxy usage).
    bool private _initialized;

    // Address of the HyperSolver (HPS) token. Used to query its totalSupply for reward calculations.
    address public hyperSolverToken;

    // Address of the Chainlink CHF/USD oracle.
    address public chfUsdOracle;

    // Maximum delay (in seconds) allowed between the last oracle update and the current block.timestamp.
    // Used to ensure price data is not stale.
    uint256 public maxDelay;

    // The ratio used to calculate rewards in Frankencoin. This is a global parameter
    // that can be set by the owner to determine how many ZCHF tokens are allocated
    // for each conceptual reward unit.
    uint256 public globalRewardRatio;

    // Epoch logic: track a single epoch start, updated in steps of 190 days if enough time has passed.
    uint256 public epoch;

    // Maps an epoch index to whether a user has already claimed rewards for that epoch.
    mapping(address => mapping(uint256 => bool)) public userClaimedEpoch;

    // The reward withdraw operation counter. Incremented each time `withdrawRewards` is called.
    uint256 public rewardWithdrawCount;

    //--------------------------------------------------------------------------------
    // Events
    //--------------------------------------------------------------------------------

    // Emitted when ownership is transferred from `previousOwner` to `newOwner`.
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // Emitted when an address is added or removed from the whitelist.
    event WhitelistUpdated(address indexed account, bool isWhitelisted);

    // Emitted when the contract is initialized.
    event TreasuryInitialized(
        address indexed owner,
        address usdcAddress,
        address aEthUSDCAddress,
        address poolV3Address,
        address frankencoinAddress,
        address hyperSolverToken
    );

    // Emitted when `afterDeposit` is called by a whitelisted contract.
    event AfterDeposit(
        address indexed depositor,
        uint256 usdcAmount,
        uint256 depositCount,
        uint256 timestamp
    );

    // Event emitted when a USDC transfer is executed as part of a deposit.
    // This logs the address from which USDC is transferred and the transferred amount.
    event TransferDeposit(address indexed from, uint256 amount);

    // Emitted after performing deposit into Aave PoolV3.
    // Logs the deposit count, current block timestamp, and new balance of aEthUSDC.
    event PoolV3Deposit(
        uint256 depositCount,
        uint256 timestamp,
        uint256 currentAethBalance
    );

    // Emitted when rewards are withdrawn to a recipient.
    // Includes the withdraw operation index and the timestamp of the operation.
    event RewardWithdrawn(
        address indexed recipient,
        uint256 amount,
        uint256 tokenType,
        uint256 withdrawOperationNumber,
        uint256 timeStamp
    );

    // Emitted when a stuck ETH is claimed by the owner.
    event StuckETHClaimed(address indexed ownerAddress, uint256 amount);

    // Emitted when stuck ERC20 tokens are claimed by the owner.
    event StuckERC20Claimed(address indexed ownerAddress, address token, uint256 amount);

    // Emitted when `owner` withdraws from Aave PoolV3.
    event PoolV3Withdrawal(uint256 amountRequested, uint256 receivedAmount);

    // Emitted when Frankencoin (ZCHF) is deposited into this contract by the owner.
    event FrankencoinToppedUp(uint256 amount);

    // Emitted when a user successfully claims Frankencoin rewards for a given epoch.
    event FrankencoinRewardClaimed(address indexed user, uint256 epoch, uint256 rewardAmount);

    // Emitted when the maximum oracle delay is updated.
    event MaxDelayUpdated(uint256 oldDelay, uint256 newDelay);

    // Emitted when the global reward ratio is updated.
    event GlobalRewardRatioUpdated(uint256 oldRatio, uint256 newRatio);

    // Emitted when the CHF/USD oracle is updated.
    event CHFUSDOracleUpdated(address oldOracle, address newOracle, uint80 roundId, int256 answer, uint256 updatedAt);

    //--------------------------------------------------------------------------------
    // Reentrancy Guard
    //--------------------------------------------------------------------------------

    // Tracks whether a function is currently executing to prevent reentrant calls.
    bool private _locked;

    /**
     * @dev Ensures that the function cannot be reentered while it is still executing.
     * If another nonReentrant function calls back into itself, it will fail.
     */
    modifier nonReentrant() {
        require(!_locked, "Treasury: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }

    //--------------------------------------------------------------------------------
    // Modifiers
    //--------------------------------------------------------------------------------

    /**
     * @dev Ensures that only the owner can call the modified function.
     */
    modifier onlyOwner() {
        require(msg.sender == owner, "Treasury: caller is not the owner");
        _;
    }

    /**
     * @dev Ensures that only whitelisted addresses can call the modified function.
     */
    modifier onlyWhitelisted() {
        require(whitelisted[msg.sender], "Treasury: caller is not whitelisted");
        _;
    }

    //--------------------------------------------------------------------------------
    // Initialization
    //--------------------------------------------------------------------------------

    /**
     * @notice Initializes the contract with critical parameters.
     * @dev This function is intended for use with a proxy pattern. It can only be called once.
     *      It also sets the initial epoch as the current block timestamp.
     * @param _owner Address that will become the owner of the contract.
     * @param _whitelist Array of addresses to be whitelisted (e.g. token contracts).
     * @param _usdc Address of the USDC token.
     * @param _aEthUSDC Address of the Aave aEthUSDC token.
     * @param _poolV3 Address of the Aave PoolV3 contract.
     * @param _frankencoin Address of the Frankencoin (ZCHF) token.
     * @param _hyperSolverToken Address of the HyperSolver (HPS) token, used to get totalSupply.
     */
    function initialize(
        address _owner,
        address[] calldata _whitelist,
        address _usdc,
        address _aEthUSDC,
        address _poolV3,
        address _frankencoin,
        address _hyperSolverToken
    ) external {
        require(!_initialized, "Treasury: already initialized");
        _initialized = true;

        owner = _owner;
        usdc = _usdc;
        aEthUSDC = _aEthUSDC;
        poolV3 = _poolV3;
        frankencoin = _frankencoin;
        hyperSolverToken = _hyperSolverToken;

        // Set the initial epoch as the current block timestamp.
        epoch = block.timestamp;

        // Populate whitelist.
        for (uint256 i = 0; i < _whitelist.length; i++) {
            whitelisted[_whitelist[i]] = true;
            emit WhitelistUpdated(_whitelist[i], true);
        }

        emit TreasuryInitialized(
            _owner,
            _usdc,
            _aEthUSDC,
            _poolV3,
            _frankencoin,
            _hyperSolverToken
        );
    }

    //--------------------------------------------------------------------------------
    // Ownership Management
    //--------------------------------------------------------------------------------

    /**
     * @notice Transfers the contract ownership to a new owner.
     * @param newOwner Address to be set as the new owner.
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Treasury: new owner is the zero address");
        address oldOwner = owner;
        owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }

    //--------------------------------------------------------------------------------
    // Whitelist Management
    //--------------------------------------------------------------------------------

    /**
     * @notice Updates the whitelist status of multiple addresses in a single transaction.
     * @dev The same `isWL` value is applied to all provided addresses in the array.
     * @param accounts Array of addresses whose whitelist status is being updated.
     * @param isWL Whether the addresses should be whitelisted (true) or removed (false).
     */
    function updateWhitelist(address[] calldata accounts, bool isWL) external onlyOwner {
        for (uint256 i = 0; i < accounts.length; i++) {
            whitelisted[accounts[i]] = isWL;
            emit WhitelistUpdated(accounts[i], isWL);
        }
    }

    //--------------------------------------------------------------------------------
    // Oracle Management 
    //--------------------------------------------------------------------------------

    /**
     * @notice Sets the Chainlink oracle for the CHF/USD price feed. Also validates the feed data.
     * @dev Reverts if the latest price is <= 0, if roundId is 0, or if the data is stale.
     * @param _oracle Address of the Chainlink aggregator for CHF/USD.
     */
    function setCHFUSDOracle(address _oracle) external onlyOwner {
        require(_oracle != address(0), "Treasury: invalid oracle address");

        address oldOracle = chfUsdOracle;

        (
            uint80 roundId,
            int256 answer,
            ,
            uint256 updatedAt,
            
        ) = IAggregatorV3(_oracle).latestRoundData();

        require(roundId > 0, "Treasury: invalid roundId");
        require(answer > 0, "Treasury: oracle price <= 0");
        require(
            updatedAt >= block.timestamp - maxDelay,
            "Treasury: oracle data is stale"
        );

        chfUsdOracle = _oracle;

        emit CHFUSDOracleUpdated(oldOracle, _oracle, roundId, answer, updatedAt);
    }

    /**
     * @notice Sets the maximum delay (in seconds) allowed between the latest oracle update and the current time.
     * @dev The delay cannot exceed 7 days.
     * @param _delay Maximum delay in seconds.
     */
    function setMaxDelay(uint256 _delay) external onlyOwner {
        require(_delay <= 7 days, "Treasury: maxDelay cannot exceed 7 days");

        uint256 oldDelay = maxDelay;
        maxDelay = _delay;

        emit MaxDelayUpdated(oldDelay, _delay);
    }

    //--------------------------------------------------------------------------------
    // afterDeposit Logic
    //--------------------------------------------------------------------------------

    /**
     * @notice Called by whitelisted token contracts after a user deposits USDC.
     * @dev This function performs several operations in sequence:
     *      1) Emits an event logging the deposit details provided by the token contract.
     *      2) Emits a TransferDeposit event and then executes a transferFrom operation to pull the 
     *         specified USDC amount from the caller (i.e., the whitelisted token contract) into this contract.
     *      3) Updates the depositor's cumulative USDC deposits as well as the overall total deposits.
     *      4) Approves PoolV3 to spend the entire USDC balance held by this contract.
     *      5) Calls the PoolV3 deposit function to convert USDC into aEthUSDC.
     *      6) Emits an event with details of the PoolV3 deposit including the updated aEthUSDC balance.
     *
     * @param depositor The address of the user who made the deposit on the token side.
     * @param usdcAmount The amount of USDC deposited by the user.
     * @param depositCount The current deposit count for the user as provided by the token contract.
     * @param timestamp The block timestamp at which the deposit was made.
     */
    function afterDeposit(
        address depositor,
        uint256 usdcAmount,
        uint256 depositCount,
        uint256 timestamp
    ) external onlyWhitelisted nonReentrant {
        // Emit event logging the initial deposit details.
        emit AfterDeposit(depositor, usdcAmount, depositCount, timestamp);

        // Emit event for the USDC transfer operation.
        emit TransferDeposit(msg.sender, usdcAmount);
        // Transfer the specified USDC amount from the caller (whitelisted token contract) to this contract.
        IERC20(usdc).transferFrom(msg.sender, address(this), usdcAmount);

        // Update the depositor's cumulative USDC deposits and the overall deposit total.
        userTotalDepositedUSDC[depositor] += usdcAmount;
        totalDepositedUSDC += usdcAmount;

        // Retrieve the total USDC balance held by this contract.
        uint256 balance = IERC20(usdc).balanceOf(address(this));

        // Approve PoolV3 to spend the entire USDC balance in this contract.
        IERC20(usdc).approve(poolV3, balance);

        // Deposit the USDC into the Aave PoolV3, converting it into aEthUSDC.
        IPoolV3(poolV3).deposit(usdc, balance, address(this), 0);

        // Retrieve the updated aEthUSDC balance after the deposit.
        uint256 aEthBalance = IERC20(aEthUSDC).balanceOf(address(this));

        // Emit event with details of the PoolV3 deposit, including the deposit count, current timestamp, and new aEthUSDC balance.
        emit PoolV3Deposit(depositCount, block.timestamp, aEthBalance);
    }

    //--------------------------------------------------------------------------------
    // Withdraw from Aave PoolV3
    //--------------------------------------------------------------------------------

    /**
     * @notice Allows the owner to withdraw USDC from the Aave PoolV3 back to this contract.
     * @dev The PoolV3 withdraw function returns the actual withdrawn amount,
     *      which may be less than or equal to the requested `amount`.
     * @param amount The amount of USDC to withdraw from Aave. Use type(uint256).max to withdraw all.
     */
    function withdrawFromAave(uint256 amount) external onlyOwner {
        uint256 received = IPoolV3(poolV3).withdraw(usdc, amount, address(this));
        emit PoolV3Withdrawal(amount, received);
    }

    //--------------------------------------------------------------------------------
    // Withdraw Rewards
    //--------------------------------------------------------------------------------

    /**
     * @notice Distributes rewards based on the `tokenType` parameter:
     *         0 -> USDC
     *         1 -> aEthUSDC
     *         2 -> Frankencoin (ZCHF)
     *
     * @dev Function callable only by the contract owner.
     *      Increments the `rewardWithdrawCount` counter each time it is executed.
     *
     * @param recipients List of recipient addresses for the rewards.
     * @param amounts List of amounts to be transferred to each recipient.
     * @param tokenType Specifies which token will be distributed (0 = USDC, 1 = aEthUSDC, 2 = Frankencoin).
     */
    function withdrawRewards(
        address[] calldata recipients,
        uint256[] calldata amounts,
        uint256 tokenType
    ) external onlyOwner {
        require(recipients.length == amounts.length, "Treasury: arrays length mismatch");
        
        // Increment the reward withdrawal counter.
        rewardWithdrawCount++;

        // Select the appropriate token based on tokenType.
        address rewardToken;
        if (tokenType == 0) {
            rewardToken = usdc;
        } else if (tokenType == 1) {
            rewardToken = aEthUSDC;
        } else if (tokenType == 2) {
            rewardToken = frankencoin;
        } else {
            revert("Treasury: invalid token type");
        }

        uint256 currentOp = rewardWithdrawCount;
        uint256 currentTimestamp = block.timestamp;

        // Process the transfer for each recipient.
        for (uint256 i = 0; i < recipients.length; i++) {
            require(
                IERC20(rewardToken).transfer(recipients[i], amounts[i]),
                "Treasury: token transfer failed"
            );
            emit RewardWithdrawn(
                recipients[i],
                amounts[i],
                tokenType,
                currentOp,
                currentTimestamp
            );
        }
    }

    //--------------------------------------------------------------------------------
    // Reward Ratio in Frankencoin + Claim Logic
    //--------------------------------------------------------------------------------

    /**
     * @notice Sets the global reward ratio in Frankencoin (ZCHF). This ratio is used in the
     *         reward calculation formula for distributing Frankencoin to users.
     * @dev Only the owner can call this function.
     * @param newRatio The new ratio value to be used in calculations.
     */
    function setGlobalRewardRatio(uint256 newRatio) external onlyOwner {
        uint256 oldRatio = globalRewardRatio;
        globalRewardRatio = newRatio;

        emit GlobalRewardRatioUpdated(oldRatio, newRatio);
    }

    /**
     * @notice Retrieves the current reward amount in Frankencoin for a given user, based on a formula.
     * @dev The formula for the reward is:
     *      numerator = ( userTokenVested * 100000000 * globalRewardRatio )
     *      denominator = ( hpsTotalSupply * chfUsdPrice * userTotalDepositedUSDC[user] )
     *      result = numerator / denominator
     *
     *      - Validates the Chainlink roundId > 0 and the price is > 0 (answer > 0).
     *      - Also checks that updatedAt is within the allowed maxDelay to prevent stale data.
     * @param user The address for which the reward is being calculated.
     * @param userTokenVested The amount of tokens the user has vested (or minted).
     * @return rewardInZCHF The final reward amount in ZCHF according to the formula.
     */
    function getUserRewardInFrankencoin(address user, uint256 userTokenVested)
        public
        view
        returns (uint256 rewardInZCHF)
    {
        uint256 hpsTotalSupply = IERC20(hyperSolverToken).totalSupply();
        uint256 userDeposited = userTotalDepositedUSDC[user];

        // Edge cases
        if (hpsTotalSupply == 0 || userDeposited == 0 || userTokenVested == 0) {
            return 0;
        }

        // Retrieve data from oracle
        (
            uint80 roundId,
            int256 answer,
            ,
            uint256 updatedAt,
            
        ) = IAggregatorV3(chfUsdOracle).latestRoundData();

        // Validate roundId and price
        require(roundId > 0, "Treasury: invalid roundId from oracle");
        require(answer > 0, "Treasury: oracle price <= 0");
        require(
            updatedAt >= block.timestamp - maxDelay,
            "Treasury: oracle data is stale"
        );

        uint256 chfUsdPrice = uint256(answer);

        // Build numerator and denominator
        uint256 numerator = userTokenVested * 100000000 * globalRewardRatio;
        uint256 denominator = hpsTotalSupply * chfUsdPrice * userDeposited;

        // Avoid division by zero (already checked above, but double-check).
        if (denominator == 0) {
            return 0;
        }

        rewardInZCHF = numerator / denominator;
    }

    /**
     * @notice Allows a whitelisted contract (e.g., a vesting or staking contract) to claim
     *         Frankencoin (ZCHF) rewards on behalf of a user, subject to epoch-based claims.
     * @dev If the current block timestamp is >= epoch + 190 days, the epoch is incremented
     *      by 190 days. Each user can only claim once per epoch.
     * @param user The address of the user who is claiming rewards.
     * @param userTokenVested The amount of tokens the user has vested (to be used in the reward formula).
     */
    function claimFrankencoinReward(address user, uint256 userTokenVested)
        external
        onlyWhitelisted
        nonReentrant
    {
        // If 190 days have passed since the last epoch, update the epoch by +190 days.
        if (block.timestamp >= epoch + 190 days) {
            epoch += 190 days;
        }

        // Check if the user has already claimed in this epoch
        require(!userClaimedEpoch[user][epoch], "Treasury: user already claimed this epoch");

        // Calculate the reward according to the formula
        uint256 rewardAmount = getUserRewardInFrankencoin(user, userTokenVested);
        require(rewardAmount > 0, "Treasury: reward amount is zero");

        // Check if the contract has enough Frankencoin to pay out
        uint256 contractBalance = IERC20(frankencoin).balanceOf(address(this));
        require(contractBalance >= rewardAmount, "Treasury: insufficient Frankencoin balance");

        // Mark the user as having claimed in this epoch
        userClaimedEpoch[user][epoch] = true;

        // Transfer Frankencoin to the user
        require(IERC20(frankencoin).transfer(user, rewardAmount), "Treasury: Frankencoin transfer failed");

        emit FrankencoinRewardClaimed(user, epoch, rewardAmount);
    }

    //--------------------------------------------------------------------------------
    // Frankencoin Management
    //--------------------------------------------------------------------------------

    /**
     * @notice Allows the owner to deposit Frankencoin (ZCHF) into this contract.
     * @dev The owner must first `approve` this contract to pull `amount` Frankencoin.
     * @param amount The amount of Frankencoin to deposit.
     */
    function topUpFrankencoin(uint256 amount) external onlyOwner {
        require(amount > 0, "Treasury: amount must be greater than 0");
        // Pull Frankencoin from the owner
        require(
            IERC20(frankencoin).transferFrom(msg.sender, address(this), amount),
            "Treasury: Frankencoin transfer failed"
        );

        emit FrankencoinToppedUp(amount);
    }

    //--------------------------------------------------------------------------------
    // Rescue / Claim Stuck ETH or ERC20
    //--------------------------------------------------------------------------------

    /**
     * @notice Transfers all stuck ETH from the contract to the owner.
     * @dev Only callable by the owner.
     */
    function claimStuckETH() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "Treasury: no ETH to claim");
        payable(owner).transfer(balance);
        emit StuckETHClaimed(owner, balance);
    }

    /**
     * @notice Transfers stuck ERC20 tokens (that are not USDC, aEthUSDC, or Frankencoin)
     *         from the contract to the owner.
     * @dev Only callable by the owner.
     * @param tokens Array of ERC20 token addresses to claim.
     */
    function claimStuckERC20(address[] calldata tokens) external onlyOwner {
        for (uint256 i = 0; i < tokens.length; i++) {
            // Skip if the token is USDC, aEthUSDC, or Frankencoin
            if (tokens[i] == usdc || tokens[i] == aEthUSDC || tokens[i] == frankencoin) {
                continue;
            }

            uint256 balance = IERC20(tokens[i]).balanceOf(address(this));
            if (balance > 0) {
                IERC20(tokens[i]).transfer(owner, balance);
                emit StuckERC20Claimed(owner, tokens[i], balance);
            }
        }
    }
}