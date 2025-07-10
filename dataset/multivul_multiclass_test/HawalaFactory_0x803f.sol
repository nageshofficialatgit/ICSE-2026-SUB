// File: @openzeppelin/contracts/utils/ReentrancyGuard.sol


// OpenZeppelin Contracts (last updated v5.1.0) (utils/ReentrancyGuard.sol)

pragma solidity ^0.8.20;

/**
 * @dev Contract module that helps prevent reentrant calls to a function.
 *
 * Inheriting from `ReentrancyGuard` will make the {nonReentrant} modifier
 * available, which can be applied to functions to make sure there are no nested
 * (reentrant) calls to them.
 *
 * Note that because there is a single `nonReentrant` guard, functions marked as
 * `nonReentrant` may not call one another. This can be worked around by making
 * those functions `private`, and then adding `external` `nonReentrant` entry
 * points to them.
 *
 * TIP: If EIP-1153 (transient storage) is available on the chain you're deploying at,
 * consider using {ReentrancyGuardTransient} instead.
 *
 * TIP: If you would like to learn more about reentrancy and alternative ways
 * to protect against it, check out our blog post
 * https://blog.openzeppelin.com/reentrancy-after-istanbul/[Reentrancy After Istanbul].
 */
abstract contract ReentrancyGuard {
    // Booleans are more expensive than uint256 or any type that takes up a full
    // word because each write operation emits an extra SLOAD to first read the
    // slot's contents, replace the bits taken up by the boolean, and then write
    // back. This is the compiler's defense against contract upgrades and
    // pointer aliasing, and it cannot be disabled.

    // The values being non-zero value makes deployment a bit more expensive,
    // but in exchange the refund on every call to nonReentrant will be lower in
    // amount. Since refunds are capped to a percentage of the total
    // transaction's gas, it is best to keep them low in cases like this one, to
    // increase the likelihood of the full refund coming into effect.
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;

    uint256 private _status;

    /**
     * @dev Unauthorized reentrant call.
     */
    error ReentrancyGuardReentrantCall();

    constructor() {
        _status = NOT_ENTERED;
    }

    /**
     * @dev Prevents a contract from calling itself, directly or indirectly.
     * Calling a `nonReentrant` function from another `nonReentrant`
     * function is not supported. It is possible to prevent this from happening
     * by making the `nonReentrant` function external, and making it call a
     * `private` function that does the actual work.
     */
    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {
        // On the first call to nonReentrant, _status will be NOT_ENTERED
        if (_status == ENTERED) {
            revert ReentrancyGuardReentrantCall();
        }

        // Any calls to nonReentrant after this point will fail
        _status = ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == ENTERED;
    }
}

// File: @openzeppelin/contracts/utils/Context.sol


// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

pragma solidity ^0.8.20;

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}

// File: @openzeppelin/contracts/utils/Pausable.sol


// OpenZeppelin Contracts (last updated v5.0.0) (utils/Pausable.sol)

pragma solidity ^0.8.20;


/**
 * @dev Contract module which allows children to implement an emergency stop
 * mechanism that can be triggered by an authorized account.
 *
 * This module is used through inheritance. It will make available the
 * modifiers `whenNotPaused` and `whenPaused`, which can be applied to
 * the functions of your contract. Note that they will not be pausable by
 * simply including this module, only once the modifiers are put in place.
 */
abstract contract Pausable is Context {
    bool private _paused;

    /**
     * @dev Emitted when the pause is triggered by `account`.
     */
    event Paused(address account);

    /**
     * @dev Emitted when the pause is lifted by `account`.
     */
    event Unpaused(address account);

    /**
     * @dev The operation failed because the contract is paused.
     */
    error EnforcedPause();

    /**
     * @dev The operation failed because the contract is not paused.
     */
    error ExpectedPause();

    /**
     * @dev Initializes the contract in unpaused state.
     */
    constructor() {
        _paused = false;
    }

    /**
     * @dev Modifier to make a function callable only when the contract is not paused.
     *
     * Requirements:
     *
     * - The contract must not be paused.
     */
    modifier whenNotPaused() {
        _requireNotPaused();
        _;
    }

    /**
     * @dev Modifier to make a function callable only when the contract is paused.
     *
     * Requirements:
     *
     * - The contract must be paused.
     */
    modifier whenPaused() {
        _requirePaused();
        _;
    }

    /**
     * @dev Returns true if the contract is paused, and false otherwise.
     */
    function paused() public view virtual returns (bool) {
        return _paused;
    }

    /**
     * @dev Throws if the contract is paused.
     */
    function _requireNotPaused() internal view virtual {
        if (paused()) {
            revert EnforcedPause();
        }
    }

    /**
     * @dev Throws if the contract is not paused.
     */
    function _requirePaused() internal view virtual {
        if (!paused()) {
            revert ExpectedPause();
        }
    }

    /**
     * @dev Triggers stopped state.
     *
     * Requirements:
     *
     * - The contract must not be paused.
     */
    function _pause() internal virtual whenNotPaused {
        _paused = true;
        emit Paused(_msgSender());
    }

    /**
     * @dev Returns to normal state.
     *
     * Requirements:
     *
     * - The contract must be paused.
     */
    function _unpause() internal virtual whenPaused {
        _paused = false;
        emit Unpaused(_msgSender());
    }
}

// File: @openzeppelin/contracts/access/Ownable.sol


// OpenZeppelin Contracts (last updated v5.0.0) (access/Ownable.sol)

pragma solidity ^0.8.20;


/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * The initial owner is set to the address provided by the deployer. This can
 * later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
    address private _owner;

    /**
     * @dev The caller account is not authorized to perform an operation.
     */
    error OwnableUnauthorizedAccount(address account);

    /**
     * @dev The owner is not a valid owner account. (eg. `address(0)`)
     */
    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Initializes the contract setting the address provided by the deployer as the initial owner.
     */
    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if the sender is not the owner.
     */
    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby disabling any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

// File: @openzeppelin/contracts/token/ERC20/IERC20.sol


// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC-20 standard as defined in the ERC.
 */
interface IERC20 {
    /**
     * @dev Emitted when `value` tokens are moved from one account (`from`) to
     * another (`to`).
     *
     * Note that `value` may be zero.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Emitted when the allowance of a `spender` for an `owner` is set by
     * a call to {approve}. `value` is the new allowance.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);

    /**
     * @dev Returns the value of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the value of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves a `value` amount of tokens from the caller's account to `to`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address to, uint256 value) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets a `value` amount of tokens as the allowance of `spender` over the
     * caller's tokens.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * IMPORTANT: Beware that changing an allowance with this method brings the risk
     * that someone may use both the old and the new allowance by unfortunate
     * transaction ordering. One possible solution to mitigate this race
     * condition is to first reduce the spender's allowance to 0 and set the
     * desired value afterwards:
     * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
     *
     * Emits an {Approval} event.
     */
    function approve(address spender, uint256 value) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to` using the
     * allowance mechanism. `value` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

// File: contracts/contracts/AgentManager.sol


pragma solidity ^0.8.0;




contract AgentManager is Ownable, ReentrancyGuard {
    address public hawalaFactory;
    address[] public allAgents;

    mapping(address => bool) public operators;

    struct Transaction {
        address wallet;
        uint256 btcAmount;
        bool orderType;
    }

    struct Agent {
        bool isActive;
        uint256 commissionRate; // in basis points (e.g., 250 = 2.5%)
        uint256 totalCommission;
        uint256 totalBtcVolume;
        uint256 totalUsdtVolume;
    }

    mapping(address => Agent) public agents;
    mapping(address => address) public clientToAgent;
    mapping(address => Transaction[]) public agentToTransactions;

    event AgentSuspended(address indexed agent);
    event AgentApproved(address indexed agent);
    event AgentDeleted(address indexed agent);
    event AgentUpdated(address indexed agent, uint256 commissionRate);
    event AgentRegistered(address indexed agent, uint256 commissionRate);
    event ClientAssigned(address indexed client, address indexed agent);
    event CommissionEarned(
        address indexed agent,
        address indexed client,
        uint256 amount
    );
    event OperatorUpdated(address indexed operator, bool status);

    modifier onlyOperator() {
        require(
            operators[msg.sender] || msg.sender == owner(),
            "Not authorized: operator only"
        );
        _;
    }

    modifier onlyFactory() {
        require(msg.sender == hawalaFactory, "Not authorized: factory only");
        _;
    }

    constructor(address initialOwner) Ownable(initialOwner) {}

    function setHawalaFactory(address _factory) external onlyOwner {
        hawalaFactory = _factory;
    }

    function setOperator(address _operator, bool _status) external onlyOwner {
        operators[_operator] = _status;
        emit OperatorUpdated(_operator, _status);
    }
    function suspendAgent(address agent) external onlyOperator {
        require(agents[agent].isActive, "Agent not active");
        agents[agent].isActive = false;
        emit AgentSuspended(agent);
    }

    function updateAgent(
        address agent,
        uint256 newCommission
    ) external onlyOperator {
        require(agents[agent].isActive, "Agent not active");
        require(newCommission <= 7500, "Commission rate too high");
        agents[agent].commissionRate = newCommission;
        emit AgentUpdated(agent, newCommission);
    }

    function approveAgent(
        address agent,
        uint256 commissionRate
    ) external onlyOperator {
        require(agent != address(0), "Invalid agent address");
        require(commissionRate <= 7500, "Commission rate too high");
        require(!agents[agent].isActive, "Agent already registered and active");

        if (agents[agent].commissionRate == 0) {
            allAgents.push(agent);
            agents[agent] = Agent({
                isActive: true,
                commissionRate: commissionRate,
                totalCommission: 0,
                totalBtcVolume: 0,
                totalUsdtVolume: 0
            });

            emit AgentRegistered(agent, commissionRate);
        } else {
            agents[agent].isActive = true;
            emit AgentApproved(agent);
        }
    }

    function deleteAgent(address agent) external onlyOperator {
        require(agents[agent].commissionRate > 0, "Agent not registered");
        delete agents[agent];

        for (uint i = 0; i < allAgents.length; i++) {
            if (allAgents[i] == agent) {
                allAgents[i] = allAgents[allAgents.length - 1];
                allAgents.pop();
                break;
            }
        }

        emit AgentDeleted(agent);
    }

    function assignClientToAgent(
        address client,
        address agent
    ) external onlyOperator {
        require(agents[agent].isActive, "Agent not active");
        clientToAgent[client] = agent;
        emit ClientAssigned(client, agent);
    }

    function recordTrade(
        address trader,
        uint256 btcAmount,
        uint256 usdtAmount,
        bool isBTCToUSDT
    ) external onlyFactory {
        address agent = clientToAgent[trader];
        if (agent != address(0) && agents[agent].isActive) {
            agentToTransactions[agent].push(
                Transaction({
                    wallet: trader,
                    btcAmount: btcAmount,
                    orderType: isBTCToUSDT
                })
            );
            agents[agent].totalBtcVolume += btcAmount;
            agents[agent].totalUsdtVolume += usdtAmount;
        }
    }

    function getAgentTransactions(
        address agent
    ) external view returns (Transaction[] memory) {
        return agentToTransactions[agent];
    }

    function addCommission(
        address trader,
        uint256 amount
    ) external onlyFactory returns (bool, uint256) {
        address agent = clientToAgent[trader];
        if (agent != address(0) && agents[agent].isActive) {
            uint256 commission = (amount * agents[agent].commissionRate) /
                10000;
            agents[agent].totalCommission += commission;
            emit CommissionEarned(agent, trader, commission);
            return (true, commission);
        }
        return (false, 0);
    }

    function getAgentAddress(
        address trader
    ) external view onlyFactory returns (address) {
        return clientToAgent[trader];
    }

    function isClientAssigned(address client) external view returns (bool) {
        return clientToAgent[client] != address(0);
    }

    function getAllAgentsData()
        external
        view
        returns (
            address[] memory agentAddresses,
            bool[] memory isActive,
            uint256[] memory commissionRates,
            uint256[] memory totalCommissions,
            uint256[] memory btcVolumes,
            uint256[] memory usdtVolumes
        )
    {
        uint256 agentCount = allAgents.length;

        agentAddresses = new address[](agentCount);
        isActive = new bool[](agentCount);
        commissionRates = new uint256[](agentCount);
        totalCommissions = new uint256[](agentCount);
        btcVolumes = new uint256[](agentCount);
        usdtVolumes = new uint256[](agentCount);

        for (uint256 i = 0; i < agentCount; i++) {
            address agentAddr = allAgents[i];
            agentAddresses[i] = agentAddr;
            isActive[i] = agents[agentAddr].isActive;
            commissionRates[i] = agents[agentAddr].commissionRate;
            totalCommissions[i] = agents[agentAddr].totalCommission;
            btcVolumes[i] = agents[agentAddr].totalBtcVolume;
            usdtVolumes[i] = agents[agentAddr].totalUsdtVolume;
        }

        return (
            agentAddresses,
            isActive,
            commissionRates,
            totalCommissions,
            btcVolumes,
            usdtVolumes
        );
    }
}

// File: contracts/contracts/HawalaFactory.sol


pragma solidity ^0.8.0;






contract HawalaFactory is Ownable, ReentrancyGuard, Pausable {
    IERC20 public usdtToken;
    AgentManager public agentManager;
    address private operator;

    struct Trade {
        bytes32 id;
        address creator;
        uint256 btcAmount;
        uint256 usdtAmount;
        uint256 initialBtcAmount;
        uint256 initialUsdtAmount;
        uint256 price;
        bool isBTCToUSDT;
        bool isMarketPrice;
        TradeStatus status;
        uint256 creationTime;
        string btcAddress;
    }

    enum TradeStatus {
        Open,
        Completed,
        Cancelled
    }

    uint256 public constant MARKET_TRADE_TIMEOUT = 1 hours;
    uint256 public constant FIXED_TRADE_TIMEOUT = 24 hours;
    uint256 public constant MIN_RESIDUAL = 100000000000000; // 0.0001 BTC

    uint256 public marketFee = 25; // 0.25%
    uint256 public fixedFee = 200; // 2.00%
    uint256 public cashback_rate;
    uint256 public minMarketTradeSize;
    uint256 public minFixedTradeSize;
    uint256 public largeOrderThreshold;
    uint256 public platformUSDTFees;

    mapping(bytes32 => Trade) public trades;
    mapping(address => bool) public blockedWallets;
    mapping(address => bool) public operators;

    bytes32[] public allTradeIds;

    event TradeCreated(
        bytes32 indexed tradeId,
        address indexed creator,
        bool isMarketPrice,
        bool isBTCToUSDT
    );
    event TradeExecuted(
        bytes32 indexed tradeId,
        address indexed buyer,
        address indexed seller,
        uint256 executedBtcAmount,
        uint256 executedUsdtAmount,
        uint256 remainingBtcAmount,
        uint256 TradeUSDTFees
    );
    event TradeCancelled(bytes32 indexed tradeId);
    event TradeAutoCancelled(bytes32 indexed tradeId);
    event TradingPaused(uint256 timestamp);
    event TradingResumed(uint256 timestamp);
    event WalletBlocked(address indexed wallet);
    event WalletUnblocked(address indexed wallet);
    event TradeSizeUpdated(
        uint256 minMarketTradeSize,
        uint256 minFixedTradeSize
    );
    event FeesUpdated(uint256 minMarketFee, uint256 minFixedFee);
    event CashbackUpdated(uint256 cashback_rate);
    event LargeOrderThresholdUpdated(uint256 largeOrderThreshold);
    event FeesWithdrawn(uint256 usdtFees);
    event EmergencyWithdrawal(uint256 withdrawn_amount);
    event OperatorUpdated(address indexed operator, bool status);

    modifier notBlocked() {
        require(!blockedWallets[msg.sender], "Wallet is blocked");
        _;
    }

    modifier onlyOperator() {
        require(
            operators[msg.sender] || msg.sender == owner(),
            "Not authorized"
        );
        _;
    }

    constructor(
        address _usdtToken,
        address _agentManager
    ) Ownable(msg.sender) {
        usdtToken = IERC20(_usdtToken);
        agentManager = AgentManager(_agentManager);
    }

    function createTrade(
        uint256 btcAmount,
        uint256 usdtAmount,
        uint256 price,
        bool isBTCToUSDT,
        bool isMarketPrice,
        string memory btcAddress
    ) external nonReentrant whenNotPaused returns (bytes32) {
        require(
            btcAmount > 0 && usdtAmount > 0 && price > 0,
            "Invalid amounts"
        );
        require(
            usdtAmount >=
                (isMarketPrice ? minMarketTradeSize : minFixedTradeSize),
            "Below minimum trade size"
        );
        require(
            usdtAmount < largeOrderThreshold,
            "Amount exceeds large order threshold"
        );

        bytes32 tradeId = keccak256(
            abi.encodePacked(block.timestamp, msg.sender, btcAmount, usdtAmount)
        );

        trades[tradeId] = Trade({
            id: tradeId,
            creator: msg.sender,
            btcAmount: btcAmount,
            usdtAmount: usdtAmount,
            initialBtcAmount: btcAmount,
            initialUsdtAmount: usdtAmount,
            price: price,
            isBTCToUSDT: isBTCToUSDT,
            isMarketPrice: isMarketPrice,
            status: blockedWallets[msg.sender]
                ? TradeStatus.Cancelled
                : TradeStatus.Open,
            creationTime: block.timestamp,
            btcAddress: btcAddress
        });

        allTradeIds.push(tradeId);

        address agent = agentManager.getAgentAddress(msg.sender);

        if (agent != address(0)) {
            agentManager.recordTrade(
                msg.sender,
                btcAmount,
                usdtAmount,
                isBTCToUSDT
            );
        }

        if (!isBTCToUSDT) {
            require(
                usdtToken.transferFrom(msg.sender, address(this), usdtAmount),
                "USDT transfer to escrow failed"
            );
        }

        emit TradeCreated(tradeId, msg.sender, isMarketPrice, isBTCToUSDT);
        return tradeId;
    }

    function executeTrade(
        bytes32 tradeId,
        uint256 amount
    ) external nonReentrant whenNotPaused notBlocked {
        Trade storage trade = trades[tradeId];
        require(trade.status == TradeStatus.Open, "Trade not open");
        require(amount > 0, "Invalid amount");
        require(amount <= trade.btcAmount, "Execution amount high");
        require(
            block.timestamp <=
                trade.creationTime +
                    (
                        trade.isMarketPrice
                            ? MARKET_TRADE_TIMEOUT
                            : FIXED_TRADE_TIMEOUT
                    ),
            "Trade expired"
        );
        uint256 usdtAmount = (amount * trade.price) / 1e18;
        require(usdtAmount <= trade.usdtAmount, "Invalid USDT amount");

        uint256 usdtFee = calculateFees(
            amount,
            trade.price,
            trade.isMarketPrice
        );

        require(usdtFee <= usdtAmount, "Fee exceeds trade amount");
        uint256 amountAfterFee = usdtAmount - usdtFee;

        (bool hasExecutorAgent, uint256 executorCommission) = agentManager
            .addCommission(msg.sender, usdtFee);
        (bool hasCreatorAgent, uint256 creatorCommission) = agentManager
            .addCommission(trade.creator, usdtFee);

        require(
            (executorCommission + creatorCommission) <= usdtFee,
            "Total commission exceeds fee"
        );

        if (hasExecutorAgent) {
            agentManager.recordTrade(
                msg.sender,
                amount,
                usdtAmount,
                !trade.isBTCToUSDT
            );
        }

        if (trade.isBTCToUSDT) {
            require(
                usdtToken.transferFrom(msg.sender, address(this), usdtAmount),
                "USDT transfer failed"
            );
            require(
                usdtToken.transfer(trade.creator, amountAfterFee),
                "USDT transfer failed"
            );

            if (hasExecutorAgent) {
                usdtToken.transfer(
                    agentManager.getAgentAddress(msg.sender),
                    executorCommission
                );
            }
            if (hasCreatorAgent) {
                usdtToken.transfer(
                    agentManager.getAgentAddress(trade.creator),
                    creatorCommission
                );
            }

            platformUSDTFees +=
                usdtFee -
                executorCommission -
                creatorCommission;
        } else {
            require(
                usdtToken.transfer(msg.sender, amountAfterFee),
                "USDT transfer failed"
            );

            if (hasExecutorAgent) {
                usdtToken.transfer(
                    agentManager.getAgentAddress(msg.sender),
                    executorCommission
                );
            }
            if (hasCreatorAgent) {
                usdtToken.transfer(
                    agentManager.getAgentAddress(trade.creator),
                    creatorCommission
                );
            }

            platformUSDTFees +=
                usdtFee -
                executorCommission -
                creatorCommission;
        }

        if (trade.isMarketPrice && cashback_rate > 0) {
            uint256 cashback = (usdtFee * cashback_rate) / 100;
            require(
                usdtToken.transfer(msg.sender, cashback),
                "Cashback failed"
            );
        }

        trade.btcAmount -= amount;
        trade.usdtAmount -= usdtAmount;

        if (trade.btcAmount == 0 || trade.btcAmount < MIN_RESIDUAL) {
            trade.status = TradeStatus.Completed;
            if (trade.btcAmount > 0) {
                if (!trade.isBTCToUSDT) {
                    uint256 residualUSDT = (trade.btcAmount * trade.price) /
                        1e18;
                    require(
                        usdtToken.transfer(trade.creator, residualUSDT),
                        "Transfer failed"
                    );
                }

                emit TradeAutoCancelled(tradeId);
            }
        }

        emit TradeExecuted(
            tradeId,
            msg.sender,
            trade.creator,
            amount,
            usdtAmount,
            trade.btcAmount,
            usdtFee
        );
    }

    function getOpenOrders()
        external
        view
        returns (
            bytes32[] memory orderIds,
            uint256[] memory prices,
            uint256[] memory usdtAmounts,
            uint256[] memory btcAmounts,
            bool[] memory isBTCToUSDTs,
            bool[] memory isMarketPrices,
            address[] memory creators,
            string[] memory btcAddresses,
            TradeStatus[] memory statusses
        )
    {
        uint256 count = 0;
        for (uint256 i = 0; i < allTradeIds.length; i++) {
            Trade storage trade = trades[allTradeIds[i]];
            bool isExpired = block.timestamp >
                trade.creationTime +
                    (
                        trade.isMarketPrice
                            ? MARKET_TRADE_TIMEOUT
                            : FIXED_TRADE_TIMEOUT
                    );
            if (
                trades[allTradeIds[i]].status == TradeStatus.Open && !isExpired
            ) {
                count++;
            }
        }

        orderIds = new bytes32[](count);
        prices = new uint256[](count);
        usdtAmounts = new uint256[](count);
        btcAmounts = new uint256[](count);
        isBTCToUSDTs = new bool[](count);
        isMarketPrices = new bool[](count);
        creators = new address[](count);
        btcAddresses = new string[](count);
        statusses = new TradeStatus[](count);

        uint256 index = 0;
        for (uint256 i = 0; i < allTradeIds.length; i++) {
            Trade storage trade = trades[allTradeIds[i]];
            bool isExpired = block.timestamp >
                trade.creationTime +
                    (
                        trade.isMarketPrice
                            ? MARKET_TRADE_TIMEOUT
                            : FIXED_TRADE_TIMEOUT
                    );
            if (trade.status == TradeStatus.Open && !isExpired) {
                orderIds[index] = allTradeIds[i];
                prices[index] = trade.price;
                usdtAmounts[index] = trade.usdtAmount;
                btcAmounts[index] = trade.btcAmount;
                isBTCToUSDTs[index] = trade.isBTCToUSDT;
                isMarketPrices[index] = trade.isMarketPrice;
                creators[index] = trade.creator;
                btcAddresses[index] = trade.btcAddress;
                statusses[index] = trade.status;

                index++;
            }
        }
    }

    function getUserAllOrders(
        address user
    )
        external
        view
        returns (
            bytes32[] memory orderIds,
            uint256[] memory prices,
            uint256[] memory usdtAmounts,
            uint256[] memory btcAmounts,
            bool[] memory isBTCToUSDTs,
            bool[] memory isMarketPrices,
            string[] memory btcAddresses,
            TradeStatus[] memory statuses,
            uint256[] memory creationTimes
        )
    {
        uint256 count = 0;

        for (uint256 i = 0; i < allTradeIds.length; i++) {
            if (trades[allTradeIds[i]].creator == user) {
                count++;
            }
        }

        orderIds = new bytes32[](count);
        prices = new uint256[](count);
        usdtAmounts = new uint256[](count);
        btcAmounts = new uint256[](count);
        isBTCToUSDTs = new bool[](count);
        isMarketPrices = new bool[](count);
        btcAddresses = new string[](count);
        statuses = new TradeStatus[](count);
        creationTimes = new uint256[](count);

        uint256 index = 0;
        for (uint256 i = 0; i < allTradeIds.length; i++) {
            Trade storage trade = trades[allTradeIds[i]];
            if (trade.creator == user) {
                orderIds[index] = trade.id;
                prices[index] = trade.price;
                usdtAmounts[index] = trade.status == TradeStatus.Completed
                    ? trade.initialUsdtAmount
                    : trade.usdtAmount;
                btcAmounts[index] = trade.status == TradeStatus.Completed
                    ? trade.initialBtcAmount
                    : trade.btcAmount;
                isBTCToUSDTs[index] = trade.isBTCToUSDT;
                isMarketPrices[index] = trade.isMarketPrice;
                btcAddresses[index] = trade.btcAddress;
                statuses[index] = trade.status;
                creationTimes[index] = trade.creationTime;
                index++;
            }
        }
    }

    function updateOperator(
        address _operator,
        bool _status
    ) external onlyOwner {
        operators[_operator] = _status;
        emit OperatorUpdated(_operator, _status);
    }

    function setAgentManager(address _agentManager) external onlyOwner {
        agentManager = AgentManager(_agentManager);
    }

    function blockWallet(address[] calldata _addresses) external onlyOperator {
        require(_addresses.length > 0, "Empty address array");
        require(_addresses.length <= 100, "Array too large");

        for (uint256 i = 0; i < _addresses.length; i++) {
            require(!blockedWallets[_addresses[i]], "Address already blocked");
            blockedWallets[_addresses[i]] = true;
            emit WalletBlocked(_addresses[i]);
        }
    }

    function unblockWallet(
        address[] calldata _addresses
    ) external onlyOperator {
        require(_addresses.length > 0, "Empty address array");
        require(_addresses.length <= 100, "Array too large");
        for (uint256 i = 0; i < _addresses.length; i++) {
            require(blockedWallets[_addresses[i]], "Address already unblocked");
            blockedWallets[_addresses[i]] = false;
            emit WalletUnblocked(_addresses[i]);
            (_addresses[i]);
        }
    }

    function cancelTrade(bytes32 tradeId) external notBlocked {
        Trade storage trade = trades[tradeId];
        require(msg.sender == trade.creator, "Not trade creator");
        require(trade.status == TradeStatus.Open, "Trade not open");

        trade.status = TradeStatus.Cancelled;

        if (!trade.isBTCToUSDT) {
            require(
                usdtToken.transfer(trade.creator, trade.usdtAmount),
                "USDT return failed"
            );
        }
        emit TradeCancelled(tradeId);
    }

    function setMinimumTradeSizes(
        uint256 _marketMin,
        uint256 _fixedMin
    ) external onlyOwner {
        require(_marketMin > 0 && _fixedMin > 0, "Invalid Sizes");
        minMarketTradeSize = _marketMin;
        minFixedTradeSize = _fixedMin;

        emit TradeSizeUpdated(_marketMin, _fixedMin);
    }

    function setLargeOrderThreshold(uint256 _threshold) external onlyOwner {
        require(_threshold > 0, "Invalid Threshold");
        largeOrderThreshold = _threshold;

        emit LargeOrderThresholdUpdated(_threshold);
    }

    function setCashBack(uint256 _cashback) external onlyOwner {
        require(_cashback <= 100, "Cashback rate cannot exceed 100%");
        cashback_rate = _cashback;

        emit CashbackUpdated(_cashback);
    }

    function setFees(uint256 _marketFee, uint256 _fixedFee) external onlyOwner {
        require(_marketFee > 0 && _fixedFee > 0, "Invalid fee values");
        marketFee = _marketFee;
        fixedFee = _fixedFee;

        emit FeesUpdated(_marketFee, _fixedFee);
    }

    function pauseTrading() external onlyOwner {
        _pause();
    }

    function resumeTrading() external onlyOwner {
        _unpause();
    }

    function withdrawFees() external onlyOwner {
        uint256 usdtFees = platformUSDTFees;

        platformUSDTFees = 0;

        require(usdtToken.transfer(owner(), usdtFees), "Transfer failed");

        emit FeesWithdrawn(usdtFees);
    }

    function emergencyWithdraw() external onlyOwner whenPaused {
        uint256 balance = usdtToken.balanceOf(address(this));
        require(usdtToken.transfer(owner(), balance), "Transfer failed");

        emit EmergencyWithdrawal(balance);
    }

    function calculateFees(
        uint256 btcAmount,
        uint256 price,
        bool isMarketPrice
    ) internal view returns (uint256 usdtFee) {
        uint256 feeRate = isMarketPrice ? marketFee : fixedFee;
        usdtFee = (btcAmount * price * feeRate) / (10000 * 1e18);
    }
}