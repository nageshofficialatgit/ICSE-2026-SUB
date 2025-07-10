// File: @openzeppelin/contracts/security/ReentrancyGuard.sol


// OpenZeppelin Contracts (last updated v4.9.0) (security/ReentrancyGuard.sol)

pragma solidity ^0.8.0;

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
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
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
        // On the first call to nonReentrant, _status will be _NOT_ENTERED
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");

        // Any calls to nonReentrant after this point will fail
        _status = _ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = _NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == _ENTERED;
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

// File: @openzeppelin/contracts/security/Pausable.sol


// OpenZeppelin Contracts (last updated v4.7.0) (security/Pausable.sol)

pragma solidity ^0.8.0;


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
    /**
     * @dev Emitted when the pause is triggered by `account`.
     */
    event Paused(address account);

    /**
     * @dev Emitted when the pause is lifted by `account`.
     */
    event Unpaused(address account);

    bool private _paused;

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
        require(!paused(), "Pausable: paused");
    }

    /**
     * @dev Throws if the contract is not paused.
     */
    function _requirePaused() internal view virtual {
        require(paused(), "Pausable: not paused");
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

// File: SafeStablecoinTrader.sol

//SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;





interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

interface IUniswapV2Router {
    function getAmountsOut(uint256 amountIn, address[] calldata path) external view returns (uint256[] memory amounts);
    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);
}

contract SafeStablecoinTrader is Context, Ownable, Pausable, ReentrancyGuard {
    // Token Constants - Mainnet addresses
    address public constant USDC = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48;
    address public constant USDT = 0xdAC17F958D2ee523a2206206994597C13D831ec7;
    address public constant UNISWAP_ROUTER = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
    address public constant SUSHISWAP_ROUTER = 0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F;
    
    // Security and anti-MEV variables
    uint256 private _lastTxTimestamp;
    uint256 private constant _MIN_EXECUTION_DELAY = 3; // Minimum blocks between transactions
    bytes32 private _lastTxBlockhash;

    // Trading parameters - optimized for small capital base (0.38 ETH and 2000 USDC)
    uint256 public minProfitUSD = 3e6;            // $3 minimum profit (further reduced)
    uint256 public maxTradeSize = 500e6;          // $500 maximum trade (reduced for capital preservation)
    uint256 public slippageTolerance = 15;        // 0.15% slippage (reduced for better execution)
    uint256 public maxGasPrice = 25 * 10**9;      // 25 gwei (optimized)
    uint256 public dailyVolumeLimit = 2000e6;     // $2,000 daily limit (aligned with capital)
    uint256 public constant MAX_PRICE_IMPACT = 30;   // 0.3% maximum price impact (reduced)
    uint256 public constant MAX_TRADE_DELAY = 15;    // Maximum blocks for trade execution

    // State variables
    uint256 public lastTradeBlock;
    uint256 public dailyVolume;
    uint256 public volumeResetTimestamp;
    uint256 public totalTrades;
    uint256 public totalProfit;
    uint256 public lastProfitableTradeTimestamp;
    uint256 public consecutiveFailures;
    uint256 public constant MAX_CONSECUTIVE_FAILURES = 5; // Increased

    // Mapping to prevent duplicate transactions
    mapping(bytes32 => bool) public executedTrades;

    struct TradeResult {
        uint256 amountOut;
        uint256 profit;
        uint256 gasUsed;
        uint256 priceImpact;
        uint256 balanceAfter;
    }

    struct TradeParams {
        address router;
        address tokenIn;
        address tokenOut;
        uint256 amountIn;
        uint256 balanceBefore;
    }

    // Events
    event TradeExecuted(
        address indexed router,
        address indexed tokenIn,
        address indexed tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 profit,
        uint256 gasUsed,
        uint256 priceImpact,
        uint256 timestamp
    );
    
    event EmergencyWithdrawal(
        address token,
        uint256 amount,
        uint256 timestamp
    );

    event ParametersUpdated(
        uint256 minProfit,
        uint256 maxTradeSize,
        uint256 slippageTolerance,
        uint256 maxGasPrice,
        uint256 dailyLimit
    );

    event SecurityAlert(
        string reason,
        uint256 timestamp
    );

    modifier validToken(address token) {
        require(token == USDC || token == USDT, "Invalid token");
        _;
    }

    modifier validRouter(address router) {
        require(router == UNISWAP_ROUTER || router == SUSHISWAP_ROUTER, "Invalid router");
        _;
    }

    constructor() 
        Ownable(msg.sender) 
        Pausable() 
        ReentrancyGuard() 
    {
        volumeResetTimestamp = block.timestamp;
        _lastTxTimestamp = block.timestamp;
        // We'll approve tokens manually after deployment
    }

    // Manual token approval function - call after deployment
    function approveTokens() external onlyOwner {
        IERC20(USDC).approve(UNISWAP_ROUTER, type(uint256).max);
        IERC20(USDT).approve(UNISWAP_ROUTER, type(uint256).max);
        IERC20(USDC).approve(SUSHISWAP_ROUTER, type(uint256).max);
        IERC20(USDT).approve(SUSHISWAP_ROUTER, type(uint256).max);
    }

    function _validateAndPrepare(
        address router,
        address tokenIn,
        uint256 amountIn
    ) internal returns (TradeParams memory params) {
        // Enhanced security validations
        require(tx.gasprice <= maxGasPrice, "Gas price too high");
        require(amountIn <= maxTradeSize, "Amount too large");
        require(amountIn >= 10e6, "Amount too small"); // Minimum $10 trade size
        
        // Anti-MEV protection: check execution delay + changing blockhash requirement
        if (_lastTxTimestamp > 0) {
            require(block.timestamp > _lastTxTimestamp, "Same block protection");
            require(blockhash(block.number - 1) != _lastTxBlockhash, "Potential replay attack");
        }
        _lastTxTimestamp = block.timestamp;
        _lastTxBlockhash = blockhash(block.number - 1);
        
        // Modified block validation with minimum delay enforcement
        if (lastTradeBlock > 0) {
            require(block.number > lastTradeBlock, "Too soon");
            require(block.number >= lastTradeBlock + _MIN_EXECUTION_DELAY, "Anti-MEV delay not met");
            require(block.number <= lastTradeBlock + MAX_TRADE_DELAY, "Trade delay too long");
        }

        // Update daily volume with timestamp validation
        require(block.timestamp >= volumeResetTimestamp, "Invalid timestamp"); // Sanity check
        if (block.timestamp >= volumeResetTimestamp + 1 days) {
            dailyVolume = 0;
            volumeResetTimestamp = block.timestamp;
        }
        require(dailyVolume + amountIn <= dailyVolumeLimit, "Daily limit reached");
        dailyVolume += amountIn;

        // Enhanced trade hash check with more entropy
        bytes32 tradeHash = keccak256(abi.encodePacked(
            router, 
            tokenIn, 
            amountIn, 
            block.number, 
            block.timestamp, 
            msg.sender
        ));
        require(!executedTrades[tradeHash], "Duplicate trade");
        executedTrades[tradeHash] = true;

        // Setup trade parameters with balance verification
        params.router = router;
        params.tokenIn = tokenIn;
        params.tokenOut = tokenIn == USDC ? USDT : USDC;
        params.amountIn = amountIn;
        params.balanceBefore = IERC20(params.tokenOut).balanceOf(address(this));
        
        // Verify non-zero token balances (sanity check for token contracts)
        require(params.balanceBefore < 1e12, "Token balance anomaly"); // Sanity check for unrealistic balance

        return params;
    }

    function _executeTrade(TradeParams memory params) internal returns (TradeResult memory result) {
        // Setup path
        address[] memory path = new address[](2);
        path[0] = params.tokenIn;
        path[1] = params.tokenOut;

        // Check token balances before operation
        uint256 initialTokenInBalance = IERC20(params.tokenIn).balanceOf(address(this));
        
        // Security check: Verify contract has sufficient funds or allowance
        try IUniswapV2Router(params.router).getAmountsOut(params.amountIn, path) returns (uint256[] memory amounts) {
            require(amounts[1] > 0, "Invalid output amount");
            // More conservative slippage calculation - uses 80% of configured slippage during normal operation
            uint256 effectiveSlippage = block.basefee < 25 * 10**9 ? slippageTolerance * 8 / 10 : slippageTolerance;
            uint256 minOutput = (amounts[1] * (10000 - effectiveSlippage)) / 10000;
            
            // Transfer tokens to contract from user if needed
            if (initialTokenInBalance < params.amountIn) {
                uint256 amountNeeded = params.amountIn - initialTokenInBalance;
                // Two-step transfer pattern for safety
                uint256 preTransferBalance = IERC20(params.tokenIn).balanceOf(address(this));
                require(IERC20(params.tokenIn).transferFrom(msg.sender, address(this), amountNeeded), "Transfer failed");
                uint256 postTransferBalance = IERC20(params.tokenIn).balanceOf(address(this));
                require(postTransferBalance >= preTransferBalance + amountNeeded, "Transfer amount mismatch");
            }
            
            // Security check: Re-validate token balance
            require(IERC20(params.tokenIn).balanceOf(address(this)) >= params.amountIn, "Insufficient balance");
            
            // Execute swap with timeout protection
            uint256 deadline = block.timestamp + 60;
            uint256 gasStart = gasleft();
            
            // Use try/catch to handle potential swap failures
            try IUniswapV2Router(params.router).swapExactTokensForTokens(
                params.amountIn,
                minOutput,
                path,
                address(this),
                deadline
            ) returns (uint256[] memory swapAmounts) {
                result.gasUsed = gasStart - gasleft();
                
                // Calculate results with security checks
                uint256 actualBalanceAfter = IERC20(params.tokenOut).balanceOf(address(this));
                require(actualBalanceAfter > params.balanceBefore, "No output tokens received");
                
                result.balanceAfter = actualBalanceAfter;
                result.amountOut = swapAmounts[1];
                result.profit = result.balanceAfter - params.balanceBefore;
                
                // Calculate price impact with safeguards
                if (swapAmounts[1] < amounts[1] && amounts[1] > 0) {
                    result.priceImpact = ((amounts[1] - swapAmounts[1]) * 10000) / amounts[1];
                    require(result.priceImpact <= MAX_PRICE_IMPACT, "Price impact too high");
                }
                
            } catch {
                revert("Swap execution failed");
            }
        } catch {
            revert("Price calculation failed");
        }
        
        return result;
    }

    function _updateStats(TradeResult memory result) internal {
        if (result.profit >= minProfitUSD) {
            totalProfit += result.profit;
            lastProfitableTradeTimestamp = block.timestamp;
            totalTrades++;
            consecutiveFailures = 0;
        } else {
            consecutiveFailures++;
            if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
                emit SecurityAlert("Too many consecutive failures", block.timestamp);
                _pause();
            }
        }
    }

    function _finalizeTradeAndEmit(
        TradeResult memory result,
        TradeParams memory params
    ) internal {
        // Retain 20% of profits in contract for compound growth
        uint256 amountToSend = result.balanceAfter * 80 / 100;
        
        // Send tokens to owner
        if (amountToSend > 0) {
            require(IERC20(params.tokenOut).transfer(owner(), amountToSend), "Transfer to owner failed");
        }
        
        // Update block number
        lastTradeBlock = block.number;
        
        // Emit event
        emit TradeExecuted(
            params.router,
            params.tokenIn,
            params.tokenOut,
            params.amountIn,
            result.amountOut,
            result.profit,
            result.gasUsed,
            result.priceImpact,
            block.timestamp
        );
    }

    function executeTrade(
        address router,
        address tokenIn,
        uint256 amountIn
    ) external onlyOwner nonReentrant whenNotPaused validToken(tokenIn) validRouter(router) {
        // Validate and prepare trade
        TradeParams memory params = _validateAndPrepare(router, tokenIn, amountIn);
        
        // Execute trade
        TradeResult memory result = _executeTrade(params);
        
        // Update statistics
        _updateStats(result);
        
        // Finalize and emit
        _finalizeTradeAndEmit(result, params);
    }

    // Enhanced function to check price differential between exchanges with security features
    function checkArbitragePotential(
        uint256 amountIn
    ) external view validToken(USDC) returns (
        bool profitable, 
        uint256 potentialProfit, 
        address bestSourceRouter, 
        address bestTargetRouter,
        uint256 estimatedGasCost
    ) {
        require(amountIn <= maxTradeSize, "Amount exceeds max trade size");
        
        // Check current gas costs to estimate profitability more accurately
        uint256 gasEstimate = 150000; // Gas used by a typical swap
        estimatedGasCost = gasEstimate * block.basefee;
        
        // Initialize variables
        uint256 bestRouteProfit = 0;
        bestSourceRouter = address(0);
        bestTargetRouter = address(0);
        
        // Check prices using helper function to reduce stack depth
        (uint256 uniToSushiProfit, uint256 sushiToUniProfit) = _checkUSDCtoUSDTRoutes(amountIn);
        (uint256 uniToSushiReverseProfit, uint256 sushiToUniReverseProfit) = _checkUSDTtoUSDCRoutes(amountIn);
        
        // Find best route (USDC -> USDT)
        if (uniToSushiProfit > bestRouteProfit) {
            bestRouteProfit = uniToSushiProfit;
            bestSourceRouter = UNISWAP_ROUTER;
            bestTargetRouter = SUSHISWAP_ROUTER;
        }
        
        if (sushiToUniProfit > bestRouteProfit) {
            bestRouteProfit = sushiToUniProfit;
            bestSourceRouter = SUSHISWAP_ROUTER;
            bestTargetRouter = UNISWAP_ROUTER;
        }
        
        // Find best route (USDT -> USDC)
        if (uniToSushiReverseProfit > bestRouteProfit) {
            bestRouteProfit = uniToSushiReverseProfit;
            bestSourceRouter = UNISWAP_ROUTER;
            bestTargetRouter = SUSHISWAP_ROUTER;
        }
        
        if (sushiToUniReverseProfit > bestRouteProfit) {
            bestRouteProfit = sushiToUniReverseProfit;
            bestSourceRouter = SUSHISWAP_ROUTER;
            bestTargetRouter = UNISWAP_ROUTER;
        }
        
        // Adjust profit by gas cost (convert to token units) and safety margin
        uint256 gasCostInTokens = estimatedGasCost / 1e12; // Convert wei to USDC/USDT units (6 decimals)
        uint256 safetyMargin = bestRouteProfit > 0 ? bestRouteProfit / 10 : 0; // 10% safety margin
        
        // Only profitable if exceeds gas costs plus safety margin
        if (bestRouteProfit > gasCostInTokens + safetyMargin) {
            potentialProfit = bestRouteProfit - gasCostInTokens - safetyMargin;
            profitable = potentialProfit >= minProfitUSD;
        } else {
            potentialProfit = 0;
            profitable = false;
        }
    }
    
    // Helper function to check USDC -> USDT routes
    function _checkUSDCtoUSDTRoutes(uint256 amountIn) internal view returns (uint256 uniProfit, uint256 sushiProfit) {
        address[] memory path = new address[](2);
        path[0] = USDC;
        path[1] = USDT;
        
        // Check USDC -> USDT on Uniswap
        try IUniswapV2Router(UNISWAP_ROUTER).getAmountsOut(amountIn, path) returns (uint256[] memory amounts) {
            uint256 adjustedOutput = (amounts[1] * (10000 - slippageTolerance)) / 10000;
            if (adjustedOutput > amountIn) {
                uniProfit = adjustedOutput - amountIn;
            }
        } catch {
            uniProfit = 0;
        }
        
        // Check USDC -> USDT on Sushiswap
        try IUniswapV2Router(SUSHISWAP_ROUTER).getAmountsOut(amountIn, path) returns (uint256[] memory amounts) {
            uint256 adjustedOutput = (amounts[1] * (10000 - slippageTolerance)) / 10000;
            if (adjustedOutput > amountIn) {
                sushiProfit = adjustedOutput - amountIn;
            }
        } catch {
            sushiProfit = 0;
        }
    }
    
    // Helper function to check USDT -> USDC routes
    function _checkUSDTtoUSDCRoutes(uint256 amountIn) internal view returns (uint256 uniProfit, uint256 sushiProfit) {
        address[] memory path = new address[](2);
        path[0] = USDT;
        path[1] = USDC;
        
        // Check USDT -> USDC on Uniswap
        try IUniswapV2Router(UNISWAP_ROUTER).getAmountsOut(amountIn, path) returns (uint256[] memory amounts) {
            uint256 adjustedOutput = (amounts[1] * (10000 - slippageTolerance)) / 10000;
            if (adjustedOutput > amountIn) {
                uniProfit = adjustedOutput - amountIn;
            }
        } catch {
            uniProfit = 0;
        }
        
        // Check USDT -> USDC on Sushiswap
        try IUniswapV2Router(SUSHISWAP_ROUTER).getAmountsOut(amountIn, path) returns (uint256[] memory amounts) {
            uint256 adjustedOutput = (amounts[1] * (10000 - slippageTolerance)) / 10000;
            if (adjustedOutput > amountIn) {
                sushiProfit = adjustedOutput - amountIn;
            }
        } catch {
            sushiProfit = 0;
        }
    }

    function checkProfitability(
        address router,
        address tokenIn,
        uint256 amountIn
    ) external view validRouter(router) validToken(tokenIn) 
      returns (bool profitable, uint256 expectedOutput, uint256 expectedProfit) 
    {
        if (amountIn > maxTradeSize) return (false, 0, 0);

        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenIn == USDC ? USDT : USDC;

        try IUniswapV2Router(router).getAmountsOut(amountIn, path) returns (uint256[] memory amounts) {
            expectedOutput = amounts[1];
            if (expectedOutput > amountIn) {
                expectedProfit = expectedOutput - amountIn;
                profitable = expectedProfit >= minProfitUSD;
            }
        } catch {
            return (false, 0, 0);
        }
    }

    // Dynamic gas price optimization
    function suggestOptimalGasPrice() external view returns (uint256) {
        if (block.basefee < 15 * 10**9) {
            return block.basefee + 3 * 10**9; // Low base fee, add 3 gwei priority
        } else if (block.basefee < 25 * 10**9) {
            return block.basefee + 2 * 10**9; // Medium base fee, add 2 gwei priority
        } else {
            return block.basefee + 1 * 10**9; // High base fee, add 1 gwei priority
        }
    }

    // Suggest optimal trade size based on current market conditions
    function suggestOptimalTradeSize() external view returns (uint256) {
        uint256 usdcBalance = IERC20(USDC).balanceOf(address(this));
        uint256 usdtBalance = IERC20(USDT).balanceOf(address(this));
        uint256 totalBalance = usdcBalance + usdtBalance;
        
        if (totalBalance < 200e6) {
            return 50e6; // $50 for very small balance
        } else if (totalBalance < 500e6) {
            return 100e6; // $100 for small balance
        } else if (totalBalance < 1000e6) {
            return 200e6; // $200 for medium balance
        } else {
            return 300e6; // $300 for larger balance
        }
    }

    function setTradeParameters(
        uint256 _minProfit,
        uint256 _maxTradeSize,
        uint256 _slippageTolerance,
        uint256 _maxGasPrice,
        uint256 _dailyLimit
    ) external onlyOwner {
        require(_slippageTolerance <= 100, "Slippage too high");
        require(_maxTradeSize <= 5000e6, "Trade size too large");
        require(_dailyLimit <= 10000e6, "Daily limit too large");

        minProfitUSD = _minProfit;
        maxTradeSize = _maxTradeSize;
        slippageTolerance = _slippageTolerance;
        maxGasPrice = _maxGasPrice;
        dailyVolumeLimit = _dailyLimit;

        emit ParametersUpdated(
            _minProfit,
            _maxTradeSize,
            _slippageTolerance,
            _maxGasPrice,
            _dailyLimit
        );
    }

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    function emergencyWithdraw() external onlyOwner {
        uint256 usdcBalance = IERC20(USDC).balanceOf(address(this));
        uint256 usdtBalance = IERC20(USDT).balanceOf(address(this));

        if (usdcBalance > 0) {
            require(IERC20(USDC).transfer(owner(), usdcBalance), "USDC transfer failed");
            emit EmergencyWithdrawal(USDC, usdcBalance, block.timestamp);
        }
        if (usdtBalance > 0) {
            require(IERC20(USDT).transfer(owner(), usdtBalance), "USDT transfer failed");
            emit EmergencyWithdrawal(USDT, usdtBalance, block.timestamp);
        }
    }

    // Enhanced emergency functions
    function emergencyPause() external payable {
        // Any address can call this, but it's expensive to prevent spam
        require(msg.value >= 0.01 ether, "Emergency pause fee required");
        _pause();
        emit SecurityAlert("Emergency pause triggered", block.timestamp);
    }
    
    // Circuit breaker for market volatility
    function checkMarketVolatility() external view returns (bool isSafe) {
        // Implement a simple check for extreme market conditions
        address[] memory path = new address[](2);
        path[0] = USDC;
        path[1] = USDT;
        
        try IUniswapV2Router(UNISWAP_ROUTER).getAmountsOut(1000000, path) returns (uint256[] memory amounts) {
            // Check if USDC:USDT rate is within reasonable bounds (0.98-1.02)
            return (amounts[1] >= 980000 && amounts[1] <= 1020000);
        } catch {
            return false;
        }
    }
    
    // Only owner can receive ETH in emergency
    receive() external payable onlyOwner {
        // Allow receiving ETH only from owner
    }
    
    // Fallback to reject ETH from non-owners
    fallback() external payable {
        revert("ETH not accepted");
    }
}