// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

// Interfaces (unchanged)
interface IERC20 {
    function approve(address spender, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

interface ILinkToken is IERC20 {}

interface AggregatorV3Interface {
    function latestRoundData() external view returns (uint80, int256, uint256, uint256, uint80);
}

interface KeeperCompatibleInterface {
    function checkUpkeep(bytes calldata) external returns (bool, bytes memory);
    function performUpkeep(bytes calldata) external;
}

interface IProfitOracle {
    function getExpectedProfit(address tokenIn, address tokenOut, uint256 amount) external view returns (uint256);
}

interface IAavePoolAddressesProvider {
    function getPool() external view returns (address);
}

interface IAavePool {
    function flashLoanSimple(address receiverAddress, address asset, uint256 amount, bytes calldata params, uint16 referralCode) external;
}

interface IUniswapV3SwapRouter {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }
    function exactInputSingle(ExactInputSingleParams calldata params) external returns (uint256);
    function quoteExactInputSingle(address tokenIn, address tokenOut, uint24 fee, uint256 amountIn, uint160 sqrtPriceLimitX96) external view returns (uint256);
}

interface ISushiV2Router {
    function swapExactTokensForTokens(uint256 amountIn, uint256 amountOutMin, address[] calldata path, address to, uint256 deadline) external returns (uint256[] memory);
    function getAmountsOut(uint256 amountIn, address[] calldata path) external view returns (uint256[] memory);
}

interface ICurvePool {
    function exchange(int128 i, int128 j, uint256 dx, uint256 min_dy) external returns (uint256);
    function get_dy(int128 i, int128 j, uint256 dx) external view returns (uint256);
}

interface IBalancerVault {
    function swap(
        SingleSwap memory singleSwap,
        FundManagement memory funds,
        uint256 limit,
        uint256 deadline
    ) external returns (uint256);
    struct SingleSwap {
        bytes32 poolId;
        uint8 kind;
        address assetIn;
        address assetOut;
        uint256 amount;
        bytes userData;
    }
    struct FundManagement {
        address sender;
        bool fromInternalBalance;
        address payable recipient;
        bool toInternalBalance;
    }
}

// Base contracts
abstract contract ReentrancyGuard {
    bool private locked;
    modifier nonReentrant() {
        require(!locked, "Reentrancy guard triggered");
        locked = true;
        _;
        locked = false;
    }
}

abstract contract Pausable {
    bool private paused;
    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }
    function isPaused() public view returns (bool) { return paused; }
    function _pause() internal { paused = true; }
    function _unpause() internal { paused = false; }
}

// Main contract
contract AutoFlashLoanArbitrage is ReentrancyGuard, Pausable, KeeperCompatibleInterface {
    // Constant addresses
    address private immutable AAVE_POOL_ADDRESSES_PROVIDER = 0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e;
    address private immutable UNISWAP_V3_ROUTER = 0xE592427A0AEce92De3Edee1F18E0157C05861564;
    address private immutable SUSHISWAP_ROUTER = 0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F;
    address private immutable CURVE_3POOL = 0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7;
    address private immutable BALANCER_VAULT = 0xBA12222222228d8Ba445958a75a0704d566BF2C8;
    address private immutable WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address private immutable DAI = 0x6B175474E89094C44Da98b954EedeAC495271d0F;
    address private immutable USDC = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48;
    address private immutable LINK = 0x514910771AF9Ca656af840dff83E8264EcF986CA;
    address private constant FAST_GAS_FEED_ADDRESS = 0x169e633A2d1E6c10dd912952d37268d6368d37F8;

    AggregatorV3Interface public immutable fastGasFeed = AggregatorV3Interface(FAST_GAS_FEED_ADDRESS);
    address public owner;
    IAavePoolAddressesProvider public immutable poolAddressesProvider;
    IProfitOracle public profitOracle;

    // Configurable variables
    uint256 public minProfitThreshold = 0.01 ether;
    uint256 public checkInterval = 1 minutes;
    uint256 public lastCheckTimestamp;
    uint256 public gasLimitPerTx = 300_000;
    uint256 public minLinkBalance = 1 ether;
    uint256 public maxLoanAmount = 10000 ether;
    uint256 public maxSlippage = 50; // 0.5% in basis points
    uint256 public maxFeePerGas = 100 gwei;
    uint256 public timelock = 1 minutes;
    bool public emergencyMode;

    // Monitoring system
    struct EmergencyLog {
        string reason;
        uint256 timestamp;
    }
    EmergencyLog[] public emergencyLogs;
    address public alertReceiver;

    mapping(bytes32 => uint256) public lastExecution;
    address[] public dexRouters;
    mapping(address => bool) public isDexSupported;
    address[] public supportedTokens;

    bytes32 public constant BALANCER_POOL_ID = 0x0b09dea16768f0799065c475be02919403cb2a3500020000000000000000001a;

    // Events
    event ArbitrageExecuted(address indexed tokenIn, address indexed tokenOut, uint256 amountIn, uint256 profit, address dex);
    event UpkeepPerformed(address indexed token, uint256 amount, uint256 gasUsed);
    event EmergencyWithdraw(address indexed token, uint256 amount);
    event EmergencyPause(string reason);
    event EmergencyStatusChecked(bool isEmergency, uint256 logCount);
    event AlertSent(address indexed receiver, string message);
    event TokenAdded(address indexed token);
    event DexAdded(address indexed dex);
    event GasLimitAdjusted(uint256 newGasLimit);
    event LinkRequested(uint256 amount);
    event ETHWithdrawn(address indexed to, uint256 amount);

    constructor() {
        owner = msg.sender;
        poolAddressesProvider = IAavePoolAddressesProvider(AAVE_POOL_ADDRESSES_PROVIDER);
        alertReceiver = msg.sender;
        supportedTokens.push(WETH);
        supportedTokens.push(DAI);
        supportedTokens.push(USDC);
        dexRouters.push(UNISWAP_V3_ROUTER);
        dexRouters.push(SUSHISWAP_ROUTER);
        dexRouters.push(CURVE_3POOL);
        dexRouters.push(BALANCER_VAULT);
        isDexSupported[UNISWAP_V3_ROUTER] = true;
        isDexSupported[SUSHISWAP_ROUTER] = true;
        isDexSupported[CURVE_3POOL] = true;
        isDexSupported[BALANCER_VAULT] = true;
    }

    function checkUpkeep(bytes calldata) external override returns (bool upkeepNeeded, bytes memory performData) {
        if (block.timestamp < lastCheckTimestamp + checkInterval || isPaused() || emergencyMode) {
            return (false, bytes("Paused or emergency mode"));
        }

        if (ILinkToken(LINK).balanceOf(address(this)) < minLinkBalance) {
            requestLink(minLinkBalance - ILinkToken(LINK).balanceOf(address(this)));
            return (false, bytes("Insufficient LINK"));
        }

        adjustGasLimitDynamically();
        uint256 tokenLength = supportedTokens.length;

        for (uint256 i = 0; i < tokenLength; i++) {
            for (uint256 j = 0; j < tokenLength; j++) {
                if (i != j) {
                    address tokenIn = supportedTokens[i];
                    address tokenOut = supportedTokens[j];
                    uint256 amount = calculateOptimalLoanAmount(tokenIn);
                    (bool profitable, uint256 profit, address bestDex) = checkArbitrageProfit(tokenIn, tokenOut, amount);
                    if (profitable && profit >= minProfitThreshold) {
                        bytes32 key = keccak256(abi.encodePacked(tokenIn, tokenOut, amount, bestDex));
                        if (block.timestamp >= lastExecution[key] + timelock) {
                            return (true, abi.encode(tokenIn, tokenOut, amount, bestDex, block.timestamp));
                        }
                    }
                }
            }
        }
        return (false, bytes("No profitable opportunity"));
    }

    function performUpkeep(bytes calldata performData) external override whenNotPaused nonReentrant {
        require(!emergencyMode, "Emergency mode active");
        lastCheckTimestamp = block.timestamp;
        (address tokenIn, address tokenOut, uint256 amount, address bestDex,) = abi.decode(performData, (address, address, uint256, address, uint256));

        bytes32 key = keccak256(abi.encodePacked(tokenIn, tokenOut, amount, bestDex));
        require(block.timestamp >= lastExecution[key] + timelock, "Timelock active");
        lastExecution[key] = block.timestamp;

        (, int256 gasPrice,,,) = fastGasFeed.latestRoundData();
        uint256 totalFee = uint256(gasPrice) + (uint256(gasPrice) / 10);
        if (totalFee > maxFeePerGas) totalFee = maxFeePerGas;
        require(address(this).balance >= gasLimitPerTx * totalFee, "Insufficient ETH for gas");

        uint256 gasBefore = gasleft();
        try this.startFlashLoan{gas: gasLimitPerTx}(tokenIn, amount, bestDex) {
            emit UpkeepPerformed(tokenIn, amount, gasBefore - gasleft());
        } catch (bytes memory reason) {
            _pause();
            emergencyMode = true;
            emergencyLogs.push(EmergencyLog(string(reason), block.timestamp));
            emit EmergencyPause(string(reason));
            if (alertReceiver != address(0)) {
                emit AlertSent(alertReceiver, string(abi.encodePacked("Emergency: ", reason)));
            }
        }
    }

    function startFlashLoan(address token, uint256 amount, address dex) external nonReentrant {
        require(msg.sender == address(this) || msg.sender == owner, "Only contract or owner");
        IAavePool pool = IAavePool(poolAddressesProvider.getPool());
        bytes memory params = abi.encode(token, dex);
        pool.flashLoanSimple(address(this), token, amount, params, 0);
    }

    function executeOperation(address, uint256 amount, uint256 premium, address initiator, bytes calldata params)
        external
        nonReentrant
        returns (bool)
    {
        require(msg.sender == poolAddressesProvider.getPool(), "Caller must be Aave Pool");
        require(initiator == address(this), "Initiator must be this contract");

        (address tokenIn, address dex) = abi.decode(params, (address, address));
        uint256 amountOwing = amount + premium;

        require(IERC20(tokenIn).balanceOf(address(this)) >= amount, "Insufficient token balance");
        (uint256 profit, address tokenOut) = executeArbitrage(tokenIn, amount, amountOwing, dex);
        require(IERC20(tokenIn).approve(address(poolAddressesProvider.getPool()), amountOwing), "Approval failed");

        if (profit > 0) {
            if (tokenOut == address(0)) {
                (bool sent, ) = owner.call{value: profit}("");
                require(sent, "ETH profit transfer failed");
            } else {
                require(IERC20(tokenOut).transfer(owner, profit), "Token profit transfer failed");
            }
            emit ArbitrageExecuted(tokenIn, tokenOut, amount, profit, dex);
        }
        return true;
    }

    function executeArbitrage(address tokenIn, uint256 amount, uint256 amountOwing, address dex)
        internal
        returns (uint256 profit, address tokenOut)
    {
        require(IERC20(tokenIn).balanceOf(address(this)) >= amount, "Insufficient token balance");
        
        if (dex == UNISWAP_V3_ROUTER) {
            (profit, tokenOut) = tryExecuteDirectUni(tokenIn, amount, amountOwing);
        } else if (dex == SUSHISWAP_ROUTER) {
            (profit, tokenOut) = tryExecuteDirectSushi(tokenIn, amount, amountOwing);
        } else if (dex == CURVE_3POOL) {
            (profit, tokenOut) = tryExecuteCurve(tokenIn, amount, amountOwing);
        } else if (dex == BALANCER_VAULT) {
            (profit, tokenOut) = tryExecuteBalancer(tokenIn, amount, amountOwing);
        } else {
            revert("Unsupported DEX");
        }
    }

    function tryExecuteDirectUni(address tokenIn, uint256 amount, uint256 amountOwing)
        internal
        returns (uint256, address)
    {
        uint256 tokenLength = supportedTokens.length;
        IUniswapV3SwapRouter uniRouter = IUniswapV3SwapRouter(UNISWAP_V3_ROUTER);
        require(IERC20(tokenIn).approve(UNISWAP_V3_ROUTER, amount), "Uniswap approval failed");

        for (uint256 i = 0; i < tokenLength; i++) {
            if (supportedTokens[i] != tokenIn) {
                address tokenOut = supportedTokens[i];
                uint256 expectedOut;
                try uniRouter.quoteExactInputSingle(tokenIn, tokenOut, 3000, amount, 0) returns (uint256 out) {
                    expectedOut = out;
                } catch {
                    continue;
                }

                uint256 minAmountOut = expectedOut * (10000 - maxSlippage) / 10000;
                try uniRouter.exactInputSingle(IUniswapV3SwapRouter.ExactInputSingleParams({
                    tokenIn: tokenIn,
                    tokenOut: tokenOut,
                    fee: 3000,
                    recipient: address(this),
                    deadline: block.timestamp + 300,
                    amountIn: amount,
                    amountOutMinimum: minAmountOut > amountOwing ? minAmountOut : amountOwing,
                    sqrtPriceLimitX96: 0
                })) returns (uint256 amountOut) {
                    if (amountOut > amountOwing) return (amountOut - amountOwing, tokenOut);
                } catch {
                    continue;
                }
            }
        }
        return (0, address(0));
    }

    function tryExecuteDirectSushi(address tokenIn, uint256 amount, uint256 amountOwing)
        internal
        returns (uint256, address)
    {
        uint256 tokenLength = supportedTokens.length;
        ISushiV2Router sushiRouter = ISushiV2Router(SUSHISWAP_ROUTER);
        require(IERC20(tokenIn).approve(SUSHISWAP_ROUTER, amount), "Sushi approval failed");

        for (uint256 i = 0; i < tokenLength; i++) {
            if (supportedTokens[i] != tokenIn) {
                address tokenOut = supportedTokens[i];
                address[] memory path = new address[](2);
                path[0] = tokenIn;
                path[1] = tokenOut;

                uint256[] memory amounts;
                try sushiRouter.getAmountsOut(amount, path) returns (uint256[] memory out) {
                    amounts = out;
                } catch {
                    continue;
                }

                uint256 minAmountOut = amounts[1] * (10000 - maxSlippage) / 10000;
                try sushiRouter.swapExactTokensForTokens(amount, minAmountOut > amountOwing ? minAmountOut : amountOwing, path, address(this), block.timestamp + 300) returns (uint256[] memory swapAmounts) {
                    if (swapAmounts[1] > amountOwing) return (swapAmounts[1] - amountOwing, tokenOut);
                } catch {
                    continue;
                }
            }
        }
        return (0, address(0));
    }

    function tryExecuteCurve(address tokenIn, uint256 amount, uint256 amountOwing)
        internal
        returns (uint256, address)
    {
        uint256 tokenLength = supportedTokens.length;
        ICurvePool curvePool = ICurvePool(CURVE_3POOL);
        require(IERC20(tokenIn).approve(CURVE_3POOL, amount), "Curve approval failed");

        for (uint256 i = 0; i < tokenLength; i++) {
            if (supportedTokens[i] != tokenIn) {
                address tokenOut = supportedTokens[i];
                int128 inIndex = tokenIn == DAI ? int128(0) : (tokenIn == USDC ? int128(1) : int128(2));
                int128 outIndex = tokenOut == DAI ? int128(0) : (tokenOut == USDC ? int128(1) : int128(2));

                uint256 expectedOut;
                try curvePool.get_dy(inIndex, outIndex, amount) returns (uint256 dy) {
                    expectedOut = dy;
                } catch {
                    continue;
                }

                uint256 minAmountOut = expectedOut * (10000 - maxSlippage) / 10000;
                try curvePool.exchange(inIndex, outIndex, amount, minAmountOut > amountOwing ? minAmountOut : amountOwing) returns (uint256 amountOut) {
                    if (amountOut > amountOwing) return (amountOut - amountOwing, tokenOut);
                } catch {
                    continue;
                }
            }
        }
        return (0, address(0));
    }

    function tryExecuteBalancer(address tokenIn, uint256 amount, uint256 amountOwing)
        internal
        returns (uint256, address)
    {
        uint256 tokenLength = supportedTokens.length;
        IBalancerVault balancerVault = IBalancerVault(BALANCER_VAULT);
        require(IERC20(tokenIn).approve(BALANCER_VAULT, amount), "Balancer approval failed");

        for (uint256 i = 0; i < tokenLength; i++) {
            if (supportedTokens[i] != tokenIn) {
                address tokenOut = supportedTokens[i];
                IBalancerVault.SingleSwap memory singleSwap = IBalancerVault.SingleSwap({
                    poolId: BALANCER_POOL_ID,
                    kind: 0,
                    assetIn: tokenIn,
                    assetOut: tokenOut,
                    amount: amount,
                    userData: "0x"
                });
                IBalancerVault.FundManagement memory funds = IBalancerVault.FundManagement({
                    sender: address(this),
                    fromInternalBalance: false,
                    recipient: payable(address(this)),
                    toInternalBalance: false
                });

                uint256 expectedOut = getBalancerQuote(tokenIn, tokenOut, amount);
                uint256 minAmountOut = expectedOut * (10000 - maxSlippage) / 10000;
                try balancerVault.swap(singleSwap, funds, minAmountOut > amountOwing ? minAmountOut : amountOwing, block.timestamp + 300) returns (uint256 amountOut) {
                    if (amountOut > amountOwing) return (amountOut - amountOwing, tokenOut);
                } catch {
                    continue;
                }
            }
        }
        return (0, address(0));
    }

    function checkArbitrageProfit(address tokenIn, address tokenOut, uint256 amount)
        public
        view
        returns (bool profitable, uint256 bestProfit, address bestDex)
    {
        uint256 amountOwing = amount + (amount * 9 / 10000);
        if (address(profitOracle) != address(0)) {
            try profitOracle.getExpectedProfit(tokenIn, tokenOut, amount) returns (uint256 oracleProfit) {
                if (oracleProfit > minProfitThreshold) return (true, oracleProfit, UNISWAP_V3_ROUTER);
            } catch {}
        }

        (, int256 gasPrice,,,) = fastGasFeed.latestRoundData();
        uint256 totalFee = uint256(gasPrice) + (uint256(gasPrice) / 10);
        if (totalFee > maxFeePerGas) totalFee = maxFeePerGas;
        uint256 gasCost = gasLimitPerTx * totalFee;

        bestProfit = 0;
        bestDex = address(0);
        uint256 dexLength = dexRouters.length;

        for (uint256 i = 0; i < dexLength; i++) {
            uint256 amountOut = getAmountOut(dexRouters[i], tokenIn, tokenOut, amount);
            if (amountOut > amountOwing + gasCost) {
                uint256 profit = amountOut - amountOwing - gasCost;
                if (profit > bestProfit) {
                    bestProfit = profit;
                    bestDex = dexRouters[i];
                }
            }
        }
        profitable = bestProfit > minProfitThreshold;
    }

    function getAmountOut(address dex, address tokenIn, address tokenOut, uint256 amount)
        internal
        view
        returns (uint256)
    {
        if (dex == UNISWAP_V3_ROUTER) {
            try IUniswapV3SwapRouter(UNISWAP_V3_ROUTER).quoteExactInputSingle(tokenIn, tokenOut, 3000, amount, 0) returns (uint256 out) {
                return out;
            } catch {
                return 0;
            }
        } else if (dex == SUSHISWAP_ROUTER) {
            address[] memory path = new address[](2);
            path[0] = tokenIn;
            path[1] = tokenOut;
            try ISushiV2Router(SUSHISWAP_ROUTER).getAmountsOut(amount, path) returns (uint256[] memory amounts) {
                return amounts[1];
            } catch {
                return 0;
            }
        } else if (dex == CURVE_3POOL && (tokenIn == DAI || tokenOut == DAI || tokenIn == USDC || tokenOut == USDC)) {
            int128 tokenInIndex = tokenIn == DAI ? int128(0) : (tokenIn == USDC ? int128(1) : int128(2));
            int128 tokenOutIndex = tokenOut == DAI ? int128(0) : (tokenOut == USDC ? int128(1) : int128(2));
            try ICurvePool(CURVE_3POOL).get_dy(tokenInIndex, tokenOutIndex, amount) returns (uint256 dy) {
                return dy;
            } catch {
                return 0;
            }
        } else if (dex == BALANCER_VAULT) {
            return getBalancerQuote(tokenIn, tokenOut, amount);
        }
        return 0;
    }

    function calculateOptimalLoanAmount(address token) internal view returns (uint256) {
        uint256 balance = IERC20(token).balanceOf(poolAddressesProvider.getPool());
        unchecked {
            uint256 optimalAmount = balance / 10;
            return optimalAmount > maxLoanAmount ? maxLoanAmount : optimalAmount;
        }
    }

    function getBalancerQuote(address tokenIn, address tokenOut, uint256 amount) internal view returns (uint256) {
        IBalancerVault.SingleSwap memory singleSwap = IBalancerVault.SingleSwap({
            poolId: BALANCER_POOL_ID,
            kind: 0,
            assetIn: tokenIn,
            assetOut: tokenOut,
            amount: amount,
            userData: "0x"
        });
        IBalancerVault.FundManagement memory funds = IBalancerVault.FundManagement({
            sender: address(this),
            fromInternalBalance: false,
            recipient: payable(address(this)),
            toInternalBalance: false
        });
        try this.simulateBalancerSwap(singleSwap, funds) returns (uint256 amountOut) {
            return amountOut;
        } catch {
            return 0;
        }
    }

    function simulateBalancerSwap(IBalancerVault.SingleSwap memory singleSwap, IBalancerVault.FundManagement memory funds)
        external
        view
        returns (uint256)
    {
        (bool success, bytes memory data) = address(BALANCER_VAULT).staticcall(
            abi.encodeWithSelector(IBalancerVault.swap.selector, singleSwap, funds, 0, block.timestamp + 300)
        );
        return success ? abi.decode(data, (uint256)) : 0;
    }

    function requestLink(uint256 amount) internal {
        emit LinkRequested(amount);
        try ILinkToken(LINK).transferFrom(owner, address(this), amount) returns (bool success) {
            if (!success) {
                _pause();
                emergencyMode = true;
                emergencyLogs.push(EmergencyLog("LINK request failed", block.timestamp));
                emit EmergencyPause("LINK request failed");
                if (alertReceiver != address(0)) {
                    emit AlertSent(alertReceiver, "Emergency: LINK request failed");
                }
            }
        } catch {
            _pause();
            emergencyMode = true;
            emergencyLogs.push(EmergencyLog("LINK request reverted", block.timestamp));
            emit EmergencyPause("LINK request reverted");
            if (alertReceiver != address(0)) {
                emit AlertSent(alertReceiver, "Emergency: LINK request reverted");
            }
        }
    }

    function adjustGasLimitDynamically() internal {
        (, int256 gasPrice,, uint256 updatedAt,) = fastGasFeed.latestRoundData();
        if (updatedAt < block.timestamp - 1 hours) {
            gasLimitPerTx = 300_000;
            emit GasLimitAdjusted(gasLimitPerTx);
            return;
        }
        uint256 estimatedGasPrice = uint256(gasPrice) / 1 gwei;
        uint256 newGasLimit = (estimatedGasPrice < 50) ? 400_000 : (estimatedGasPrice < 100) ? 300_000 : 200_000;
        if (newGasLimit != gasLimitPerTx) {
            gasLimitPerTx = newGasLimit;
            emit GasLimitAdjusted(newGasLimit);
        }
    }

    function checkEmergencyStatus() external returns (bool) {
        bool isEmergency = emergencyMode || isPaused();
        emit EmergencyStatusChecked(isEmergency, emergencyLogs.length);
        
        if (isEmergency && alertReceiver != address(0)) {
            emit AlertSent(alertReceiver, "Contract in emergency or paused state");
        }
        return isEmergency;
    }

    function setAlertReceiver(address _receiver) external onlyOwner {
        require(_receiver != address(0), "Invalid receiver address");
        alertReceiver = _receiver;
    }

    function getEmergencyLogs(uint256 index) external view returns (string memory reason, uint256 timestamp) {
        require(index < emergencyLogs.length, "Invalid index");
        EmergencyLog memory log = emergencyLogs[index];
        return (log.reason, log.timestamp);
    }

    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        require(emergencyMode, "Emergency mode not active");
        if (token == address(0)) {
            require(address(this).balance >= amount, "Insufficient ETH");
            (bool sent, ) = owner.call{value: amount}("");
            require(sent, "ETH withdrawal failed");
        } else {
            require(IERC20(token).balanceOf(address(this)) >= amount, "Insufficient token balance");
            require(IERC20(token).transfer(owner, amount), "Token withdrawal failed");
        }
        emergencyLogs.push(EmergencyLog("Funds withdrawn in emergency", block.timestamp));
        emit EmergencyWithdraw(token, amount);
        if (alertReceiver != address(0)) {
            emit AlertSent(alertReceiver, "Emergency withdrawal executed");
        }
    }

    // New function to withdraw ETH
    function withdrawETH(address payable to, uint256 amount) external onlyOwner nonReentrant {
        require(to != address(0), "Invalid recipient address");
        require(address(this).balance >= amount, "Insufficient ETH balance");
        (bool sent, ) = to.call{value: amount}("");
        require(sent, "ETH withdrawal failed");
        emit ETHWithdrawn(to, amount);
    }

    // Management functions
    function addToken(address token) external onlyOwner {
        require(token != address(0), "Invalid token address");
        supportedTokens.push(token);
        emit TokenAdded(token);
    }

    function addDex(address dex) external onlyOwner {
        require(dex != address(0), "Invalid DEX address");
        dexRouters.push(dex);
        isDexSupported[dex] = true;
        emit DexAdded(dex);
    }

    function updateParameters(
        uint256 _minProfitThreshold,
        uint256 _checkInterval,
        uint256 _gasLimit,
        uint256 _minLinkBalance,
        uint256 _maxLoanAmount,
        uint256 _maxSlippage,
        uint256 _timelock
    ) external onlyOwner {
        minProfitThreshold = _minProfitThreshold;
        checkInterval = _checkInterval;
        gasLimitPerTx = _gasLimit;
        minLinkBalance = _minLinkBalance;
        maxLoanAmount = _maxLoanAmount;
        maxSlippage = _maxSlippage;
        timelock = _timelock;
        emit GasLimitAdjusted(_gasLimit);
    }

    function setProfitOracle(address _profitOracle) external onlyOwner {
        require(_profitOracle != address(0), "Invalid oracle address");
        profitOracle = IProfitOracle(_profitOracle);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    receive() external payable {}
}