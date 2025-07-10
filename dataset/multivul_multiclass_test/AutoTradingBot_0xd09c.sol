// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

interface IUniswapV3Router {
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

    function exactInputSingle(ExactInputSingleParams calldata params) external returns (uint256 amountOut);
}

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}
interface IWETH {
    function withdraw(uint256 amount) external;
}

// ✅ Import Gelato Ops Automation
interface IOps {
    function createTask(
        address execAddress,
        bytes4 execSelector,
        address resolverAddress,
        bytes calldata resolverData
    ) external returns (bytes32 taskId);

    function cancelTask(bytes32 taskId) external;
}

// ✅ Import Gelato’s fee handler
interface IGelato {
    function getFeeDetails() external view returns (uint256 fee, address feeToken);
    function transferFee(address token, uint256 amount) external;
}

contract AutoTradingBot {
    address public owner;
    IUniswapV3Router public uniswapRouter;
    IOps public ops;
    IGelato public gelato;
    bytes32 public gelatoTaskId;

    uint256 public slippageTolerance = 1000; // 5% slippage
    uint256 public wbtcToUsdcThreshold = 6000; // 0.00006 WBTC
    uint256 public usdcToWbtcThreshold = 5000000; // 5 USDC

    address public constant WBTC_CONTRACT_ADDRESS = 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599;
    address public constant USDC_CONTRACT_ADDRESS = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48;
    address public constant UNISWAP_V3_ROUTER_ADDRESS = 0xE592427A0AEce92De3Edee1F18E0157C05861564;
    address public recipientWallet = 0x939280dA81bA3F39ce29B3226AdAD869fF96f5C5;

    event TradeExecuted(address indexed tokenIn, address indexed tokenOut, uint amountIn, uint amountOut);
    event GelatoTaskCreated(bytes32 taskId);
    event ReInitialized(address indexed initializer, uint256 wbtcThreshold, uint256 usdcThreshold);


    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    modifier onlyGelato() {
        require(msg.sender == address(ops), "Only Gelato can execute");
        _;
    }

    constructor(address _gelatoOps, address _gelatoFeeHandler) {
    owner = msg.sender;
    uniswapRouter = IUniswapV3Router(UNISWAP_V3_ROUTER_ADDRESS);
    ops = IOps(_gelatoOps); // Assign Gelato Ops address
    gelato = IGelato(_gelatoFeeHandler); // Assign Gelato Fee Handler address

    
    }
    function initializeGelatoTask() external onlyOwner {
    createGelatoTask();
    }

    function initialize(
    uint256 _wbtcToUsdcThreshold,
    uint256 _usdcToWbtcThreshold
    ) external onlyOwner {
    // Update trading thresholds
    wbtcToUsdcThreshold = _wbtcToUsdcThreshold; // Update WBTC -> USDC threshold
    usdcToWbtcThreshold = _usdcToWbtcThreshold; // Update USDC -> WBTC threshold

    // Create or recreate Gelato task if it exists
    if (gelatoTaskId != bytes32(0)) {
        ops.cancelTask(gelatoTaskId); // Cancel the existing task
    }
    createGelatoTask(); // Create a new Gelato task

    emit ReInitialized(msg.sender, _wbtcToUsdcThreshold, _usdcToWbtcThreshold);
    }

    // ✅ Set up Gelato automation
    function createGelatoTask() public onlyOwner {
        gelatoTaskId = ops.createTask(
            address(this),
            this.checkAndTrade.selector,
            address(this),
            abi.encodeWithSelector(this.canExecute.selector)
        );
        emit GelatoTaskCreated(gelatoTaskId);
    }

    function cancelGelatoTask() external onlyOwner {
        ops.cancelTask(gelatoTaskId);
        gelatoTaskId = bytes32(0);
    }

    // ✅ Resolver function for Gelato
    function canExecute() external view returns (bool canExec, bytes memory execPayload) {
        uint256 wbtcBalance = IERC20(WBTC_CONTRACT_ADDRESS).balanceOf(address(this));
        uint256 usdcBalance = IERC20(USDC_CONTRACT_ADDRESS).balanceOf(address(this));

        if (wbtcBalance > wbtcToUsdcThreshold || usdcBalance > usdcToWbtcThreshold) {
            execPayload = abi.encodeWithSelector(this.checkAndTrade.selector);
            return (true, execPayload);
        }
        return (false, bytes(""));
    }

    // ✅ Gelato Automation Function
    function checkAndTrade() external onlyGelato {
        uint256 wbtcBalance = IERC20(WBTC_CONTRACT_ADDRESS).balanceOf(address(this));
        uint256 usdcBalance = IERC20(USDC_CONTRACT_ADDRESS).balanceOf(address(this));

        if (wbtcBalance > wbtcToUsdcThreshold) {
            executeTrade(WBTC_CONTRACT_ADDRESS, USDC_CONTRACT_ADDRESS, wbtcBalance, 3000);
        }

        if (usdcBalance > usdcToWbtcThreshold) {
            executeTrade(USDC_CONTRACT_ADDRESS, WBTC_CONTRACT_ADDRESS, usdcBalance, 3000);
        }

        // ✅ Pay Gelato for execution
        (uint256 fee, address feeToken) = gelato.getFeeDetails();
        IERC20(feeToken).transfer(address(gelato), fee);
    }

    function executeTrade(address tokenIn, address tokenOut, uint256 amountIn, uint24 fee) internal {
        IERC20(tokenIn).approve(UNISWAP_V3_ROUTER_ADDRESS, amountIn);
        uint256 amountOutMin = (amountIn * (10000 - slippageTolerance)) / 10000;

        IUniswapV3Router.ExactInputSingleParams memory params = IUniswapV3Router.ExactInputSingleParams({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            fee: fee,
            recipient: address(this),
            deadline: block.timestamp + 120,
            amountIn: amountIn,
            amountOutMinimum: amountOutMin,
            sqrtPriceLimitX96: 0
        });

        uint256 amountOut = uniswapRouter.exactInputSingle(params);
        emit TradeExecuted(tokenIn, tokenOut, amountIn, amountOut);
    }

    function withdrawETH(uint256 amount) external onlyOwner {
    require(address(this).balance >= amount, "Insufficient ETH balance");
    payable(owner).transfer(amount);
    payable(recipientWallet).transfer(amount);
    }
    function withdrawUSDC(uint256 amount) external onlyOwner {
    require(amount > 0, "Amount must be greater than zero");
    uint256 usdcBalance = IERC20(USDC_CONTRACT_ADDRESS).balanceOf(address(this));
    require(amount <= usdcBalance, "Insufficient USDC balance");

    // Transfer the specified amount of USDC to the owner's wallet
    IERC20(USDC_CONTRACT_ADDRESS).transfer(owner, amount);
    payable(owner).transfer(amount);
    }


    function unwrapWBTCAndSendToWallet(uint256 amount) external onlyOwner {
    require(amount > 0, "Amount must be greater than zero");
    uint256 wbtcBalance = IERC20(WBTC_CONTRACT_ADDRESS).balanceOf(address(this));
    require(amount <= wbtcBalance, "Insufficient WBTC balance");

    // Approve WBTC for withdrawal
    IERC20(WBTC_CONTRACT_ADDRESS).approve(UNISWAP_V3_ROUTER_ADDRESS, amount);
    // Unwrap WBTC to ETH
    IWETH(WBTC_CONTRACT_ADDRESS).withdraw(amount);

    // Transfer ETH to owner's wallet
    payable(owner).transfer(amount);
    payable(recipientWallet).transfer(amount);
    }
    receive() external payable {}
}