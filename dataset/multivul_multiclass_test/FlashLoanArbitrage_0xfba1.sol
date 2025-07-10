// SPDX-License-Identifier: MIT
pragma solidity 0.6.12;

// Flattened version of OpenZeppelin contracts

/**
 * @dev Interface of the ERC20 standard as defined in the EIP.
 */
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 */
abstract contract Context {
    function _msgSender() internal view virtual returns (address payable) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes memory) {
        this; // silence state mutability warning without generating bytecode - see https://github.com/ethereum/solidity/issues/2691
        return msg.data;
    }
}

contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () internal {
        address msgSender = _msgSender();
        _owner = msgSender;
        emit OwnershipTransferred(address(0), msgSender);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

// Aave and DEX interfaces
interface ILendingPool {
    function flashLoan(
        address receiverAddress,
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata modes,
        address onBehalfOf,
        bytes calldata params,
        uint16 referralCode
    ) external;
}

interface ILendingPoolAddressesProvider {
    function getLendingPool() external view returns (address);
}

interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    
    function getAmountsOut(
        uint amountIn, 
        address[] memory path
    ) external view returns (uint[] memory amounts);
}

/**
 * @title FlashLoanArbitrage
 * @dev Gas-optimized contract for cross-DEX arbitrage using Aave flash loans
 */
contract FlashLoanArbitrage is Ownable {
    // Reference to the LendingPool contract
    ILendingPool public immutable LENDING_POOL; // immutable saves gas
    
    // Track whether a flash loan is being executed
    bool private _inFlashLoan;
    
    // Define structs in memory to save gas
    struct ArbitrageParams {
        address tokenIn;
        address tokenOut;
        address buyRouter;
        address sellRouter;
        uint256 flashLoanAmount;
        uint256 minProfit;
    }
    
    // Use this for the current trade parameters during flash loan
    ArbitrageParams private _currentTrade;
    
    // Events to track arbitrage results
    event ArbitrageExecuted(address tokenIn, address tokenOut, uint256 profit);
    
    /**
     * @dev Constructor
     * @param _addressProvider Address of the Aave LendingPoolAddressesProvider
     */
    constructor(address _addressProvider) public {
        ILendingPoolAddressesProvider provider = ILendingPoolAddressesProvider(_addressProvider);
        LENDING_POOL = ILendingPool(provider.getLendingPool());
    }
    
    /**
     * @dev Function called by Aave after flash loan is provided
     */
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external returns (bool) {
        // Ensure this is being called by the lending pool during a flash loan
        require(msg.sender == address(LENDING_POOL), "Not called by lending pool");
        require(_inFlashLoan, "Not in flash loan");
        
        // Get the borrowed token and amount
        address tokenIn = assets[0];
        uint256 borrowedAmount = amounts[0];
        uint256 fee = premiums[0];
        
        // Get reference to current trade params
        ArbitrageParams memory tradeParams = _currentTrade;
        
        // Quick validation
        require(tokenIn == tradeParams.tokenIn, "Token mismatch");
        require(borrowedAmount == tradeParams.flashLoanAmount, "Amount mismatch");
        
        // Execute the arbitrage
        uint256 profit = _executeArbitrage(tradeParams);
        
        // Verify the profit covers the fee and meets minimum
        require(profit >= tradeParams.minProfit, "Arbitrage failed: profit less than minimum");
        require(profit >= fee, "Arbitrage failed: insufficient profit to cover fee");
        
        // Approve the LendingPool to withdraw the borrowed amount + fee
        IERC20(tokenIn).approve(address(LENDING_POOL), borrowedAmount + fee);
        
        // Reset the flash loan flag
        _inFlashLoan = false;
        
        return true;
    }
    
    /**
     * @dev Execute the arbitrage between DEXes
     */
    function _executeArbitrage(ArbitrageParams memory tradeParams) internal returns (uint256) {
        // Get token addresses
        address tokenIn = tradeParams.tokenIn;
        address tokenOut = tradeParams.tokenOut;
        
        // Initial balance
        uint256 initialBalance = IERC20(tokenIn).balanceOf(address(this));
        
        // Setup paths for swaps - reuse array to save gas
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        // 1. Approve and swap on first DEX
        IERC20(tokenIn).approve(tradeParams.buyRouter, tradeParams.flashLoanAmount);
        
        // Execute first swap
        IUniswapV2Router(tradeParams.buyRouter).swapExactTokensForTokens(
            tradeParams.flashLoanAmount,
            0, // Min output amount (we're trusting the calculation that led us here)
            path,
            address(this),
            block.timestamp + 60 // Shorter deadline saves gas
        );
        
        // Get tokenOut amount
        uint256 tokenOutAmount = IERC20(tokenOut).balanceOf(address(this));
        
        // 2. Approve and swap on second DEX
        IERC20(tokenOut).approve(tradeParams.sellRouter, tokenOutAmount);
        
        // Update path for second swap
        path[0] = tokenOut;
        path[1] = tokenIn;
        
        // Execute second swap
        IUniswapV2Router(tradeParams.sellRouter).swapExactTokensForTokens(
            tokenOutAmount,
            0, // Min output amount (we've already verified profitability)
            path,
            address(this),
            block.timestamp + 60
        );
        
        // Calculate profit
        uint256 finalBalance = IERC20(tokenIn).balanceOf(address(this));
        
        // Return profit (final - initial)
        uint256 profit = finalBalance > initialBalance ? finalBalance - initialBalance : 0;
        
        // Emit event for tracking
        emit ArbitrageExecuted(tokenIn, tokenOut, profit);
        
        return profit;
    }
    
    /**
     * @dev Initiate a flash loan arbitrage
     */
    function executeFlashLoanArbitrage(
        address tokenIn,
        address tokenOut,
        address buyRouter,
        address sellRouter,
        uint256 amount,
        uint256 minProfit
    ) external onlyOwner {
        // Set current trade parameters
        _currentTrade = ArbitrageParams({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            buyRouter: buyRouter,
            sellRouter: sellRouter,
            flashLoanAmount: amount,
            minProfit: minProfit
        });
        
        // Prepare flash loan parameters
        address[] memory assets = new address[](1);
        assets[0] = tokenIn;
        
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = amount;
        
        uint256[] memory modes = new uint256[](1);
        modes[0] = 0; // 0 = no debt
        
        // Set flash loan flag
        _inFlashLoan = true;
        
        // Execute flash loan
        LENDING_POOL.flashLoan(
            address(this),
            assets,
            amounts,
            modes,
            address(this),
            bytes("ARBITRAGE"),
            0 // No referral code
        );
    }
    
    /**
     * @dev Withdraw ERC20 tokens from the contract (rescue function)
     */
    function withdrawToken(address token) external onlyOwner {
        uint256 balance = IERC20(token).balanceOf(address(this));
        require(balance > 0, "No balance to withdraw");
        
        IERC20(token).transfer(owner(), balance);
    }
    
    /**
     * @dev Withdraw ETH from the contract (rescue function)
     */
    function withdrawETH() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH balance to withdraw");
        
        payable(owner()).transfer(balance);
    }

       /**
     * @dev Receive function to allow contract to receive ETH from DEX operations
     */
    receive() external payable {}
}