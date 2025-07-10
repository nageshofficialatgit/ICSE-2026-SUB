// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;



/*
    Rafael Baena Alvarez  
    Founder  
    FFOLLOWME OÜ (16785919)  
    Harju maakond, Tallinn, Lasnamäe linnaosa, Lõõtsa tn 5 // Sepapaja tn 4, 11415  
    Tallinn (Estonia)  
    LinkedIn: https://www.linkedin.com/in/rafael-baena-828b06181/  
    Email: info@ffollowme.com  
    GitHub: https://github.com/followme-dot  
*/





// File: contracts/IERC20.sol


pragma solidity ^0.8.20;

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

// File: contracts/Ownable.sol


pragma solidity ^0.8.20;

contract Ownable {
    address public owner;
    address public pendingOwner;
    
    // Mapeo para los roles
    mapping(address => bool) public isAdmin;
    mapping(address => bool) public isAuditor;
    
    // Eventos
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event OwnershipTransferStarted(address indexed newOwner);
    event AdminAdded(address indexed admin);
    event AdminRemoved(address indexed admin);
    event AuditorAdded(address indexed auditor);
    event AuditorRemoved(address indexed auditor);
    
    // Modificadores
    modifier onlyOwner() {
        require(msg.sender == owner, "Ownable: caller is not the owner");
        _;
    }

    modifier onlyAdmin() {
        require(isAdmin[msg.sender], "Ownable: caller is not an admin");
        _;
    }

    modifier onlyAuditor() {
        require(isAuditor[msg.sender], "Ownable: caller is not an auditor");
        _;
    }

    modifier notOwner(address newOwner) {
        require(newOwner != owner, "Ownable: new owner is the current owner");
        _;
    }

    modifier hasPendingOwner() {
        require(pendingOwner != address(0), "Ownable: no pending owner");
        _;
    }

    // Constructor
    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }

    // Funciones de administración de la propiedad
    function startOwnershipTransfer(address newOwner) external onlyOwner notOwner(newOwner) {
        pendingOwner = newOwner;
        emit OwnershipTransferStarted(newOwner);
    }

    function completeOwnershipTransfer() external hasPendingOwner {
        require(msg.sender == pendingOwner, "Ownable: only the pending owner can complete the transfer");
        address oldOwner = owner;
        owner = pendingOwner;
        pendingOwner = address(0);
        emit OwnershipTransferred(oldOwner, owner);
    }

    function cancelOwnershipTransfer() external onlyOwner hasPendingOwner {
        pendingOwner = address(0);
    }

    // Funciones de rol de Administrador
    function addAdmin(address admin) external onlyOwner {
        require(admin != address(0), "Ownable: invalid address");
        require(!isAdmin[admin], "Ownable: address is already an admin");
        isAdmin[admin] = true;
        emit AdminAdded(admin);
    }

    function removeAdmin(address admin) external onlyOwner {
        require(isAdmin[admin], "Ownable: address is not an admin");
        isAdmin[admin] = false;
        emit AdminRemoved(admin);
    }

    // Funciones de rol de Auditor
    function addAuditor(address auditor) external onlyOwner {
        require(auditor != address(0), "Ownable: invalid address");
        require(!isAuditor[auditor], "Ownable: address is already an auditor");
        isAuditor[auditor] = true;
        emit AuditorAdded(auditor);
    }

    function removeAuditor(address auditor) external onlyOwner {
        require(isAuditor[auditor], "Ownable: address is not an auditor");
        isAuditor[auditor] = false;
        emit AuditorRemoved(auditor);
    }

    // Función de transferencia de propiedad a una nueva dirección
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    // Renunciar a la propiedad (solo si no hay una transferencia pendiente)
    function renounceOwnership() external onlyOwner {
        require(pendingOwner == address(0), "Ownable: cannot renounce ownership while transfer is pending");
        address previousOwner = owner;
        owner = address(0);
        emit OwnershipTransferred(previousOwner, address(0));
    }
}

// File: contracts/ReentrancyGuard.sol


pragma solidity ^0.8.20;

abstract contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
    }

    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }
}

// File: contracts/IUniswapV2Router02.sol


pragma solidity ^0.8.20;

// This interface defines the functions of the UniswapV2Router02 contract.
// The UniswapV2Router02 contract is used for swapping tokens, adding/removing liquidity, and interacting with the Uniswap V2 protocol.
interface IUniswapV2Router02 {
    
    // Function to get the address of the Uniswap factory.
    function factory() external pure returns (address);

    // Function to get the address of the Wrapped Ether (WETH) token used by Uniswap.
    function WETH() external pure returns (address);
    function getWETH() external pure returns (address); // Alternative function name

    // Function to add liquidity to a token pair on Uniswap.
    function addLiquidity(
        address tokenA,
        address tokenB,
        uint amountA,
        uint amountB,
        uint amountAMin,
        uint amountBMin,
        address to,
        uint deadline
    ) external returns (uint amountAActual, uint amountBActual, uint liquidity);

    // Function to add liquidity with ETH and a token.
    function addLiquidityETH(
        address token,
        uint amountToken,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountTokenActual, uint amountETHActual, uint liquidity);

    // Function to remove liquidity from a token pair on Uniswap.
    function removeLiquidity(
        address tokenA,
        address tokenB,
        uint liquidity,
        uint amountAMin,
        uint amountBMin,
        address to,
        uint deadline
    ) external returns (uint amountA, uint amountB);

    // Function to remove liquidity with ETH and a token.
    function removeLiquidityETH(
        address token,
        uint liquidity,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external returns (uint amountToken, uint amountETH);

    // Function to swap an exact amount of tokens for another token.
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    // Function to swap tokens for an exact amount of another token.
    function swapTokensForExactTokens(
        uint amountOut,
        uint amountInMax,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    // Function to swap ETH for an exact amount of another token.
    function swapExactETHForTokens(
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable returns (uint[] memory amounts);

    // Function to swap tokens for an exact amount of ETH.
    function swapTokensForExactETH(
        uint amountOut,
        uint amountInMax,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    // Function to swap an exact amount of tokens for ETH.
    function swapExactTokensForETH(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    // Function to swap ETH for an exact amount of tokens.
    function swapETHForExactTokens(
        uint amountOut,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable returns (uint[] memory amounts);
}

// File: contracts/IUniswapV2Factory.sol


pragma solidity ^0.8.20;

// This interface defines the functions of the UniswapV2Factory contract. 
// The UniswapV2Factory is used to create and manage liquidity pairs for token swaps in the Uniswap V2 protocol.
interface IUniswapV2Factory {

    // Function to get the address of the liquidity pool (pair) for two tokens.
    // `tokenA` and `tokenB` are the addresses of the two tokens for which the liquidity pair is sought.
    // This function allows users to find the existing liquidity pool for a token pair (if it exists).
    // Returns the address of the pair contract for the tokens, or the zero address if no pair exists.
    function getPair(address tokenA, address tokenB) external view returns (address pair);

    // Function to create a liquidity pair for two tokens.
    // `tokenA` and `tokenB` are the two tokens for which the liquidity pair is being created.
    // This function is typically called by the factory to create a new pair of tokens for trading.
    // The function returns the address of the newly created pair contract.
    function createPair(address tokenA, address tokenB) external returns (address pair);

    // Function to get the address of the owner of the factory contract.
    // This function returns the address of the person or entity that owns and controls the factory contract.
    // The owner is responsible for managing and deploying liquidity pairs.
    function owner() external view returns (address);
}
// File: contracts/FlashLoan.sol


pragma solidity ^0.8.20;






contract FlashLoan is Ownable, ReentrancyGuard {
    address public liquidityPool;
    uint256 public loanFeePercentage = 3; // 3% fee for the loan
    uint256 public maxLoanAmount = 51000000 * 10**10; // Max loan amount
     address[] private path;  // Path for token swaps
    // Uniswap Router for liquidity borrowing
    IUniswapV2Router02 public uniswapRouter;
    
    mapping(address => bool) public approvedTokens;
    mapping(address => uint256) public loanBalances;

    event LoanTaken(address indexed borrower, uint256 amount, address token);
    event LoanRepaid(address indexed borrower, uint256 amount, address token, uint256 fee);
    event LoanFeeChanged(uint256 newFee);
    event TokenApproved(address token);
    event TokenRemoved(address token);

    modifier onlyLiquidityPool() {
        require(msg.sender == liquidityPool, "Not liquidity pool");
        _;
    }

    modifier onlyApprovedToken(address token) {
        require(approvedTokens[token], "Token not approved");
        _;
    }

    constructor(address _liquidityPool, address _uniswapRouter) {
        liquidityPool = _liquidityPool;
        uniswapRouter = IUniswapV2Router02(_uniswapRouter);
    }

    function setLiquidityPool(address _liquidityPool) external onlyOwner {
        liquidityPool = _liquidityPool;
    }

    // Set the loan fee percentage
    function setLoanFeePercentage(uint256 _fee) external onlyOwner {
        loanFeePercentage = _fee;
        emit LoanFeeChanged(_fee);
    }

    // Add an approved token for flash loan
    function approveToken(address token) external onlyOwner {
        approvedTokens[token] = true;
        emit TokenApproved(token);
    }

    // Remove an approved token for flash loan
    function removeToken(address token) external onlyOwner {
        approvedTokens[token] = false;
        emit TokenRemoved(token);
    }

    // Multi-token flash loan
    function multiTokenFlashLoan(address[] calldata tokens, uint256[] calldata amounts) external nonReentrant {
        require(tokens.length == amounts.length, "Tokens and amounts length mismatch");

        // Total fee to be repaid for each token
        uint256 totalFee;
        for (uint256 i = 0; i < tokens.length; i++) {
            require(approvedTokens[tokens[i]], "Token not approved");
            require(amounts[i] <= maxLoanAmount, "Loan exceeds max limit");
            require(amounts[i] > 0, "Invalid loan amount");

            // Transfer loaned tokens to the borrower
            IERC20(tokens[i]).transfer(msg.sender, amounts[i]);

            totalFee += (amounts[i] * loanFeePercentage) / 100;
            loanBalances[msg.sender] += amounts[i]; // Record the loan balance
            emit LoanTaken(msg.sender, amounts[i], tokens[i]);
        }

        // Borrow liquidity from Uniswap if needed
        borrowFromUniswapIfNeeded(tokens, amounts);

        // Execute borrower logic here (do whatever you want with the loan)

        // Repayment process for multi-token
        repayLoan(tokens, amounts, totalFee);
    }

    // Function to borrow liquidity from Uniswap if it's not available locally
function borrowFromUniswapIfNeeded(address[] calldata tokens, uint256[] calldata amounts) internal {
    for (uint256 i = 0; i < tokens.length; i++) {
        // Check if the contract has enough balance of the token
        if (IERC20(tokens[i]).balanceOf(address(this)) < amounts[i]) {
            
            // Declare the path array (token -> WETH)
            
            path[0] = tokens[i];  // Token to swap
            path[1] = uniswapRouter.WETH();  // WETH (Wrapped ETH), obtained dynamically

            // Calculate the missing amount needed
            uint256 amountIn = amounts[i] - IERC20(tokens[i]).balanceOf(address(this));

            // Ensure the path has exactly 2 elements
            require(path.length == 2, "Path should have two elements");

            // Approve Uniswap Router to spend the token
            IERC20(tokens[i]).approve(address(uniswapRouter), amountIn);

            // Swap tokens for ETH
            uniswapRouter.swapExactTokensForETH(
                amountIn, // The amount of tokens to swap
                0,        // Minimum ETH received (set to 0 for now, consider slippage protection)
                path,     // Swap path
                address(this), // Recipient of ETH
                block.timestamp // Deadline for the swap
            );
        }
    }
}

    // Function to repay multi-token loan
    function repayLoan(address[] calldata tokens, uint256[] calldata amounts, uint256 /*totalFee*/) internal {
        uint256 feePaid;
        for (uint256 i = 0; i < tokens.length; i++) {
            uint256 repaymentAmount = amounts[i] + ((amounts[i] * loanFeePercentage) / 100);
            feePaid += (amounts[i] * loanFeePercentage) / 100;

            // Repay the loan
            IERC20(tokens[i]).transferFrom(msg.sender, address(this), repaymentAmount);
            emit LoanRepaid(msg.sender, repaymentAmount, tokens[i], feePaid);
        }
    }

    // Function to check if a loan is repaid
    function checkLoanRepayment(address borrower, uint256 amount, address token) external view returns (bool) {
    uint256 totalAmount = amount + ((amount * loanFeePercentage) / 100);
    return IERC20(token).balanceOf(borrower) >= totalAmount;
}

    // Function to withdraw collected fees
    function withdrawFees(address token, uint256 amount) external onlyOwner {
        IERC20(token).transfer(owner, amount);
    }
}