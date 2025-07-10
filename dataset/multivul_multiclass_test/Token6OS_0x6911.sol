/*

             ██████╗ ███████╗███╗   ██╗███████╗███████╗    ████████╗ ██████╗ ██╗  ██╗███████╗███╗   ██╗
            ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔════╝    ╚══██╔══╝██╔═══██╗██║ ██╔╝██╔════╝████╗  ██║
            ███████╗ █████╗  ██╔██╗ ██║███████╗█████╗         ██║   ██║   ██║█████╔╝ █████╗  ██╔██╗ ██║
            ██╔═══██╗██╔══╝  ██║╚██╗██║╚════██║██╔══╝         ██║   ██║   ██║██╔═██╗ ██╔══╝  ██║╚██╗██║
            ╚██████╔╝███████╗██║ ╚████║███████║███████╗       ██║   ╚██████╔╝██║  ██╗███████╗██║ ╚████║
             ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝       ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝

        ”Sometimes when you're in a dark place you think you've been buried, but Actually you’ve been planted.“
                                                                                                                                                                       
*/

// SPDX-License-Identifier: MIT

pragma solidity 0.8.26;

interface IERC20 {

    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the amount of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves `amount` tokens from the caller's account to `recipient`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address recipient, uint256 amount) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets `amount` as the allowance of `spender` over the caller's tokens.
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
    function approve(address spender, uint256 amount) external returns (bool);

    /**
     * @dev Moves `amount` tokens from `sender` to `recipient` using the
     * allowance mechanism. `amount` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

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
}

/**
 * @title Context
 * @dev The base contract that provides information about the message sender
 * and the calldata in the current transaction.
 */

abstract contract Context {
    function _msgSender() internal view virtual returns (address payable) {
        return payable(msg.sender);
    }

    function _msgData() internal view virtual returns (bytes memory) {
        this; // silence state mutability warning without generating bytecode - see https://github.com/ethereum/solidity/issues/2691
        return msg.data;
    }
}

/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * By default, the owner account will be the one that deploys the contract. This
 * can later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
contract Ownable is Context {
    address private _owner;
    address private _previousOwner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Locked(address owner, address newOwner,uint256 lockTime);

    /**
     * @dev Initializes the contract setting the deployer as the initial owner.
     */
    constructor ()  {
        address msgSender = _msgSender();
        _owner = msgSender;
        emit OwnershipTransferred(address(0), msgSender);
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

     /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions anymore. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby removing any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

/**
 * @title IUniswapV2Factory
 * @dev Interface for the Uniswap V2 Factory contract.
 */

interface IUniswapV2Factory {
    event PairCreated(address indexed token0, address indexed token1, address pair, uint);

    function feeTo() external view returns (address);
    function feeToSetter() external view returns (address);

    function getPair(address tokenA, address tokenB) external view returns (address pair);
    function allPairs(uint) external view returns (address pair);
    function allPairsLength() external view returns (uint);

    function createPair(address tokenA, address tokenB) external returns (address pair);

    function setFeeTo(address) external;
    function setFeeToSetter(address) external;
}

/**
 * @title IUniswapV2Router01
 * @dev Interface for the Uniswap V2 Router version 01 contract.
*/

interface IUniswapV2Router01 {
    function factory() external pure returns (address);
    //WETH function that return const value,  rather than performing some state-changing operation. 
    function WETH() external pure returns (address);

    function addLiquidity(
        address tokenA,
        address tokenB,
        uint amountADesired,
        uint amountBDesired,
        uint amountAMin,
        uint amountBMin,
        address to,
        uint deadline
    ) external returns (uint amountA, uint amountB, uint liquidity);
    function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);
    function removeLiquidity(
        address tokenA,
        address tokenB,
        uint liquidity,
        uint amountAMin,
        uint amountBMin,
        address to,
        uint deadline
    ) external returns (uint amountA, uint amountB);
    function removeLiquidityETH(
        address token,
        uint liquidity,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external returns (uint amountToken, uint amountETH);
    function removeLiquidityWithPermit(
        address tokenA,
        address tokenB,
        uint liquidity,
        uint amountAMin,
        uint amountBMin,
        address to,
        uint deadline,
        bool approveMax, uint8 v, bytes32 r, bytes32 s
    ) external returns (uint amountA, uint amountB);
    function removeLiquidityETHWithPermit(
        address token,
        uint liquidity,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline,
        bool approveMax, uint8 v, bytes32 r, bytes32 s
    ) external returns (uint amountToken, uint amountETH);
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    function swapTokensForExactTokens(
        uint amountOut,
        uint amountInMax,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    function swapExactETHForTokens(uint amountOutMin, address[] calldata path, address to, uint deadline)
        external
        payable
        returns (uint[] memory amounts);
    function swapTokensForExactETH(uint amountOut, uint amountInMax, address[] calldata path, address to, uint deadline)
        external
        returns (uint[] memory amounts);
    function swapExactTokensForETH(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline)
        external
        returns (uint[] memory amounts);
    function swapETHForExactTokens(uint amountOut, address[] calldata path, address to, uint deadline)
        external
        payable
        returns (uint[] memory amounts);

    function quote(uint amountA, uint reserveA, uint reserveB) external pure returns (uint amountB);
    function getAmountOut(uint amountIn, uint reserveIn, uint reserveOut) external pure returns (uint amountOut);
    function getAmountIn(uint amountOut, uint reserveIn, uint reserveOut) external pure returns (uint amountIn);
    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);
    function getAmountsIn(uint amountOut, address[] calldata path) external view returns (uint[] memory amounts);
}


interface IUniswapV2Router02 is IUniswapV2Router01 {
    function removeLiquidityETHSupportingFeeOnTransferTokens(
        address token,
        uint liquidity,
        uint amountTokenMin,
        uint amountETHMin, 
        address to,
        uint deadline
    ) external returns (uint amountETH);
    function removeLiquidityETHWithPermitSupportingFeeOnTransferTokens(
        address token,
        uint liquidity,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline,
        bool approveMax, uint8 v, bytes32 r, bytes32 s
    ) external returns (uint amountETH);

    function swapExactTokensForTokensSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;
    function swapExactETHForTokensSupportingFeeOnTransferTokens(
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable;
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;
}


contract Token6OS is Context, IERC20, Ownable {
    mapping (address => uint256) private _rOwned;
    mapping (address => uint256) private _tOwned;
    mapping (address => mapping (address => uint256)) private _allowances;

    mapping (address => bool) public isExcludedFromFee;
    mapping (address => bool) private _isExcluded;
    address[] private _excluded;
   
    uint256 private constant MAX = ~uint256(0);
    uint256 private _tTotal = 963 * 10**7 * 10**18;
    uint256 private _rTotal = (MAX - (MAX % _tTotal));
    uint256 private _tFeeTotal;
   
    string  private constant NAME = "6ENSE";
    string  private  constant SYMBOL = "6OS";
    uint8  private constant DECIMALS = 18;
 
    bool private swapping; 
    bool public tradeEnabled;
    
    uint256 private taxThreshold = 1 * 10**3 * 10**18;

    uint256 refAmt;
    uint256 plantoFee;

    uint256 public reflectionTax=0;
    uint256 public plantoGroupTax=1; 
   
    uint256 public  maxTransferAmount = 1926 * 10*5 * 10**18; // Max transfer Limit 

    IUniswapV2Router02 public immutable uniswapV2Router;
    address public immutable uniswapV2Pair;
    address public immutable usdcAddress = 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48;

    address public plantoGroupWallet;
    
    event PlantoGroupWalletChange(address wallet);
    event ThresholdUpdated(uint256 amount);
    event ReflectedFee(uint256 totalReflectFee);
    event UpdatedMaxAmount(uint256 updatedMaxAmount);
    event UpdatedReflectionTax(uint256 reflectionTax); 
    event UpdatedWalletTax(uint256 walletTax);
    event TradeEnabled(bool enabled);
    event IncludedInFee(address account);
    event ExcludedFromFee(address account);
    event IncludedInReward(address account);
    event ExcludedFromReward(address account);

    /**
    * @dev Constructor for the token contract.
    * @param _wallet Address of the PlantoGroup wallet, which cannot be the zero address.
    * 
    * Initializes the contract by setting the initial token supply to the deployer, 
    * setting up the Uniswap V2 router and liquidity pair, and excluding certain addresses 
    * (owner, contract, and liquidity pair) from fees.
    */
    constructor(address _wallet)  {
        require(_wallet != address(0),"plantoGroup wallet can not be zero");
        _rOwned[_msgSender()] = _rTotal;
        plantoGroupWallet = _wallet;
        IUniswapV2Router02 _uniswapV2Router = IUniswapV2Router02(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D); //etherum mainnet
        
        
        // Create a uniswap pair for this new token
        uniswapV2Pair = IUniswapV2Factory(_uniswapV2Router.factory())
            .createPair(address(this), usdcAddress);

        // set the rest of the contract variables
        uniswapV2Router = _uniswapV2Router;
        
        //exclude owner and this contract from fee
        isExcludedFromFee[owner()] = true;
        isExcludedFromFee[address(this)] = true;

        excludeFromReward(uniswapV2Pair);
        
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    /**
    * @notice Retrieves the name of the token.
    * @dev This function returns the name of the token, which is often used for identification.
    * It is commonly displayed in user interfaces and provides a human-readable name for the token.
    * @return The name of the token.
    */
    function name() external pure returns (string memory) {
        return NAME;
    }

    /**
    * @notice Retrieves the symbol or ticker of the token.
    * @dev This function returns the symbol or ticker that represents the token.
    * It is commonly used for identifying the token in user interfaces and exchanges.
    * @return The symbol or ticker of the token.
    */
    function symbol() external pure returns (string memory) {
        return SYMBOL;
    }

    /**
    * @notice Retrieves the number of decimal places used in the token representation.
    * @dev This function returns the number of decimal places used to represent the token balances.
    * It is commonly used to interpret the token amounts correctly in user interfaces.
    * @return The number of decimal places used in the token representation.
    */
    function decimals() external pure returns (uint8) {
        return DECIMALS;
    }

    /**
    * @notice Retrieves the total supply of tokens.
    * @dev This function returns the total supply of tokens in circulation.
    * @return The total supply of tokens.
    */
    function totalSupply() external view override returns (uint256) {
        return _tTotal;
    }

    /**
    * @notice Retrieves the token balance of a specified account.
    * @dev This function returns the token balance of the specified account.
    * If the account is excluded, it directly returns the token balance.
    * If the account is not excluded, it converts the reflection balance to token balance using the current rate.
    * @param account The address of the account whose token balance is being queried.
    * @return The token balance of the specified account.
    */
    function balanceOf(address account) public view override returns (uint256) {
        if (_isExcluded[account]) return _tOwned[account];//exculded
        return tokenFromReflection(_rOwned[account]);//not excluded
    }

    /**
    * @notice Transfers a specified amount of tokens to a recipient.
    * @dev This function transfers tokens from the sender's account to the specified recipient.
    * If successful, it returns true.
    * @param recipient The address of the recipient to whom tokens are being transferred.
    * @param amount The amount of tokens to be transferred.
    * @return A boolean indicating the success of the transfer operation.
     */
    function transfer(address recipient, uint256 amount) external override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    /**
    * @notice Retrieves the remaining allowance for a spender to spend tokens on behalf of an owner.
    * @dev This function returns the current allowance set for the specified spender to spend tokens
    * from the specified owner's account.
    * @param owner The address of the owner whose allowance is being queried.
    * @param spender The address of the spender for whom the allowance is queried.
    * @return The remaining allowance for the specified spender to spend tokens on behalf of the owner.
    */
    function allowance(address owner, address spender) external view override returns (uint256) {
        return _allowances[owner][spender];
    }

    /**
    * @notice Approves a spender to spend a specified amount of tokens on behalf of the owner.
    * @dev This function sets or updates the allowance for a spender to spend tokens
    * from the owner's account. If successful, it returns true.
    * @param spender The address of the spender to be approved.
    * @param amount The amount of tokens to approve for spending.
    * @return A boolean indicating the success of the approval operation.
    */
    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    /**
    * @notice Transfers tokens from one address to another on behalf of a third-party.
    * @dev This function allows a designated spender to transfer tokens from the sender's account
    * to the recipient's account. It also ensures that the allowance is updated correctly.
    * If successful, it returns true.
    * @param sender The address from which tokens are being transferred.
    * @param recipient The address to which tokens are being transferred.
    * @param amount The amount of tokens to be transferred.
    * @return A boolean indicating the success of the transfer operation.
    */
    function transferFrom(address sender, address recipient, uint256 amount) external override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), _allowances[sender][_msgSender()]-amount);
        return true;
    }

    /**
    * @notice Increases the allowance granted to a spender by a specified amount.
    * @dev This function increases the allowance for the specified spender by the given value.
    * It ensures that the updated allowance is correctly set. If successful, it returns true.
    * @param spender The address of the spender whose allowance is being increased.
    * @param addedValue The amount by which to increase the allowance.
    * @return A boolean indicating the success of the operation.
    */
    function increaseAllowance(address spender, uint256 addedValue) external virtual returns (bool) {
        _approve(_msgSender(), spender, _allowances[_msgSender()][spender]+addedValue);
        return true;
    }

    /**
    * @notice Reduces the allowance granted to a spender by a specified amount.
    * @dev This function decreases the allowance for the specified spender by the given value.
    * It ensures that the allowance does not go below zero. If successful, it returns true.
    * @param spender The address of the spender whose allowance is being reduced.
    * @param subtractedValue The amount by which to reduce the allowance.
    * @return A boolean indicating the success of the operation.
    */
    function decreaseAllowance(address spender, uint256 subtractedValue) external virtual returns (bool) {
        _approve(_msgSender(), spender, _allowances[_msgSender()][spender]-subtractedValue);
        return true;
    }

    /**
    * @notice Checks if the specified address is excluded from earning reflections.
    * @dev Excluded addresses do not receive reflections in certain tokenomics designs.
    * This function returns true if the address is excluded, and false otherwise.
    * @param account The address to check for exclusion from reflections.
    * @return A boolean indicating whether the address is excluded from earning reflections.
    */
    function isExcludedFromReward(address account) external view returns (bool) {
        return _isExcluded[account];
    }

    /**
    * @notice Retrieves the total amount of fees collected in tokens.
    * @dev This function returns the cumulative sum of fees collected during transactions.
    * The fees are often used for various purposes like liquidity provision, rewards, or burns.
    * @return The total amount of fees collected in tokens.
    */
    function totalFees() external view returns (uint256) {
        return _tFeeTotal;
    }

    /**
    * @notice Distributes the specified amount of tokens as reflections to the reward pool.
    * @dev This function is typically used to convert a portion of tokens into reflections
    * and add them to a reward pool. Excluded addresses cannot call this function.
    * @param tAmount The amount of tokens to be converted and added to reflections.
    */
    function deliver(uint256 tAmount) external {
        address sender = _msgSender();
        require(!_isExcluded[sender], "Excluded addresses cannot call this function");
        (uint256 rAmount,) = _getValue(tAmount);
        _rOwned[sender] = _rOwned[sender]-rAmount;
        _rTotal = _rTotal-rAmount;
        _tFeeTotal = _tFeeTotal+tAmount;
    }

    /**
    * @notice Converts the given token amount to its equivalent reflection amount.
    * @dev Reflections are often used in tokenomics to calculate rewards or balances.
    * This function converts a token amount to its corresponding reflection amount
    * based on the current rate. Optionally, it deducts the transfer fee from the calculation.
    * @param tAmount The token amount to be converted to reflections.
    * @param deductTransferFee A boolean indicating whether to deduct the transfer fee from the calculation.
    * @return The equivalent reflection amount corresponding to the given token amount.
    */
    function reflectionFromToken(uint256 tAmount, bool deductTransferFee) external view returns(uint256) {
        require(tAmount <= _tTotal, "Amount must be less than supply");
        if (!deductTransferFee) {
            (uint256 rAmount,) = _getValue(tAmount);
             return rAmount;
        } else {
            (,uint256 rTransferAmount) = _getValue(tAmount);
             return rTransferAmount;
        }
    }

    /**
    * @notice Converts the given reflection amount to its equivalent token amount.
    * @dev Reflections are often used in tokenomics to calculate rewards or balances.
    * This function converts a reflection amount to its corresponding token amount
    * based on the current rate.
    * @param rAmount The reflection amount to be converted to tokens.
    * @return The equivalent token amount corresponding to the given reflection amount.
    */
    function tokenFromReflection(uint256 rAmount) public view returns(uint256) {
        require(rAmount <= _rTotal, "Amount must be less than total reflections");
        uint256 currentRate =  _getRate();
        return rAmount / currentRate;
    }

    /**
    * @notice Grants the owner the ability to exclude an address from earning reflections.
    * @dev Reflections are often used in tokenomics to distribute rewards to holders.
    * This function excludes the specified address from receiving reflections.
    * @param account The address to be excluded from earning reflections.
    */
    function excludeFromReward(address account) public onlyOwner() {
        require(!_isExcluded[account], "Account is already excluded");
        if(_rOwned[account] > 0) {
            _tOwned[account] = tokenFromReflection(_rOwned[account]);
        }
        _isExcluded[account] = true;
        _excluded.push(account);
        emit ExcludedFromReward(account);
    }

    /**
    * @dev External function for including an account in the reward distribution.
    * @param account The address to be included in the reward distribution.
    * 
    * The function can only be called by the owner of the contract.
    * Requires that the specified account is currently excluded.
    * Iterates through the list of excluded accounts, finds the specified account, and removes it from the exclusion list.
    * Resets the token balance of the specified account to 0 and updates the exclusion status.
    * 
    * @notice Only the owner of the contract can call this function.
    * @notice Requires that the specified account is currently excluded.
    */
    function includeInReward(address account) external onlyOwner() {
        require(_isExcluded[account], "Account is already Included");
        for (uint256 i = 0; i < _excluded.length; i++) {
            if (_excluded[i] == account) {
                _excluded[i] = _excluded[_excluded.length - 1];
                _tOwned[account] = 0;
                _isExcluded[account] = false;
                _excluded.pop();
                break;
            }
        }
        emit IncludedInReward(account);
    }

    /**
    * @notice Grants the owner the ability to exclude an address from transaction fees.
    * @dev Transaction fees are often applied in decentralized finance (DeFi) projects
    * to support various mechanisms like liquidity provision, rewards, or token burns.
    * @param account The address to exclude from transaction fees.
    */
     function excludeFromFee(address account) external  onlyOwner {
        require(!isExcludedFromFee[account],"Alreay excluded from fee");
        isExcludedFromFee[account] = true;
        emit ExcludedFromFee(account);
    }

    /**
    * @notice Grants the owner the ability to include an address in transaction fees.
    * @dev Transaction fees are often applied in decentralized finance (DeFi) projects
    * to support various mechanisms like liquidity provision, rewards, or token burns.
    * @param account The address to include in transaction fees.
    */
    
    function includeInFee(address account) external onlyOwner {
        require(isExcludedFromFee[account],"Alreay included in fee");
        isExcludedFromFee[account] = false;
        emit IncludedInFee(account);
    }

    /**
    * @dev Sets the address of the fund wallet.
    * @param wallet The new address to be set as the fund wallet.
    *
    * Requirements:
    * - Only the contract owner can call this function.
    *
    * Emits a {plantoGroupWalletChange} event with the updated wallet address on successful execution.
    */
    function setplantoGroupWallet(address wallet) external onlyOwner{
        require(wallet != address(0),"wallet can not be zero");
        plantoGroupWallet = wallet;
        emit PlantoGroupWalletChange(wallet);
    }                                                                                                                  

    /**
    * @dev External function for updating the threshold amount required for triggering liquidity addition.
    * @param amount The new threshold amount.
    * 
    * The function can only be called by the owner of the contract.
    * Requires that the provided threshold amount (amount) is greater than 0.
    * Updates the taxThreshold with the new threshold amount.
    * @notice Only the owner of the contract can call this function.
    * @notice Requires a positive amount for successful execution.
    */
    function updateThreshold(uint256 amount) external onlyOwner {
        require(amount > 0 && amount <= 5 * 10**5 * 10**18,"Amount should be more than zero and less than 500k tokens");
        taxThreshold = amount;
        emit ThresholdUpdated(amount);
    }
    
    //to recieve ETH from uniswapV2Router when swaping
    receive() external payable {}

    /**
    * @dev Private function for approving a spender to spend a certain amount on behalf of the owner.
    * @param owner The address that owns the tokens.
    * @param spender The address that is approved to spend the tokens.
    * @param amount The amount of tokens to be approved for spending.
    * 
    * Requires that both the owner and spender addresses are not the zero address.
    * Sets the allowance for the spender on behalf of the owner to the specified amount.
    * Emits an `Approval` event with details about the approval.
    * 
    * @notice This function is intended for internal use and should not be called directly.
    */
    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    /**
    * @dev Reduces the total reflection supply by `rFee` and adds `tFee` to the total fee collected.
    * @param rFee The reflection fee deducted from the total supply.
    * @param tFee The transaction fee added to the fee total.
    */
    function _reflectFee(uint256 rFee, uint256 tFee) private {
        _rTotal = _rTotal - rFee ;
        _tFeeTotal = _tFeeTotal + tFee;

        emit ReflectedFee(tFee);    
    }

    /**
    * @dev Allocates a portion of the transaction amount to the plantoGroup fund.
    * @param tPlantoFee The amount of tokens allocated for plantoGroup.
    * 
    * Converts `_takePlantoFee` to reflected value and stores it in the contract balance.
    */
    function _takePlantoFee(uint256 tPlantoFee) private {
        uint256 currentRate =  _getRate();
        uint256 rPlantoFee = tPlantoFee * currentRate;
        _rOwned[address(this)] = _rOwned[address(this)] + rPlantoFee;
        if(_isExcluded[address(this)])
          _tOwned[address(this)] = _tOwned[address(this)] + tPlantoFee;
    }
     
    /**
    * @dev Calculates and returns the transfer amount, tax fee, and plantoGroup fee for a given transaction.
    * @param tAmount The total transaction amount.
    * @return tTransferAmount The amount remaining after tax deductions.
    * @return tFee The transaction fee deducted.
    * @return tCoinOperation The fee allocated for plantoGroups.
    */
    function _getValues(uint256 tAmount) private view returns (uint256, uint256, uint256) {
        (uint256 tTransferAmount, uint256 tFee,uint256 tPlantoFee) = _getTValues(tAmount); 
        return ( tTransferAmount, tFee,  tPlantoFee);
    }

    /**
    * @dev Returns the reflected amount and reflected transfer amount for a given transaction.
    * @param tAmount The total transaction amount.
    * @return rAmount The reflected total amount.
    * @return rTransferAmount The reflected transfer amount after tax deductions.
    */
    function _getValue(uint256 tAmount) private view returns(uint256, uint256){
        (, uint256 tFee, uint256 tPlantoFee) = _getTValues(tAmount);
        (uint256 rAmount, uint256 rTransferAmount,) = _getRValues(tAmount, tFee, tPlantoFee);
         return (rAmount, rTransferAmount);
    }
 
    /**
    * @dev Calculates the tax and plantoGroup fees and returns the transfer amount.
    * @param tAmount The total transaction amount.
    * @return tTransferAmount The amount left after deducting all taxes.
    * @return tFee The tax fee deducted from the transaction.
    * @return tPlantoFee The plantoGroup fee deducted from the transaction.
    */
    function _getTValues(uint256 tAmount) private view returns (uint256, uint256, uint256) {
        uint256 tFee = calculateTaxFee(tAmount);
        uint256 tPlantoFee = calculatePlantoTax(tAmount);
        uint256 allTax = tFee + tPlantoFee;
        uint256 tTransferAmount = tAmount - allTax;
        return (tTransferAmount, tFee, tPlantoFee);
    }
    
    /**
    * @dev Converts token values to reflected values and calculates the reflected transfer amount.
    * @param tAmount The total transaction amount.
    * @param tFee The transaction fee in token value.
    * @param tPlantoFee The plantoGroup fee in token value.
    * @return rAmount The reflected total amount.
    * @return rTransferAmount The reflected transfer amount after tax deductions.
    * @return rFee The reflected tax fee deducted from the transaction.
    */
    function _getRValues(uint256 tAmount, uint256 tFee, uint256 tPlantoFee) private view returns (uint256, uint256, uint256) {
        uint256 currentRate = _getRate();
        uint256 rAmount = tAmount * currentRate;
        uint256 rFee = tFee * currentRate;
        uint256 rPlantoFee = tPlantoFee * currentRate;
        uint256 allTax = rFee + rPlantoFee;
        uint256 rTransferAmount = rAmount - allTax;
        return (rAmount, rTransferAmount, rFee);
    }

    /**
    * @dev Private function for retrieving the current conversion rate between reflection and token balances.
    * @return rate Current conversion rate.
    * 
    * @notice Internal use only.
    */
    function _getRate() private view returns(uint256) {
        (uint256 rSupply, uint256 tSupply) = _getCurrentSupply();
        return rSupply / tSupply;
    }

    /**
    * @dev Private function for retrieving the current supply of both reflection and token balances.
    * @return rSupply Current reflection supply.
    * @return tSupply Current token supply.
    * 
    * @notice Internal use only.
    */
    function _getCurrentSupply() private view returns(uint256, uint256) {
        uint256 rSupply = _rTotal;
        uint256 tSupply = _tTotal;      
        for (uint256 i = 0; i < _excluded.length; i++) {
            if (_rOwned[_excluded[i]] > rSupply || _tOwned[_excluded[i]] > tSupply) return (_rTotal, _tTotal);
            rSupply = rSupply - _rOwned[_excluded[i]];
            tSupply = tSupply - _tOwned[_excluded[i]];
        }
        if (rSupply < _rTotal / _tTotal) return (_rTotal, _tTotal);
        return (rSupply, tSupply);
    }

    /**
    * @dev Calculates the tax fee for reflection based on a specified amount.
    * @param amount Amount for tax fee calculation.
    * @return Calculated tax fee amount.
    * 
    * @notice Internal use only.
    */
    function calculateTaxFee(uint256 amount) private view returns (uint256) {
        return amount * refAmt / 10**2;
    }

    /**
    * @dev Calculates the plantoGroup tax based on a specified amount.
    * @param amount Amount for plantoGroup tax calculation.
    * @return Calculated plantoGroup tax amount.
    * 
    * @notice Internal use only.
    */
    function calculatePlantoTax(uint256 amount) private view returns (uint256) {
        return amount * plantoFee / 10**2;
    }

    /**
    * @dev Removes all fees by setting `refAmt` and `plantoFee` to zero.
    * 
    * This function is typically used for transactions where fees should be excluded, 
    * such as transfers between specific addresses.
    */
    function removeAllFee() private {
        refAmt = 0;
        plantoFee = 0;
    }

    /**
     * @notice Enables or disables trading functionality based on the input parameter.
     * @dev Only callable by the owner of the contract.
     * @param _enable A boolean value: `true` to enable trading, `false` to disable trading.
     */
    function setTrading(bool _enable) external onlyOwner {
        require(tradeEnabled != _enable, "Trading is already in the desired state");
        tradeEnabled = _enable;
        emit TradeEnabled(tradeEnabled);
    }

    /**
    * @notice Updates the reflection tax (max 6%).
    */
    function updateReflectionTaxPer(uint256 reflectionPercent) external onlyOwner {
        require(reflectionPercent <= 6,"You can not set reflection tax more then 6%");     
        reflectionTax = reflectionPercent;
        emit UpdatedReflectionTax(reflectionTax);
    }

    /**
    * @notice Updates the PlantoGroup tax (max 6%).
    */
    function updatePlantoGroupTax(uint256 walletTax) external onlyOwner {
        require(walletTax <= 6,"You can not set plantoGroup tax more then 6%");     
        plantoGroupTax = walletTax;
        emit UpdatedWalletTax(plantoGroupTax);
    }

    /**
     * @dev Sets the maximum buy limit per transaction. Can only be called by the contract owner.
     * 
     * The `amount` entered should include the token's decimal places.
     * For example, if the token has 18 decimals, to set a limit of 500,000 tokens,
     * the `amount` should be entered as 500,000 * 10^18 (i.e., 500k tokens with decimals).
     * 
     * The function enforces a minimum buy limit of 500,000 tokens (accounting for decimals).
     * 
     * @param amount The new maximum amount allowed per transaction. This value must include decimals.
     * Emits an {UpdatedMaxAmount} event indicating the new maximum buy amount.
     */
    function setMaxTransferLimit(uint256 amount) external onlyOwner {
        require(amount >= 500000 * 10**18, "Max Transfer limit can not be less than 500,000 tokens");
        maxTransferAmount = amount;
        emit UpdatedMaxAmount(maxTransferAmount);
    }

    /**
    * @notice Transfers tokens while applying tax rules.
    * @dev Ensures transaction limits, tax deductions, and trading status.
    * @param from The address sending tokens.
    * @param to The address receiving tokens.
    * @param amount The amount of tokens to transfer.
    */
    function _transfer( address from, address to, uint256 amount ) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(amount > 0, "Transfer amount must be greater than zero");

        if (from == owner() || to == owner()){
            _tokenTransfer(from,to,amount,false);
            return;
        }

        require(amount <= maxTransferAmount, "Transaction limit exceed");
        uint256 contractTokenBalance = balanceOf(address(this));
        
        bool overMinTokenBalance = contractTokenBalance >= taxThreshold;
        if (
            overMinTokenBalance &&
            !swapping &&
            from != uniswapV2Pair 
        ) {
            swapping = true;
            swapAndLiquify();
            swapping = false;
        }
        
        //indicates if fee should be deducted from transfer
        bool takeFee = true;
        
        //if any account belongs to isExcludedFromFee account then remove the fee
        if(isExcludedFromFee[from] || isExcludedFromFee[to]){
            takeFee = false;
        }

        //if takeFee is true then set sell or buy tax percentage
        if(takeFee){
            refAmt = reflectionTax; 
            plantoFee = plantoGroupTax; 
        }
       _tokenTransfer(from,to,amount,takeFee);
    }

    /**
    * @dev Swaps contract token balance for USDC and limits swap amount to maxTransferAmount.
    * Ensures that excessive tokens are not swapped in a single transaction.
    */
    function swapAndLiquify() private{

        uint256 contractTokenBalance = balanceOf(address(this));
        
        if(contractTokenBalance > maxTransferAmount){
            contractTokenBalance = maxTransferAmount;
        }
        swapTokensForUsdc(contractTokenBalance);   
    }

    /**
    * @dev Swaps a specified amount of tokens for USDC using Uniswap.
    * The swapped USDC is sent to the `plantoGroupWallet`.
    * 
    * @param tokenAmount The amount of tokens to swap for USDC.
    */
    function swapTokensForUsdc(uint256 tokenAmount) private {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = usdcAddress;

        _approve(address(this), address(uniswapV2Router), tokenAmount);

        // make the swap
        uniswapV2Router.swapExactTokensForTokens(
            tokenAmount,
            0, // accept any amount of Usdc
            path,
            plantoGroupWallet,
            block.timestamp
        );
    }

    /**
    * @notice Handles token transfers, applying tax rules if necessary.
    * @dev Determines the transfer type based on sender/recipient exclusions and applies fees accordingly.
    * @param sender The address sending tokens.
    * @param recipient The address receiving tokens.
    * @param amount The amount of tokens being transferred.
    * @param takeFee Boolean indicating whether tax should be applied.
    */
    function _tokenTransfer(address sender, address recipient, uint256 amount,bool takeFee) private {
        if(!takeFee)
            removeAllFee();
        
          if (_isExcluded[sender] && !_isExcluded[recipient]) {
            _transferFromExcluded(sender, recipient, amount);
        } else if (!_isExcluded[sender] && _isExcluded[recipient]) {
            _transferToExcluded(sender, recipient, amount);
        } else if (!_isExcluded[sender] && !_isExcluded[recipient]) {
            _transferStandard(sender, recipient, amount);
        } else if (_isExcluded[sender] && _isExcluded[recipient]) {
            _transferBothExcluded(sender, recipient, amount);
        } else {
            _transferStandard(sender, recipient, amount);
        }  
    }
     
    /**
    * @notice Handles transfers between two excluded accounts.
    * @dev Excluded accounts hold both reflected and total token balances.
    * @param sender The address sending tokens.
    * @param recipient The address receiving tokens.
    * @param tAmount The amount of tokens being transferred.
    */
    function _transferBothExcluded(address sender, address recipient, uint256 tAmount) private {
        (uint256 tTransferAmount, uint256 tFee, uint256 tPlantoFee) = _getValues(tAmount);
        (uint256 rAmount, uint256 rTransferAmount, uint256 rFee) = _getRValues(tAmount, tFee, tPlantoFee);
        _tOwned[sender] = _tOwned[sender]-tAmount;
        _rOwned[sender] = _rOwned[sender]-rAmount;
        _tOwned[recipient] = _tOwned[recipient]+tTransferAmount;
        _rOwned[recipient] = _rOwned[recipient]+rTransferAmount;        
        _reflectFee(rFee, tFee);
        _takePlantoFee(tPlantoFee);
        emit Transfer(sender, recipient, tTransferAmount);
    }

    /**
    * @notice Handles standard transfers between two non-excluded accounts.
    * @dev Only reflected balances are updated, and transaction fees are deducted.
    * @param sender The address sending tokens.
    * @param recipient The address receiving tokens.
    * @param tAmount The amount of tokens being transferred.
    */
    function _transferStandard(address sender, address recipient, uint256 tAmount) private {
        (uint256 tTransferAmount, uint256 tFee,  uint256 tPlantoFee) = _getValues(tAmount);
        (uint256 rAmount, uint256 rTransferAmount, uint256 rFee) = _getRValues(tAmount, tFee, tPlantoFee);
        _rOwned[sender] = _rOwned[sender]-rAmount;
        _rOwned[recipient] = _rOwned[recipient]+rTransferAmount;
        _reflectFee(rFee, tFee);
        _takePlantoFee(tPlantoFee);
        emit Transfer(sender, recipient, tTransferAmount);
    }

    /**
    * @notice Handles transfers where the recipient is excluded from rewards.
    * @dev Excluded recipients maintain a total balance but do not participate in reflections.
    * @param sender The address sending tokens.
    * @param recipient The address receiving tokens.
    * @param tAmount The amount of tokens being transferred.
    */
    function _transferToExcluded(address sender, address recipient, uint256 tAmount) private {
        (uint256 tTransferAmount, uint256 tFee, uint256 tPlantoFee) = _getValues(tAmount);
        (uint256 rAmount, uint256 rTransferAmount, uint256 rFee) = _getRValues(tAmount, tFee,tPlantoFee);
        _rOwned[sender] = _rOwned[sender]-rAmount;
        _tOwned[recipient] = _tOwned[recipient]+tTransferAmount;
        _rOwned[recipient] = _rOwned[recipient]+rTransferAmount;           
        _reflectFee(rFee, tFee);
        _takePlantoFee(tPlantoFee);
        emit Transfer(sender, recipient, tTransferAmount);
    }

    /**
    * @notice Handles transfers where the sender is excluded from rewards.
    * @dev Excluded senders maintain a total balance but do not participate in reflections.
    * @param sender The address sending tokens.
    * @param recipient The address receiving tokens.
    * @param tAmount The amount of tokens being transferred.
    */
    function _transferFromExcluded(address sender, address recipient, uint256 tAmount) private {
        (uint256 tTransferAmount, uint256 tFee, uint256 tPlantoFee) = _getValues(tAmount);
        (uint256 rAmount, uint256 rTransferAmount, uint256 rFee) = _getRValues(tAmount, tFee, tPlantoFee);
        _tOwned[sender] = _tOwned[sender]-tAmount;
        _rOwned[sender] = _rOwned[sender]-rAmount;
        _rOwned[recipient] = _rOwned[recipient]+rTransferAmount;   
        _reflectFee(rFee, tFee);
        _takePlantoFee(tPlantoFee);
        emit Transfer(sender, recipient, tTransferAmount);
    }
}