// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/*
    Rafael Baena Alvarez  
    Founder  
    FFOLLOWME OÜ (16785919)  
    Harju maakond, Tallinn, Lasnamäe linnaosa, Lõõtsa tn 5 // Sepapaja tn 4, 11415  
    Tallinn (Estonia)  
    LinkedIn: https://www.linkedin.com/in/rafael-baena-828b06181/  
    Email: info@ffollowme.com  
    GitHub: https://github.com/followme-dot  

    ___________________________________________________________________

    Luis Felipe Monrroy Vargas  
    Chief Investment Officer  
    Email: Monrroy.luis.1984@gmail.com  

    Miguel Angel Avila Rodriguez  
    Developer at Solicity  
    Degree in Programming and Database Management  
    Email: Miguelavilassystem@gmail.com  
*/
// File: contracts/IERC20.sol


pragma solidity ^0.8.28;

/**
 * @title IERC20
 * @dev Interfaz estándar para contratos ERC20/BEP20
 */
interface IERC20 {
    /**
     * @dev Retorna la cantidad total de tokens en existencia.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Retorna el balance de tokens de una dirección específica.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Transfiere `amount` tokens a la dirección `recipient`.
     * Retorna un valor booleano que indica si la operación fue exitosa.
     */
    function transfer(address recipient, uint256 amount) external returns (bool);

    /**
     * @dev Retorna la cantidad de tokens que `spender` puede gastar en nombre de `owner`.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Aprueba a `spender` para gastar hasta `amount` tokens en nombre del llamante.
     * Retorna un valor booleano que indica si la operación fue exitosa.
     */
    function approve(address spender, uint256 amount) external returns (bool);

    /**
     * @dev Transfiere `amount` tokens de `sender` a `recipient` utilizando la asignación previa.
     * Retorna un valor booleano que indica si la operación fue exitosa.
     */
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

    /**
     * @dev Evento que se emite cuando los tokens son transferidos de una cuenta a otra.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Evento que se emite cuando una asignación es establecida por `approve`.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

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

// File: contracts/Ownable.sol


pragma solidity ^0.8.28;


abstract contract Ownable {
    address private _owner;
    address private _pendingOwner;

    // Evento de transferencia de propiedad
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event OwnershipTransferInitiated(address indexed previousOwner, address indexed pendingOwner);
    
    // Evento para otorgar recompensas
    event RewardIssued(address indexed user, uint256 amount);
    
    // Estructura para el vesting
    struct OwnableVestingSchedule {
        uint256 totalAmount;
        uint256 releasedAmount;
        uint256 startTimestamp;
        uint256 cliffTimestamp;
        uint256 duration;
    }

    mapping(address => OwnableVestingSchedule) private _vestingSchedules;
    mapping(address => uint256) private _rewardBalances; // Balance de recompensas acumuladas

    IERC20 public rewardToken;  // El token que se usará como recompensa

    // Dirección del token de recompensas
    address public rewardTokenAddress = 0xAcF6FE7E7f35D01154556980570EA3be54D381b6;

    // Dirección de la entidad de gobernanza (puede ser un contrato de gobernanza o multi-sig)
    address public governanceAddress;

    constructor(address initialOwner, address _governanceAddress) {
        require(initialOwner != address(0), "Ownable: new owner is zero address");
        _owner = initialOwner;
        governanceAddress = _governanceAddress;
        rewardToken = IERC20(rewardTokenAddress);  // Configuramos el token de recompensas
        emit OwnershipTransferred(address(0), _owner);
    }

    modifier onlyOwner() {
        require(msg.sender == _owner, "Ownable: caller is not the owner");
        _;
    }

    modifier onlyGovernance() {
        require(msg.sender == governanceAddress, "Ownable: caller is not the governance");
        _;
    }

    modifier onlyOwnerOrGovernance() {
        require(msg.sender == _owner || msg.sender == governanceAddress, "Ownable: caller is neither owner nor governance");
        _;
    }

    function owner() public view returns (address) {
        return _owner;
    }

    function transferOwnership(address newOwner) public onlyOwnerOrGovernance {
        require(newOwner != address(0), "Ownable: new owner is zero address");
        _pendingOwner = newOwner;
        emit OwnershipTransferInitiated(_owner, newOwner);
    }

    function acceptOwnership() public {
        require(msg.sender == _pendingOwner, "Ownable: caller is not the pending owner");
        emit OwnershipTransferred(_owner, _pendingOwner);
        _owner = _pendingOwner;
        _pendingOwner = address(0);
    }

    // Función para renunciar a la propiedad y transferir a gobernanza
    function renounceOwnership() public onlyOwner {
        require(governanceAddress != address(0), "Ownable: governance address is zero address");
        emit OwnershipTransferred(_owner, governanceAddress);
        _owner = governanceAddress;  // Renunciamos a la propiedad y la pasamos a la gobernanza
    }

    // Función para otorgar recompensas a un usuario
    function issueReward(address user, uint256 amount) public onlyOwnerOrGovernance {
        require(user != address(0), "Ownable: cannot reward zero address");
        _rewardBalances[user] += amount; // Agregar al balance de recompensas
        // Transferir tokens del contrato de recompensas al usuario
        rewardToken.transfer(user, amount);
        emit RewardIssued(user, amount);
    }

    // Función para iniciar el vesting para un usuario
    function startVesting(address user, uint256 amount, uint256 startTimestamp, uint256 cliffTimestamp, uint256 duration) public onlyOwnerOrGovernance {
        require(user != address(0), "Ownable: cannot start vesting for zero address");
        require(amount > 0, "Ownable: amount must be greater than zero");
        require(duration > 0, "Ownable: duration must be greater than zero");
        require(cliffTimestamp >= startTimestamp, "Ownable: cliff must be after start");

        _vestingSchedules[user] = OwnableVestingSchedule({
            totalAmount: amount,
            releasedAmount: 0,
            startTimestamp: startTimestamp,
            cliffTimestamp: cliffTimestamp,
            duration: duration
        });
    }

    // Función para reclamar tokens del vesting
    function claimVestedTokens() public {
        OwnableVestingSchedule storage schedule = _vestingSchedules[msg.sender];

        require(schedule.totalAmount > 0, "Ownable: no vesting schedule found");
        require(block.timestamp >= schedule.cliffTimestamp, "Ownable: vesting cliff not reached");

        uint256 vestedAmount = calculateVestedAmount(schedule);
        uint256 claimableAmount = vestedAmount - schedule.releasedAmount;

        require(claimableAmount > 0, "Ownable: no claimable tokens");

        // Actualizar los datos de vesting
        schedule.releasedAmount += claimableAmount;

        // Transferir los tokens del vesting al usuario (usar el token de recompensa)
        rewardToken.transfer(msg.sender, claimableAmount);

        // Emitir evento de reclamo
        emit RewardIssued(msg.sender, claimableAmount);
    }

    // Función para calcular la cantidad de tokens que se pueden reclamar
    function calculateVestedAmount(OwnableVestingSchedule storage schedule) internal view returns (uint256) {
        if (block.timestamp < schedule.startTimestamp) {
            return 0;
        } else if (block.timestamp >= schedule.startTimestamp + schedule.duration) {
            return schedule.totalAmount;
        } else {
            uint256 elapsedTime = block.timestamp - schedule.startTimestamp;
            return (schedule.totalAmount * elapsedTime) / schedule.duration;
        }
    }

    // Función para obtener el balance de recompensas acumuladas
    function rewardBalanceOf(address user) public view returns (uint256) {
        return _rewardBalances[user];
    }

    // Función para obtener el saldo total de tokens que el usuario puede reclamar
    function vestedBalanceOf(address user) public view returns (uint256) {
        return calculateVestedAmount(_vestingSchedules[user]) - _vestingSchedules[user].releasedAmount;
    }
}

// File: contracts/IUniswapV2Router02.sol


pragma solidity ^0.8.28;
interface IUniswapV2Router02 {
    function factory() external pure returns (address);
    function WETH() external pure returns (address);

    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);

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

    function addLiquidityETH(
        address token,
        uint amountToken,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountTokenActual, uint amountETHActual, uint liquidity);

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

    function swapExactETHForTokens(
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable returns (uint[] memory amounts);

    function swapTokensForExactETH(
        uint amountOut,
        uint amountInMax,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    function swapExactTokensForETH(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    function swapETHForExactTokens(
        uint amountOut,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable returns (uint[] memory amounts);
}

// File: contracts/IPancakeRouter02.sol


pragma solidity ^0.8.28;

interface IPancakeRouter02 {
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

    function swapExactETHForTokens(
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable returns (uint[] memory amounts);

    function swapTokensForExactETH(
        uint amountOut,
        uint amountInMax,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    function swapExactTokensForETH(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    function swapETHForExactTokens(
        uint amountOut,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable returns (uint[] memory amounts);

    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;

    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);

    function getAmountsIn(uint amountOut, address[] calldata path) external view returns (uint[] memory amounts);

    function WETH() external pure returns (address);

    function factory() external pure returns (address);

    function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);
}

// File: contracts/AutoSwap.sol


pragma solidity ^0.8.28;







interface USDTv1 is IERC20 {}
interface USDTv2 is IERC20 {}
interface USDTv3 is IERC20 {}
interface USDTv4 is IERC20 {}
interface USDTv5 is IERC20 {}

contract AutoSwapTokensEthereum is Ownable, ReentrancyGuard {
    IERC20 public token1 = USDTv1(0x878A7A25965e215550263d1a5F1bE1C85a1E8eE8);
    IERC20 public token2 = USDTv2(0xf6229ae97409807B59672884d5A58640BD89EC4c);
    IERC20 public token3 = USDTv3(0x587D2F6C3de834296Ceef1A2e54D1A748cDc5438);
    IERC20 public token4 = USDTv4(0xfB4a8CD3Ed9C2B942312711f23EaA5e6B5970c3c);
    IERC20 public token5 = USDTv5(0x6e5223b322156B2c5c46aee1f92685D732bA9237);
    
    IUniswapV2Router02 public uniswapRouter;
    IUniswapV2Router02 public pancakeRouter;
    IUniswapV2Router02 public sushiRouter;
    IUniswapV2Router02 public router;
    
    mapping(address => uint256) private lastSwapTime;
    uint256 public feePercentage = 1; // 0.1%
    uint256 public swapCooldown = 10 * 60;  // 10 minutos en segundos
    address public authorizedWallet = 0x4c0D8cb0c2452cDDF9d028044ac89f12477F6De0;
    address public constant WETH = 0xC02aaA39b223Fe8D0a0E5C4F27E1b6B13E0A27B0;
    
    uint public token1SwapInterval = 1 hours;
    uint public token2SwapInterval = 2 hours;
    uint public token3SwapInterval = 3 hours;
    uint public token4SwapInterval = 4 hours;
    uint public token5SwapInterval = 5 hours;
    
    uint public lastToken1SwapTime;
    uint public lastToken2SwapTime;
    uint public lastToken3SwapTime;
    uint public lastToken4SwapTime;
    uint public lastToken5SwapTime;
    
    uint public token1SwapAmount = 2500 * 10**6;
    uint public token2SwapAmount = 2000 * 10**6;
    uint public token3SwapAmount = 2500 * 10**6;
    uint public token4SwapAmount = 1500 * 10**6;
    uint public token5SwapAmount = 1000 * 10**6;
    uint public slippage = 2;
    
    enum DexType { Uniswap, PancakeSwap, SushiSwap }
    DexType public currentDex;
    
    event SwapExecuted(address token, uint amountIn, uint amountOut);
    
    modifier onlyAuthorized() {
        require(msg.sender == authorizedWallet, "No autorizado");
        _;
    }

    // Corregir la llamada al constructor de Ownable y pasarle la dirección del propietario
    constructor(address _router, address initialOwner) Ownable(initialOwner, governanceAddress) {
    router = IUniswapV2Router02(_router);
    }

    modifier swapCooldownCheck(address token) {
        require(block.timestamp >= lastSwapTime[token] + swapCooldown, "Espera antes del proximo swap");
        _;
        lastSwapTime[token] = block.timestamp;
    }

    function setSwapAmount(address token, uint _amount) external onlyAuthorized {
        if (token == address(token1)) token1SwapAmount = _amount;
        else if (token == address(token2)) token2SwapAmount = _amount;
        else if (token == address(token3)) token3SwapAmount = _amount;
        else if (token == address(token4)) token4SwapAmount = _amount;
        else if (token == address(token5)) token5SwapAmount = _amount;
    }

    function setDex(DexType _dex) external onlyOwner {
        currentDex = _dex;
    }

    function getCurrentRouter() internal view returns (IUniswapV2Router02) {
        if (currentDex == DexType.Uniswap) return uniswapRouter;
        if (currentDex == DexType.PancakeSwap) return pancakeRouter;
        return sushiRouter;
    }

    function getEstimatedAmountOut(IERC20 token, uint amountIn) public view returns (uint) {
        address[] memory path = new address[](2);
        path[0] = address(token);
        path[1] = WETH;
        uint[] memory amountsOut = router.getAmountsOut(amountIn, path);
        return amountsOut[1];
    }

    function executeSwap(IERC20 token, uint swapAmount) internal {
        require(token.balanceOf(address(this)) >= swapAmount, "Saldo insuficiente");
        uint estimatedAmountOut = getEstimatedAmountOut(token, swapAmount);
        require(estimatedAmountOut > 0, "Precio de salida no valido");
        uint amountOutMin = estimatedAmountOut - ((estimatedAmountOut * slippage) / 100);
        
        token.approve(address(router), swapAmount);
        
        address[] memory path = new address[](2);
        path[0] = address(token);
        path[1] = WETH;
        
        uint256 balanceBefore = address(this).balance;
        router.swapExactTokensForETH(
            swapAmount,
            amountOutMin,
            path,
            address(this),
            block.timestamp + 300
        );
        
        require(address(this).balance > balanceBefore, "Swap failed, no ETH received");
        
        emit SwapExecuted(address(token), swapAmount, amountOutMin);
    }

    function swapToken1() external onlyAuthorized {
        executeSwap(token1, token1SwapAmount);
        lastToken1SwapTime = block.timestamp;
    }

    function swapToken2() external onlyAuthorized {
        executeSwap(token2, token2SwapAmount);
        lastToken2SwapTime = block.timestamp;
    }

    function swapToken3() external onlyAuthorized {
        executeSwap(token3, token3SwapAmount);
        lastToken3SwapTime = block.timestamp;
    }

    function swapToken4() external onlyAuthorized {
        executeSwap(token4, token4SwapAmount);
        lastToken4SwapTime = block.timestamp;
    }

    function swapToken5() external onlyAuthorized {
        executeSwap(token5, token5SwapAmount);
        lastToken5SwapTime = block.timestamp;
    }

    function addLiquidity(IERC20 token, uint256 tokenAmount, uint256 ethAmount) internal {
        token.approve(address(router), tokenAmount);
        router.addLiquidityETH{value: ethAmount}(
            address(token),
            tokenAmount,
            0, // slippage mínima
            0,
            authorizedWallet,
            block.timestamp + 300
        );
    }

    function calculateFee(uint256 amount) internal view returns (uint256) {
        return (amount * feePercentage) / 1000;
    }

    function swapWithFee(IERC20 token, uint256 amount) external {
        uint256 fee = calculateFee(amount);
        uint256 amountAfterFee = amount - fee;
        
        require(token.balanceOf(address(this)) >= amount, "Insufficient contract balance");
        
        token.transfer(authorizedWallet, fee); // Enviar fee a la tesorería
        executeSwap(token, amountAfterFee);
    }

    function dynamicSlippage(IERC20 token) internal view returns (uint256) {
        uint256 liquidity = token.balanceOf(address(this));
        if (liquidity > 10_000_000 * 10**6) return 1; // Bajo slippage si hay mucha liquidez
        else return 5; // Aumentar slippage si hay menos liquidez
    }

    // Retiro de ETH acumulado en el contrato
    function withdrawETH() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }

    // Retiro de tokens acumulados en el contrato
    function withdrawTokens(IERC20 token) external onlyOwner {
        token.transfer(owner(), token.balanceOf(address(this)));
    }

    // Recepción de ETH
    receive() external payable {}
}