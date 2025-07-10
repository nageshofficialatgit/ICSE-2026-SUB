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

// File: contracts/Ownable.sol


pragma solidity ^0.8.20;

abstract contract Ownable {
    address private _owner;
    address private _pendingOwner;
    mapping(address => bool) private _authorized;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event OwnershipRenounced(address indexed previousOwner);
    event OwnershipTransferInitiated(address indexed currentOwner, address indexed pendingOwner);
    event AuthorizationGranted(address indexed account);
    event AuthorizationRevoked(address indexed account);

    modifier onlyOwner() {
        require(msg.sender == _owner, "Ownable: caller is not the owner");
        _;
    }

    modifier onlyAuthorized() {
        require(_authorized[msg.sender] || msg.sender == _owner, "Ownable: caller is not authorized");
        _;
    }

    constructor() {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    function pendingOwner() public view returns (address) {
        return _pendingOwner;
    }

    function initiateOwnershipTransfer(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _pendingOwner = newOwner;
        emit OwnershipTransferInitiated(_owner, newOwner);
    }

    function acceptOwnership() external {
        require(msg.sender == _pendingOwner, "Ownable: caller is not the pending owner");
        emit OwnershipTransferred(_owner, _pendingOwner);
        _owner = _pendingOwner;
        _pendingOwner = address(0);
    }

    function renounceOwnership() external onlyOwner {
        emit OwnershipRenounced(_owner);
        _owner = address(0);
    }

    function grantAuthorization(address account) external onlyOwner {
        require(account != address(0), "Ownable: cannot authorize zero address");
        _authorized[account] = true;
        emit AuthorizationGranted(account);
    }

    function revokeAuthorization(address account) external onlyOwner {
        require(account != address(0), "Ownable: cannot revoke zero address");
        _authorized[account] = false;
        emit AuthorizationRevoked(account);
    }

    function isAuthorized(address account) public view returns (bool) {
        return _authorized[account];
    }
}

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

// File: contracts/IERC20Metadata.sol


pragma solidity ^0.8.20;


interface IERC20Metadata is IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}

// File: contracts/IUniswapV2Router02.sol


pragma solidity ^0.8.20;

interface IUniswapV2Router02 {
    // Intercambia una cantidad exacta de tokens por tokens de salida
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    // Obtiene las cantidades de salida para un intercambio
    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);

    // Obtiene las cantidades de entrada necesarias para recibir una cantidad de salida
    function getAmountsIn(uint amountOut, address[] calldata path) external view returns (uint[] memory amounts);

    // Añadir liquidez a un par de tokens
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

    // Eliminar liquidez de un par de tokens
    function removeLiquidity(
        address tokenA,
        address tokenB,
        uint liquidity,
        uint amountAMin,
        uint amountBMin,
        address to,
        uint deadline
    ) external returns (uint amountA, uint amountB);
    
    // Swap de Tokens Exactos por ETH (WETH)
    function swapExactTokensForETH(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    // Obtener la cantidad de salida para un swap de tokens a ETH
    function getAmountsOutForETH(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);

    // Añadir liquidez para tokens y ETH (WETH)
    function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);

    // Eliminar liquidez de un par de tokens y ETH (WETH)
    function removeLiquidityETH(
        address token,
        uint liquidity,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external returns (uint amountToken, uint amountETH);
}


// File: contracts/ISushiRouter02.sol


pragma solidity ^0.8.20;

interface ISushiRouter02 {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
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

    function swapExactTokensForETH(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
}

// File: contracts/IUniswapV2Factory.sol


pragma solidity ^0.8.20;

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

// File: contracts/USDTv1.sol


pragma solidity ^0.8.20;


contract USDTv1 is Ownable, ReentrancyGuard, IERC20, IERC20Metadata {
    string public  _name = "USDTv1";
    string public  _symbol = "USDTv1";
    uint8 public  _decimals = 6;
    uint256 public  _totalSupply = 10_000_000 * 10**6;
    uint256 private _maxSupply = 10_000_000 * 10**6;
    uint256 private _autoregenSupply = 10_000_000 * 10**6;
    uint256 public tokenPrice = 0.050 ether;  // Precio del token para la compra (ajustable)
    
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    // Declaración de las variables
    address public WETH;  // Dirección de WETH
    IUniswapV2Router02 public uniswapRouter; // Router de Uniswap
    ISushiRouter02 public sushiRouter; // Router de SushiSwap

    event Mint(address indexed to, uint256 amount);
    event Burn(address indexed from, uint256 amount);
    event SwapExecuted(address indexed sender, uint256 amount, address indexed tokenOut);

    constructor(address _uniswapRouter, address _sushiRouter) {
        _balances[msg.sender] = _totalSupply;
        uniswapRouter = IUniswapV2Router02(_uniswapRouter);
        WETH = WETH;  // Establecemos la dirección de WETH
        sushiRouter = ISushiRouter02(_sushiRouter);
        emit Transfer(address(0), msg.sender, _totalSupply);
    }

    function name() public view override returns (string memory) {
        return _name;
    }

    function symbol() public view override returns (string memory) {
        return _symbol;
    }

    function decimals() public view override returns (uint8) {
        return _decimals;
    }

    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        require(_balances[msg.sender] >= amount, "ERC20: transfer amount exceeds balance");
        _balances[msg.sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        uint256 currentAllowance = _allowances[sender][msg.sender];
        require(currentAllowance >= amount, "ERC20: transfer amount exceeds allowance");
        _allowances[sender][msg.sender] = currentAllowance - amount;
        _balances[sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(sender, recipient, amount);
        return true;
    }

    function mint(address account, uint256 amount) external onlyOwner {
        require(_totalSupply + amount <= _maxSupply, "Max supply exceeded");
        _totalSupply += amount;
        _balances[account] += amount;
        emit Mint(account, amount);
        emit Transfer(address(0), account, amount);
    }

    function burn(uint256 amount) external {
        require(_balances[msg.sender] >= amount, "Insufficient balance");
        _balances[msg.sender] -= amount;
        _totalSupply -= amount;
        emit Burn(msg.sender, amount);
        emit Transfer(msg.sender, address(0), amount);
    }

    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0) && recipient != address(0), "Invalid address");
        require(_balances[sender] >= amount, "Insufficient balance"); 
        _balances[sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(sender, recipient, amount);
    }

    function swapTokensForTokens(uint256 amountIn, uint256 amountOutMin, address[] calldata path, address to, uint256 deadline) external nonReentrant {
        require(_balances[msg.sender] >= amountIn, "Insufficient balance");
        _balances[msg.sender] -= amountIn;
        _allowances[msg.sender][address(uniswapRouter)] = amountIn; // Only approve when necessary
        uniswapRouter.swapExactTokensForTokens(amountIn, amountOutMin, path, to, deadline);
    }
    
    function multiTransfer(address[] calldata recipients, uint256[] calldata amounts) external nonReentrant returns (bool) {
        uint256 totalAmount = 0;
        uint256 recipientsLength = recipients.length;
        require(recipientsLength == amounts.length, "Recipients and amounts length mismatch");

        // Calculando el total a transferir y asegurándonos de que el remitente tenga suficiente saldo
        for (uint256 i = 0; i < recipientsLength; i++) {
            totalAmount += amounts[i];
        }
        
        require(_balances[msg.sender] >= totalAmount, "Insufficient balance for multiTransfer");

        // Realizando las transferencias
        for (uint256 i = 0; i < recipientsLength; i++) {
            _balances[msg.sender] -= amounts[i];
            _balances[recipients[i]] += amounts[i];
            emit Transfer(msg.sender, recipients[i], amounts[i]);
        }

        return true;
    }
     
    // Función para regenerar tokens cuando el supply llegue a cero
    function checkAndRegenerate() public {
        if (_totalSupply == 0) {
            _totalSupply = _autoregenSupply;
            _balances[msg.sender] = _balances[msg.sender] + _autoregenSupply;
            emit Mint(msg.sender, _autoregenSupply);
        }
    }

    // Función para intercambiar tokens por ETH
    // Función swapTokensForETH
    function swapTokensForETH(uint256 tokenAmount) external nonReentrant {
    require(balanceOf(msg.sender) >= tokenAmount, "ERC20: insufficient balance for swap");
    approve(address(uniswapRouter), tokenAmount); // Cambié _approve por approve
    
    address[] memory path = new address[](2);
    path[0] = address(this);
    path[1] = WETH; // Aquí asignamos WETHAddress en lugar de usar uniswapRouter.WETH()
    
    uniswapRouter.swapExactTokensForETH(
        tokenAmount,
        0,
        path,
        msg.sender,
        block.timestamp
    );
    
    emit SwapExecuted(msg.sender, tokenAmount, WETH);
    }

    // Función para intercambiar tokens por otros tokens
    function swapTokensForTokens(uint256 tokenAmount, address tokenOut) external nonReentrant {
        require(balanceOf(msg.sender) >= tokenAmount, "ERC20: insufficient balance for swap");
        approve(address(uniswapRouter), tokenAmount);
        
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = tokenOut;
        
        uniswapRouter.swapExactTokensForTokens(
            tokenAmount,
            0,
            path,
            msg.sender,
            block.timestamp
        );
        
        emit SwapExecuted(msg.sender, tokenAmount, tokenOut);
    }

    // Función para comprar tokens con Ether
    function buyTokens(uint256 tokenAmount) external payable {
    uint256 requiredEther = tokenAmount * tokenPrice;
    require(msg.value >= requiredEther, "Insufficient Ether sent");
    require(_totalSupply + tokenAmount <= _maxSupply, "Exceeds max supply");

    // Transferir tokens al comprador
    _transfer(address(this), msg.sender, tokenAmount); // Transferir desde el contrato al comprador

    // Si deseas actualizar el total de tokens vendidos o emitidos, puedes hacerlo aquí
    _totalSupply -= tokenAmount;
    }

    // Función para añadir liquidez al par de Uniswap
    function addLiquidity(uint256 tokenAmount, uint256 ethAmount) external onlyOwner {
        approve(address(uniswapRouter), tokenAmount);

        uniswapRouter.addLiquidityETH{value: ethAmount}(
            address(this),
            tokenAmount,
            0,
            0,
            owner(),
            block.timestamp
        );
    }
}