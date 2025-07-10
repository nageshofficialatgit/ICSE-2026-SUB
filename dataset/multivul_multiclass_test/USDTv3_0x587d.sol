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
// File: contracts/IERC20Metadata.sol


pragma solidity ^0.8.20;

interface IERC20Metadata is IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}
// File: contracts/Context.sol


pragma solidity ^0.8.20;


abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
    
    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}
// File: contracts/Ownable.sol


pragma solidity ^0.8.20;

abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Establece la dirección inicial como el propietario del contrato.
     */
    constructor() {
        _transferOwnership(_msgSender());
    }

    /**
     * @dev Devuelve la dirección del propietario del contrato.
     */
    function owner() public view returns (address) {
        return _owner;
    }

    /**
     * @dev Modifier que permite solo al propietario ejecutar ciertas funciones.
     */
    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    /**
     * @dev Permite al propietario transferir la propiedad a una nueva dirección.
     * @param newOwner La dirección que será el nuevo propietario.
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    /**
     * @dev Permite al propietario renunciar a su propiedad del contrato.
     * Esto hará que el contrato ya no tenga propietario.
     */
    function renounceOwnership() public onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Internamente realiza la transferencia de la propiedad.
     * @param newOwner La nueva dirección del propietario.
     */
    function _transferOwnership(address newOwner) internal {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}
// File: contracts/ERC20.sol


pragma solidity ^0.8.20;





contract ERC20 is Context, IERC20, IERC20Metadata, Ownable {
    string private _name;
    string private _symbol;
    uint8 private _decimals;
    uint256 public constant MAX_SUPPLY = 100_000_000 * 10**6; // 100 billones de tokens con 10 decimales
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;

    constructor(string memory name_, string memory symbol_, uint8 decimals_) {
        _name = name_;
        _symbol = symbol_;
        _decimals = decimals_;
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

    function transfer(address recipient, uint256 amount) public virtual override returns (bool) {
        address sender = _msgSender();
        _transfer(sender, recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        address owner = _msgSender();
        _approve(owner, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public virtual override returns (bool) {
        address spender = _msgSender();
        _spendAllowance(sender, spender, amount);
        _transfer(sender, recipient, amount);
        return true;
    }

    function _transfer(address sender, address recipient, uint256 amount) internal virtual {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");

        uint256 senderBalance = _balances[sender];
        require(senderBalance >= amount, "ERC20: transfer amount exceeds balance");

        unchecked {
            _balances[sender] = senderBalance - amount;
        }
        _balances[recipient] += amount;

        emit Transfer(sender, recipient, amount);
    }
    // Añadir la función mint
    // Función de mint mejorada con límite de suministro
    function mint(address to, uint256 amount) public virtual onlyOwner {
    require(to != address(0), "ERC20: mint to the zero address");
    require(_totalSupply + amount <= MAX_SUPPLY, "Minting would exceed MAX_SUPPLY");

    _totalSupply += amount;
    _balances[to] += amount;
    emit Transfer(address(0), to, amount);
    }
    
    

    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _spendAllowance(address owner, address spender, uint256 amount) internal {
        uint256 currentAllowance = allowance(owner, spender);
        require(currentAllowance >= amount, "ERC20: insufficient allowance");
        _approve(owner, spender, currentAllowance - amount);
    }
}
// File: contracts/IUniswapV2Router02.sol


pragma solidity ^0.8.20;

interface IUniswapV2Router02 {
    function factory() external pure returns (address);
    function WETH() external pure returns (address);

    // Add Liquidity
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

    // Remove Liquidity
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

    function removeLiquidityETHSupportingFeeOnTransferTokens(
        address token,
        uint liquidity,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external returns (uint amountETH);

    function removeLiquidityETHWithPermit(
        address token,
        uint liquidity,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline,
        bool approveMax,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external returns (uint amountToken, uint amountETH);

    // Swap
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

    function swapETHForExactTokens(uint amountOut, address[] calldata path, address to, uint deadline)
        external
        payable
        returns (uint[] memory amounts);

    // Utils
    function quote(uint amountA, uint reserveA, uint reserveB) external pure returns (uint amountB);
    function getAmountOut(uint amountIn, uint reserveIn, uint reserveOut) external pure returns (uint amountOut);
    function getAmountIn(uint amountOut, uint reserveIn, uint reserveOut) external pure returns (uint amountIn);
    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);
    function getAmountsIn(uint amountOut, address[] calldata path) external view returns (uint[] memory amounts);
}

// File: contracts/IUniswapV2Factory.sol


pragma solidity ^0.8.20;

interface IUniswapV2Factory {
    event PairCreated(address indexed token0, address indexed token1, address pair, uint);

    function feeTo() external view returns (address);
    function feeToSetter() external view returns (address);

    function getPair(address tokenA, address tokenB) external view returns (address pair);
    function createPair(address tokenA, address tokenB) external returns (address pair);

    function setFeeTo(address _feeTo) external;
    function setFeeToSetter(address _feeToSetter) external;
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

// File: contracts/USDTv2.sol


pragma solidity ^0.8.20;








contract USDTv3 is Ownable, ERC20, ReentrancyGuard {
    uint256 public constant MAX_USDT_SUPPLY = 100_000_000 * 10**6;
    uint256 public totalMinted;
    bool public saleEnabled = false;
    uint256 public tokenPrice = 1e6; // Precio por token en wei (ajustable)
    event TokenPurchased(address indexed buyer, uint256 amount, uint256 price);
    IUniswapV2Router02 public uniswapRouter;
    
    event SaleStatusChanged(bool saleEnabled);
    event Mint(address indexed to, uint256 amount);
    event SwapExecuted(address indexed user, uint256 amountIn, address tokenOut);

    constructor(address initialOwner, address _uniswapRouter) ERC20("USDTv3", "USDTv3", 6) Ownable() {
        transferOwnership(initialOwner);
        mint(msg.sender, MAX_USDT_SUPPLY);
        totalMinted = MAX_USDT_SUPPLY;
        uniswapRouter = IUniswapV2Router02(_uniswapRouter);
    }

    function _transfer(address sender, address recipient, uint256 amount) internal override {
        super._transfer(sender, recipient, amount);
    }

    function multiTransfer(address[] calldata recipients, uint256[] calldata amounts) external returns (bool) {
        require(recipients.length == amounts.length, "ERC20: recipients and amounts mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            require(recipients[i] != address(0), "ERC20: transfer to the zero address");
            require(balanceOf(msg.sender) >= amounts[i], "ERC20: insufficient balance for transfer");
            _transfer(msg.sender, recipients[i], amounts[i]);
        }
        return true;
    }

    function burn(uint256 amount) external {
        require(balanceOf(msg.sender) >= amount, "ERC20: burn amount exceeds balance");
        _transfer(msg.sender, address(0), amount);
    }

    function swapTokensForETH(uint256 tokenAmount) external nonReentrant {
        require(balanceOf(msg.sender) >= tokenAmount, "ERC20: insufficient balance for swap");
        _approve(msg.sender, address(uniswapRouter), tokenAmount);
        
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = uniswapRouter.WETH();
        
        uniswapRouter.swapExactTokensForETH(
            tokenAmount,
            0,
            path,
            msg.sender,
            block.timestamp
        );
        
        emit SwapExecuted(msg.sender, tokenAmount, uniswapRouter.WETH());
    }

    function swapTokensForTokens(uint256 tokenAmount, address tokenOut) external nonReentrant {
        require(balanceOf(msg.sender) >= tokenAmount, "ERC20: insufficient balance for swap");
        _approve(msg.sender, address(uniswapRouter), tokenAmount);
        
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = tokenOut;
        
        uniswapRouter.swapExactTokensForETH(
            tokenAmount,
            0,
            path,
            msg.sender,
            block.timestamp
        );
        
        emit SwapExecuted(msg.sender, tokenAmount, tokenOut);
    }
    function buyTokens(uint256 tokenAmount) external payable {
    uint256 requiredEther = tokenAmount * tokenPrice;
    require(msg.value >= requiredEther, "Insufficient Ether sent");

    // Asegurar que no se supera el suministro máximo
    require(totalMinted + tokenAmount <= MAX_USDT_SUPPLY, "Exceeds max supply");

    mint(msg.sender, tokenAmount);
    totalMinted += tokenAmount;
    } 
    function addLiquidity(uint256 tokenAmount, uint256 ethAmount) external onlyOwner {
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
    





}