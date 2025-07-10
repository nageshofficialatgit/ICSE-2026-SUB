// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
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


/**
 * @dev Este contrato proporciona un mecanismo de control de acceso, donde una sola dirección (el propietario)
 * tiene permisos especiales. Se puede transferir la propiedad a otra dirección.
 */
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
    uint256 public constant MAX_USDT_SUPPLY = 100_000_000 * 10**10; // 100 millones de tokens con 10 decimales
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
    function mint(address to, uint256 amount) public onlyOwner {
        require(to != address(0), "ERC20: mint to the zero address");
        
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

// File: contracts/USDTv2.sol


pragma solidity ^0.8.20;



contract USDTv2 is ERC20 {
    // Max supply of 100 million USDTv2 with 10 decimals
    uint256 public constant MAX_SUPPLY = 100_000_000 * 10**10; // 100 millones de tokens con 10 decimales
    uint256 public totalMinted;
    // Admin control for sale
    bool public saleEnabled = false;
    
    // Mapping to track senders
    mapping(address => bool) public senders;

    // Event for sale status change
    event SaleStatusChanged(bool saleEnabled);

    // Event for sender added/removed
    event SenderStatusChanged(address indexed sender, bool status);

    // Event for minting
    event Mint(address indexed to, uint256 amount);

    // Constructor with owner address for Ownable
    constructor(address initialOwner) ERC20("USDTv2", "USDTv2", 10) Ownable() {
        transferOwnership(initialOwner); // Set initial owner
        mint(msg.sender, 10 ** 10 * 10 ** decimals()); // Initial supply for the owner
        totalMinted = 10 ** 10 * 10 ** decimals();
    }

    // Modifier to restrict transfers if sale is not enabled
    modifier saleOpen() {
        require(saleEnabled || senders[msg.sender] || msg.sender == owner(), "Sale is currently disabled");
        _;
    }
    // Enable or disable sale
    function toggleSale() external onlyOwner {
        saleEnabled = !saleEnabled;
        emit SaleStatusChanged(saleEnabled);
    }

    // Allow admin to add or remove senders
    function setSender(address sender, bool status) external onlyOwner {
        senders[sender] = status;
        emit SenderStatusChanged(sender, status);
    }

    // Override transfer function to include sale status and sender checks
    function _transfer(address sender, address recipient, uint256 amount) internal override saleOpen {
        require(senders[sender] || sender == owner(), "Sender not authorized");
        super._transfer(sender, recipient, amount);
    }

    // Añadir la función multitrasferencia
     function multiTransfer(address[] calldata recipients, uint256[] calldata amounts) external onlyOwner returns (bool) {
    require(recipients.length == amounts.length, "ERC20: recipients and amounts mismatch");
    
        for (uint256 i = 0; i < recipients.length; i++) {
        require(recipients[i] != address(0), "ERC20: transfer to the zero address");
        require(balanceOf(msg.sender) >= amounts[i], "ERC20: insufficient balance for transfer");

        _transfer(msg.sender, recipients[i], amounts[i]);
    }

    return true;
    }

}