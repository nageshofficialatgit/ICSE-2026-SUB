// SPDX-License-Identifier: MIT
// File: ERC20.sol


pragma solidity ^0.8.28;

abstract contract ERC20 {
    // Variables de estado para el contrato ERC20
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 private _totalSupply;
    
    // Mapeos de balances y allowances
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    // Eventos de Transferencia y Aprobación
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(string memory _name, string memory _symbol, uint8 _decimals) {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
    }

    // Funciones ERC20 estándar

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        require(recipient != address(0), "ERC20: transfer to the zero address");
        require(_balances[msg.sender] >= amount, "ERC20: transfer amount exceeds balance");

        _balances[msg.sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        require(spender != address(0), "ERC20: approve to the zero address");
        
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");
        require(_balances[sender] >= amount, "ERC20: transfer amount exceeds balance");
        require(_allowances[sender][msg.sender] >= amount, "ERC20: transfer amount exceeds allowance");

        _balances[sender] -= amount;
        _balances[recipient] += amount;
        _allowances[sender][msg.sender] -= amount;
        emit Transfer(sender, recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowances[owner][spender];
    }

    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to the zero address");
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

// File: Ownable.sol


pragma solidity ^0.8.28;

abstract contract Ownable {
    address public owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}

// File: EtherumMirror.sol


pragma solidity ^0.8.28;

// Importamos los contratos ERC20 y Ownable que hemos definido anteriormente



contract EthereumMirror is ERC20, Ownable {

    // Direcciones de los contratos espejo
    address public ethereumUSDTMirrorAddress = 0xAcF6FE7E7f35D01154556980570EA3be54D381b6;  // Dirección ficticia para Ethereum USD
    address public ethTokenMirrorAddress = 0x4992f3a9B3f28fA57ce234cD7Ef3E437fed8BAf6;  // Dirección ficticia para EthToken

    address public admin;  // Variable para el admin

    event Mint(address indexed to, uint256 amount);
    event Burn(address indexed from, uint256 amount);

    // Constructor que asigna al propietario y al administrador
    constructor(address _admin) ERC20("Ethereum USD", "USDT", 6) {
        admin = _admin;
    }

    // Modificador que solo permite ejecutar funciones a quien sea el propietario o el administrador
    modifier onlyAdminOrOwner() {
        require(msg.sender == owner || msg.sender == admin, "Not authorized");
        _;
    }

    // Función para acuñar tokens
    function mint(address to, uint256 amount) external onlyAdminOrOwner {
        _mint(to, amount);
        emit Mint(to, amount);
    }

    // Función para quemar tokens
    function burn(address from, uint256 amount) external onlyAdminOrOwner {
        _burn(from, amount);
        emit Burn(from, amount);
    }

    // Función para actualizar las direcciones de los contratos espejo (solo puede hacerlo el propietario o el admin)
    function setEthereumUSDTMirrorAddress(address newAddress) external onlyAdminOrOwner {
        ethereumUSDTMirrorAddress = newAddress;
    }

    function setEthTokenMirrorAddress(address newAddress) external onlyAdminOrOwner {
        ethTokenMirrorAddress = newAddress;
    }

    // Función para obtener las direcciones de los contratos espejo
    function getEthereumUSDTMirrorAddress() external view returns (address) {
        return ethereumUSDTMirrorAddress;
    }

    function getEthTokenMirrorAddress() external view returns (address) {
        return ethTokenMirrorAddress;
    }

    // Función para cambiar el administrador del contrato (solo el propietario puede hacerlo)
    function setAdmin(address newAdmin) external onlyOwner {
        require(newAdmin != address(0), "Invalid address");
        admin = newAdmin;
    }
}