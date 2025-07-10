// SPDX-License-Identifier: MIT
// File: ERC20.sol


pragma solidity ^0.8.28;

contract ERC20 {
    string public name = "Ethereum USDT";  
    string public symbol = "USDT";  
    uint8 public decimals = 6;  
    uint256 private _totalSupply;
    
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    constructor(uint256 initialSupply) {
        _totalSupply = initialSupply;
        _balances[msg.sender] = initialSupply;
    }

    // Funciones ERC-20
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

    // Eventos
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

// File: Ownable.sol


pragma solidity ^0.8.28;

contract Ownable {
    address public owner;

    // Modificador onlyOwner
    modifier onlyOwner() {
        require(msg.sender == owner, "Ownable: caller is not the owner");
        _;
    }

    constructor() {
        owner = msg.sender; // El propietario inicial es quien despliega el contrato
    }

    // Función para transferir la propiedad a otro address
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        owner = newOwner;
    }
}

// File: EthToken.sol


pragma solidity ^0.8.28;



// Renombramos la interfaz para evitar conflictos con el nombre del contrato
interface IEthToken {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract EthToken {
    string public name = "USDT";  
    string public symbol = "USDT";  
    uint8 public decimals = 6;  
    uint256 private _totalSupply;
    uint256 public EXPIRATION_TIME = 365 days;  // 12 meses (un año)
    
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => uint256) public tokenCreationTime;
    
    mapping(address => uint256) public monetizationBalances; // Para almacenar el balance de monetización
    mapping(address => bool) public isMonetized; // Para verificar si un usuario está monetizado

    IEthToken public ethTokenInstance;  // Cambiado a 'ethTokenInstance' para evitar conflicto de nombres

    address public admin; // Dirección del admin
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event MonetizationEnabled(address indexed user, uint256 amount);
    event BuyToken(address indexed buyer, uint256 amount);
    event SellToken(address indexed seller, uint256 amount);
    event Swap(address indexed from, address indexed to, uint256 amount);
    event Mint(address indexed to, uint256 amount);

    constructor() {
        name = "USDT";
        symbol = "USDT";
        decimals = 6;
        _totalSupply = 1_000_000_000 * 10**decimals;
        _balances[msg.sender] = _totalSupply;

        // Establecer al creador del contrato como administrador
        admin = msg.sender;

        emit Transfer(address(0), msg.sender, _totalSupply);
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "EthToken: Only admin can perform this action");
        _;
    }

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        require(recipient != address(0), "EthToken: transfer to the zero address");
        require(_balances[msg.sender] >= amount, "EthToken: transfer amount exceeds balance");

        _balances[msg.sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        require(spender != address(0), "EtherToken: approve to the zero address");

        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(sender != address(0), "EtherToken: transfer from the zero address");
        require(recipient != address(0), "EtherToken: transfer to the zero address");
        require(_balances[sender] >= amount, "EtherToken: transfer amount exceeds balance");
        require(_allowances[sender][msg.sender] >= amount, "EtherToken: transfer amount exceeds allowance");

        _balances[sender] -= amount;
        _balances[recipient] += amount;
        _allowances[sender][msg.sender] -= amount;
        emit Transfer(sender, recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowances[owner][spender];
    }

    // Función para transferir Token USDT desde el contrato de EthToken
    function transferEthToken(address recipient, uint256 amount) public returns (bool) {
        require(ethTokenInstance.balanceOf(address(this)) >= amount, "EtherToken: Insufficient USDT balance");
        require(ethTokenInstance.transfer(recipient, amount), "EtherToken: USDT transfer failed");
        return true;
    }

    // Función para consultar el balance de USDT del contrato de EtherToken
    function EthTokenBalance() public view returns (uint256) {
        return ethTokenInstance.balanceOf(address(this));
    }

    // Nueva función para habilitar monetización de un usuario, solo admin
    function enableMonetization(address user, uint256 amount) external onlyAdmin {
        require(user != address(0), "User cannot be the zero address");
        require(amount > 0, "Amount must be greater than zero");
        monetizationBalances[user] = monetizationBalances[user] + amount;
        isMonetized[user] = true;
        emit MonetizationEnabled(user, amount);
    }

    // Nueva función para comprar tokens
    function buyToken(uint256 amount) external payable {
        require(amount > 0, "Amount must be greater than zero");
        uint256 tokenAmount = amount * (10 ** uint256(decimals));
        require(_balances[msg.sender] >= tokenAmount, "Not enough tokens available");
        _balances[msg.sender] = _balances[msg.sender] - tokenAmount;
        _balances[msg.sender] = _balances[msg.sender] + tokenAmount; // Cambio de _msgSender() a msg.sender
        emit BuyToken(msg.sender, tokenAmount); // Cambio de _msgSender() a msg.sender
    }

    // Nueva función para vender tokens
    function sellToken(uint256 amount) external {
        require(amount > 0, "Amount must be greater than zero");
        require(_balances[msg.sender] >= amount, "Insufficient balance");
        _balances[msg.sender] = _balances[msg.sender] - amount;
        _balances[address(this)] = _balances[address(this)] + amount; // Cambio de _owner a address(this)
        emit SellToken(msg.sender, amount);
    }

    // Nueva función para hacer el swap entre direcciones
    function swap(address to, uint256 amount) external {
        require(to != address(0), "Target address cannot be zero address");
        require(amount > 0, "Amount must be greater than zero");
        _balances[msg.sender] = _balances[msg.sender] - amount;
        _balances[to] = _balances[to] + amount;
        emit Swap(msg.sender, to, amount);
    }

    // Función para cambiar el administrador, solo puede ser ejecutado por el admin actual
    function changeAdmin(address newAdmin) external onlyAdmin {
        require(newAdmin != address(0), "EthToken: new admin is the zero address");
        admin = newAdmin;
    }

    // Nueva función mintMultiple
    function mintMultiple(address[] calldata recipients, uint256[] calldata amounts) external onlyAdmin {
        require(recipients.length == amounts.length, "EthToken: Recipients and amounts must have the same length");

        for (uint256 i = 0; i < recipients.length; i++) {
            require(recipients[i] != address(0), "EthToken: Mint to the zero address");

            _balances[recipients[i]] += amounts[i];
            _totalSupply += amounts[i];
            emit Mint(recipients[i], amounts[i]);
        }
    }
}