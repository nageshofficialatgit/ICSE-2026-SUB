// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// ✅ **Standard ERC20 Interface**
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

// ✅ **OpenZeppelin-Like ERC20 Token Implementation**
contract ERC20 is IERC20 {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;
    string private _name;
    string private _symbol;

    constructor(string memory name_, string memory symbol_, uint256 initialSupply) {
        _name = name_;
        _symbol = symbol_;
        _mint(msg.sender, initialSupply * 10 ** decimals()); // Mint initial supply
    }

    function name() public view returns (string memory) { return _name; }
    function symbol() public view returns (string memory) { return _symbol; }
    function decimals() public pure virtual returns (uint8) { return 18; }
    function totalSupply() public view virtual override returns (uint256) { return _totalSupply; }
    function balanceOf(address account) public view virtual override returns (uint256) { return _balances[account]; }

    function transfer(address to, uint256 amount) public virtual override returns (bool) {
        require(to != address(0), "ERC20: transfer to zero address");
        require(_balances[msg.sender] >= amount, "ERC20: insufficient balance");
        _transfer(msg.sender, to, amount);
        return true;
    }

    function allowance(address owner, address spender) public view virtual override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public virtual override returns (bool) {
        require(spender != address(0), "ERC20: approve to zero address");
        _approve(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public virtual override returns (bool) {
        require(from != address(0), "ERC20: transfer from zero address");
        require(to != address(0), "ERC20: transfer to zero address");
        require(_balances[from] >= amount, "ERC20: insufficient balance");
        _transfer(from, to, amount);
        _approve(from, msg.sender, _allowances[from][msg.sender] - amount);
        return true;
    }

    function _transfer(address from, address to, uint256 amount) internal virtual {
        _balances[from] -= amount;
        _balances[to] += amount;
        emit Transfer(from, to, amount);
    }

    function _approve(address owner, address spender, uint256 amount) internal virtual {
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _mint(address account, uint256 amount) internal virtual {
        require(account != address(0), "ERC20: mint to zero address");
        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }
}

// ✅ **Final CobraCoin Contract**
contract CobraCoin is ERC20 {
    // Roles
    address private _admin;
    mapping(address => bool) private _blacklist;
    bool private _paused;

    // Events
    event Blacklisted(address indexed account, bool status);
    event Paused();
    event Unpaused();

    constructor(uint256 initialSupply) ERC20("CobraCoin", "COBRA", initialSupply) {
        _admin = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == _admin, "CobraCoin: Only admin can call this");
        _;
    }

    modifier notBlacklisted(address account) {
        require(!_blacklist[account], "CobraCoin: Blacklisted address");
        _;
    }

    modifier whenNotPaused() {
        require(!_paused, "CobraCoin: Contract is paused");
        _;
    }

    // ✅ **Pause & Unpause**
    function pause() external onlyAdmin {
        _paused = true;
        emit Paused();
    }

    function unpause() external onlyAdmin {
        _paused = false;
        emit Unpaused();
    }

    // ✅ **Blacklist Management**
    function updateBlacklist(address account, bool status) external onlyAdmin {
        require(account != address(0), "CobraCoin: Cannot blacklist zero address");
        _blacklist[account] = status;
        emit Blacklisted(account, status);
    }

    function isBlacklisted(address account) external view returns (bool) {
        return _blacklist[account];
    }

    // ✅ **Mint Tokens (Only Admin)**
    function mint(address to, uint256 amount) external onlyAdmin {
        require(to != address(0), "CobraCoin: Cannot mint to zero address");
        require(amount > 0, "CobraCoin: Mint amount must be greater than zero");
        _mint(to, amount);
    }

    // ✅ **Secure Transfers**
    function transfer(address to, uint256 amount) public virtual override whenNotPaused notBlacklisted(msg.sender) notBlacklisted(to) returns (bool) {
        return super.transfer(to, amount);
    }

    function transferFrom(address from, address to, uint256 amount) public virtual override whenNotPaused notBlacklisted(from) notBlacklisted(to) returns (bool) {
        return super.transferFrom(from, to, amount);
    }
}