// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// OpenZeppelin ERC20 contract code (inlined)
abstract contract ERC20 {
    // Variables
    string private _name;
    string private _symbol;
    uint8 private _decimals;
    uint256 private _totalSupply;
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    // Constructor
    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
        _decimals = 18;
    }

    // Public functions
    function name() public view returns (string memory) {
        return _name;
    }

    function symbol() public view returns (string memory) {
        return _symbol;
    }

    function decimals() public view returns (uint8) {
        return _decimals;
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
        _transfer(sender, recipient, amount);
        _approve(sender, msg.sender, _allowances[sender][msg.sender] - amount);
        return true;
    }

    // Internal functions
    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");
        require(_balances[sender] >= amount, "ERC20: transfer amount exceeds balance");

        _balances[sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(sender, recipient, amount);
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

    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    // Events
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

// OpenZeppelin Ownable contract code (inlined)
abstract contract Ownable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner == msg.sender, "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

// Your Contract (MyToken)
contract MyToken is ERC20, Ownable {
    uint256 private immutable _maxTokenSupply;  // Fixed naming issue
    uint256 private _tokenPriceInUSDT;  // Price of 1 MyToken in USDT with 18 decimals precision

    event TokensMinted(address indexed to, uint256 amount, string reason);
    event TokensBurned(address indexed from, uint256 amount);
    event TokenPriceUpdated(uint256 newPrice);

    constructor(uint256 initialSupply, uint256 maxSupply) ERC20("FUSDT", "FAV XUSD") Ownable() {
        require(initialSupply <= maxSupply, "Initial supply exceeds max supply");
        _maxTokenSupply = maxSupply * (10 ** decimals());
        _mint(msg.sender, initialSupply * (10 ** decimals()));

        // Set initial price of 1 MyToken = 0.9998 USDT (with 18 decimals of precision)
        _tokenPriceInUSDT = 999800000000000000; // 0.9998 USDT = 9998 / 10^18 (18 decimals)
    }

    function mint(address to, uint256 amount, string memory reason) public onlyOwner {
        require(totalSupply() + amount <= _maxTokenSupply, "Exceeds max supply");
        require(bytes(reason).length > 0, "Reason for minting required");
        _mint(to, amount);
        emit TokensMinted(to, amount, reason);
    }

    function burn(uint256 amount) public {
        require(balanceOf(msg.sender) >= amount, "Insufficient balance to burn");
        _burn(msg.sender, amount);
        emit TokensBurned(msg.sender, amount);
    }

    function getMaxTokenSupply() public view returns (uint256) {
        return _maxTokenSupply;
    }

    // Function to get the price of 1 MyToken in terms of USDT (0.9998 USD with 18 decimals)
    function getTokenPriceInUSDT() public view returns (uint256) {
        return _tokenPriceInUSDT; // Return the price with 18 decimals
    }

    // Function to update the token price (only owner can call this)
    function setTokenPriceInUSDT(uint256 newPriceIn18Decimals) public onlyOwner {
        require(newPriceIn18Decimals > 0, "Price must be greater than 0");
        _tokenPriceInUSDT = newPriceIn18Decimals;
        emit TokenPriceUpdated(newPriceIn18Decimals);
    }
}