// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// OpenZeppelin ERC20 ve Ownable Kodlarını Dahil Ettik

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(address initialOwner) {
        _transferOwnership(initialOwner);
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

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

contract ERC20 is Context, IERC20 {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;
    string private _name;
    string private _symbol;
    uint8 private _decimals;

    constructor(string memory name_, string memory symbol_, uint8 decimals_) {
        _name = name_;
        _symbol = symbol_;
        _decimals = decimals_;
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

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _update(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _update(sender, recipient, amount);
        _approve(sender, _msgSender(), _allowances[sender][_msgSender()] - amount);
        return true;
    }

    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to the zero address");
        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    function _update(address sender, address recipient, uint256 amount) internal virtual {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");
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
}

contract ZOOToken is ERC20, Ownable {
    mapping(address => bool) public isExempt;
    mapping(address => bool) public isDexPair;
    bool public tradingEnabled = false;
    uint256 public taxRate;
    address public liquidityWallet;

    constructor(
        uint256 _initialSupply,
        uint256 _taxRate,
        address _owner
    ) ERC20("ZOO", "ZOO", 18) Ownable(_owner) {
        _mint(_owner, _initialSupply * 10**18);
        taxRate = _taxRate;
        liquidityWallet = 0x59B966c224F457554685B0731a3D21298cc9fbfd;
        isExempt[_owner] = true;
        isExempt[liquidityWallet] = true;
    }

    function enableTrading() external onlyOwner {
        tradingEnabled = true;
    }

    function setDexPair(address pair, bool status) external onlyOwner {
        isDexPair[pair] = status;
    }

    function setExempt(address account, bool exempt) external onlyOwner {
        isExempt[account] = exempt;
    }

    function _update(address sender, address recipient, uint256 amount) internal override {
        if ((isDexPair[sender] || isDexPair[recipient]) && !isExempt[sender]) {
            require(tradingEnabled, "Trading is not enabled yet");
            require(!isDexPair[recipient], "Selling is restricted");
        }
        
        uint256 fee = 0;
        if (!isExempt[sender] && !isExempt[recipient] && taxRate > 0) {
            fee = (amount * taxRate) / 10000;
            super._update(sender, liquidityWallet, fee);
        }
        
        super._update(sender, recipient, amount - fee);
    }

    function withdrawLiquidity(address to, uint256 amount) external onlyOwner {
        require(balanceOf(liquidityWallet) >= amount, "Insufficient balance!");
        super._update(liquidityWallet, to, amount);
    }
}