// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

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

contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _transferOwnership(_msgSender());
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
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

contract HouseADS is Context, IERC20, Ownable {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _isExcludedFromFee;
    mapping(address => bool) public blacklisted;

    uint256 private _totalSupply;
    uint8 private constant _decimals = 18;
    string private constant _name = "HouseADS";
    string private constant _symbol = "HADS";
    
    uint256 public constant TOTAL_SUPPLY = 1_000_000_000 * 10**18; // 1 billion tokens
    
    uint256 public buyFee = 300;  // 3%
    uint256 public sellFee = 300; // 3%
    uint256 private constant FEE_DENOMINATOR = 10000;
    
    address public marketingWallet;
    address public devWallet;
    
    bool public tradingEnabled;
    
    uint256 public maxTxAmount;
    uint256 public maxWalletAmount;
    
    event TradingEnabled(bool enabled);
    
    constructor(address _marketingWallet, address _devWallet) {
        require(_marketingWallet != address(0), "Marketing wallet cannot be zero");
        require(_devWallet != address(0), "Dev wallet cannot be zero");
        
        marketingWallet = _marketingWallet;
        devWallet = _devWallet;
        
        _totalSupply = TOTAL_SUPPLY;
        _balances[_msgSender()] = TOTAL_SUPPLY;
        
        maxTxAmount = TOTAL_SUPPLY / 200;    // 0.5% of total supply
        maxWalletAmount = TOTAL_SUPPLY / 50; // 2% of total supply
        
        _isExcludedFromFee[owner()] = true;
        _isExcludedFromFee[address(this)] = true;
        
        emit Transfer(address(0), _msgSender(), TOTAL_SUPPLY);
    }
    
    function name() public pure returns (string memory) {
        return _name;
    }
    
    function symbol() public pure returns (string memory) {
        return _symbol;
    }
    
    function decimals() public pure returns (uint8) {
        return _decimals;
    }
    
    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }
    
    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }
    
    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
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
        _transfer(sender, recipient, amount);
        
        uint256 currentAllowance = _allowances[sender][_msgSender()];
        require(currentAllowance >= amount, "ERC20: transfer amount exceeds allowance");
        unchecked {
            _approve(sender, _msgSender(), currentAllowance - amount);
        }
        
        return true;
    }
    
    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from zero address");
        require(spender != address(0), "ERC20: approve to zero address");
        
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
    
    function _transfer(address sender, address recipient, uint256 amount) private {
        require(sender != address(0), "ERC20: transfer from zero address");
        require(recipient != address(0), "ERC20: transfer to zero address");
        require(amount > 0, "Transfer amount must be greater than zero");
        require(!blacklisted[sender] && !blacklisted[recipient], "Address is blacklisted");
        require(tradingEnabled || _isExcludedFromFee[sender] || _isExcludedFromFee[recipient], "Trading not enabled");
        
        if (!_isExcludedFromFee[sender] && !_isExcludedFromFee[recipient]) {
            require(amount <= maxTxAmount, "Transfer amount exceeds the maxTxAmount");
            
            if (recipient != owner() && recipient != address(this)) {
                uint256 recipientBalance = balanceOf(recipient);
                require(recipientBalance + amount <= maxWalletAmount, "Max wallet limit exceeded");
            }
        }
        
        uint256 fee;
        
        // Calculate fee
        if (!_isExcludedFromFee[sender] && !_isExcludedFromFee[recipient]) {
            fee = (amount * (sender == owner() ? buyFee : sellFee)) / FEE_DENOMINATOR;
            
            if (fee > 0) {
                _balances[address(this)] += fee;
                uint256 marketingShare = fee / 2;
                uint256 devShare = fee - marketingShare;
                
                _balances[marketingWallet] += marketingShare;
                _balances[devWallet] += devShare;
                
                emit Transfer(sender, address(this), fee);
                emit Transfer(address(this), marketingWallet, marketingShare);
                emit Transfer(address(this), devWallet, devShare);
            }
        }
        
        uint256 transferAmount = amount - fee;
        
        _balances[sender] = _balances[sender] - amount;
        _balances[recipient] = _balances[recipient] + transferAmount;
        
        emit Transfer(sender, recipient, transferAmount);
    }
    
    // Owner functions
    function enableTrading() external onlyOwner {
        tradingEnabled = true;
        emit TradingEnabled(true);
    }
    
    function setFees(uint256 _buyFee, uint256 _sellFee) external onlyOwner {
        require(_buyFee <= 2500 && _sellFee <= 2500, "Fee cannot exceed 25%");
        buyFee = _buyFee;
        sellFee = _sellFee;
    }
    
    function setWallets(address _marketingWallet, address _devWallet) external onlyOwner {
        require(_marketingWallet != address(0) && _devWallet != address(0), "Cannot set zero address");
        marketingWallet = _marketingWallet;
        devWallet = _devWallet;
    }
    
    function setMaxTxAmount(uint256 _maxTxAmount) external onlyOwner {
        require(_maxTxAmount >= TOTAL_SUPPLY / 1000, "Max TX amount too low");
        maxTxAmount = _maxTxAmount;
    }
    
    function setMaxWalletAmount(uint256 _maxWalletAmount) external onlyOwner {
        require(_maxWalletAmount >= TOTAL_SUPPLY / 100, "Max wallet amount too low");
        maxWalletAmount = _maxWalletAmount;
    }
    
    function setExcludedFromFee(address account, bool excluded) external onlyOwner {
        _isExcludedFromFee[account] = excluded;
    }
    
    function setBlacklist(address account, bool blacklist) external onlyOwner {
        blacklisted[account] = blacklist;
    }
    
    function isExcludedFromFee(address account) public view returns (bool) {
        return _isExcludedFromFee[account];
    }
    
    // Emergency functions
    function withdrawStuckTokens(address token) external onlyOwner {
        require(token != address(this), "Cannot withdraw native tokens");
        uint256 amount = IERC20(token).balanceOf(address(this));
        IERC20(token).transfer(owner(), amount);
    }
}