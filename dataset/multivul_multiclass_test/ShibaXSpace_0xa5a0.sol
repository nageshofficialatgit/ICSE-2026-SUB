// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

contract Ownable is Context {
    address private _owner;

    constructor() {
        _owner = _msgSender();
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Je jo-autorizuar.");
        _;
    }

    function owner() public view returns (address) {
        return _owner;
    }
}

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

contract ShibaXSpace is Context, IERC20, Ownable {
    string public name = "ShibaXSpace";
    string public symbol = "SXS";
    uint8 public decimals = 18;
    uint256 private _totalSupply;
    bool public tradingEnabled = false;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    constructor(uint256 initialSupply) {
        _mint(_msgSender(), initialSupply * 10 ** uint256(decimals));
    }

    function enableTrading() external onlyOwner {
        tradingEnabled = true;
    }

    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        require(tradingEnabled || _msgSender() == owner(), "Shitja eshte e fikur per momentin.");
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner_, address spender) public view override returns (uint256) {
        return _allowances[owner_][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        require(tradingEnabled || sender == owner(), "Shitja eshte e fikur per momentin.");
        _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), _allowances[sender][_msgSender()] - amount);
        return true;
    }

    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0) && recipient != address(0), "Adrese jo e vlefshme.");
        require(_balances[sender] >= amount, "Balanca e pamjaftueshme.");

        _balances[sender] -= amount;
        _balances[recipient] += amount;
    }

    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "Adrese jo e vlefshme.");
        _totalSupply += amount;
        _balances[account] += amount;
    }

    function _approve(address owner_, address spender, uint256 amount) internal {
        require(owner_ != address(0) && spender != address(0), "Adrese jo e vlefshme.");
        _allowances[owner_][spender] = amount;
    }

    // Funksioni për blerjen e tokenëve
    function buyTokens() external payable {
        uint256 amountToBuy = msg.value; // Përdorimi i ETH për të blerë tokenë
        uint256 tokenPrice = 0.1 ether; // Çmimi i tokenit (për 1 token = 0.1 ETH)
        uint256 amountOfTokens = amountToBuy / tokenPrice;

        uint256 ownerBalance = balanceOf(owner()); // Balanca e pronarit të kontratës

        // Kontrollo nëse ka mjaftueshëm token-a për të blerë
        require(amountOfTokens <= ownerBalance, "Not enough tokens available.");

        // Transferon token-et tek përdoruesi
        _transfer(owner(), msg.sender, amountOfTokens);

        // Dërgon ETH-në në wallet-in e pronarit
        payable(owner()).transfer(msg.value);
    }
}