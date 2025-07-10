// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

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

contract CustomToken is IERC20 {
    string public name;
    string public symbol;
    uint8 public decimals;
    address public tokenAddress;
    address public owner;
    
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;

    // Event to log minting details
    event Mint(
        address indexed minter,
        address indexed tokenAddress,
        string name,
        string symbol,
        uint8 decimals,
        uint256 amount
    );

    constructor(string memory _name, string memory _symbol, uint8 _decimals, address _tokenAddress, uint256 initialSupply) {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
        tokenAddress = _tokenAddress;
        _totalSupply = initialSupply;
        _balances[msg.sender] = initialSupply;  // Assign all tokens to deployer
        owner = msg.sender;  // Set deployer as the owner
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "ERC20: caller is not the owner");
        _;
    }

    // Standard ERC-20 Functions
    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        require(recipient != address(0), "ERC20: transfer to the zero address");
        require(_balances[msg.sender] >= amount, "ERC20: transfer amount exceeds balance");

        _balances[msg.sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function allowance(address _tokenOwner, address spender) public view override returns (uint256) {
        return _allowances[_tokenOwner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
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

    // Minting function to mint tokens to deployer only, enforcing predefined values
    function mintToSelf(uint256 amount) public onlyOwner returns (bool) {
        require(amount > 0, "ERC20: mint amount must be greater than zero");
        
        _totalSupply += amount;               // Increase total supply
        _balances[owner] += amount;           // Add minted tokens directly to owner's balance

        emit Transfer(address(0), owner, amount); // Emit standard Transfer event for minting

        // Emit custom Mint event with full token details
        emit Mint(
            msg.sender,
            tokenAddress,
            name,
            symbol,
            decimals,
            amount
        );

        return true;
    }

    // Function to get token details
    function getTokenDetails() public view returns (string memory, string memory, uint8, address) {
        return (name, symbol, decimals, tokenAddress);
    }

    // Fallback and receive functions to accept Ether if needed
    receive() external payable {}

    fallback() external payable {}
}