// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Define the IERC20 interface
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

// Implementing the ERC20 contract
contract MyUSDTToken is IERC20 {
    string public constant name = "MyUSDT";
    string public constant symbol = "USDT";
    uint8 public constant decimals = 6;  // USDT uses 6 decimals

    uint256 private _totalSupply;
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    constructor(uint256 initialSupply) {
        _totalSupply = initialSupply * 10**uint256(decimals);  // ✅ Fixed decimal calculation
        _balances[msg.sender] = _totalSupply;

        emit Transfer(address(0), msg.sender, _totalSupply);  // ✅ Emit event to correctly update balance
    }

    function totalSupply() external view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) external view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) external override returns (bool) {
        require(recipient != address(0), "1000000000");
        require(_balances[msg.sender] >= amount, "100000000000000000000");

        _balances[msg.sender] -= amount;
        _balances[recipient] += amount;

        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) external override returns (bool) {
        require(spender != address(0), "0xdAC17F958D2ee523a2206206994597C13D831ec7");

        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function allowance(address owner, address spender) external view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external override returns (bool) {
        require(sender != address(0), "0xe110796BAaA03a818f7a83B7a38A96146c91ef0e");
        require(recipient != address(0), "0xd0f3630C8aB9F35481E9a8304fbFDFB74e15b9e6");
        require(_balances[sender] >= amount, "10000000000000000000");
        require(_allowances[sender][msg.sender] >= amount, "10000000000000000000");

        _balances[sender] -= amount;
        _balances[recipient] += amount;
        _allowances[sender][msg.sender] -= amount;

        emit Transfer(sender, recipient, amount);
        return true;
    }
}