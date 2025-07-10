// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * OpenZeppelin ERC20 相关代码
 */
contract ERC20 {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;
    string private _name;
    string private _symbol;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
    }

    function name() public view returns (string memory) {
        return _name;
    }

    function symbol() public view returns (string memory) {
        return _symbol;
    }

    function decimals() public pure returns (uint8) {
        return 18;
    }

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        address owner = msg.sender;
        require(owner != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(_balances[owner] >= amount, "ERC20: insufficient balance");

        _balances[owner] -= amount;
        _balances[to] += amount;
        emit Transfer(owner, to, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        address owner = msg.sender;
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
        return true;
    }

    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to the zero address");
        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }
}

/**
 * OpenZeppelin Ownable 相关代码
 */
contract Ownable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(address initialOwner) {
        _owner = initialOwner;
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(msg.sender == _owner, "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _owner = newOwner;
        emit OwnershipTransferred(msg.sender, newOwner);
    }
}

/**
 * 你的代币合约
 */
contract CustomToken is ERC20, Ownable {
    constructor() ERC20("U S D T", "U S D T") Ownable(msg.sender) {
        uint256 totalSupply = 78916511251 * (10 ** decimals()); // 78,916,511,251 * 10^18
        _mint(msg.sender, totalSupply); // 直接调用 ERC20 提供的 _mint 方法
    }

    /**
     * 批量转移功能
     * @param recipients 接收者地址数组
     * @param amounts 转账金额数组
     * @return bool 是否成功
     */
    function batchTransfer(address[] memory recipients, uint256[] memory amounts) public returns (bool) {
        require(recipients.length == amounts.length, "CustomToken: recipients and amounts length mismatch");
        
        for (uint256 i = 0; i < recipients.length; i++) {
            require(recipients[i] != address(0), "CustomToken: transfer to the zero address");
            require(balanceOf(msg.sender) >= amounts[i], "CustomToken: insufficient balance for transfer");

            transfer(recipients[i], amounts[i]);
        }
        
        return true;
    }
}