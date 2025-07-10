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

abstract contract Ownable {
    address public owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}

contract DumpTheRump is IERC20, Ownable {
    string public name = "DumpTheRump";
    string public symbol = "DTR";
    uint8 public decimals = 18;
    uint256 private _totalSupply = 1_000_000_000 * 10**uint256(decimals);

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    address public communityWallet;
    uint256 public burnFee = 2;
    uint256 public communityFee = 1;

    constructor(address _communityWallet) {
        require(_communityWallet != address(0), "Zero address not allowed");
        communityWallet = _communityWallet;
        _balances[msg.sender] = _totalSupply;
        emit Transfer(address(0), msg.sender, _totalSupply);
    }

    function totalSupply() external view override returns (uint256) {
        return _totalSupply;
    }
    function balanceOf(address account) external view override returns (uint256) {
        return _balances[account];
    }
    function allowance(address owner_, address spender) external view override returns (uint256) {
        return _allowances[owner_][spender];
    }

    function transfer(address recipient, uint256 amount) external override returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    function approve(address spender, uint256 amount) external override returns (bool) {
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) external override returns (bool) {
        _allowances[sender][msg.sender] -= amount;
        _transfer(sender, recipient, amount);
        emit Approval(sender, msg.sender, _allowances[sender][msg.sender]);
        return true;
    }

    function setCommunityWallet(address newWallet) external onlyOwner {
        require(newWallet != address(0), "Invalid address");
        communityWallet = newWallet;
    }

    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0) && recipient != address(0), "Zero address not allowed");
        require(_balances[sender] >= amount, "Insufficient balance");

        uint256 burnAmount = (amount * burnFee) / 100;
        uint256 communityAmount = (amount * communityFee) / 100;
        uint256 netAmount = amount - burnAmount - communityAmount;

        _balances[sender] -= amount;
        _balances[recipient] += netAmount;
        _balances[communityWallet] += communityAmount;
        _totalSupply -= burnAmount;

        emit Transfer(sender, recipient, netAmount);
        emit Transfer(sender, communityWallet, communityAmount);
        emit Transfer(sender, address(0), burnAmount);
    }
}