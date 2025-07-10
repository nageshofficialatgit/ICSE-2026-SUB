// SPDX-License-Identifier: MIT
pragma solidity ^0.8.21;

interface IERC20 {
  function totalSupply() external view returns(uint256);
  function balanceOf(address account) external view returns(uint256);
  function transfer(address recipient, uint256 amount) external returns(bool);
  function allowance(address owner, address spender) external view returns(uint256);
  function approve(address spender, uint256 amount) external returns(bool);
  function transferFrom(address sender, address recipient, uint256 amount) external returns(bool);
}

contract ERC20 is IERC20 {
  event Transfer(address indexed from, address indexed to, uint256 value);
  event Approval(address indexed owner, address indexed spender, uint256 value);

  uint256 public totalSupply;
  mapping(address => uint256) public balanceOf;
  mapping(address => mapping(address => uint256)) public allowance;
  string public name;
  string public symbol;
  uint8 public decimals;

  constructor(string memory _name, string memory _symbol, uint8 _decimals) {
    name = _name;
    symbol = _symbol;
    decimals = _decimals;
  }

  function transfer(address recipient, uint256 amount) external returns(bool) {
    require(recipient != address(0), "ERC20: transfer to the zero address");
    balanceOf[msg.sender] -= amount;
    balanceOf[recipient] += amount;
    emit Transfer(msg.sender, recipient, amount);
    return true;
  }

  function approve(address spender, uint256 amount) external returns(bool) {
    require(spender != address(0), "ERC20: approve to the zero address");
    allowance[msg.sender][spender] = amount;
    emit Approval(msg.sender, spender, amount);
    return true;
  }

  function transferFrom(address sender, address recipient, uint256 amount) external returns(bool) {
    require(sender != address(0), "ERC20: transfer from the zero address");
    require(recipient != address(0), "ERC20: transfer to the zero address");
    allowance[sender][msg.sender] -= amount;
    balanceOf[sender] -= amount;
    balanceOf[recipient] += amount;
    emit Transfer(sender, recipient, amount);
    return true;
  }
}

contract TOKEN is ERC20 {
  constructor(string memory name, string memory symbol, uint8 decimals) ERC20(name, symbol, decimals) {
    require(msg.sender != address(0), "ERC20: mint to the zero address");
    totalSupply = 10000000000 * (10 ** uint256(decimals));
    balanceOf[msg.sender] = totalSupply;
    emit Transfer(address(0), msg.sender, totalSupply);
  }
}