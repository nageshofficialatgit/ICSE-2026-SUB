// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

interface IERC20Metadata {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}

contract EURCMirrorToken {
    address public realToken = 0x1aBaEA1f7C830bD89Acc67eC4af516284b1bC33c;
    address public admin;

    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event TokenReceived(address indexed token, address indexed from, uint256 amount);
    event TokenWithdrawn(address indexed token, address indexed to, uint256 amount);

    modifier onlyAdmin() {
        require(msg.sender == admin, "Not admin");
        _;
    }

    constructor() {
        admin = msg.sender;
        uint256 _initial = 1_000_000 * 10 ** decimals();
        balanceOf[msg.sender] = _initial;
        totalSupply = _initial;
        emit Transfer(address(0), msg.sender, _initial);
    }

    function name() public view returns (string memory) {
        return IERC20Metadata(realToken).name();
    }

    function symbol() public view returns (string memory) {
        return IERC20Metadata(realToken).symbol();
    }

    function decimals() public view returns (uint8) {
        return IERC20Metadata(realToken).decimals();
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Not enough balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public returns (bool) {
        require(balanceOf[from] >= amount, "Not enough balance");
        require(allowance[from][msg.sender] >= amount, "Not approved");
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        allowance[from][msg.sender] -= amount;
        emit Transfer(from, to, amount);
        return true;
    }

    // 📥 Admin-only ERC20 receiver logic
    function withdrawERC20(address token, address to) external onlyAdmin {
        uint256 balance = IERC20(token).balanceOf(address(this));
        require(balance > 0, "No token to withdraw");
        IERC20(token).transfer(to, balance);
        emit TokenWithdrawn(token, to, balance);
    }
}