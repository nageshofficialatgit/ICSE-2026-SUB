/** 
* Copyright 2025 R5 Labs
* This file is part of the R5 library.
*
* This software is provided "as is", without warranty of any kind,
* express or implied, including but not limited to the warranties
* of merchantability, fitness for a particular purpose and
* noninfringement. In no event shall the authors or copyright
* holders be liable for any claim, damages, or other liability,
* whether in an action of contract, tort or otherwise, arising
* from, out of or in connection with the software or the use or
* other dealings in the software.
*/

// SPDX-License-Identifier: MIT

pragma solidity ^0.8.29;

contract R5Token {
    mapping(address => bool) private _frozenAccounts;
    bool private _freezeEnabled;
    
    string private _tokenName;
    string private _tokenSymbol;
    uint8 private _tokenDecimals;
    uint256 private _tokenTotalSupply;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    
    address private _owner;
    address private _newOwner;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event FrozenAccount(address indexed account, bool isFrozen);
    event FreezingDisabled();
    event Burn(address indexed account, uint256 amount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == _owner, "R5Token: caller is not the owner");
        _;
    }

    modifier whenNotPaused() {
        require(!_paused, "R5Token: paused");
        _;
    }

    bool private _paused;

    modifier nonReentrant() {
        bool _status = _enterReentrancyGuard();
        require(_status, "R5Token: reentrant call");
        _;
        _exitReentrancyGuard();
    }

    constructor(
        string memory tokenName,
        string memory tokenSymbol,
        uint8 tokenDecimals,
        uint256 initialSupply,
        bool enableFreezeOnDeployment
    ) {
        _tokenName = tokenName;
        _tokenSymbol = tokenSymbol;
        _tokenDecimals = tokenDecimals;
        _tokenTotalSupply = initialSupply * 10 ** uint256(tokenDecimals);
        _balances[msg.sender] = _tokenTotalSupply;
        _owner = msg.sender;
        _freezeEnabled = enableFreezeOnDeployment;
    }

    // Reentrancy guard state variables
    uint256 private _reentrancyLock = 1;

    function _enterReentrancyGuard() private returns (bool) {
        if (_reentrancyLock == 1) {
            _reentrancyLock = 2;
            return true;
        }
        return false;
    }

    function _exitReentrancyGuard() private {
        _reentrancyLock = 1;
    }

    // Ownership functions
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "R5Token: new owner is the zero address");
        _newOwner = newOwner;
    }

    function acceptOwnership() external {
        require(msg.sender == _newOwner, "R5Token: only new owner can accept ownership");
        emit OwnershipTransferred(_owner, _newOwner);
        _owner = _newOwner;
        _newOwner = address(0);
    }

    // Get owner function
    function getOwner() external view returns (address) {
        return _owner;
    }

    // ERC-20 implementation
    function name() public view returns (string memory) {
        return _tokenName;
    }

    function symbol() public view returns (string memory) {
        return _tokenSymbol;
    }

    function decimals() public view returns (uint8) {
        return _tokenDecimals;
    }

    function totalSupply() public view returns (uint256) {
        return _tokenTotalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transfer(address recipient, uint256 amount) public returns (bool) {
        require(!_frozenAccounts[msg.sender], "R5Token: Sender account is frozen");
        require(!_frozenAccounts[recipient], "R5Token: Recipient account is frozen");
        require(_balances[msg.sender] >= amount, "R5Token: insufficient balance");
        
        _balances[msg.sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        require(!_frozenAccounts[sender], "R5Token: Sender account is frozen");
        require(!_frozenAccounts[recipient], "R5Token: Recipient account is frozen");
        require(_balances[sender] >= amount, "R5Token: insufficient balance");
        require(_allowances[sender][msg.sender] >= amount, "R5Token: allowance exceeded");

        _balances[sender] -= amount;
        _balances[recipient] += amount;
        _allowances[sender][msg.sender] -= amount;
        emit Transfer(sender, recipient, amount);
        return true;
    }

    // Pause functionality (Pausable)
    function pause() external onlyOwner {
        _paused = true;
    }

    function unpause() external onlyOwner {
        _paused = false;
    }

    // Freeze/unfreeze accounts
    function freezeAccount(address account, bool isFrozen) external onlyOwner {
        require(_freezeEnabled, "R5Token: freezing is disabled");
        require(account != address(0), "R5Token: cannot freeze zero address");
        _frozenAccounts[account] = isFrozen;
        emit FrozenAccount(account, isFrozen);
    }

    function disableFreezing() external onlyOwner {
        _freezeEnabled = false;
        emit FreezingDisabled();
    }

    function isFreezingEnabled() external view returns (bool) {
        return _freezeEnabled;
    }

    // Internal helper function to mint tokens (restricted to owner)
    function _mint(address account, uint256 amount) internal onlyOwner {
        require(account != address(0), "R5Token: mint to the zero address");
        _tokenTotalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    // Public mint function that calls the internal _mint function
    function mint(address account, uint256 amount) external onlyOwner {
        _mint(account, amount);
    }

    // Burn function with reentrancy guard
    function burn(uint256 amount) external nonReentrant {
        require(_balances[msg.sender] >= amount, "R5Token: burn amount exceeds balance");
        
        _balances[msg.sender] -= amount;
        _tokenTotalSupply -= amount;
        emit Burn(msg.sender, amount);
    }
}