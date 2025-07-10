// SPDX-License-Identifier: BSD-3-Clause
pragma solidity ^0.8.28;

contract Securency {
    string public constant name = "Securency";
    string public constant symbol = "SEC";
    uint8 public constant decimals = 9;
    uint256 public constant totalSupply = 6000000000 * 10**decimals;
    uint256 public constant maxTransferAmount = 100000000 * 10**decimals;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    bool public paused = false;

    modifier whenNotPaused() {
        require(!paused, "Kontrak sedang dihentikan sementara");
        _;
    }

    modifier whenPaused() {
        require(paused, "Kontrak tidak sedang dihentikan sementara");
        _;
    }

    uint256 private _status;

    modifier nonReentrant() {
        require(_status != 1, "Panggilan berulang");
        _status = 1;
        _;
        _status = 2;
    }

    address public owner;

    constructor() {
        _balances[0x5CceA49380c0d1DFda7F8d9F0dc154A3dC6f7A86] = totalSupply;
        emit Transfer(address(0), 0x5CceA49380c0d1DFda7F8d9F0dc154A3dC6f7A86, totalSupply);
        owner = 0x5CceA49380c0d1DFda7F8d9F0dc154A3dC6f7A86;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Hanya pemilik yang dapat memanggil fungsi ini");
        _;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public whenNotPaused nonReentrant returns (bool) {
        require(amount <= maxTransferAmount, "Jumlah transfer melebihi batas");
        require(_balances[msg.sender] >= amount, "Saldo tidak mencukupi");
        _balances[msg.sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    function allowance(address ownerAddress, address spender) public view returns (uint256) {
        return _allowances[ownerAddress][spender];
    }

    function approve(address spender, uint256 amount) public whenNotPaused nonReentrant returns (bool) {
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public whenNotPaused nonReentrant returns (bool) {
        require(_allowances[sender][msg.sender] >= amount, "Jatah tidak mencukupi");
        require(_balances[sender] >= amount, "Saldo tidak mencukupi");
        _balances[sender] -= amount;
        _balances[recipient] += amount;
        _allowances[sender][msg.sender] -= amount;
        emit Transfer(sender, recipient, amount);
        return true;
    }

    function pause() public onlyOwner whenNotPaused {
        paused = true;
    }

    function unpause() public onlyOwner whenPaused {
        paused = false;
    }

    function tokenInfo() public pure returns (string memory, string memory, uint8, uint256, string memory) {
        return (
            name,
            symbol,
            decimals,
            totalSupply,
            "Securency (SEC) is an innovative crypto token that aims to create companies that facilitate space travel, planetary exploration, and understanding of the universe by starting new lives on spaceships for eons-long journeys. \
This token is projected to become the official currency for space travel and provides exclusive access to its holders. \
Securency is also committed to keeping the Earth green and peaceful for future generations so that each generation can still start space travel or return. \
Earth is not the center of the universe and we are certainly not the only ones."
        );
    }
}