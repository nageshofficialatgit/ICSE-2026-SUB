// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () {
        _owner = _msgSender();
        emit OwnershipTransferred(address(0), _owner);
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

abstract contract Pausable is Ownable {
    event Paused(address account);
    event Unpaused(address account);
    bool private _paused;

    constructor() {
        _paused = false;
    }

    modifier whenNotPaused() {
        require(!_paused, "Paused");
        _;
    }

    function pause() public onlyOwner {
        _paused = true;
        emit Paused(_msgSender());
    }

    function unpause() public onlyOwner {
        _paused = false;
        emit Unpaused(_msgSender());
    }
}

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

contract USDTf is Context, IERC20, Ownable, Pausable {
    string public constant name = "Tether USD Bridged FED20";
    string public constant symbol = "USDT.f";
    uint8 public constant decimals = 6;

    uint256 private _totalSupply;
    uint256 public transferTaxBasisPoints = 50; // 0.5%
    address public feeCollector;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    constructor(address _feeCollector) {
        require(_feeCollector != address(0), "Fee collector cannot be zero");
        uint256 initialSupply = 100000000 * 10 ** decimals;
        _mint(_msgSender(), initialSupply);
        feeCollector = _feeCollector;
    }

    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public override whenNotPaused returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner_, address spender) public view override returns (uint256) {
        return _allowances[owner_][spender];
    }

    function approve(address spender, uint256 amount) public override whenNotPaused returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override whenNotPaused returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), _allowances[sender][_msgSender()] - amount);
        return true;
    }

    function mint(address to, uint256 amount) public onlyOwner {
        _mint(to, amount * 10 ** decimals);
    }

    function burn(uint256 amount) public {
        _burn(_msgSender(), amount * 10 ** decimals);
    }

    function setTransferTax(uint256 basisPoints) public onlyOwner {
        require(basisPoints <= 500, "Max tax is 5%");
        transferTaxBasisPoints = basisPoints;
    }

    function setFeeCollector(address newCollector) public onlyOwner {
        require(newCollector != address(0), "Zero address");
        feeCollector = newCollector;
    }

    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "From zero");
        require(recipient != address(0), "To zero");

        uint256 fee = (amount * transferTaxBasisPoints) / 10000;
        uint256 netAmount = amount - fee;

        _balances[sender] -= amount;
        _balances[recipient] += netAmount;
        if (fee > 0) {
            _balances[feeCollector] += fee;
            emit Transfer(sender, feeCollector, fee);
        }

        emit Transfer(sender, recipient, netAmount);
    }

    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "Mint to zero");
        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    function _burn(address account, uint256 amount) internal {
        require(account != address(0), "Burn from zero");
        _balances[account] -= amount;
        _totalSupply -= amount;
        emit Transfer(account, address(0), amount);
    }

    function _approve(address owner_, address spender, uint256 amount) internal {
        require(owner_ != address(0), "Approve from zero");
        require(spender != address(0), "Approve to zero");
        _allowances[owner_][spender] = amount;
        emit Approval(owner_, spender, amount);
    }
}