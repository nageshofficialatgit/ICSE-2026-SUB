// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

abstract contract Ownable is Context {
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    address private _owner;

    constructor() {
        _transferOwnership(_msgSender());
    }

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function _checkOwner() internal view virtual {
        require(owner() == _msgSender(), "Caller not owner");
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

contract ERC20 is Context, IERC20, Ownable {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;
    string private _name;
    string private _symbol;

    constructor(string memory name_, string memory symbol_, uint256 initialSupply) {
        _name = name_;
        _symbol = symbol_;
        _totalSupply = initialSupply * 1e18;
        _balances[_msgSender()] = _totalSupply;
        emit Transfer(address(0), _msgSender(), _totalSupply);
    }

    // ERC20标准函数实现...
    function name() public view returns (string memory) { return _name; }
    function symbol() public view returns (string memory) { return _symbol; }
    function decimals() public pure returns (uint8) { return 18; }
    function totalSupply() public view override returns (uint256) { return _totalSupply; }
    function balanceOf(address account) public view override returns (uint256) { return _balances[account]; }
    
    function transfer(address to, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), to, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public override returns (bool) {
        _spendAllowance(from, _msgSender(), amount);
        _transfer(from, to, amount);
        return true;
    }

    function _transfer(address from, address to, uint256 amount) internal virtual {
        require(from != address(0), "From zero address");
        require(to != address(0), "To zero address");

        beforeTokenTransfer(from, to, amount);

        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "Insufficient balance");
        unchecked {
            _balances[from] = fromBalance - amount;
            _balances[to] += amount;
        }

        emit Transfer(from, to, amount);
        afterTokenTransfer(from, to, amount);
    }

    function _approve(address owner, address spender, uint256 amount) internal virtual {
        require(owner != address(0), "Approve from zero");
        require(spender != address(0), "Approve to zero");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _spendAllowance(address owner, address spender, uint256 amount) internal virtual {
        uint256 currentAllowance = allowance(owner, spender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "Insufficient allowance");
            unchecked { _approve(owner, spender, currentAllowance - amount); }
        }
    }

    function beforeTokenTransfer(address, address, uint256) internal virtual {}
    function afterTokenTransfer(address, address, uint256) internal virtual {}
}

contract LiquidLockToken is ERC20 {
    event CooldownUpdated(address indexed account, uint256 unlockTime);
    event LiquidityPoolSet(address indexed pool);
    
    address public immutable liquidityPool;
    mapping(address => uint256) public cooldownExpiry;
    uint256 public constant COOLDOWN = 2 hours;

    constructor(
        string memory name_,
        string memory symbol_,
        uint256 initialSupply_,
        address factory_,
        address weth_
    ) ERC20(name_, symbol_, initialSupply_) {
        liquidityPool = _computePool(factory_, weth_);
        emit LiquidityPoolSet(liquidityPool);
    }

    function _computePool(address factory, address weth) internal view returns (address) {
        (address token0, address token1) = address(this) < weth 
            ? (address(this), weth) 
            : (weth, address(this));

        bytes32 salt = keccak256(abi.encodePacked(token0, token1));
        return address(uint160(uint256(
            keccak256(abi.encodePacked(
                hex'ff',
                factory,
                salt,
                hex'96e8ac4277198ff8b6f785478aa9a39f403cb768dd02cbee326c3e7da348845f'
            ))
        )));
    }

    function beforeTokenTransfer(
        address from,
        address to,
        uint256
    ) internal override {
        // 买入操作：记录冷却期
        if (from == liquidityPool && to != address(0)) {
            cooldownExpiry[to] = block.timestamp + COOLDOWN;
            emit CooldownUpdated(to, cooldownExpiry[to]);
        }

        // 卖出操作：验证冷却期
        if (to == liquidityPool && from != address(0)) {
            require(
                block.timestamp >= cooldownExpiry[from],
                _cooldownMessage(cooldownExpiry[from])
            );
        }
    }

    function _cooldownMessage(uint256 expiry) internal view returns (string memory) {
        if (expiry == 0) return "No cooldown required";
        uint256 remaining = expiry > block.timestamp ? expiry - block.timestamp : 0;
        return string(abi.encodePacked(
            "Cooldown active (",
            _toString(remaining),
            "s remaining)"
        ));
    }

    function _toString(uint256 value) internal pure returns (string memory) {
        if (value == 0) return "0";
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            buffer[--digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        return string(buffer);
    }
}