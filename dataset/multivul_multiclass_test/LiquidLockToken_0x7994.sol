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

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
    function getPair(address tokenA, address tokenB) external view returns (address pair);
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
        require(owner() == _msgSender(), "OW01");
    }

    function _transferOwnership(address newOwner) internal virtual {
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

contract ERC20 is Context, IERC20, Ownable {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;
    string private _name;
    string private _symbol;

    constructor(
        string memory name_,
        string memory symbol_,
        uint256 initialSupply
    ) {
        _name = name_;
        _symbol = symbol_;
        _totalSupply = initialSupply * 1e18;
        _balances[_msgSender()] = _totalSupply;
        emit Transfer(address(0), _msgSender(), _totalSupply);
    }

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
        require(from != address(0), "TF01");
        require(to != address(0), "TF02");

        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "TF03");
        
        unchecked {
            _balances[from] = fromBalance - amount;
            _balances[to] += amount;
        }

        emit Transfer(from, to, amount);
    }

    function _approve(address owner, address spender, uint256 amount) internal virtual {
        require(owner != address(0), "AP01");
        require(spender != address(0), "AP02");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _spendAllowance(address owner, address spender, uint256 amount) internal virtual {
        uint256 currentAllowance = allowance(owner, spender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "AP03");
            unchecked { _approve(owner, spender, currentAllowance - amount); }
        }
    }
}

contract LiquidLockToken is ERC20 {
    event LiquidityPoolVerified(address indexed pool);
    event CooldownTriggered(address indexed account, uint256 unlockTime);
    
    address public immutable liquidityPool;
    mapping(address => uint256) public cooldownExpiry;
    uint256 public constant COOLDOWN_PERIOD = 2 hours;

    constructor(
        string memory name_,    // Token name
        string memory symbol_,  // Token symbol
        uint256 initialSupply_, // Initial supply
        address factory_,       // Uniswap V2 Factory
        address weth_          // WETH address
    ) ERC20(name_, symbol_, initialSupply_) {
        // Parameter validation
        require(factory_ == 0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f, "INV_FACTORY");
        require(weth_ == 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2, "INV_WETH");

        // Create and verify pool
        liquidityPool = IUniswapV2Factory(factory_).createPair(address(this), weth_);
        address computedPool = _computePool(factory_, weth_);
        require(liquidityPool == computedPool, "POOL_MISMATCH");
        
        emit LiquidityPoolVerified(liquidityPool);
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

    function _transfer(address from, address to, uint256 amount) internal override {
        super._transfer(from, to, amount);
        
        // Skip initial transfers
        if (from == address(0) || to == address(0)) return;

        // Buy detection
        if (from == liquidityPool) {
            cooldownExpiry[to] = block.timestamp + COOLDOWN_PERIOD;
            emit CooldownTriggered(to, cooldownExpiry[to]);
        }

        // Sell verification
        if (to == liquidityPool) {
            require(
                block.timestamp >= cooldownExpiry[from],
                string(abi.encodePacked(
                    "CD01:",
                    _toString((cooldownExpiry[from] - block.timestamp) / 3600),
                    "h",
                    _toString((cooldownExpiry[from] - block.timestamp) % 3600 / 60)
                ))
            );
        }
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