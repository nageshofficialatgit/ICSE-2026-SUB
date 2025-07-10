// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// IERC20 接口定义
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

// Context 抽象合约，用于获取消息发送者
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

// ERC20 合约实现
contract ERC20 is Context, IERC20 {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;
    string private _name;
    string private _symbol;

    constructor(string memory name_, string memory symbol_, uint256 initialSupply) {
        _name = name_;
        _symbol = symbol_;
        _totalSupply = initialSupply;
        _balances[msg.sender] = initialSupply;
        emit Transfer(address(0), msg.sender, initialSupply);
    }

    function name() public view virtual returns (string memory) {
        return _name;
    }

    function symbol() public view virtual returns (string memory) {
        return _symbol;
    }

    function decimals() public view virtual returns (uint8) {
        return 18;
    }

    function totalSupply() public view virtual override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view virtual override returns (uint256) {
        return _balances[account];
    }

    function transfer(address to, uint256 amount) public virtual override returns (bool) {
        address owner = _msgSender();
        _transfer(owner, to, amount);
        return true;
    }

    function allowance(address owner, address spender) public view virtual override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public virtual override returns (bool) {
        address owner = _msgSender();
        _approve(owner, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public virtual override returns (bool) {
        address spender = _msgSender();
        _spendAllowance(from, spender, amount);
        _transfer(from, to, amount);
        return true;
    }

    function _transfer(address from, address to, uint256 amount) internal virtual {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        beforeTokenTransfer(from, to, amount);
        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "ERC20: transfer amount exceeds balance");
        unchecked {
            _balances[from] = fromBalance - amount;
            _balances[to] += amount;
        }
        emit Transfer(from, to, amount);
        afterTokenTransfer(from, to, amount);
    }

    function _approve(address owner, address spender, uint256 amount) internal virtual {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _spendAllowance(address owner, address spender, uint256 amount) internal virtual {
        uint256 currentAllowance = allowance(owner, spender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "ERC20: insufficient allowance");
            unchecked {
                _approve(owner, spender, currentAllowance - amount);
            }
        }
    }

    // 声明为 virtual 函数，方便子类重写
    function beforeTokenTransfer(address from, address to, uint256 amount) internal virtual {}
    function afterTokenTransfer(address from, address to, uint256 amount) internal virtual {}
}

// 实现 Ownable 合约
contract Ownable is Context {
    address private _owner;
    bool private _ownershipRenounced = false;
    event OwnershipRenounced(address indexed previousOwner);

    constructor(address initialOwner) {
        _owner = initialOwner;
    }

    modifier onlyOwner() {
        require(_owner == _msgSender() &&!_ownershipRenounced, "Ownable: caller is not the owner or ownership has been renounced");
        _;
    }

    function renounceOwnership() public onlyOwner {
        _ownershipRenounced = true;
        emit OwnershipRenounced(_owner);
        _owner = address(0);
    }

    function isOwnershipRenounced() public view returns (bool) {
        return _ownershipRenounced;
    }
}

// LiquidLockToken 合约
contract LiquidLockToken is ERC20, Ownable {
    address public liquidityPool;
    mapping(address => uint256) private _lastBuyTime; // 记录每个地址的最后买入时间
    uint256 private constant COOLDOWN_PERIOD = 2 hours; // 冷却时间为 2 小时

    constructor(
        string memory name_,
        string memory symbol_,
        uint256 initialSupply_,
        address initialOwner
    ) ERC20(name_, symbol_, initialSupply_) Ownable(initialOwner) {}

    function setLiquidityPool(address _liquidityPool) external onlyOwner {
        liquidityPool = _liquidityPool;
    }

    // 重写 beforeTokenTransfer 函数
    function beforeTokenTransfer(address from, address to, uint256 amount) internal virtual override {
        if (from == liquidityPool && to != liquidityPool) {
            // 用户从流动性池买入，更新冷却期
            _lastBuyTime[to] = block.timestamp;
        } else if (from != liquidityPool && to == liquidityPool) {
            // 用户向流动性池卖出，检查冷却期
            if (block.timestamp < _lastBuyTime[from] + COOLDOWN_PERIOD) {
                uint256 remainingTime = (_lastBuyTime[from] + COOLDOWN_PERIOD) - block.timestamp;
                revert(string(abi.encodePacked("Selling is not allowed during the cooldown period. Remaining time: ", toString(remainingTime), " seconds")));
            }
        }
        // 调用父类的 beforeTokenTransfer 函数，确保原有的逻辑也能执行
        super.beforeTokenTransfer(from, to, amount);
    }

    // 辅助函数，将 uint256 转换为 string
    function toString(uint256 value) internal pure returns (string memory) {
        if (value == 0) {
            return "0";
        }
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        return string(buffer);
    }
}