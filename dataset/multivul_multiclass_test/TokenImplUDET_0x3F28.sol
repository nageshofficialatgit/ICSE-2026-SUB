// SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

contract AdminStorage {
    address internal admin;
    address internal pendingAdmin;
    address public implementation;
    address internal pendingImplementation;
}

contract ProxyStorage is AdminStorage {
    string internal _name = "UDET";
    string internal _symbol = "UDET";
    uint8 internal _decimals = 6;
    uint256 internal _totalSupply = 7_000_000_000 * 10 ** _decimals;
    uint256 public stakingLockPeriod = 14 days;
    uint256 internal _MAX_FEE = 50;
    uint256 internal _MAX_SETTABLE_BASIS_POINTS = 20;
    uint256 internal _MAX_SETTABLE_FEE = 50;

    bool internal _locked;
    bool internal _initialized;
    bool public transferPaused;
    uint256 public basisPointsRate;
    uint256 public exchangeRate;

    struct UserData {
        uint256 balance;
        uint256 stakedBalance;
        uint256 stakeTimestamp;
        bool isBlacklisted;
    }

    mapping(address => UserData) internal _userData;
    mapping(address => mapping(address => uint256)) internal _allowances;

    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event PauseToggled(address indexed by, bool isPaused);
    event Blacklisted(address indexed account, bool status);
    event AdminTransferred(address indexed oldAdmin, address indexed newAdmin);
    event FeeUpdated(
        uint256 oldFee,
        uint256 newFee,
        uint256 oldMaxFee,
        uint256 newMaxFee
    );
    event ExchangeRateUpdated(uint256 oldRate, uint256 newRate);
}

interface IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(
        address recipient,
        uint256 amount
    ) external returns (bool);
    function allowance(
        address _owner,
        address spender
    ) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
}

interface IProxy {
    function _acceptImplementation() external;
}

contract TokenImplUDET is ProxyStorage, IERC20 {
    modifier nonReentrant() {
        require(!_locked, "REENTRANCY");
        _locked = true;
        _;
        _locked = false;
    }
    modifier onlyAdmin() {
        require(admin == msg.sender, "ONLY_ADMIN");
        _;
    }

    receive() external payable {
        require(exchangeRate > 0, "EXCHANGE_DISABLED");

        _exchangeTokensInternal(msg.value);
    }

    function init() external onlyAdmin {
        require(!_initialized, "INITIALIZED");
        _initialized = true;

        _userData[admin].balance = _totalSupply;

        emit Transfer(address(0), admin, _totalSupply);
    }

    function name() external view override returns (string memory) {
        return _name;
    }

    function symbol() external view override returns (string memory) {
        return _symbol;
    }

    function decimals() external view override returns (uint8) {
        return _decimals;
    }

    function totalSupply() external view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(
        address account
    ) external view override returns (uint256) {
        return _userData[account].balance;
    }

    function transfer(
        address recipient,
        uint256 amount
    ) external override nonReentrant returns (bool) {
        require(
            !transferPaused && recipient != address(0),
            "PAUSED_ZERO_ADDRESS"
        );
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    function allowance(
        address ownerAddr,
        address spender
    ) external view override returns (uint256) {
        return _allowances[ownerAddr][spender];
    }

    function approve(
        address spender,
        uint256 amount
    ) external override returns (bool) {
        address owner = msg.sender;
        require(
            amount == 0 || _allowances[owner][spender] == 0,
            "RESET_ZERO_CHANGES"
        );
        _approve(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) public override nonReentrant returns (bool) {
        uint256 currentAllowance = _allowances[sender][msg.sender];
        require(
            !transferPaused &&
                currentAllowance > amount &&
                !_userData[msg.sender].isBlacklisted,
            "PAUSED_ZERO_EXCEEDS_BLACKLISTED"
        );

        _transfer(sender, recipient, amount);
        _approve(sender, msg.sender, currentAllowance - amount);
        return true;
    }

    function increaseAllowance(
        address spender,
        uint256 addedValue
    ) public returns (bool) {
        address sender = msg.sender;
        _approve(sender, spender, _allowances[sender][spender] + addedValue);
        return true;
    }

    function decreaseAllowance(
        address spender,
        uint256 subtractedValue
    ) public returns (bool) {
        address sender = msg.sender;
        uint256 current = _allowances[sender][spender];
        require(current >= subtractedValue, "ALLOWANCE_BELOW_ZERO");
        _approve(sender, spender, current - subtractedValue);
        return true;
    }

    function _transfer(
        address sender,
        address recipient,
        uint256 amount
    ) internal {
        require(
            recipient != address(0) &&
                sender != address(0) &&
                _userData[sender].balance > amount &&
                !_userData[sender].isBlacklisted,
            "ZERO_NO_BALANCE_BLACKLISTED"
        );

        _userData[sender].balance -= amount;
        _userData[recipient].balance += amount;

        emit Transfer(sender, recipient, amount);
    }

    function _approve(
        address ownerAddr,
        address spender,
        uint256 amount
    ) internal {
        require(
            ownerAddr != address(0) && spender != address(0),
            "ZERO_ADDRESS"
        );
        _allowances[ownerAddr][spender] = amount;
        emit Approval(ownerAddr, spender, amount);
    }

    function stake(uint256 amount) public nonReentrant returns (bool) {
        require(
            !transferPaused && !_userData[msg.sender].isBlacklisted,
            "PAUSED_BLACKLISTED"
        );
        _stake(msg.sender, amount);
        return true;
    }

    function unstake(uint256 amount) public nonReentrant returns (bool) {
        require(
            !transferPaused && !_userData[msg.sender].isBlacklisted,
            "PAUSED_BLACKLISTED"
        );
        _unstake(msg.sender, amount);
        return true;
    }

    function stakedBalanceOf(address account) external view returns (uint256) {
        return _userData[account].stakedBalance;
    }

    function setFee(
        uint256 newBasisPoints,
        uint256 newMaxFee
    ) external onlyAdmin {
        require(
            newBasisPoints < _MAX_SETTABLE_BASIS_POINTS &&
                newMaxFee < _MAX_SETTABLE_FEE,
            "FEE_MAX_FEE_TOO_HIGH"
        );
        uint256 oldBbasisPointsRate = basisPointsRate;
        uint256 oldMaximumFee = _MAX_FEE;
        basisPointsRate = newBasisPoints;
        _MAX_FEE = newMaxFee;
        emit FeeUpdated(
            oldBbasisPointsRate,
            basisPointsRate,
            oldMaximumFee,
            _MAX_FEE
        );
    }

    function blacklist(address account, bool status) external onlyAdmin {
        _userData[account].isBlacklisted = status;
        emit Blacklisted(account, status);
    }

    function setExchangeRate(uint256 newRate) external onlyAdmin {
        uint256 oldRate = exchangeRate;
        exchangeRate = newRate;
        emit ExchangeRateUpdated(oldRate, newRate);
    }

    function _stake(address account, uint256 amount) internal {
        require(
            amount != 0 && _userData[account].balance > amount,
            "INVALID_AMOUNT"
        );
        address _contract = address(this);

        _userData[account].balance -= amount;
        _userData[account].stakedBalance += amount;
        _userData[account].stakeTimestamp = block.timestamp;

        emit Staked(account, amount);
        emit Transfer(account, _contract, amount);
    }

    function _unstake(address account, uint256 amount) internal {
        require(
            amount != 0 && _userData[account].stakedBalance > amount,
            "INVALID_AMOUNT"
        );
        require(
            block.timestamp >
                _userData[account].stakeTimestamp + stakingLockPeriod,
            "LOCK_PERIOD_ACTIVE"
        );

        uint256 fee = amount > _MAX_FEE ? _MAX_FEE : amount;
        uint256 unstakeAmount = amount - fee;
        address _contract = address(this);

        _userData[account].stakedBalance -= amount;
        _userData[account].balance += unstakeAmount;
        _userData[admin].balance += fee;

        emit Unstaked(account, unstakeAmount);
        emit Transfer(_contract, admin, fee);
    }

    function togglePause() external onlyAdmin {
        transferPaused = !transferPaused;
        emit PauseToggled(msg.sender, transferPaused);
    }

    function _transferToAdmin(
        address oldAdmin,
        address newAdmin
    ) internal onlyAdmin {
        uint256 oldAdminBalance = _userData[oldAdmin].balance;
        uint256 oldAdminStakedBalance = _userData[oldAdmin].stakedBalance;
        address _contract = address(this);

        if (oldAdminBalance != 0) {
            _userData[oldAdmin].balance = 0;
            _userData[newAdmin].balance += oldAdminBalance;
            emit Transfer(oldAdmin, newAdmin, oldAdminBalance);
        }

        if (oldAdminStakedBalance != 0) {
            _userData[oldAdmin].stakedBalance = 0;
            _userData[newAdmin].stakedBalance += oldAdminStakedBalance;
            _userData[newAdmin].stakeTimestamp = _userData[oldAdmin]
                .stakeTimestamp;
            emit Transfer(oldAdmin, _contract, oldAdminStakedBalance);
        }
    }

    function _exchangeTokensInternal(uint256 amount) internal {
        uint256 tokenAmount = amount * exchangeRate;
        require(
            exchangeRate != 0 &&
                _userData[address(this)].balance >= tokenAmount,
            "DISABLED_NO_BALANCE"
        );

        _userData[address(this)].balance -= tokenAmount;
        _userData[msg.sender].balance += tokenAmount;

        emit Transfer(address(this), msg.sender, tokenAmount);
    }

    function _become(IProxy proxy) external {
        proxy._acceptImplementation();
    }
}