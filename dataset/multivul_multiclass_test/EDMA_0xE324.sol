// File: @openzeppelin/contracts/token/ERC20/IERC20.sol


// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC-20 standard as defined in the ERC.
 */
interface IERC20 {
    /**
     * @dev Emitted when `value` tokens are moved from one account (`from`) to
     * another (`to`).
     *
     * Note that `value` may be zero.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Emitted when the allowance of a `spender` for an `owner` is set by
     * a call to {approve}. `value` is the new allowance.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);

    /**
     * @dev Returns the value of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the value of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves a `value` amount of tokens from the caller's account to `to`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address to, uint256 value) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets a `value` amount of tokens as the allowance of `spender` over the
     * caller's tokens.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * IMPORTANT: Beware that changing an allowance with this method brings the risk
     * that someone may use both the old and the new allowance by unfortunate
     * transaction ordering. One possible solution to mitigate this race
     * condition is to first reduce the spender's allowance to 0 and set the
     * desired value afterwards:
     * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
     *
     * Emits an {Approval} event.
     */
    function approve(address spender, uint256 value) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to` using the
     * allowance mechanism. `value` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

// File: Mainnet/EDMANew.sol


pragma solidity ^0.8.20;



abstract contract Context {
    function _msgSender() internal view virtual returns (address payable) {
        return payable(msg.sender);
    }
}

contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    constructor() {
        _owner = _msgSender();
        emit OwnershipTransferred(address(0), _owner);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(
            newOwner != address(0),
            "Ownable: zero address"
        );
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

contract EDMA is IERC20, Ownable {
    struct VestingSchedule {
        uint256 totalLocked;
        uint256 totalReleased;
        uint256 lastRelasedTime;
        bool isFirstReleased;
    }
    uint256 public vestingInterval = 90 days;  // 7776000;

    string private _name = "EDMA";
    string private _symbol = "EDM";
    uint8 private _decimals = 18;
    uint256 private _totalSupply = 500_000_000 * 10 ** _decimals;

    bool public tradingEnabled;
    bool public  burningEnabled;
    address public preSaleAddress;

    modifier onlyPreSale() {
        require(preSaleAddress == _msgSender(), "EDMA: caller is not the presale");
        _;
    }


    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) public whitelist;
    mapping(address => VestingSchedule) public vesting;
    
    
    

    event TradingEnabled();
    event Burned(address indexed from, uint256 amount);
    event TokensRecovered(address token, address recipient, uint256 amount);
    event ETHRecovered(address recipient, uint256 amount);
    event TokensVested(address indexed beneficiary, uint256 amount);
    event Received(address sender, uint256 value);

    constructor() {
        whitelist[_msgSender()] = true;
        _balances[_msgSender()] = _totalSupply;
        emit Transfer(address(0), _msgSender(), _totalSupply);
    }

    // Fallback to receive ETH
    receive() external payable {
        emit Received(msg.sender, msg.value);
    }

    // Token Information
    function name() public view returns (string memory) {
        return _name;
    }

    function symbol() public view returns (string memory) {
        return _symbol;
    }

    function decimals() public view returns (uint8) {
        return _decimals;
    }

    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function burn(uint256 amount) external {
        require(burningEnabled, "Burning is not enabled");
        require(amount > 0 && _balances[msg.sender] >= amount, "Invalid token amount");
        checkIfCanSpend(msg.sender, amount);
        _balances[msg.sender] -= amount;
        _totalSupply -= amount;
        emit Burned(msg.sender, amount);
    }

    // Enable trading
    function enableTrading() external onlyOwner {
        require(!tradingEnabled, "Trading is already enabled");
        tradingEnabled = true;
        emit TradingEnabled();
    }

    // toggle burning
    function toggleBurning() external onlyOwner {
        burningEnabled = !burningEnabled;
    }

    function transferAndVest(address recipient, uint256 amount) external onlyPreSale returns (bool) {
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        if(vesting[recipient].totalLocked == 0) {
            vesting[recipient] = VestingSchedule(amount, 0, block.timestamp, false);
        } else {
            vesting[recipient].totalLocked =  vesting[recipient].totalLocked + amount;
            vesting[recipient].lastRelasedTime =  block.timestamp;
        }
        emit TokensVested(recipient, amount);
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function calculateToBeUnlockNext(address recepient) public view  returns (uint256 toUnlockAmount) {
        VestingSchedule storage vestinDetails = vesting[recepient];
        if(vestinDetails.totalLocked == 0 || vestinDetails.totalLocked - vestinDetails.totalReleased <= 0) {
            return 0;
        }
        if(vestinDetails.isFirstReleased && block.timestamp < vestinDetails.lastRelasedTime + vestingInterval) {
            return  0;
        }
        uint256 intervalPassed = (block.timestamp - vestinDetails.lastRelasedTime) / vestingInterval;
        if(tradingEnabled && !vestinDetails.isFirstReleased) {
            if(intervalPassed <= 0) {
                intervalPassed = 1;
            }
        }
        uint256 toRelease  = (vestinDetails.totalLocked * (intervalPassed * 20)) / 100;
        // uint256 toRelease = ((vestinDetails.totalLocked * 20) / 100);
        if(toRelease > vestinDetails.totalLocked - vestinDetails.totalReleased) {
            return vestinDetails.totalLocked - vestinDetails.totalReleased;
        }
        return toRelease;
    }

    

    // Whitelist control
    function setWhitelist(address user, bool isWhitelisted) external onlyOwner {
        whitelist[user] = isWhitelisted;
    }

    // set Presale address control
    function setPresale(address presale) external onlyOwner {
        preSaleAddress = presale;
        whitelist[presale] = true;
    }

    // set Vesting Interval time (second based 1minute = 60);
    function setvestingInterval(uint256 second) external onlyOwner {
        vestingInterval = second;
    }

    // ERC20 functions
    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), _allowances[sender][_msgSender()] - amount);
        return true;
    }

    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
        _approve(_msgSender(), spender, _allowances[_msgSender()][spender] + addedValue);
        return true;
    }

    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
        _approve(
            _msgSender(),
            spender,
            _allowances[_msgSender()][spender] - subtractedValue
        );
        return true;
    }

    // Utility functions
    function recoverStuckTokens(address token, address recipient, uint256 amount) external onlyOwner {
        require(IERC20(token).transfer(recipient, amount), "EDMA: Token recovery failed");
        emit TokensRecovered(token, recipient, amount);
    }

    function recoverStuckETH(address recipient) external onlyOwner {
        uint256 balnce = address(this).balance;
        require(balnce > 0, "EDMA: No ETH to recover");
        payable(recipient).transfer(balnce);
        emit ETHRecovered(recipient, balnce);
    }

    function checkIfCanSpend(address sender, uint256 amount) internal  returns(bool) {
        uint256 toRelease = calculateToBeUnlockNext(sender);
        if (toRelease > 0) {
            vesting[sender].totalReleased = vesting[sender].totalReleased + toRelease;
            vesting[sender].lastRelasedTime = block.timestamp;
        }
        
        uint256 lockedBalance = vesting[sender].totalLocked - vesting[sender].totalReleased;
        if(lockedBalance > 0) {
            require(amount <= balanceOf(sender) - lockedBalance, "Amount exceeds unlocked balance");
            if(!vesting[sender].isFirstReleased && tradingEnabled) {
                vesting[sender].isFirstReleased = true;
            }
        }
        return  true;
    }

    // Internal functions
    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "EDMA: transfer from zero address");
        require(recipient != address(0), "EDMA: transfer to zero address");
        require(amount > 0, "EDMA: amount must not be zero");

        if (!whitelist[sender]) {
            require(tradingEnabled, "EDMA: trading is not enabled");
        }
        
        checkIfCanSpend(sender, amount);
        _balances[sender] -= amount;
        _balances[recipient] += amount;

        emit Transfer(sender, recipient, amount);
    }

    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0), "EDMA: approve from zero address");
        require(spender != address(0), "EDMA: approve to zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
}