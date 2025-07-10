pragma solidity ^0.4.24;
pragma solidity 0.4.24;
interface IETokenProxy {
    function nameProxy(address sender) external view returns(string);
    function symbolProxy(address sender)
        external
        view
        returns(string);
    function decimalsProxy(address sender)
        external
        view
        returns(uint8);
    function totalSupplyProxy(address sender)
        external
        view
        returns (uint256);
    function balanceOfProxy(address sender, address who)
        external
        view
        returns (uint256);
    function allowanceProxy(address sender,
                            address owner,
                            address spender)
        external
        view
        returns (uint256);
    function transferProxy(address sender, address to, uint256 value)
        external
        returns (bool);
    function approveProxy(address sender,
                          address spender,
                          uint256 value)
        external
        returns (bool);
    function transferFromProxy(address sender,
                               address from,
                               address to,
                               uint256 value)
        external
        returns (bool);
    function mintProxy(address sender, address to, uint256 value)
        external
        returns (bool);
    function changeMintingRecipientProxy(address sender,
                                         address mintingRecip)
        external;
    function burnProxy(address sender, uint256 value) external;
    function burnFromProxy(address sender,
                           address from,
                           uint256 value)
        external;
    function increaseAllowanceProxy(address sender,
                                    address spender,
                                    uint addedValue)
        external
        returns (bool success);
    function decreaseAllowanceProxy(address sender,
                                    address spender,
                                    uint subtractedValue)
        external
        returns (bool success);
    function pauseProxy(address sender) external;
    function unpauseProxy(address sender) external;
    function pausedProxy(address sender) external view returns (bool);
    function finalizeUpgrade() external;
}
pragma solidity 0.4.24;
interface IEToken {
    function upgrade(IETokenProxy upgradedToken) external;
    function name() external view returns(string);
    function symbol() external view returns(string);
    function decimals() external view returns(uint8);
    function totalSupply() external view returns (uint256);
    function balanceOf(address who) external view returns (uint256);
    function allowance(address owner, address spender)
        external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function approve(address spender, uint256 value)
        external
        returns (bool);
    function transferFrom(address from, address to, uint256 value)
        external
        returns (bool);
    function mint(address to, uint256 value) external returns (bool);
    function burn(uint256 value) external;
    function burnFrom(address from, uint256 value) external;
    function increaseAllowance(
        address spender,
        uint addedValue
    )
        external
        returns (bool success);
    function pause() external;
    function unpause() external;
    function paused() external view returns (bool);
    function decreaseAllowance(
        address spender,
        uint subtractedValue
    )
        external
        returns (bool success);
    event Transfer(
        address indexed from,
        address indexed to,
        uint256 value
    );
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
}
contract Ownable {
  address private _owner;
  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );
  constructor() internal {
    _owner = msg.sender;
    emit OwnershipTransferred(address(0), _owner);
  }
  function owner() public view returns(address) {
    return _owner;
  }
  modifier onlyOwner() {
    require(isOwner());
    _;
  }
  function isOwner() public view returns(bool) {
    return msg.sender == _owner;
  }
  function renounceOwnership() public onlyOwner {
    emit OwnershipTransferred(_owner, address(0));
    _owner = address(0);
  }
  function transferOwnership(address newOwner) public onlyOwner {
    _transferOwnership(newOwner);
  }
  function _transferOwnership(address newOwner) internal {
    require(newOwner != address(0));
    emit OwnershipTransferred(_owner, newOwner);
    _owner = newOwner;
  }
}
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    require(c / a == b);
    return c;
  }
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b > 0); 
    uint256 c = a / b;
    return c;
  }
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b <= a);
    uint256 c = a - b;
    return c;
  }
  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    require(c >= a);
    return c;
  }
  function mod(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b != 0);
    return a % b;
  }
}
pragma solidity 0.4.24;
contract Storage is Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private balances;
    mapping (address => mapping (address => uint256)) private allowed;
    uint256 private totalSupply;
    address private _implementor;
    event StorageImplementorTransferred(address indexed from,
                                        address indexed to);
    constructor(address owner, address implementor) public {
        require(
            owner != address(0),
            "Owner should not be the zero address"
        );
        require(
            implementor != address(0),
            "Implementor should not be the zero address"
        );
        transferOwnership(owner);
        _implementor = implementor;
    }
    function isImplementor() public view returns(bool) {
        return msg.sender == _implementor;
    }
    function setBalance(address owner,
                        uint256 value)
        public
        onlyImplementor
    {
        balances[owner] = value;
    }
    function increaseBalance(address owner, uint256 addedValue)
        public
        onlyImplementor
    {
        balances[owner] = balances[owner].add(addedValue);
    }
    function decreaseBalance(address owner, uint256 subtractedValue)
        public
        onlyImplementor
    {
        balances[owner] = balances[owner].sub(subtractedValue);
    }
    function getBalance(address owner)
        public
        view
        returns (uint256)
    {
        return balances[owner];
    }
    function setAllowed(address owner,
                        address spender,
                        uint256 value)
        public
        onlyImplementor
    {
        allowed[owner][spender] = value;
    }
    function increaseAllowed(
        address owner,
        address spender,
        uint256 addedValue
    )
        public
        onlyImplementor
    {
        allowed[owner][spender] = allowed[owner][spender].add(addedValue);
    }
    function decreaseAllowed(
        address owner,
        address spender,
        uint256 subtractedValue
    )
        public
        onlyImplementor
    {
        allowed[owner][spender] = allowed[owner][spender].sub(subtractedValue);
    }
    function getAllowed(address owner,
                        address spender)
        public
        view
        returns (uint256)
    {
        return allowed[owner][spender];
    }
    function setTotalSupply(uint256 value)
        public
        onlyImplementor
    {
        totalSupply = value;
    }
    function getTotalSupply()
        public
        view
        returns (uint256)
    {
        return totalSupply;
    }
    function transferImplementor(address newImplementor)
        public
        requireNonZero(newImplementor)
        onlyImplementorOrOwner
    {
        require(newImplementor != _implementor,
                "Cannot transfer to same implementor as existing");
        address curImplementor = _implementor;
        _implementor = newImplementor;
        emit StorageImplementorTransferred(curImplementor, newImplementor);
    }
    modifier onlyImplementorOrOwner() {
        require(isImplementor() || isOwner(), "Is not implementor or owner");
        _;
    }
    modifier onlyImplementor() {
        require(isImplementor(), "Is not implementor");
        _;
    }
    modifier requireNonZero(address addr) {
        require(addr != address(0), "Expected a non-zero address");
        _;
    }
}
pragma solidity 0.4.24;
contract ERC20 {
    using SafeMath for uint256;
    Storage private externalStorage;
    string private name_;
    string private symbol_;
    uint8 private decimals_;
    event Transfer(
        address indexed from,
        address indexed to,
        uint256 value
    );
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
    constructor(
        string name,
        string symbol,
        uint8 decimals,
        Storage _externalStorage,
        bool initialDeployment
    )
        public
    {
        require(
            (_externalStorage != address(0) && (!initialDeployment)) ||
            (_externalStorage == address(0) && initialDeployment),
            "Cannot both create external storage and use the provided one.");
        name_ = name;
        symbol_ = symbol;
        decimals_ = decimals;
        if (initialDeployment) {
            externalStorage = new Storage(msg.sender, this);
        } else {
            externalStorage = _externalStorage;
        }
    }
    function getExternalStorage() public view returns(Storage) {
        return externalStorage;
    }
    function _name() internal view returns(string) {
        return name_;
    }
    function _symbol() internal view returns(string) {
        return symbol_;
    }
    function _decimals() internal view returns(uint8) {
        return decimals_;
    }
    function _totalSupply() internal view returns (uint256) {
        return externalStorage.getTotalSupply();
    }
    function _balanceOf(address owner) internal view returns (uint256) {
        return externalStorage.getBalance(owner);
    }
    function _allowance(address owner, address spender)
        internal
        view
        returns (uint256)
    {
        return externalStorage.getAllowed(owner, spender);
    }
    function _transfer(address originSender, address to, uint256 value)
        internal
        returns (bool)
    {
        require(to != address(0));
        externalStorage.decreaseBalance(originSender, value);
        externalStorage.increaseBalance(to, value);
        emit Transfer(originSender, to, value);
        return true;
    }
    function _approve(address originSender, address spender, uint256 value)
        internal
        returns (bool)
    {
        require(spender != address(0));
        externalStorage.setAllowed(originSender, spender, value);
        emit Approval(originSender, spender, value);
        return true;
    }
    function _transferFrom(
        address originSender,
        address from,
        address to,
        uint256 value
    )
        internal
        returns (bool)
    {
        externalStorage.decreaseAllowed(from, originSender, value);
        _transfer(from, to, value);
        emit Approval(
            from,
            originSender,
            externalStorage.getAllowed(from, originSender)
        );
        return true;
    }
    function _increaseAllowance(
        address originSender,
        address spender,
        uint256 addedValue
    )
        internal
        returns (bool)
    {
        require(spender != address(0));
        externalStorage.increaseAllowed(originSender, spender, addedValue);
        emit Approval(
            originSender, spender,
            externalStorage.getAllowed(originSender, spender)
        );
        return true;
    }
    function _decreaseAllowance(
        address originSender,
        address spender,
        uint256 subtractedValue
    )
        internal
        returns (bool)
    {
        require(spender != address(0));
        externalStorage.decreaseAllowed(originSender,
                                        spender,
                                        subtractedValue);
        emit Approval(
            originSender, spender,
            externalStorage.getAllowed(originSender, spender)
        );
        return true;
    }
    function _mint(address account, uint256 value) internal returns (bool)
    {
        require(account != 0);
        externalStorage.setTotalSupply(
            externalStorage.getTotalSupply().add(value));
        externalStorage.increaseBalance(account, value);
        emit Transfer(address(0), account, value);
        return true;
    }
    function _burn(address originSender, uint256 value) internal returns (bool)
    {
        require(originSender != 0);
        externalStorage.setTotalSupply(
            externalStorage.getTotalSupply().sub(value));
        externalStorage.decreaseBalance(originSender, value);
        emit Transfer(originSender, address(0), value);
        return true;
    }
    function _burnFrom(address originSender, address account, uint256 value)
        internal
        returns (bool)
    {
        require(value <= externalStorage.getAllowed(account, originSender));
        externalStorage.decreaseAllowed(account, originSender, value);
        _burn(account, value);
        emit Approval(account, originSender,
                      externalStorage.getAllowed(account, originSender));
        return true;
    }
}
pragma solidity 0.4.24;
contract UpgradeSupport is Ownable, ERC20 {
    event Upgraded(address indexed to);
    event UpgradeFinalized(address indexed upgradedFrom);
    address private _upgradedFrom;
    bool private enabled;
    IETokenProxy private upgradedToken;
    constructor(bool initialDeployment, address upgradedFrom) internal {
        require((upgradedFrom != address(0) && (!initialDeployment)) ||
                (upgradedFrom == address(0) && initialDeployment),
                "Cannot both be upgraded and initial deployment.");
        if (! initialDeployment) {
            enabled = false;
            _upgradedFrom = upgradedFrom;
        } else {
            enabled = true;
        }
    }
    modifier upgradeExists() {
        require(_upgradedFrom != address(0),
                "Must have a contract to upgrade from");
        _;
    }
    function finalizeUpgrade()
        external
        upgradeExists
        onlyProxy
    {
        enabled = true;
        emit UpgradeFinalized(msg.sender);
    }
    function upgrade(IETokenProxy _upgradedToken) public onlyOwner {
        require(!isUpgraded(), "Token is already upgraded");
        require(_upgradedToken != IETokenProxy(0),
                "Cannot upgrade to null address");
        require(_upgradedToken != IETokenProxy(this),
                "Cannot upgrade to myself");
        require(getExternalStora