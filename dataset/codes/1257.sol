pragma solidity ^0.4.23;
contract TrueCoinReceiver {
    function tokenFallback( address from, uint256 value ) external;
}
contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}
contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender) public view returns (uint256);
  function transferFrom(address from, address to, uint256 value) public returns (bool);
  function approve(address spender, uint256 value) public returns (bool);
  event Approval(address indexed owner, address indexed spender, uint256 value);
}
interface RegistryClone {
    function syncAttributeValue(address _who, bytes32 _attribute, uint256 _value) external;
}
contract Registry {
    struct AttributeData {
        uint256 value;
        bytes32 notes;
        address adminAddr;
        uint256 timestamp;
    }
    address public owner;
    address public pendingOwner;
    bool initialized;
    mapping(address => mapping(bytes32 => AttributeData)) attributes;
    bytes32 constant WRITE_PERMISSION = keccak256("canWriteTo-");
    mapping(bytes32 => RegistryClone[]) subscribers;
    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );
    event SetAttribute(address indexed who, bytes32 attribute, uint256 value, bytes32 notes, address indexed adminAddr);
    event SetManager(address indexed oldManager, address indexed newManager);
    event StartSubscription(bytes32 indexed attribute, RegistryClone indexed subscriber);
    event StopSubscription(bytes32 indexed attribute, RegistryClone indexed subscriber);
    function confirmWrite(bytes32 _attribute, address _admin) internal view returns (bool) {
        return (_admin == owner || hasAttribute(_admin, keccak256(WRITE_PERMISSION ^ _attribute)));
    }
    function setAttribute(address _who, bytes32 _attribute, uint256 _value, bytes32 _notes) public {
        require(confirmWrite(_attribute, msg.sender));
        attributes[_who][_attribute] = AttributeData(_value, _notes, msg.sender, block.timestamp);
        emit SetAttribute(_who, _attribute, _value, _notes, msg.sender);
        RegistryClone[] storage targets = subscribers[_attribute];
        uint256 index = targets.length;
        while (index --> 0) {
            targets[index].syncAttributeValue(_who, _attribute, _value);
        }
    }
    function subscribe(bytes32 _attribute, RegistryClone _syncer) external onlyOwner {
        subscribers[_attribute].push(_syncer);
        emit StartSubscription(_attribute, _syncer);
    }
    function unsubscribe(bytes32 _attribute, uint256 _index) external onlyOwner {
        uint256 length = subscribers[_attribute].length;
        require(_index < length);
        emit StopSubscription(_attribute, subscribers[_attribute][_index]);
        subscribers[_attribute][_index] = subscribers[_attribute][length - 1];
        subscribers[_attribute].length = length - 1;
    }
    function subscriberCount(bytes32 _attribute) public view returns (uint256) {
        return subscribers[_attribute].length;
    }
    function setAttributeValue(address _who, bytes32 _attribute, uint256 _value) public {
        require(confirmWrite(_attribute, msg.sender));
        attributes[_who][_attribute] = AttributeData(_value, "", msg.sender, block.timestamp);
        emit SetAttribute(_who, _attribute, _value, "", msg.sender);
        RegistryClone[] storage targets = subscribers[_attribute];
        uint256 index = targets.length;
        while (index --> 0) {
            targets[index].syncAttributeValue(_who, _attribute, _value);
        }
    }
    function hasAttribute(address _who, bytes32 _attribute) public view returns (bool) {
        return attributes[_who][_attribute].value != 0;
    }
    function getAttribute(address _who, bytes32 _attribute) public view returns (uint256, bytes32, address, uint256) {
        AttributeData memory data = attributes[_who][_attribute];
        return (data.value, data.notes, data.adminAddr, data.timestamp);
    }
    function getAttributeValue(address _who, bytes32 _attribute) public view returns (uint256) {
        return attributes[_who][_attribute].value;
    }
    function getAttributeAdminAddr(address _who, bytes32 _attribute) public view returns (address) {
        return attributes[_who][_attribute].adminAddr;
    }
    function getAttributeTimestamp(address _who, bytes32 _attribute) public view returns (uint256) {
        return attributes[_who][_attribute].timestamp;
    }
    function syncAttribute(bytes32 _attribute, uint256 _startIndex, address[] _addresses) external {
        RegistryClone[] storage targets = subscribers[_attribute];
        uint256 index = targets.length;
        while (index --> _startIndex) {
            RegistryClone target = targets[index];
            for (uint256 i = _addresses.length; i --> 0; ) {
                address who = _addresses[i];
                target.syncAttributeValue(who, _attribute, attributes[who][_attribute].value);
            }
        }
    }
    function reclaimEther(address _to) external onlyOwner {
        _to.transfer(address(this).balance);
    }
    function reclaimToken(ERC20 token, address _to) external onlyOwner {
        uint256 balance = token.balanceOf(this);
        token.transfer(_to, balance);
    }
    modifier onlyOwner() {
        require(msg.sender == owner, "only Owner");
        _;
    }
    modifier onlyPendingOwner() {
        require(msg.sender == pendingOwner);
        _;
    }
    function transferOwnership(address newOwner) public onlyOwner {
        pendingOwner = newOwner;
    }
    function claimOwnership() public onlyPendingOwner {
        emit OwnershipTransferred(owner, pendingOwner);
        owner = pendingOwner;
        pendingOwner = address(0);
    }
}
contract Ownable {
  address public owner;
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
  function Ownable() public {
    owner = msg.sender;
  }
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }
  function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }
}
contract Claimable is Ownable {
  address public pendingOwner;
  modifier onlyPendingOwner() {
    require(msg.sender == pendingOwner);
    _;
  }
  function transferOwnership(address newOwner) onlyOwner public {
    pendingOwner = newOwner;
  }
  function claimOwnership() onlyPendingOwner public {
    emit OwnershipTransferred(owner, pendingOwner);
    owner = pendingOwner;
    pendingOwner = address(0);
  }
}
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256 c) {
    if (a == 0) {
      return 0;
    }
    c = a * b;
    assert(c / a == b);
    return c;
  }
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    return a / b;
  }
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }
  function add(uint256 a, uint256 b) internal pure returns (uint256 c) {
    c = a + b;
    assert(c >= a);
    return c;
  }
}
contract BalanceSheet is Claimable {
    using SafeMath for uint256;
    mapping (address => uint256) public balanceOf;
    function addBalance(address _addr, uint256 _value) public onlyOwner {
        balanceOf[_addr] = balanceOf[_addr].add(_value);
    }
    function subBalance(address _addr, uint256 _value) public onlyOwner {
        balanceOf[_addr] = balanceOf[_addr].sub(_value);
    }
    function setBalance(address _addr, uint256 _value) public onlyOwner {
        balanceOf[_addr] = _value;
    }
}
contract AllowanceSheet is Claimable {
    using SafeMath for uint256;
    mapping (address => mapping (address => uint256)) public allowanceOf;
    function addAllowance(address _tokenHolder, address _spender, uint256 _value) public onlyOwner {
        allowanceOf[_tokenHolder][_spender] = allowanceOf[_tokenHolder][_spender].add(_value);
    }
    function subAllowance(address _tokenHolder, address _spender, uint256 _value) public onlyOwner {
        allowanceOf[_tokenHolder][_spender] = allowanceOf[_tokenHolder][_spender].sub(_value);
    }
    function setAllowance(address _tokenHolder, address _spender, uint256 _value) public onlyOwner {
        allowanceOf[_tokenHolder][_spender] = _value;
    }
}
contract ProxyStorage {
    address public owner;
    address public pendingOwner;
    bool initialized;
    BalanceSheet balances_Deprecated;
    AllowanceSheet allowances_Deprecated;
    uint256 totalSupply_;
    bool private paused_Deprecated = false;
    address private globalPause_Deprecated;
    uint256 public burnMin = 0;
    uint256 public burnMax = 0;
    Registry public registry;
    string name_Deprecated;
    string symbol_Deprecated;
    uint[] gasRefundPool_Deprecated;
    uint256 private redemptionAddressCount_Deprecated;
    uint256 public minimumGasPriceForFutureRefunds;
    mapping (address => uint256) _balanceOf;
    mapping (address => mapping (address => uint256)) _allowance;
    mapping (bytes32 => mapping (address => uint256)) attributes;
}
contract HasOwner is ProxyStorage {
    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );
    constructor() public {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), owner);
    }
    modifier onlyOwner() {
        require(msg.sender == owner, "only Owner");
        _;
    }
    modifier onlyPendingOwner() {
        require(msg.sender == pendingOwner);
        _;
    }
    function transferOwnership(address newOwner) public onlyOwner {
        pendingOwner = newOwner;
    }
    function claimOwnership() public onlyPendingOwner {
        emit OwnershipTransferred(owner, pendingOwner);
        owner = pendingOwner;
        pendingOwner = address(0);
    }
}
contract ReclaimerToken is HasOwner {
    function reclaimEther(address _to) external onlyOwner {
        _to.transfer(address(this).balance);
    }
    function reclaimToken(ERC20 token, address _to) external onlyOwner {
        uint256 balance = token.balanceOf(this);
        token.transfer(_to, balance);
    }
    function reclaimContract(Ownable _ownable) external onlyOwner {
        _ownable.transferOwnership(owner);
    }
}
contract ModularBasicToken is HasOwner {
    using SafeMath for uint256;
    event Transfer(address indexed from, address indexed to, uint256 value);
    function totalSupply() public view returns (uint256) {
        return totalSupply_;
    }
    function balanceOf(address _who) public view returns (uint256) {
        return _getBalance(_who);
    }
    function _getBalance(address _who) internal view returns (uint256) {
        return _balanceOf[_who];
    }
    function _addBalance(address _who, uint256 _value) internal returns (uint256 priorBalance) {
        priorBalance = _balanceOf[_who];
        _balanceOf[_who] = priorBalance.add(_value);
    }
    function _subBalance(address _who, uint256 _value) internal returns (uint256 result) {
        result = _balanceOf[_who].sub(_value);
        _balanceOf[_who] = result;
    }
    function _setBalance(address _who, uint256 _value) internal {
        _balanceOf[_who] = _value;
    }
}
contract ModularStandardToken is ModularBasicToken {
    using SafeMath for uint256;
    event Approval(address indexed owner, address indexed spender, uint256 value);
    function approve(address _spender, uint256 _value) public returns (bool) {
        _approveAllArgs(_spender, _value, msg.sender);
        return true;
    }
    function _approveAllArgs(address _spender, uint256 _value, address _tokenHolder) internal {
        _setAllowance(_tokenHolder, _spender, _value);
        emit Approval(_tokenHolder, _spender, _value);
    }
    function increaseApproval(address _spender, uint _addedValue) public returns (bool) {
        _increaseApprovalAllArgs(_spender, _addedValue, msg.sender);
        return true;
    }
    function _increaseApprovalAllArgs(address _spender, uint256 _addedValue, address _tokenHolder) internal {
        _addAllowance(_tokenHolder, _spender, _addedValue);
        emit Approval(_tokenHolder, _spender, _getAllowance(_tokenHolder, _spender));
    }
    function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
        _decreaseApprovalAllArgs(_spender, _subtractedValue, msg.sender);
        return true;
    }
    function _decreaseApprovalAllArgs(address _spender, uint256 _subtractedValue, address _tokenHolder) internal {
        uint256 oldValue = _getAllowance(_tokenHolder, _spender);
        uint256 newValue;
        if (_subtractedValue > oldValue) {
            newValue = 0;
        } else {
            newValue = oldValue - _subtractedValue;
        }
        _setAllowance(_tokenHolder, _spender, newValue);
        emit Approval(_tokenHolder,_spender, newValue);
    }
    function allowance(address _who, address _spender) public view returns (uint256) {
        return _getAllowance(_who, _spender);
    }
    function _getAllowance(address _who, address _spender) internal view returns (uint256 value) {
        return _allowance[_who][_spender];
    }
    function _addAllowance(address _who, address _spender, uint256 _value) internal {
        _allowance[_who][_spender] = _allowance[_who][_spender].add(_value);
    }
    function _subAllowance(address _who, address _spender, uint256 _value) internal returns (uint256 newAllowance){
        newAllowance = _allowance[_who][_spender].sub(_value);
        _allowance[_who][_spender] = newAllowance;
    }
    function _setAllowance(address _who, address _spender, uint256 _value) internal {
        _allowance[_who][_spender] = _value;
    }
}
contract ModularBurnableToken is ModularStandardToken {
    event Burn(address indexed burner, uint256 value);
    event Mint(address indexed to, uint256 value);
    uint256 constant CENT = 10 ** 16;
    function burn(uint256 _value) external {
        _burnAllArgs(msg.sender, _value - _value % CENT);
    }
    function _burnAllArgs(address _from, uint256 _value) internal {
        _subBalance(_from, _value);
        totalSupply_ = totalSupply_.sub(_value);
        emit Burn(_from, _value);
        emit Transfer(_from, address(0), _value);
    }
}
contract BurnableTokenWithBounds is ModularBurnableToken {
    event SetBurnBounds(uint256 newMin, uint256 newMax);
    function _burnAllArgs(address _burner, uint256 _value) internal {
        require(_value >= burnMin, "below min burn bound");
        require(_value <= burnMax, "exceeds max burn bound");
        super._burnAllArgs(_burner, _value);
    }
    function setBurnBounds(uint256 _min, uint256 _max) external onlyOwner {
        require(_min <= _max, "min > max");
        burnMin = _min;
        burnMax = _max;
        emit SetBurnBounds(_min, _max);
    }
}
contract GasRefundToken is ProxyStorage {
    function sponsorGas2() external {
        assembly {
            mstore(0, or(0x601f8060093d393df33d33730000000000000000000000000000000000000000, address))
            mstore(32,   0x14601d5780fd5bff000000000000000000000000000000000000000000000000)
            let sheep := create(0, 0, 0x28)
            let offset := sload(0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff)
            let location := sub(0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe, offset)
            sstore(location, sheep)
            sheep := create(0, 0, 0x28)
            sstore(sub(location, 1), sheep)
            sheep := create(0, 0, 0x28)
            sstore(sub(location, 2), sheep)
            sstore(0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff, add(offset, 3))
        }
    }
    function gasRefund39() internal {
        assembly {
            let offset := sload(0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff)
            if gt(offset, 0) {
              let location := sub(0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff,offset)
              sstore(0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff, sub(offset, 1))
              let sheep := sload(location)
              pop(call(gas, sheep, 0, 0, 0, 0, 0))
              sstore(location, 0)
            }
        }
    }
    function sponsorGas() external {
        uint256 refundPrice = minimumGasPriceForFutureRefunds;
        require(refundPrice > 0);
        assembly {
            let offset := sload(0xfffff)
            let result := add(offset, 9)
            sstore(0xfffff, result)
            let position := add(offset, 0x100000)
            sstore(position, refundPrice)
            position := add(position, 1)
            sstore(position, refundPrice)
            position := add(position, 1)
            sstore(position, refundPrice)
            position := add(position, 1)
            sstore(position, refundPrice)
            position := add(position, 1)
            sstore(position, refundPrice)
            position := add(position, 1)
            sstore(position, refundPrice)
            position := add(position, 1)
            sstore(position, refundPrice)
            position := add(position, 1)
            sstore(position, refundPrice)
            position := add(position, 1)
            sstore(position, refundPrice)
        }
    }
    function minimumGasPriceForRefund() public view returns (uint256 result) {
        assembly {
            let offset := sload(0xfffff)
            let location := add(offset, 0xfffff)
            result := add(sload(location), 1)
        }
    }
    function gasRefund30() internal {
        assembly {
            let offset := sload(0xfffff)
            if gt(offset, 1) {
                let location := add(offset, 0xfffff)
                if gt(gasprice,sload(location)) {
                    sstore(location, 0)
                    location := sub(location, 1)
                    sstore(location, 0)
                    sstore(0xfffff, sub(offset, 2))
                }
            }
        }
    }
    function gasRefund15() internal {
        assembly {
            let offset := sload(0xfffff)
            if gt(offset, 1) {
                let location := add(offset, 0xfffff)
                if gt(gasprice,sload(location)) {
                    sstore(location, 0)
                    sstore(0xfffff, sub(offset, 1))
                }
            }
        }
    }
    function remainingGasRefundPool() public view returns (uint length) {
        assembly {
            length := sload(0xfffff)
        }
    }
    function gasRefundPool(uint256 _index) public view returns (uint256 gasPrice) {
        assembly {
            gasPrice := sload(add(0x100000, _index))
        }
    }
    bytes32 constant CAN_SET_FUTURE_REFUND_MIN_GAS_PRICE = "canSetFutureRefundMinGasPrice";
    function setMinimumGasPriceForFutureRefunds(uint256 _minimumGasPriceForFutureRefunds) public {
        require(registry.hasAttribute(msg.sender, CAN_SET_FUTURE_REFUND_MIN_GAS_PRICE));
        minimumGasPriceForFutureRefunds = _minimumGasPriceForFutureRefunds;
    }
}
contract CompliantDepositTokenWithHook is ReclaimerToken, RegistryClone, BurnableTokenWithBounds, GasRefundToken {
    bytes32 constant IS_REGISTERED_CONTRACT = "isRegisteredContract";
    bytes32 constant IS_DEPOSIT_ADDRESS = "isDepositAddress";
    uint256 constant REDEMPTION_ADDRESS_COUNT = 0x100000;
    bytes32 constant IS_BLACKLISTED = "isBlacklisted";
    function canBurn() internal pure returns (bytes32);
    function transfer(address _to, uint256 _value) public returns (bool) {
        _transferAllArgs(msg.sender, _to, _value);
        return true;
    }
    /**
     * @dev Transfer tokens from one address to another
     * @param _from address The address which you want to send tokens from
     * @param _to address The address which you want to transfer to
     * @param _value uint256 the amoun