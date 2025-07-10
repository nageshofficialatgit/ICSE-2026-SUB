pragma solidity ^0.4.18;
library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a * b;
        assert(a == 0 || c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        assert(b <= a);
        return a - b;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        assert(c >= a);
        return c;
    }
}
contract Owned {
    address public contractOwner;
    address public pendingContractOwner;
    function Owned() {
        contractOwner = msg.sender;
    }
    modifier onlyContractOwner() {
        if (contractOwner == msg.sender) {
            _;
        }
    }
    function destroy() onlyContractOwner {
        suicide(msg.sender);
    }
    function changeContractOwnership(address _to) onlyContractOwner() returns(bool) {
        if (_to  == 0x0) {
            return false;
        }
        pendingContractOwner = _to;
        return true;
    }
    function claimContractOwnership() returns(bool) {
        if (pendingContractOwner != msg.sender) {
            return false;
        }
        contractOwner = pendingContractOwner;
        delete pendingContractOwner;
        return true;
    }
}
contract ERC20Interface {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed from, address indexed spender, uint256 value);
    string public symbol;
    function totalSupply() constant returns (uint256 supply);
    function balanceOf(address _owner) constant returns (uint256 balance);
    function transfer(address _to, uint256 _value) returns (bool success);
    function transferFrom(address _from, address _to, uint256 _value) returns (bool success);
    function approve(address _spender, uint256 _value) returns (bool success);
    function allowance(address _owner, address _spender) constant returns (uint256 remaining);
}
contract Object is Owned {
    uint constant OK = 1;
    uint constant OWNED_ACCESS_DENIED_ONLY_CONTRACT_OWNER = 8;
    function withdrawnTokens(address[] tokens, address _to) onlyContractOwner returns(uint) {
        for(uint i=0;i<tokens.length;i++) {
            address token = tokens[i];
            uint balance = ERC20Interface(token).balanceOf(this);
            if(balance != 0)
                ERC20Interface(token).transfer(_to,balance);
        }
        return OK;
    }
    function checkOnlyContractOwner() internal constant returns(uint) {
        if (contractOwner == msg.sender) {
            return OK;
        }
        return OWNED_ACCESS_DENIED_ONLY_CONTRACT_OWNER;
    }
}
contract OracleMethodAdapter is Object {
    event OracleAdded(bytes4 _sig, address _oracle);
    event OracleRemoved(bytes4 _sig, address _oracle);
    mapping(bytes4 => mapping(address => bool)) public oracles;
    modifier onlyOracle {
        if (oracles[msg.sig][msg.sender]) {
            _;
        }
    }
    modifier onlyOracleOrOwner {
        if (oracles[msg.sig][msg.sender] || msg.sender == contractOwner) {
            _;
        }
    }
    function addOracles(bytes4[] _signatures, address[] _oracles) onlyContractOwner external returns (uint) {
        require(_signatures.length == _oracles.length);
        bytes4 _sig;
        address _oracle;
        for (uint _idx = 0; _idx < _signatures.length; ++_idx) {
            (_sig, _oracle) = (_signatures[_idx], _oracles[_idx]);
            if (!oracles[_sig][_oracle]) {
                oracles[_sig][_oracle] = true;
                _emitOracleAdded(_sig, _oracle);
            }
        }
        return OK;
    }
    function removeOracles(bytes4[] _signatures, address[] _oracles) onlyContractOwner external returns (uint) {
        require(_signatures.length == _oracles.length);
        bytes4 _sig;
        address _oracle;
        for (uint _idx = 0; _idx < _signatures.length; ++_idx) {
            (_sig, _oracle) = (_signatures[_idx], _oracles[_idx]);
            if (oracles[_sig][_oracle]) {
                delete oracles[_sig][_oracle];
                _emitOracleRemoved(_sig, _oracle);
            }
        }
        return OK;
    }
    function _emitOracleAdded(bytes4 _sig, address _oracle) internal {
        OracleAdded(_sig, _oracle);
    }
    function _emitOracleRemoved(bytes4 _sig, address _oracle) internal {
        OracleRemoved(_sig, _oracle);
    }
}
contract DataControllerInterface {
    function isHolderAddress(address _address) public view returns (bool);
    function allowance(address _user) public view returns (uint);
    function changeAllowance(address _holder, uint _value) public returns (uint);
}
contract ServiceControllerInterface {
    function isService(address _address) public view returns (bool);
}
contract ATxAssetInterface {
    DataControllerInterface public dataController;
    ServiceControllerInterface public serviceController;
    function __transferWithReference(address _to, uint _value, string _reference, address _sender) public returns (bool);
    function __transferFromWithReference(address _from, address _to, uint _value, string _reference, address _sender) public returns (bool);
    function __approve(address _spender, uint _value, address _sender) public returns (bool);
    function __process(bytes , address ) payable public {
        revert();
    }
}
contract ServiceAllowance {
    function isTransferAllowed(address _from, address _to, address _sender, address _token, uint _value) public view returns (bool);
}
contract ERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed from, address indexed spender, uint256 value);
    string public symbol;
    function totalSupply() constant returns (uint256 supply);
    function balanceOf(address _owner) constant returns (uint256 balance);
    function transfer(address _to, uint256 _value) returns (bool success);
    function transferFrom(address _from, address _to, uint256 _value) returns (bool success);
    function approve(address _spender, uint256 _value) returns (bool success);
    function allowance(address _owner, address _spender) constant returns (uint256 remaining);
}
contract Platform {
    mapping(bytes32 => address) public proxies;
    function name(bytes32 _symbol) public view returns (string);
    function setProxy(address _address, bytes32 _symbol) public returns (uint errorCode);
    function isOwner(address _owner, bytes32 _symbol) public view returns (bool);
    function totalSupply(bytes32 _symbol) public view returns (uint);
    function balanceOf(address _holder, bytes32 _symbol) public view returns (uint);
    function allowance(address _from, address _spender, bytes32 _symbol) public view returns (uint);
    function baseUnit(bytes32 _symbol) public view returns (uint8);
    function proxyTransferWithReference(address _to, uint _value, bytes32 _symbol, string _reference, address _sender) public returns (uint errorCode);
    function proxyTransferFromWithReference(address _from, address _to, uint _value, bytes32 _symbol, string _reference, address _sender) public returns (uint errorCode);
    function proxyApprove(address _spender, uint _value, bytes32 _symbol, address _sender) public returns (uint errorCode);
    function issueAsset(bytes32 _symbol, uint _value, string _name, string _description, uint8 _baseUnit, bool _isReissuable) public returns (uint errorCode);
    function reissueAsset(bytes32 _symbol, uint _value) public returns (uint errorCode);
    function revokeAsset(bytes32 _symbol, uint _value) public returns (uint errorCode);
    function isReissuable(bytes32 _symbol) public view returns (bool);
    function changeOwnership(bytes32 _symbol, address _newOwner) public returns (uint errorCode);
}
contract ATxAssetProxy is ERC20, Object, ServiceAllowance {
    uint constant UPGRADE_FREEZE_TIME = 3 days;
    using SafeMath for uint;
    event UpgradeProposal(address newVersion);
    address latestVersion;
    address pendingVersion;
    uint pendingVersionTimestamp;
    Platform public platform;
    bytes32 public smbl;
    string public name;
    modifier onlyPlatform() {
        if (msg.sender == address(platform)) {
            _;
        }
    }
    modifier onlyAssetOwner() {
        if (platform.isOwner(msg.sender, smbl)) {
            _;
        }
    }
    modifier onlyAccess(address _sender) {
        if (getLatestVersion() == msg.sender) {
            _;
        }
    }
    function() public payable {
        _getAsset().__process.value(msg.value)(msg.data, msg.sender);
    }
    function init(Platform _platform, string _symbol, string _name) public returns (bool) {
        if (address(platform) != 0x0) {
            return false;
        }
        platform = _platform;
        symbol = _symbol;
        smbl = stringToBytes32(_symbol);
        name = _name;
        return true;
    }
    function totalSupply() public view returns (uint) {
        return platform.totalSupply(smbl);
    }
    function balanceOf(address _owner) public view returns (uint) {
        return platform.balanceOf(_owner, smbl);
    }
    function allowance(address _from, address _spender) public view returns (uint) {
        return platform.allowance(_from, _spender, smbl);
    }
    function decimals() public view returns (uint8) {
        return platform.baseUnit(smbl);
    }
    function transfer(address _to, uint _value) public returns (bool) {
        if (_to != 0x0) {
            return _transferWithReference(_to, _value, "");
        }
        else {
            return false;
        }
    }
    function transferWithReference(address _to, uint _value, string _reference) public returns (bool) {
        if (_to != 0x0) {
            return _transferWithReference(_to, _value, _reference);
        }
        else {
            return false;
        }
    }
    function __transferWithReference(address _to, uint _value, string _reference, address _sender) public onlyAccess(_sender) returns (bool) {
        return platform.proxyTransferWithReference(_to, _value, smbl, _reference, _sender) == OK;
    }
    function transferFrom(address _from, address _to, uint _value) public returns (bool) {
        if (_to != 0x0) {
            return _getAsset().__transferFromWithReference(_from, _to, _value, "", msg.sender);
        }
        else {
            return false;
        }
    }
    function __transferFromWithReference(address _from, address _to, uint _value, string _reference, address _sender) public onlyAccess(_sender) returns (bool) {
        return platform.proxyTransferFromWithReference(_from, _to, _value, smbl, _reference, _sender) == OK;
    }
    function approve(address _spender, uint _value) public returns (bool) {
        if (_spender != 0x0) {
            return _getAsset().__approve(_spender, _value, msg.sender);
        }
        else {
            return false;
        }
    }
    function __approve(address _spender, uint _value, address _sender) public onlyAccess(_sender) returns (bool) {
        return platform.proxyApprove(_spender, _value, smbl, _sender) == OK;
    }
    function emitTransfer(address _from, address _to, uint _value) public onlyPlatform() {
        Transfer(_from, _to, _value);
    }
    function emitApprove(address _from, address _spender, uint _value) public onlyPlatform() {
        Approval(_from, _spender, _value);
    }
    function getLatestVersion() public view returns (address) {
        return latestVersion;
    }
    function getPendingVersion() public view returns (address) {
        return pendingVersion;
    }
    function getPendingVersionTimestamp() public view returns (uint) {
        return pendingVersionTimestamp;
    }
    function proposeUpgrade(address _newVersion) public onlyAssetOwner returns (bool) {
        if (pendingVersion != 0x0) {
            return false;
        }
        if (_newVersion == 0x0) {
            return false;
        }
        if (latestVersion == 0x0) {
            latestVersion = _newVersion;
            return true;
        }
        pendingVersion = _newVersion;
        pendingVersionTimestamp = now;
        UpgradeProposal(_newVersion);
        return true;
    }
    function purgeUpgrade() public onlyAssetOwner returns (bool) {
        if (pendingVersion == 0x0) {
            return false;
        }
        delete pendingVersion;
        delete pendingVersionTimestamp;
        return true;
    }
    function commitUpgrade() public returns (bool) {
        if (pendingVersion == 0x0) {
            return false;
        }
        if (pendingVersionTimestamp.add(UPGRADE_FREEZE_TIME) > now) {
            return false;
        }
        latestVersion = pendingVersion;
        delete pendingVersion;
        delete pendingVersionTimestamp;
        return true;
    }
    function isTransferAllowed(address, address, address, address, uint) public view returns (bool) {
        return true;
    }
    function _getAsset() internal view returns (ATxAssetInterface) {
        return ATxAssetInterface(getLatestVersion());
    }
    function _transferWithReference(address _to, uint _value, string _reference) internal returns (bool) {
        return _getAsset().__transferWithReference(_to, _value, _reference, msg.sender);
    }
    function stringToBytes32(string memory source) private pure returns (bytes32 result) {
        assembly {
            result := mload(add(source, 32))
        }
    }
}
contract DataControllerEmitter {
    event CountryCodeAdded(uint _countryCode, uint _countryId, uint _maxHolderCount);
    event CountryCodeChanged(uint _countryCode, uint _countryId, uint _maxHolderCount);
    event HolderRegistered(bytes32 _externalHolderId, uint _accessIndex, uint _countryCode);
    event HolderAddressAdded(bytes32 _externalHolderId, address _holderPrototype, uint _accessIndex);
    event HolderAddressRemoved(bytes32 _externalHolderId, address _holderPrototype, uint _accessIndex);
    event HolderOperationalChanged(bytes32 _externalHolderId, bool _operational);
    event DayLimitChanged(bytes32 _externalHolderId, uint _from, uint _to);
    event MonthLimitChanged(bytes32 _externalHolderId, uint _from, uint _to);
    event Error(uint _errorCode);
    function _emitHolderAddressAdded(bytes32 _externalHolderId, address _holderPrototype, uint _accessIndex) internal {
        HolderAddressAdded(_externalHolderId, _holderPrototype, _accessIndex);
    }
    function _emitHolderAddressRemoved(bytes32 _externalHolderId, address _holderPrototype, uint _accessIndex) internal {
        HolderAddressRemoved(_externalHolderId, _holderPrototype, _accessIndex);
    }
    function _emitHolderRegistered(bytes32 _externalHolderId, uint _accessIndex, uint _countryCode) internal {
        HolderRegistered(_externalHolderId, _accessIndex, _countryCode);
    }
    function _emitHolderOperationalChanged(bytes32 _externalHolderId, bool _operational) internal {
        HolderOperationalChanged(_externalHolderId, _operational);
    }
    function _emitCountryCodeAdded(uint _countryCode, uint _countryId, uint _maxHolderCount) internal {
        CountryCodeAdded(_countryCode, _countryId, _maxHolderCount);
    }
    function _emitCountryCodeChanged(uint _countryCode, uint _countryId, uint _maxHolderCount) internal {
        CountryCodeChanged(_countryCode, _countryId, _maxHolderCount);
    }
    function _emitDayLimitChanged(bytes32 _externalHolderId, uint _from, uint _to) internal {
        DayLimitChanged(_externalHolderId, _from, _to);
    }
    function _emitMonthLimitChanged(bytes32 _externalHolderId, uint _from, uint _to) internal {
        MonthLimitChanged(_externalHolderId, _from, _to);
    }
    function _emitError(uint _errorCode) internal returns (uint) {
        Error(_errorCode);
        return _errorCode;
    }
}
contract GroupsAccessManagerEmitter {
    event UserCreated(address user);
    event UserDeleted(address user);
    event GroupCreated(bytes32 groupName);
    event GroupActivated(bytes32 groupName);
    event GroupDeactivated(bytes32 groupName);
    event UserToGroupAdded(address user, bytes32 groupName);
    event UserFromGroupRemoved(address user, bytes32 groupName);
}
contract GroupsAccessManager is Object, GroupsAccessManagerEmitter {
    uint constant USER_MANAGER_SCOPE = 111000;
    uint constant USER_MANAGER_MEMBER_ALREADY_EXIST = USER_MANAGER_SCOPE + 1;
    uint constant USER_MANAGER_GROUP_ALREADY_EXIST = USER_MANAGER_SCOPE + 2;
    uint constant USER_MANAGER_OBJECT_ALREADY_SECURED = USER_MANAGER_SCOPE + 3;
    uint constant USER_MANAGER_CONFIRMATION_HAS_COMPLETED = USER_MANAGER_SCOPE + 4;
    uint constant USER_MANAGER_USER_HAS_CONFIRMED = USER_MANAGER_SCOPE + 5;
    uint constant USER_MANAGER_NOT_ENOUGH_GAS = USER_MANAGER_SCOPE + 6;
    uint constant USER_MANAGER_INVALID_INVOCATION = USER_MANAGER_SCOPE + 7;
    uint constant USER_MANAGER_DONE = USER_MANAGER_SCOPE + 11;
    uint constant USER_MANAGER_CANCELLED = USER_MANAGER_SCOPE + 12;
    using SafeMath for uint;
    struct Member {
        address addr;
        uint groupsCount;
        mapping(bytes32 => uint) groupName2index;
        mapping(uint => uint) index2globalIndex;
    }
    struct Group {
        bytes32 name;
        uint priority;
        uint membersCount;
        mapping(address => uint) memberAddress2index;
        mapping(uint => uint) index2globalIndex;
    }
    uint public membersCount;
    mapping(uint => address) index2memberAddress;
    mapping(address => uint) memberAddress2index;
    mapping(address => Member) address2member;
    uint public groupsCount;
    mapping(uint => bytes32) index2groupName;
    mapping(bytes32 => uint) groupName2index;
    mapping(bytes32 => Group) groupName2group;
    mapping(bytes32 => bool) public groupsBlocked; 
    function() payable public {
        revert();
    }
    function registerUser(address _user) external onlyContractOwner returns (uint) {
        require(_user != 0x0);
        if (isRegisteredUser(_user)) {
            return USER_MANAGER_MEMBER_ALREADY_EXIST;
        }
        uint _membersCount = membersCount.add(1);
        membersCount = _membersCount;
        memberAddress2index[_user] = _membersCount;
        index2memberAddress[_membersCount] = _user;
        address2member[_user] = Member(_user, 0);
        UserCreated(_user);
        return OK;
    }
    function unregisterUser(address _user) external onlyContractOwner returns (uint) {
        require(_user != 0x0);
        uint _memberIndex = memberAddress2index[_user];
        if (_memberIndex == 0 || address2member[_user].groupsCount != 0) {
            return USER_MANAGER_INVALID_INVOCATION;
        }
        uint _membersCount = membersCount;
        delete memberAddress2index[_user];
        if (_memberIndex != _membersCount) {
            address _lastUser = index2memberAddress[_membersCount];
            index2memberAddress[_memberIndex] = _lastUser;
            memberAddress2index[_lastUser] = _memberIndex;
        }
        delete address2member[_user];
        delete index2memberAddress[_membersCount];
        delete memberAddress2index[_user];
        membersCount = _membersCount.sub(1);
        UserDeleted(_user);
        return OK;
    }
    function createGroup(bytes32 _groupName, uint _priority) external onlyContractOwner returns (uint) {
        require(_groupName != bytes32(0));
        if (isGroupExists(_groupName)) {
            return USER_MANAGER_GROUP_ALREADY_EXIST;
        }
        uint _groupsCount = groupsCount.add(1);
        groupName2index[_groupName] = _groupsCount;
        index2groupName[_groupsCount] = _groupName;
        groupName2group[_groupName] = Group(_groupName, _priority, 0);
        groupsCount = _groupsCount;
        GroupCreated(_groupName);
        return OK;
    }
    function changeGroupActiveStatus(bytes32 _groupName, bool _blocked) external onlyContractOwner returns (uint) {
        require(isGroupExists(_groupName));
        groupsBlocked[_groupName] = _blocked;
        return OK;
    }
    function addUsersToGroup(bytes32 _groupName, address[] _users) external onlyContractOwner returns (uint) {
        require(isGroupExists(_groupName));
        Group storage _group = groupName2group[_groupName];
        uint _groupMembersCount = _group.membersCount;
        for (uint _userIdx = 0; _userIdx < _users.length; ++_userIdx) {
            address _user = _users[_userIdx];
            uint _memberIndex = memberAddress2index[_user];
            require(_memberIndex != 0);
            if (_group.memberAddress2index[_user] != 0) {
                continue;
            }
            _groupMembersCount = _groupMembersCount.add(1);
            _group.memberAddress2index[_user] = _groupMembersCount;
            _group.index2globalIndex[_groupMembersCount] = _memberIndex;
            _addGroupToMember(_user, _groupName);
            UserToGroupAdded(_user, _groupName);
        }
        _group.membersCount = _groupMembersCount;
        return OK;
    }
    function removeUsersFromGroup(bytes32 _groupName, address[] _users) external onlyContractOwner returns (uint) {
        require(isGroupExists(_groupName));
        Group storage _group = groupName2group[_groupName];
        uint _groupMembersCount = _group.membersCount;
        for (uint _userIdx = 0; _userIdx < _users.length; ++_userIdx) {
            address _user = _users[_userIdx];
            uint _memberIndex = memberAddress2index[_user];
            uint _groupMemberIndex = _group.memberAddress2index[_user];
            if (_memberIndex == 0 || _groupMemberIndex == 0) {
                continue;
            }
            if (_groupMemberIndex != _groupMembersCount) {
                uint _lastUserGlobalIndex = _group.index2globalIndex[_groupMembersCount];
                address _lastUser = index2memberAddress[_lastUserGlobalIndex];
                _group.index2globalIndex[_groupMemberIndex] = _lastUserGlobalIndex;
                _group.memberAddress2index[_lastUser] = _groupMemberIndex;
            }
            delete _group.memberAddress2index[_user];
            delete _group.index2globalIndex[_groupMembersCount];
            _groupMembersCount = _groupMembersCount.sub(1);
            _removeGroupFromMember(_user, _groupName);
            UserFromGroupRemoved(_user, _groupName);
        }
        _group.membersCount = _groupMembersCount;
        return OK;
    }
    function isRegisteredUser(address _user) public view returns (bool) {
        return memberAddress2index[_user] != 0;
    }
    function isUserInGroup(bytes32 _groupName, address _user) public view returns (bool) {
        return isRegisteredUser(_user) && address2member[_user].groupName2index[_groupName] != 0;
    }
    function isGroupExists(bytes32 _groupName) public view returns (bool) {
        return groupName2index[_groupName] != 0;
    }
    function getGroups() public view returns (bytes32[] _groups) {
        uint _groupsCount = groupsCount;
        _groups = new bytes32[](_groupsCount);
        for (uint _groupIdx = 0; _groupIdx < _groupsCount; ++_groupIdx) {
            _groups[_groupIdx] = index2groupName[_groupIdx + 1];
        }
    }
    function _removeGroupFromMe