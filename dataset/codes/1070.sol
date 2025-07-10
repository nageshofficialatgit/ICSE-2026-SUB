pragma solidity ^0.4.24;
library SafeMath {
  function mul(uint256 _a, uint256 _b) internal pure returns (uint256 c) {
    if (_a == 0) {
      return 0;
    }
    c = _a * _b;
    assert(c / _a == _b);
    return c;
  }
  function div(uint256 _a, uint256 _b) internal pure returns (uint256) {
    return _a / _b;
  }
  function sub(uint256 _a, uint256 _b) internal pure returns (uint256) {
    assert(_b <= _a);
    return _a - _b;
  }
  function add(uint256 _a, uint256 _b) internal pure returns (uint256 c) {
    c = _a + _b;
    assert(c >= _a);
    return c;
  }
}
library Roles {
  struct Role {
    mapping (address => bool) bearer;
  }
  function add(Role storage _role, address _addr)
    internal
  {
    _role.bearer[_addr] = true;
  }
  function remove(Role storage _role, address _addr)
    internal
  {
    _role.bearer[_addr] = false;
  }
  function check(Role storage _role, address _addr)
    internal
    view
  {
    require(has(_role, _addr));
  }
  function has(Role storage _role, address _addr)
    internal
    view
    returns (bool)
  {
    return _role.bearer[_addr];
  }
}
contract RBAC {
  using Roles for Roles.Role;
  mapping (string => Roles.Role) private roles;
  event RoleAdded(address indexed operator, string role);
  event RoleRemoved(address indexed operator, string role);
  function checkRole(address _operator, string _role)
    public
    view
  {
    roles[_role].check(_operator);
  }
  function hasRole(address _operator, string _role)
    public
    view
    returns (bool)
  {
    return roles[_role].has(_operator);
  }
  function addRole(address _operator, string _role)
    internal
  {
    roles[_role].add(_operator);
    emit RoleAdded(_operator, _role);
  }
  function removeRole(address _operator, string _role)
    internal
  {
    roles[_role].remove(_operator);
    emit RoleRemoved(_operator, _role);
  }
  modifier onlyRole(string _role)
  {
    checkRole(msg.sender, _role);
    _;
  }
}
contract Ownable {
  address public owner;
  event OwnershipRenounced(address indexed previousOwner);
  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );
  constructor() public {
    owner = msg.sender;
  }
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }
  function renounceOwnership() public onlyOwner {
    emit OwnershipRenounced(owner);
    owner = address(0);
  }
  function transferOwnership(address _newOwner) public onlyOwner {
    _transferOwnership(_newOwner);
  }
  function _transferOwnership(address _newOwner) internal {
    require(_newOwner != address(0));
    emit OwnershipTransferred(owner, _newOwner);
    owner = _newOwner;
  }
}
contract Contributions is RBAC, Ownable {
  using SafeMath for uint256;
  string public constant ROLE_MINTER = "minter";
  modifier onlyMinter () {
    checkRole(msg.sender, ROLE_MINTER);
    _;
  }
  mapping(address => uint256) public tokenBalances;
  mapping(address => uint256) public ethContributions;
  address[] public addresses;
  constructor() public {}
  function addBalance(
    address _address,
    uint256 _weiAmount,
    uint256 _tokenAmount
  )
  public
  onlyMinter
  {
    if (ethContributions[_address] == 0) {
      addresses.push(_address);
    }
    ethContributions[_address] = ethContributions[_address].add(_weiAmount);
    tokenBalances[_address] = tokenBalances[_address].add(_tokenAmount);
  }
  function addMinter(address _minter) public onlyOwner {
    addRole(_minter, ROLE_MINTER);
  }
  function addMinters(address[] _minters) public onlyOwner {
    require(_minters.length > 0);
    for (uint i = 0; i < _minters.length; i++) {
      addRole(_minters[i], ROLE_MINTER);
    }
  }
  function removeMinter(address _minter) public onlyOwner {
    removeRole(_minter, ROLE_MINTER);
  }
  function getContributorsLength() public view returns (uint) {
    return addresses.length;
  }
}