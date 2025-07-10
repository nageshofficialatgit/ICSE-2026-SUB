pragma solidity ^0.4.18;
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    assert(c / a == b);
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
    OwnershipTransferred(owner, newOwner);
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
    OwnershipTransferred(owner, pendingOwner);
    owner = pendingOwner;
    pendingOwner = address(0);
  }
}
contract Pausable is Ownable {
  event Pause();
  event Unpause();
  bool public paused = false;
  modifier whenNotPaused() {
    require(!paused);
    _;
  }
  modifier whenPaused() {
    require(paused);
    _;
  }
  function pause() onlyOwner whenNotPaused public {
    paused = true;
    Pause();
  }
  function unpause() onlyOwner whenPaused public {
    paused = false;
    Unpause();
  }
}
contract ERC721 {
  event Transfer(address indexed _from, address indexed _to, uint256 _tokenId);
  event Approval(address indexed _owner, address indexed _approved, uint256 _tokenId);
  function balanceOf(address _owner) public view returns (uint256 _balance);
  function ownerOf(uint256 _tokenId) public view returns (address _owner);
  function transfer(address _to, uint256 _tokenId) public;
  function approve(address _to, uint256 _tokenId) public;
  function takeOwnership(uint256 _tokenId) public;
}
contract ERC721Token is ERC721 {
  using SafeMath for uint256;
  uint256 private totalTokens;
  mapping (uint256 => address) private tokenOwner;
  mapping (uint256 => address) private tokenApprovals;
  mapping (address => uint256[]) private ownedTokens;
  mapping(uint256 => uint256) private ownedTokensIndex;
  modifier onlyOwnerOf(uint256 _tokenId) {
    require(ownerOf(_tokenId) == msg.sender);
    _;
  }
  function totalSupply() public view returns (uint256) {
    return totalTokens;
  }
  function balanceOf(address _owner) public view returns (uint256) {
    return ownedTokens[_owner].length;
  }
  function tokensOf(address _owner) public view returns (uint256[]) {
    return ownedTokens[_owner];
  }
  function ownerOf(uint256 _tokenId) public view returns (address) {
    address owner = tokenOwner[_tokenId];
    require(owner != address(0));
    return owner;
  }
  function approvedFor(uint256 _tokenId) public view returns (address) {
    return tokenApprovals[_tokenId];
  }
  function transfer(address _to, uint256 _tokenId) public onlyOwnerOf(_tokenId) {
    clearApprovalAndTransfer(msg.sender, _to, _tokenId);
  }
  function approve(address _to, uint256 _tokenId) public onlyOwnerOf(_tokenId) {
    address owner = ownerOf(_tokenId);
    require(_to != owner);
    if (approvedFor(_tokenId) != 0 || _to != 0) {
      tokenApprovals[_tokenId] = _to;
      Approval(owner, _to, _tokenId);
    }
  }
  function takeOwnership(uint256 _tokenId) public {
    require(isApprovedFor(msg.sender, _tokenId));
    clearApprovalAndTransfer(ownerOf(_tokenId), msg.sender, _tokenId);
  }
  function _mint(address _to, uint256 _tokenId) internal {
    require(_to != address(0));
    addToken(_to, _tokenId);
    Transfer(0x0, _to, _tokenId);
  }
  function _burn(uint256 _tokenId) onlyOwnerOf(_tokenId) internal {
    if (approvedFor(_tokenId) != 0) {
      clearApproval(msg.sender, _tokenId);
    }
    removeToken(msg.sender, _tokenId);
    Transfer(msg.sender, 0x0, _tokenId);
  }
  function isApprovedFor(address _owner, uint256 _tokenId) internal view returns (bool) {
    return approvedFor(_tokenId) == _owner;
  }
  function clearApprovalAndTransfer(address _from, address _to, uint256 _tokenId) internal {
    require(_to != address(0));
    require(_to != ownerOf(_tokenId));
    require(ownerOf(_tokenId) == _from);
    clearApproval(_from, _tokenId);
    removeToken(_from, _tokenId);
    addToken(_to, _tokenId);
    Transfer(_from, _to, _tokenId);
  }
  function clearApproval(address _owner, uint256 _tokenId) private {
    require(ownerOf(_tokenId) == _owner);
    tokenApprovals[_tokenId] = 0;
    Approval(_owner, 0, _tokenId);
  }
  function addToken(address _to, uint256 _tokenId) private {
    require(tokenOwner[_tokenId] == address(0));
    tokenOwner[_tokenId] = _to;
    uint256 length = balanceOf(_to);
    ownedTokens[_to].push(_tokenId);
    ownedTokensIndex[_tokenId] = length;
    totalTokens = totalTokens.add(1);
  }
  function removeToken(address _from, uint256 _tokenId) private {
    require(ownerOf(_tokenId) == _from);
    uint256 tokenIndex = ownedTokensIndex[_tokenId];
    uint256 lastTokenIndex = balanceOf(_from).sub(1);
    uint256 lastToken = ownedTokens[_from][lastTokenIndex];
    tokenOwner[_tokenId] = 0;
    ownedTokens[_from][tokenIndex] = lastToken;
    ownedTokens[_from][lastTokenIndex] = 0;
    ownedTokens[_from].length--;
    ownedTokensIndex[_tokenId] = 0;
    ownedTokensIndex[lastToken] = tokenIndex;
    totalTokens = totalTokens.sub(1);
  }
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
contract BasicToken is ERC20Basic {
  using SafeMath for uint256;
  mapping(address => uint256) balances;
  uint256 totalSupply_;
  function totalSupply() public view returns (uint256) {
    return totalSupply_;
  }
  function transfer(address _to, uint256 _value) public returns (bool) {
    require(_to != address(0));
    require(_value <= balances[msg.sender]);
    balances[msg.sender] = balances[msg.sender].sub(_value);
    balances[_to] = balances[_to].add(_value);
    Transfer(msg.sender, _to, _value);
    return true;
  }
  function balanceOf(address _owner) public view returns (uint256 balance) {
    return balances[_owner];
  }
}
contract StandardToken is ERC20, BasicToken {
  mapping (address => mapping (address => uint256)) internal allowed;
  function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
    require(_to != address(0));
    require(_value <= balances[_from]);
    require(_value <= allowed[_from][msg.sender]);
    balances[_from] = balances[_from].sub(_value);
    balances[_to] = balances[_to].add(_value);
    allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
    Transfer(_from, _to, _value);
    return true;
  }
  function approve(address _spender, uint256 _value) public returns (bool) {
    allowed[msg.sender][_spender] = _value;
    Approval(msg.sender, _spender, _value);
    return true;
  }
  function allowance(address _owner, address _spender) public view returns (uint256) {
    return allowed[_owner][_spender];
  }
  function increaseApproval(address _spender, uint _addedValue) public returns (bool) {
    allowed[msg.sender][_spender] = allowed[msg.sender][_spender].add(_addedValue);
    Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }
  function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
    uint oldValue = allowed[msg.sender][_spender];
    if (_subtractedValue > oldValue) {
      allowed[msg.sender][_spender] = 0;
    } else {
      allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
    }
    Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }
}
contract AccessDeposit is Claimable {
  mapping(address => bool) private depositAccess;
  modifier onlyAccessDeposit {
    require(msg.sender == owner || depositAccess[msg.sender] == true);
    _;
  }
  function grantAccessDeposit(address _address)
    onlyOwner
    public
  {
    depositAccess[_address] = true;
  }
  function revokeAccessDeposit(address _address)
    onlyOwner
    public
  {
    depositAccess[_address] = false;
  }
}
contract AccessDeploy is Claimable {
  mapping(address => bool) private deployAccess;
  modifier onlyAccessDeploy {
    require(msg.sender == owner || deployAccess[msg.sender] == true);
    _;
  }
  function grantAccessDeploy(address _address)
    onlyOwner
    public
  {
    deployAccess[_address] = true;
  }
  function revokeAccessDeploy(address _address)
    onlyOwner
    public
  {
    deployAccess[_address] = false;
  }
}
contract AccessMint is Claimable {
  mapping(address => bool) private mintAccess;
  modifier onlyAccessMint {
    require(msg.sender == owner || mintAccess[msg.sender] == true);
    _;
  }
  function grantAccessMint(address _address)
    onlyOwner
    public
  {
    mintAccess[_address] = true;
  }
  function revokeAccessMint(address _address)
    onlyOwner
    public
  {
    mintAccess[_address] = false;
  }
}
contract Gold is StandardToken, Claimable, AccessMint {
  string public constant name = "Gold";
  string public constant symbol = "G";
  uint8 public constant decimals = 18;
  event Mint(
    address indexed _to,
    uint256 indexed _tokenId
  );
  function mint(address _to, uint256 _amount) 
    onlyAccessMint
    public 
    returns (bool) 
  {
    totalSupply_ = totalSupply_.add(_amount);
    balances[_to] = balances[_to].add(_amount);
    Mint(_to, _amount);
    Transfer(address(0), _to, _amount);
    return true;
  }
}
contract CryptoSagaCard is ERC721Token, Claimable, AccessMint {
  string public constant name = "CryptoSaga Card";
  string public constant symbol = "CARD";
  mapping(uint256 => uint8) public tokenIdToRank;
  uint256 public numberOfTokenId;
  CryptoSagaCardSwap private swapContract;
  event CardSwap(address indexed _by, uint256 _tokenId, uint256 _rewardId);
  function setCryptoSagaCardSwapContract(address _contractAddress)
    public
    onlyOwner
  {
    swapContract = CryptoSagaCardSwap(_contractAddress);
  }
  function rankOf(uint256 _tokenId) 
    public view
    returns (uint8)
  {
    return tokenIdToRank[_tokenId];
  }
  function mint(address _beneficiary, uint256 _amount, uint8 _rank)
    onlyAccessMint
    public
  {
    for (uint256 i = 0; i < _amount; i++) {
      _mint(_beneficiary, numberOfTokenId);
      tokenIdToRank[numberOfTokenId] = _rank;
      numberOfTokenId ++;
    }
  }
  function swap(uint256 _tokenId)
    onlyOwnerOf(_tokenId)
    public
    returns (uint256)
  {
    require(address(swapContract) != address(0));
    var _rank = tokenIdToRank[_tokenId];
    var _rewardId = swapContract.swapCardForReward(this, _rank);
    CardSwap(ownerOf(_tokenId), _tokenId, _rewardId);
    _burn(_tokenId);
    return _rewardId;
  }
}
contract CryptoSagaCardSwap is Ownable {
  address internal cardAddess;
  modifier onlyCard {
    require(msg.sender == cardAddess);
    _;
  }
  function setCardContract(address _contractAddress)
    public
    onlyOwner
  {
    cardAddess = _contractAddress;
  }
  function swapCardForReward(address _by, uint8 _rank)
    onlyCard
    public 
    returns (uint256);
}
contract CryptoSagaHero is ERC721Token, Claimable, Pausable, AccessMint, AccessDeploy, AccessDeposit {
  string public constant name = "CryptoSaga Hero";
  string public constant symbol = "HERO";
  struct HeroClass {
    string className;
    uint8 classRank;
    uint8 classRace;
    uint32 classAge;
    uint8 classType;
    uint32 maxLevel; 
    uint8 aura; 
    uint32[5] baseStats;
    uint32[5] minIVForStats;
    uint32[5] maxIVForStats;
    uint32 currentNumberOfInstancedHeroes;
  }
  struct HeroInstance {
    uint32 heroClassId;
    string heroName;
    uint32 currentLevel;
    uint32 currentExp;
    uint32 lastLocationId;
    uint256 availableAt;
    uint32[5] currentStats;
    uint32[5] ivForStats;
  }
  uint32 public requiredExpIncreaseFactor = 100;
  uint256 public requiredGoldIncreaseFactor = 1000000000000000000;
  mapping(uint32 => HeroClass) public heroClasses;
  uint32 public numberOfHeroClasses;
  mapping(uint256 => HeroInstance) public tokenIdToHeroInstance;
  uint256 public numberOfTokenIds;
  Gold public goldContract;
  mapping(address => uint256) public addressToGoldDeposit;
  uint32 private seed = 0;
  event DefineType(
    address indexed _by,
    uint32 indexed _typeId,
    string _className
  );
  event LevelUp(
    address indexed _by,
    uint256 indexed _tokenId,
    uint32 _newLevel
  );
  event Deploy(
    address indexed _by,
    uint256 indexed _tokenId,
    uint32 _locationId,
    uint256 _duration
  );
  function getClassInfo(uint32 _classId)
    external view
    returns (string className, uint8 classRank, uint8 classRace, uint32 classAge, uint8 classType, uint32 maxLevel, uint8 aura, uint32[5] baseStats, uint32[5] minIVs, uint32[5] maxIVs) 
  {
    var _cl = heroClasses[_classId];
    return (_cl.className, _cl.classRank, _cl.classRace, _cl.classAge, _cl.classType, _cl.maxLevel, _cl.aura, _cl.baseStats, _cl.minIVForStats, _cl.maxIVForStats);
  }
  function getClassName(uint32 _classId)
    external view
    returns (string)
  {
    return heroClasses[_classId].className;
  }
  function getClassRank(uint32 _classId)
    external view
    returns (uint8)
  {
    return heroClasses[_classId].classRank;
  }
  function getClassMintCount(uint32 _classId)
    external view
    returns (uint32)
  {
    return heroClasses[_classId].currentNumberOfInstancedHeroes;
  }
  function getHeroInfo(uint256 _tokenId)
    external view
    returns (uint32 classId, string heroName, uint32 currentLevel, uint32 currentExp, uint32 lastLocationId, uint256 availableAt, uint32[5] currentStats, uint32[5] ivs, uint32 bp)
  {
    HeroInstance memory _h = tokenIdToHeroInstance[_tokenId];
    var _bp = _h.currentStats[0] + _h.currentStats[1] + _h.currentStats[2] + _h.currentStats[3] + _h.currentStats[4];
    return (_h.heroClassId, _h.heroName, _h.currentLevel, _h.currentExp, _h.lastLocationId, _h.availableAt, _h.currentStats, _h.ivForStats, _bp);
  }
  function getHeroClassId(uint256 _tokenId)
    external view
    returns (uint32)
  {
    return tokenIdToHeroInstance[_tokenId].heroClassId;
  }
  function getHeroName(uint256 _tokenId)
    external view
    returns (string)
  {
    return tokenIdToHeroInstance[_tokenId].heroName;
  }
  function getHeroLevel(uint256 _tokenId)
    external view
    returns (uint32)
  {
    return tokenIdToHeroInstance[_tokenId].currentLevel;
  }
  function getHeroLocation(uint256 _tokenId)
    external view
    returns (uint32)
  {
    return tokenIdToHeroInstance[_tokenId].lastLocationId;
  }
  function getHeroAvailableAt(uint256 _tokenId)
    external view
    returns (uint256)
  {
    return tokenIdToHeroInstance[_tokenId].availableAt;
  }
  function getHeroBP(uint256 _tokenId)
    public view
    returns (uint32)
  {
    var _tmp = tokenIdToHeroInstance[_tokenId].currentStats;
    return (_tmp[0] + _tmp[1] + _tmp[2] + _tmp[3] + _tmp[4]);
  }
  function getHeroRequiredGoldForLevelUp(uint256 _tokenId)
    public view
    returns (uint256)
  {
    return (uint256(2) ** (tokenIdToHeroInstance[_tokenId].currentLevel / 10)) * requiredGoldIncreaseFactor;
  }
  function getHeroRequiredExpForLevelUp(uint256 _tokenId)
    public view
    returns (uint32)
  {
    return ((tokenIdToHeroInstance[_tokenId].currentLevel + 2) * requiredExpIncreaseFactor);
  }
  function getGoldDepositOfAddress(address _address)
    external view
    returns (uint256)
  {
    return addressToGoldDeposit[_address];
  }
  function getTokenIdOfAddressAndIndex(address _address, uint256 _index)
    external view
    returns (uint256)
  {
    return tokensOf(_address)[_index];
  }
  function getTotalBPOfAddress(address _address)
    external view
    returns (uint32)
  {
    var _tokens = tokensOf(_address);
    uint32 _totalBP = 0;
    for (uint256 i = 0; i < _tokens.length; i ++) {
      _totalBP += getHeroBP(_tokens[i]);
    }
    return _totalBP;
  }
  function setHeroName(uint256 _tokenId, string _name)
    onlyOwnerOf(_tokenId)
    public
  {
    tokenIdToHeroInstance[_tokenId].heroName = _name;
  }
  function setGoldContract(address _contractAddress)
    onlyOwner
    public
  {
    goldContract = Gold(_contractAddress);
  }
  function setRequiredExpIncreaseFactor(uint32 _value)
    onlyOwner
    public
  {
    requiredExpIncreaseFactor = _value;
  }
  function setRequiredGoldIncreaseFactor(uint256 _value)
    onlyOwner
    public
  {
    requiredGoldIncreaseFactor = _value;
  }
  function CryptoSagaHero(address _goldAddress)
    public
  {
    require(_goldAddress != address(0));
    setGoldContract(_goldAddress);
    defineType("Archangel", 4, 1, 13540, 0, 99, 3, [uint32(74), 75, 57, 99, 95], [uint32(8), 6, 8, 5, 5], [uint32(8), 10, 10, 6, 6]);
    defineType("Shadowalker", 3, 4, 134, 1, 75, 4, [uint32(45), 35, 60, 80, 40], [uint32(3), 2, 10, 4, 5], [uint32(5), 5, 10, 7, 5]);
    defineType("Pyromancer", 2, 0, 14, 2, 50, 1, [uint32(50), 28, 17, 40, 35], [uint32(5), 3, 2, 3, 3], [uint32(8), 4, 3, 4, 5]);
    defineType("Magician", 1, 3, 224, 2, 30, 0, [uint32(35), 15, 25, 25, 30], [uint32(3), 1, 2, 2, 2], [uint32(5), 2, 3, 3, 3]);
    defineType("Farmer", 0, 0, 59, 0, 15, 2