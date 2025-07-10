pragma solidity ^0.4.23;
contract ParsecReferralTracking {
  mapping (address => address) public referrer;
  event ReferrerUpdated(address indexed _referee, address indexed _referrer);
  function _updateReferrerFor(address _referee, address _referrer) internal {
    if (_referrer != address(0) && _referrer != _referee) {
      referrer[_referee] = _referrer;
      emit ReferrerUpdated(_referee, _referrer);
    }
  }
}
contract ParsecShipInfo {
  uint256 public constant TOTAL_SHIP = 900;
  uint256 public constant TOTAL_ARK = 100;
  uint256 public constant TOTAL_HAWKING = 400;
  uint256 public constant TOTAL_SATOSHI = 400;
  uint256 public constant NAME_NOT_AVAILABLE = 0;
  uint256 public constant NAME_ARK = 1;
  uint256 public constant NAME_HAWKING = 2;
  uint256 public constant NAME_SATOSHI = 3;
  uint256 public constant TYPE_NOT_AVAILABLE = 0;
  uint256 public constant TYPE_EXPLORER_FREIGHTER = 1;
  uint256 public constant TYPE_EXPLORER = 2;
  uint256 public constant TYPE_FREIGHTER = 3;
  uint256 public constant COLOR_NOT_AVAILABLE = 0;
  uint256 public constant COLOR_CUSTOM = 1;
  uint256 public constant COLOR_BLACK = 2;
  uint256 public constant COLOR_BLUE = 3;
  uint256 public constant COLOR_BROWN = 4;
  uint256 public constant COLOR_GOLD = 5;
  uint256 public constant COLOR_GREEN = 6;
  uint256 public constant COLOR_GREY = 7;
  uint256 public constant COLOR_PINK = 8;
  uint256 public constant COLOR_RED = 9;
  uint256 public constant COLOR_SILVER = 10;
  uint256 public constant COLOR_WHITE = 11;
  uint256 public constant COLOR_YELLOW = 12;
  function getShip(uint256 _shipId)
    external
    pure
    returns (
      uint256 ,
      uint256 ,
      uint256 
    )
  {
    return (
      _getShipName(_shipId),
      _getShipType(_shipId),
      _getShipColor(_shipId)
    );
  }
  function _getShipName(uint256 _shipId) internal pure returns (uint256 ) {
    if (_shipId < 1) {
      return NAME_NOT_AVAILABLE;
    } else if (_shipId <= TOTAL_ARK) {
      return NAME_ARK;
    } else if (_shipId <= TOTAL_ARK + TOTAL_HAWKING) {
      return NAME_HAWKING;
    } else if (_shipId <= TOTAL_SHIP) {
      return NAME_SATOSHI;
    } else {
      return NAME_NOT_AVAILABLE;
    }
  }
  function _getShipType(uint256 _shipId) internal pure returns (uint256 ) {
    if (_shipId < 1) {
      return TYPE_NOT_AVAILABLE;
    } else if (_shipId <= TOTAL_ARK) {
      return TYPE_EXPLORER_FREIGHTER;
    } else if (_shipId <= TOTAL_ARK + TOTAL_HAWKING) {
      return TYPE_EXPLORER;
    } else if (_shipId <= TOTAL_SHIP) {
      return TYPE_FREIGHTER;
    } else {
      return TYPE_NOT_AVAILABLE;
    }
  }
  function _getShipColor(uint256 _shipId) internal pure returns (uint256 ) {
    if (_shipId < 1) {
      return COLOR_NOT_AVAILABLE;
    } else if (_shipId == 1) {
      return COLOR_CUSTOM;
    } else if (_shipId <= 23) {
      return COLOR_BLACK;
    } else if (_shipId <= 37) {
      return COLOR_BLUE;
    } else if (_shipId <= 42) {
      return COLOR_BROWN;
    } else if (_shipId <= 45) {
      return COLOR_GOLD;
    } else if (_shipId <= 49) {
      return COLOR_GREEN;
    } else if (_shipId <= 64) {
      return COLOR_GREY;
    } else if (_shipId <= 67) {
      return COLOR_PINK;
    } else if (_shipId <= 77) {
      return COLOR_RED;
    } else if (_shipId <= 83) {
      return COLOR_SILVER;
    } else if (_shipId <= 93) {
      return COLOR_WHITE;
    } else if (_shipId <= 100) {
      return COLOR_YELLOW;
    } else if (_shipId <= 140) {
      return COLOR_BLACK;
    } else if (_shipId <= 200) {
      return COLOR_BLUE;
    } else if (_shipId <= 237) {
      return COLOR_BROWN;
    } else if (_shipId <= 247) {
      return COLOR_GOLD;
    } else if (_shipId <= 330) {
      return COLOR_GREEN;
    } else if (_shipId <= 370) {
      return COLOR_GREY;
    } else if (_shipId <= 380) {
      return COLOR_PINK;
    } else if (_shipId <= 440) {
      return COLOR_RED;
    } else if (_shipId <= 460) {
      return COLOR_SILVER;
    } else if (_shipId <= 500) {
      return COLOR_WHITE;
    } else if (_shipId <= 540) {
      return COLOR_BLACK;
    } else if (_shipId <= 600) {
      return COLOR_BLUE;
    } else if (_shipId <= 637) {
      return COLOR_BROWN;
    } else if (_shipId <= 647) {
      return COLOR_GOLD;
    } else if (_shipId <= 730) {
      return COLOR_GREEN;
    } else if (_shipId <= 770) {
      return COLOR_GREY;
    } else if (_shipId <= 780) {
      return COLOR_PINK;
    } else if (_shipId <= 840) {
      return COLOR_RED;
    } else if (_shipId <= 860) {
      return COLOR_SILVER;
    } else if (_shipId <= TOTAL_SHIP) {
      return COLOR_WHITE;
    } else {
      return COLOR_NOT_AVAILABLE;
    }
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
contract ParsecShipPricing {
  using SafeMath for uint256;
  uint256 public constant TOTAL_PARSEC_CREDIT_SUPPLY = 30856775800000000;
  uint256[18] private _multipliers = [
    30841347412100000,
    30825926738393950,
    307951085181372406484875,
    3073356447836811380826526098454678,
    306108451404054441555510982248498,
    3036687456535506201905741115048326,
    2988475130535213555319509943479229,
    2894334671812167005118183115407839,
    2714856939931502657115329246779589,
    2388599590594375264119680273916152,
    1848996810673789555394216521160879,
    1107954125875278770222144092290365,
    3978258626243293616409580784511455,
    5129032858085962996925781077178762,
    8525510970373470528186667481043039,
    2355538951219861249087266462563245,
    1798167049816644768546906209889074
  ];
  uint256[18] private _decimals = [
    0, 0, 7, 17, 16,
    17, 17, 17, 17, 17,
    17, 17, 18, 19, 21,
    24, 31
  ];
  function _getShipPrice(
    uint256 _initialPrice,
    uint256 _minutesPassed
  )
    internal
    view
    returns (uint256 )
  {
    require(
      _initialPrice <= TOTAL_PARSEC_CREDIT_SUPPLY,
      "Initial ship price must not be greater than total Parsec Credit."
    );
    if (_minutesPassed >> _multipliers.length > 0) {
      return 0;
    }
    uint256 _price = _initialPrice;
    for (uint256 _powerOfTwo = 0; _powerOfTwo < _multipliers.length; _powerOfTwo++) {
      if (_minutesPassed >> _powerOfTwo & 1 > 0) {
        _price = _price
          .mul(_multipliers[_powerOfTwo])
          .div(TOTAL_PARSEC_CREDIT_SUPPLY)
          .div(10 ** _decimals[_powerOfTwo]);
      }
    }
    return _price;
  }
}
interface TokenRecipient {
  function receiveApproval(
    address _from,
    uint256 _value,
    address _token,
    bytes _extraData
  )
    external;
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
    emit Pause();
  }
  function unpause() onlyOwner whenPaused public {
    paused = false;
    emit Unpause();
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
contract ERC721Basic {
  event Transfer(address indexed _from, address indexed _to, uint256 _tokenId);
  event Approval(address indexed _owner, address indexed _approved, uint256 _tokenId);
  event ApprovalForAll(address indexed _owner, address indexed _operator, bool _approved);
  function balanceOf(address _owner) public view returns (uint256 _balance);
  function ownerOf(uint256 _tokenId) public view returns (address _owner);
  function exists(uint256 _tokenId) public view returns (bool _exists);
  function approve(address _to, uint256 _tokenId) public;
  function getApproved(uint256 _tokenId) public view returns (address _operator);
  function setApprovalForAll(address _operator, bool _approved) public;
  function isApprovedForAll(address _owner, address _operator) public view returns (bool);
  function transferFrom(address _from, address _to, uint256 _tokenId) public;
  function safeTransferFrom(address _from, address _to, uint256 _tokenId) public;
  function safeTransferFrom(
    address _from,
    address _to,
    uint256 _tokenId,
    bytes _data
  )
    public;
}
contract ERC721Enumerable is ERC721Basic {
  function totalSupply() public view returns (uint256);
  function tokenOfOwnerByIndex(address _owner, uint256 _index) public view returns (uint256 _tokenId);
  function tokenByIndex(uint256 _index) public view returns (uint256);
}
contract ERC721Metadata is ERC721Basic {
  function name() public view returns (string _name);
  function symbol() public view returns (string _symbol);
  function tokenURI(uint256 _tokenId) public view returns (string);
}
contract ERC721 is ERC721Basic, ERC721Enumerable, ERC721Metadata {
}
library AddressUtils {
  function isContract(address addr) internal view returns (bool) {
    uint256 size;
    assembly { size := extcodesize(addr) }  
    return size > 0;
  }
}
contract ERC721Receiver {
  bytes4 constant ERC721_RECEIVED = 0xf0b9e5ba;
  function onERC721Received(address _from, uint256 _tokenId, bytes _data) public returns(bytes4);
}
contract ERC721BasicToken is ERC721Basic {
  using SafeMath for uint256;
  using AddressUtils for address;
  bytes4 constant ERC721_RECEIVED = 0xf0b9e5ba;
  mapping (uint256 => address) internal tokenOwner;
  mapping (uint256 => address) internal tokenApprovals;
  mapping (address => uint256) internal ownedTokensCount;
  mapping (address => mapping (address => bool)) internal operatorApprovals;
  modifier onlyOwnerOf(uint256 _tokenId) {
    require(ownerOf(_tokenId) == msg.sender);
    _;
  }
  modifier canTransfer(uint256 _tokenId) {
    require(isApprovedOrOwner(msg.sender, _tokenId));
    _;
  }
  function balanceOf(address _owner) public view returns (uint256) {
    require(_owner != address(0));
    return ownedTokensCount[_owner];
  }
  function ownerOf(uint256 _tokenId) public view returns (address) {
    address owner = tokenOwner[_tokenId];
    require(owner != address(0));
    return owner;
  }
  function exists(uint256 _tokenId) public view returns (bool) {
    address owner = tokenOwner[_tokenId];
    return owner != address(0);
  }
  function approve(address _to, uint256 _tokenId) public {
    address owner = ownerOf(_tokenId);
    require(_to != owner);
    require(msg.sender == owner || isApprovedForAll(owner, msg.sender));
    if (getApproved(_tokenId) != address(0) || _to != address(0)) {
      tokenApprovals[_tokenId] = _to;
      emit Approval(owner, _to, _tokenId);
    }
  }
  function getApproved(uint256 _tokenId) public view returns (address) {
    return tokenApprovals[_tokenId];
  }
  function setApprovalForAll(address _to, bool _approved) public {
    require(_to != msg.sender);
    operatorApprovals[msg.sender][_to] = _approved;
    emit ApprovalForAll(msg.sender, _to, _approved);
  }
  function isApprovedForAll(address _owner, address _operator) public view returns (bool) {
    return operatorApprovals[_owner][_operator];
  }
  function transferFrom(address _from, address _to, uint256 _tokenId) public canTransfer(_tokenId) {
    require(_from != address(0));
    require(_to != address(0));
    clearApproval(_from, _tokenId);
    removeTokenFrom(_from, _tokenId);
    addTokenTo(_to, _tokenId);
    emit Transfer(_from, _to, _tokenId);
  }
  function safeTransferFrom(
    address _from,
    address _to,
    uint256 _tokenId
  )
    public
    canTransfer(_tokenId)
  {
    safeTransferFrom(_from, _to, _tokenId, "");
  }
  function safeTransferFrom(
    address _from,
    address _to,
    uint256 _tokenId,
    bytes _data
  )
    public
    canTransfer(_tokenId)
  {
    transferFrom(_from, _to, _tokenId);
    require(checkAndCallSafeTransfer(_from, _to, _tokenId, _data));
  }
  function isApprovedOrOwner(address _spender, uint256 _tokenId) internal view returns (bool) {
    address owner = ownerOf(_tokenId);
    return _spender == owner || getApproved(_tokenId) == _spender || isApprovedForAll(owner, _spender);
  }
  function _mint(address _to, uint256 _tokenId) internal {
    require(_to != address(0));
    addTokenTo(_to, _tokenId);
    emit Transfer(address(0), _to, _tokenId);
  }
  function _burn(address _owner, uint256 _tokenId) internal {
    clearApproval(_owner, _tokenId);
    removeTokenFrom(_owner, _tokenId);
    emit Transfer(_owner, address(0), _tokenId);
  }
  function clearApproval(address _owner, uint256 _tokenId) internal {
    require(ownerOf(_tokenId) == _owner);
    if (tokenApprovals[_tokenId] != address(0)) {
      tokenApprovals[_tokenId] = address(0);
      emit Approval(_owner, address(0), _tokenId);
    }
  }
  function addTokenTo(address _to, uint256 _tokenId) internal {
    require(tokenOwner[_tokenId] == address(0));
    tokenOwner[_tokenId] = _to;
    ownedTokensCount[_to] = ownedTokensCount[_to].add(1);
  }
  function removeTokenFrom(address _from, uint256 _tokenId) internal {
    require(ownerOf(_tokenId) == _from);
    ownedTokensCount[_from] = ownedTokensCount[_from].sub(1);
    tokenOwner[_tokenId] = address(0);
  }
  function checkAndCallSafeTransfer(
    address _from,
    address _to,
    uint256 _tokenId,
    bytes _data
  )
    internal
    returns (bool)
  {
    if (!_to.isContract()) {
      return true;
    }
    bytes4 retval = ERC721Receiver(_to).onERC721Received(_from, _tokenId, _data);
    return (retval == ERC721_RECEIVED);
  }
}
contract ERC721Token is ERC721, ERC721BasicToken {
  string internal name_;
  string internal symbol_;
  mapping (address => uint256[]) internal ownedTokens;
  mapping(uint256 => uint256) internal ownedTokensIndex;
  uint256[] internal allTokens;
  mapping(uint256 => uint256) internal allTokensIndex;
  mapping(uint256 => string) internal tokenURIs;
  function ERC721Token(string _name, string _symbol) public {
    name_ = _name;
    symbol_ = _symbol;
  }
  function name() public view returns (string) {
    return name_;
  }
  function symbol() public view returns (string) {
    return symbol_;
  }
  function tokenURI(uint256 _tokenId) public view returns (string) {
    require(exists(_tokenId));
    return tokenURIs[_tokenId];
  }
  function tokenOfOwnerByIndex(address _owner, uint256 _index) public view returns (uint256) {
    require(_index < balanceOf(_owner));
    return ownedTokens[_owner][_index];
  }
  function totalSupply() public view returns (uint256) {
    return allTokens.length;
  }
  function tokenByIndex(uint256 _index) public view returns (uint256) {
    require(_index < totalSupply());
    return allTokens[_index];
  }
  function _setTokenURI(uint256 _tokenId, string _uri) internal {
    require(exists(_tokenId));
    tokenURIs[_tokenId] = _uri;
  }
  function addTokenTo(address _to, uint256 _tokenId) internal {
    super.addTokenTo(_to, _tokenId);
    uint256 length = ownedTokens[_to].length;
    ownedTokens[_to].push(_tokenId);
    ownedTokensIndex[_tokenId] = length;
  }
  function removeTokenFrom(address _from, uint256 _tokenId) internal {
    super.removeTokenFrom(_from, _tokenId);
    uint256 tokenIndex = ownedTokensIndex[_tokenId];
    uint256 lastTokenIndex = ownedTokens[_from].length.sub(1);
    uint256 lastToken = ownedTokens[_from][lastTokenIndex];
    ownedTokens[_from][tokenIndex] = lastToken;
    ownedTokens[_from][lastTokenIndex] = 0;
    ownedTokens[_from].length--;
    ownedTokensIndex[_tokenId] = 0;
    ownedTokensIndex[lastToken] = tokenIndex;
  }
  /**
   * @dev Internal function to mint a new token
   * @dev Reverts if the given token ID already exists
   * @param _to address the beneficiary that will own the minted token
   * @param _token