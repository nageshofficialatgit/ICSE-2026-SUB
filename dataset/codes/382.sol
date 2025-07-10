pragma solidity ^0.4.24;
interface IERC165 {
  function supportsInterface(bytes4 interfaceId)
    external
    view
    returns (bool);
}
contract ERC165 is IERC165 {
  bytes4 private constant _InterfaceId_ERC165 = 0x01ffc9a7;
  mapping(bytes4 => bool) private _supportedInterfaces;
  constructor()
    internal
  {
    _registerInterface(_InterfaceId_ERC165);
  }
  function supportsInterface(bytes4 interfaceId)
    external
    view
    returns (bool)
  {
    return _supportedInterfaces[interfaceId];
  }
  function _registerInterface(bytes4 interfaceId)
    internal
  {
    require(interfaceId != 0xffffffff);
    _supportedInterfaces[interfaceId] = true;
  }
}
contract IERC721 is IERC165 {
  event Transfer(
    address indexed from,
    address indexed to,
    uint256 indexed tokenId
  );
  event Approval(
    address indexed owner,
    address indexed approved,
    uint256 indexed tokenId
  );
  event ApprovalForAll(
    address indexed owner,
    address indexed operator,
    bool approved
  );
  function balanceOf(address owner) public view returns (uint256 balance);
  function ownerOf(uint256 tokenId) public view returns (address owner);
  function approve(address to, uint256 tokenId) public;
  function getApproved(uint256 tokenId)
    public view returns (address operator);
  function setApprovalForAll(address operator, bool _approved) public;
  function isApprovedForAll(address owner, address operator)
    public view returns (bool);
  function transferFrom(address from, address to, uint256 tokenId) public;
  function safeTransferFrom(address from, address to, uint256 tokenId)
    public;
  function safeTransferFrom(
    address from,
    address to,
    uint256 tokenId,
    bytes data
  )
    public;
}
contract IERC721Receiver {
  function onERC721Received(
    address operator,
    address from,
    uint256 tokenId,
    bytes data
  )
    public
    returns(bytes4);
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
library Address {
  function isContract(address account) internal view returns (bool) {
    uint256 size;
    assembly { size := extcodesize(account) }
    return size > 0;
  }
}
contract ERC721_custom is ERC165, IERC721 {
  using SafeMath for uint256;
  using Address for address;
  bytes4 private constant _ERC721_RECEIVED = 0x150b7a02;
  mapping (uint256 => address) private _tokenOwner;
  mapping (uint256 => address) private _tokenApprovals;
  mapping (address => uint256) private _ownedTokensCount;
  mapping (address => mapping (address => bool)) private _operatorApprovals;
  bytes4 private constant _InterfaceId_ERC721 = 0x80ac58cd;
  constructor()
    public
  {
    _registerInterface(_InterfaceId_ERC721);
  }
  function balanceOf(address owner) public view returns (uint256) {
    require(owner != address(0));
    return _ownedTokensCount[owner];
  }
  function ownerOf(uint256 tokenId) public view returns (address) {
    address owner = _tokenOwner[tokenId];
    require(owner != address(0));
    return owner;
  }
  function approve(address to, uint256 tokenId) public {
    address owner = ownerOf(tokenId);
    require(to != owner);
    require(msg.sender == owner || isApprovedForAll(owner, msg.sender));
    _tokenApprovals[tokenId] = to;
    emit Approval(owner, to, tokenId);
  }
  function getApproved(uint256 tokenId) public view returns (address) {
    require(_exists(tokenId));
    return _tokenApprovals[tokenId];
  }
  function setApprovalForAll(address to, bool approved) public {
    require(to != msg.sender);
    _operatorApprovals[msg.sender][to] = approved;
    emit ApprovalForAll(msg.sender, to, approved);
  }
  function isApprovedForAll(
    address owner,
    address operator
  )
    public
    view
    returns (bool)
  {
    return _operatorApprovals[owner][operator];
  }
  function transferFrom(
    address from,
    address to,
    uint256 tokenId
  )
    public
  {
    require(_isApprovedOrOwner(msg.sender, tokenId));
    require(to != address(0));
    _clearApproval(from, tokenId);
    _removeTokenFrom(from, tokenId);
    _addTokenTo(to, tokenId);
    emit Transfer(from, to, tokenId);
  }
    function internal_transferFrom(
        address _from,
        address to,
        uint256 tokenId
    )
    internal
  {
    require(to != address(0));
    if (_tokenApprovals[tokenId] != address(0)) {
      _tokenApprovals[tokenId] = address(0);
    }
    if(_ownedTokensCount[_from] > 1) {
    _ownedTokensCount[_from] = _ownedTokensCount[_from] -1; 
    }
    _tokenOwner[tokenId] = address(0); 
    _addTokenTo(to, tokenId); 
    emit Transfer(_from, to, tokenId);
  }
  function safeTransferFrom(
    address from,
    address to,
    uint256 tokenId
  )
    public
  {
    safeTransferFrom(from, to, tokenId, "");
  }
  function safeTransferFrom(
    address from,
    address to,
    uint256 tokenId,
    bytes _data
  )
    public
  {
    transferFrom(from, to, tokenId);
    require(_checkOnERC721Received(from, to, tokenId, _data));
  }
  function _exists(uint256 tokenId) internal view returns (bool) {
    address owner = _tokenOwner[tokenId];
    return owner != address(0);
  }
  function _isApprovedOrOwner(
    address spender,
    uint256 tokenId
  )
    internal
    view
    returns (bool)
  {
    address owner = ownerOf(tokenId);
    return (
      spender == owner ||
      getApproved(tokenId) == spender ||
      isApprovedForAll(owner, spender)
    );
  }
  function _mint(address to, uint256 tokenId) internal {
    require(to != address(0));
    _addTokenTo(to, tokenId);
    emit Transfer(address(0), to, tokenId);
  }
  function _burn(address owner, uint256 tokenId) internal {
    _clearApproval(owner, tokenId);
    _removeTokenFrom(owner, tokenId);
    emit Transfer(owner, address(0), tokenId);
  }
  function _addTokenTo(address to, uint256 tokenId) internal {
    require(_tokenOwner[tokenId] == address(0));
    _tokenOwner[tokenId] = to;
    _ownedTokensCount[to] = _ownedTokensCount[to].add(1);
  }
  function _removeTokenFrom(address from, uint256 tokenId) internal {
    require(ownerOf(tokenId) == from);
    _ownedTokensCount[from] = _ownedTokensCount[from].sub(1);
    _tokenOwner[tokenId] = address(0);
  }
  function _checkOnERC721Received(
    address from,
    address to,
    uint256 tokenId,
    bytes _data
  )
    internal
    returns (bool)
  {
    if (!to.isContract()) {
      return true;
    }
    bytes4 retval = IERC721Receiver(to).onERC721Received(
      msg.sender, from, tokenId, _data);
    return (retval == _ERC721_RECEIVED);
  }
  function _clearApproval(address owner, uint256 tokenId) private {
    require(ownerOf(tokenId) == owner);
    if (_tokenApprovals[tokenId] != address(0)) {
      _tokenApprovals[tokenId] = address(0);
    }
  }
}
contract IERC721Enumerable is IERC721 {
  function totalSupply() public view returns (uint256);
  function tokenOfOwnerByIndex(
    address owner,
    uint256 index
  )
    public
    view
    returns (uint256 tokenId);
  function tokenByIndex(uint256 index) public view returns (uint256);
}
contract ERC721Enumerable_custom is ERC165, ERC721_custom, IERC721Enumerable {
  mapping(address => uint256[]) private _ownedTokens;
  mapping(uint256 => uint256) private _ownedTokensIndex;
  uint256[] private _allTokens;
  mapping(uint256 => uint256) private _allTokensIndex;
  bytes4 private constant _InterfaceId_ERC721Enumerable = 0x780e9d63;
  constructor() public {
    _registerInterface(_InterfaceId_ERC721Enumerable);
  }
  function tokenOfOwnerByIndex(
    address owner,
    uint256 index
  )
    public
    view
    returns (uint256)
  {
    require(index < balanceOf(owner));
    return _ownedTokens[owner][index];
  }
  function totalSupply() public view returns (uint256) {
    return _allTokens.length;
  }
  function tokenByIndex(uint256 index) public view returns (uint256) {
    require(index < totalSupply());
    return _allTokens[index];
  }
  function _addTokenTo(address to, uint256 tokenId) internal {
    super._addTokenTo(to, tokenId);
    uint256 length = _ownedTokens[to].length;
    _ownedTokens[to].push(tokenId);
    _ownedTokensIndex[tokenId] = length;
  }
  function _removeTokenFrom(address from, uint256 tokenId) internal {
    super._removeTokenFrom(from, tokenId);
    uint256 tokenIndex = _ownedTokensIndex[tokenId];
    uint256 lastTokenIndex = _ownedTokens[from].length.sub(1);
    uint256 lastToken = _ownedTokens[from][lastTokenIndex];
    _ownedTokens[from][tokenIndex] = lastToken;
    _ownedTokens[from].length--;
    _ownedTokensIndex[tokenId] = 0;
    _ownedTokensIndex[lastToken] = tokenIndex;
  }
  function _mint(address to, uint256 tokenId) internal {
    super._mint(to, tokenId);
    _allTokensIndex[tokenId] = _allTokens.length;
    _allTokens.push(tokenId);
  }
  function _burn(address owner, uint256 tokenId) internal {
    super._burn(owner, tokenId);
    uint256 tokenIndex = _allTokensIndex[tokenId];
    uint256 lastTokenIndex = _allTokens.length.sub(1);
    uint256 lastToken = _allTokens[lastTokenIndex];
    _allTokens[tokenIndex] = lastToken;
    _allTokens[lastTokenIndex] = 0;
    _allTokens.length--;
    _allTokensIndex[tokenId] = 0;
    _allTokensIndex[lastToken] = tokenIndex;
  }
}
contract IERC721Metadata is IERC721 {
  function name() external view returns (string);
  function symbol() external view returns (string);
  function tokenURI(uint256 tokenId) external view returns (string);
}
contract ERC721Metadata_custom is ERC165, ERC721_custom, IERC721Metadata {
  string private _name;
  string private _symbol;
  mapping(uint256 => string) private _tokenURIs;
  bytes4 private constant InterfaceId_ERC721Metadata = 0x5b5e139f;
  constructor(string name, string symbol) public {
    _name = name;
    _symbol = symbol;
    _registerInterface(InterfaceId_ERC721Metadata);
  }
  function name() external view returns (string) {
    return _name;
  }
  function symbol() external view returns (string) {
    return _symbol;
  }
  function tokenURI(uint256 tokenId) external view returns (string) {
    require(_exists(tokenId));
    return _tokenURIs[tokenId];
  }
  function _setTokenURI(uint256 tokenId, string uri) internal {
    require(_exists(tokenId));
    _tokenURIs[tokenId] = uri;
  }
  function _burn(address owner, uint256 tokenId) internal {
    super._burn(owner, tokenId);
    if (bytes(_tokenURIs[tokenId]).length != 0) {
      delete _tokenURIs[tokenId];
    }
  }
}
contract ERC721Full_custom is ERC721_custom, ERC721Enumerable_custom, ERC721Metadata_custom {
  constructor(string name, string symbol) ERC721Metadata_custom(name, symbol)
    public
  {
  }
}
interface PlanetCryptoCoin_I {
    function balanceOf(address owner) external returns(uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns(bool);
}
interface PlanetCryptoUtils_I {
    function validateLand(address _sender, int256[] plots_lat, int256[] plots_lng) external returns(bool);
    function validatePurchase(address _sender, uint256 _value, int256[] plots_lat, int256[] plots_lng) external returns(bool);
    function validateTokenPurchase(address _sender, int256[] plots_lat, int256[] plots_lng) external returns(bool);
    function validateResale(address _sender, uint256 _value, uint256 _token_id) external returns(bool);
    function strConcat(string _a, string _b, string _c, string _d, string _e, string _f) external view returns (string);
    function strConcat(string _a, string _b, string _c, string _d, string _e) external view returns (string);
    function strConcat(string _a, string _b, string _c, string _d) external view returns (string);
    function strConcat(string _a, string _b, string _c) external view returns (string);
    function strConcat(string _a, string _b) external view returns (string);
    function int2str(int i) external view returns (string);
    function uint2str(uint i) external view returns (string);
    function substring(string str, uint startIndex, uint endIndex) external view returns (string);
    function utfStringLength(string str) external view returns (uint length);
    function ceil1(int256 a, int256 m) external view returns (int256 );
    function parseInt(string _a, uint _b) external view returns (uint);
}
library Percent {
  struct percent {
    uint num;
    uint den;
  }
  function mul(percent storage p, uint a) internal view returns (uint) {
    if (a == 0) {
      return 0;
    }
    return a*p.num/p.den;
  }
  function div(percent storage p, uint a) internal view returns (uint) {
    return a/p.num*p.den;
  }
  function sub(percent storage p, uint a) internal view returns (uint) {
    uint b = mul(p, a);
    if (b >= a) return 0;
    return a - b;
  }
  function add(percent storage p, uint a) internal view returns (uint) {
    return a + mul(p, a);
  }
}
contract PlanetCryptoToken is ERC721Full_custom{
    using Percent for Percent.percent;
    event referralPaid(address indexed search_to,
                    address to, uint256 amnt, uint256 timestamp);
    event issueCoinTokens(address indexed searched_to, 
                    address to, uint256 amnt, uint256 timestamp);
    event landPurchased(uint256 indexed search_token_id, address indexed search_buyer, 
            uint256 token_id, address buyer, bytes32 name, int256 center_lat, int256 center_lng, uint256 size, uint256 bought_at, uint256 empire_score, uint256 timestamp);
    event taxDistributed(uint256 amnt, uint256 total_players, uint256 timestamp);
    event cardBought(
                    uint256 indexed search_token_id, address indexed search_from, address indexed search_to,
                    uint256 token_id, address from, address to, 
                    bytes32 name,
                    uint256 orig_value, 
                    uint256 new_value,
                    uint256 empireScore, uint256 newEmpireScore, uint256 now);
    address owner;
    address devBankAddress; 
    address tokenBankAddress; 
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }
    modifier validateLand(int256[] plots_lat, int256[] plots_lng) {
        require(planetCryptoUtils_interface.validateLand(msg.sender, plots_lat, plots_lng) == true, "Some of this land already owned!");
        _;
    }
    modifier validatePurchase(int256[] plots_lat, int256[] plots_lng) {
        require(planetCryptoUtils_interface.validatePurchase(msg.sender, msg.value, plots_lat, plots_lng) == true, "Not enough ETH!");
        _;
    }
    modifier validateTokenPurchase(int256[] plots_lat, int256[] plots_lng) {
        require(planetCryptoUtils_interface.validateTokenPurchase(msg.sender, plots_lat, plots_lng) == true, "Not enough COINS to buy these plots!");
        require(planetCryptoCoin_interface.transferFrom(msg.sender, tokenBankAddress, plots_lat.length) == true, "Token transfer failed");
        _;
    }
    modifier validateResale(uint256 _token_id) {
        require(planetCryptoUtils_interface.validateResale(msg.sender, msg.value, _token_id) == true, "Not enough ETH to buy this card!");
        _;
    }
    modifier updateUsersLastAccess() {
        uint256 allPlyersIdx = playerAddressToPlayerObjectID[msg.sender];
        if(allPlyersIdx == 0){
            all_playerObjects.push(player(msg.sender,now,0,0));
            playerAddressToPlayerObjectID[msg.sender] = all_playerObjects.length-1;
        } else {
            all_playerObjects[allPlyersIdx].lastAccess = now;
        }
        _;
    }
    struct plotDetail {
        bytes32 name;
        uint256 orig_value;
        uint256 current_value;
        uint256 empire_score;
        int256[] plots_lat;
        int256[] plots_lng;
    }
    struct plotBasic {
        int256 lat;
        int256 lng;
    }
    struct player {
        address playerAddress;
        uint256 lastAccess;
        uint256 totalEmpireScore;
        uint256 totalLand;
    }
    address planetCryptoCoinAddress = 0xEF5B27207bD9Cd7B7708755CDF1A4523CB0b1f7D;
    PlanetCryptoCoin_I internal planetCryptoCoin_interface;
    address planetCryptoUtilsAddress = 0x6cd067629939290C5aBFf20E1F6fC8bd750508C0;
    PlanetCryptoUtils_I internal planetCryptoUtils_interface;
    Percent.percent pr