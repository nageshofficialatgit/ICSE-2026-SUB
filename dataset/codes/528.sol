pragma solidity ^0.4.24;
library SafeMath {
  function mul(uint a, uint b) internal pure returns (uint) {
    if (a == 0) {
      return 0;
    }
    uint c = a * b;
    assert(c / a == b);
    return c;
  }
  function div(uint a, uint b) internal pure returns (uint) {
    uint c = a / b;
    return c;
  }
  function sub(uint a, uint b) internal pure returns (uint) {
    assert(b <= a);
    return a - b;
  }
  function add(uint a, uint b) internal pure returns (uint) {
    uint c = a + b;
    assert(c >= a);
    return c;
  }
}
library ZethrTierLibrary {
  uint constant internal magnitude = 2 ** 64;
  function getTier(uint divRate) internal pure returns (uint8) {
    uint actualDiv = divRate / magnitude;
    if (actualDiv >= 30) {
      return 6;
    } else if (actualDiv >= 25) {
      return 5;
    } else if (actualDiv >= 20) {
      return 4;
    } else if (actualDiv >= 15) {
      return 3;
    } else if (actualDiv >= 10) {
      return 2;
    } else if (actualDiv >= 5) {
      return 1;
    } else if (actualDiv >= 2) {
      return 0;
    } else {
      revert();
    }
  }
  function getDivRate(uint _tier)
  internal pure
  returns (uint8)
  {
    if (_tier == 0) {
      return 2;
    } else if (_tier == 1) {
      return 5;
    } else if (_tier == 2) {
      return 10;
    } else if (_tier == 3) {
      return 15;
    } else if (_tier == 4) {
      return 20;
    } else if (_tier == 5) {
      return 25;
    } else if (_tier == 6) {
      return 33;
    } else {
      revert();
    }
  }
}
contract ERC223Receiving {
  function tokenFallback(address _from, uint _amountOfTokens, bytes _data) public returns (bool);
}
contract ZethrMultiSigWallet is ERC223Receiving {
  using SafeMath for uint;
  event Confirmation(address indexed sender, uint indexed transactionId);
  event Revocation(address indexed sender, uint indexed transactionId);
  event Submission(uint indexed transactionId);
  event Execution(uint indexed transactionId);
  event ExecutionFailure(uint indexed transactionId);
  event Deposit(address indexed sender, uint value);
  event OwnerAddition(address indexed owner);
  event OwnerRemoval(address indexed owner);
  event WhiteListAddition(address indexed contractAddress);
  event WhiteListRemoval(address indexed contractAddress);
  event RequirementChange(uint required);
  event BankrollInvest(uint amountReceived);
  mapping (uint => Transaction) public transactions;
  mapping (uint => mapping (address => bool)) public confirmations;
  mapping (address => bool) public isOwner;
  address[] public owners;
  uint public required;
  uint public transactionCount;
  bool internal reEntered = false;
  uint constant public MAX_OWNER_COUNT = 15;
  struct Transaction {
    address destination;
    uint value;
    bytes data;
    bool executed;
  }
  struct TKN {
    address sender;
    uint value;
  }
  modifier onlyWallet() {
    if (msg.sender != address(this))
      revert();
    _;
  }
  modifier isAnOwner() {
    address caller = msg.sender;
    if (isOwner[caller])
      _;
    else
      revert();
  }
  modifier ownerDoesNotExist(address owner) {
    if (isOwner[owner]) 
      revert();
      _;
  }
  modifier ownerExists(address owner) {
    if (!isOwner[owner])
      revert();
    _;
  }
  modifier transactionExists(uint transactionId) {
    if (transactions[transactionId].destination == 0)
      revert();
    _;
  }
  modifier confirmed(uint transactionId, address owner) {
    if (!confirmations[transactionId][owner])
      revert();
    _;
  }
  modifier notConfirmed(uint transactionId, address owner) {
    if (confirmations[transactionId][owner])
      revert();
    _;
  }
  modifier notExecuted(uint transactionId) {
    if (transactions[transactionId].executed)
      revert();
    _;
  }
  modifier notNull(address _address) {
    if (_address == 0)
      revert();
    _;
  }
  modifier validRequirement(uint ownerCount, uint _required) {
    if ( ownerCount > MAX_OWNER_COUNT
      || _required > ownerCount
      || _required == 0
      || ownerCount == 0)
      revert();
    _;
  }
  constructor (address[] _owners, uint _required)
    public
    validRequirement(_owners.length, _required)
  {
    for (uint i=0; i<_owners.length; i++) {
      if (isOwner[_owners[i]] || _owners[i] == 0)
        revert();
      isOwner[_owners[i]] = true;
    }
    owners = _owners;
    required = _required;
  }
  function()
    public
    payable
  {
  }
  function addOwner(address owner)
    public
    onlyWallet
    ownerDoesNotExist(owner)
    notNull(owner)
    validRequirement(owners.length + 1, required)
  {
    isOwner[owner] = true;
    owners.push(owner);
    emit OwnerAddition(owner);
  }
  function removeOwner(address owner)
    public
    onlyWallet
    ownerExists(owner)
    validRequirement(owners.length, required)
  {
    isOwner[owner] = false;
    for (uint i=0; i<owners.length - 1; i++)
      if (owners[i] == owner) {
        owners[i] = owners[owners.length - 1];
        break;
      }
    owners.length -= 1;
    if (required > owners.length)
      changeRequirement(owners.length);
    emit OwnerRemoval(owner);
  }
  function replaceOwner(address owner, address newOwner)
    public
    onlyWallet
    ownerExists(owner)
    ownerDoesNotExist(newOwner)
  {
    for (uint i=0; i<owners.length; i++)
      if (owners[i] == owner) {
        owners[i] = newOwner;
        break;
      }
    isOwner[owner] = false;
    isOwner[newOwner] = true;
    emit OwnerRemoval(owner);
    emit OwnerAddition(newOwner);
  }
  function changeRequirement(uint _required)
    public
    onlyWallet
    validRequirement(owners.length, _required)
  {
    required = _required;
    emit RequirementChange(_required);
  }
  function submitTransaction(address destination, uint value, bytes data)
    public
    returns (uint transactionId)
  {
    transactionId = addTransaction(destination, value, data);
    confirmTransaction(transactionId);
  }
  function confirmTransaction(uint transactionId)
    public
    ownerExists(msg.sender)
    transactionExists(transactionId)
    notConfirmed(transactionId, msg.sender)
  {
    confirmations[transactionId][msg.sender] = true;
    emit Confirmation(msg.sender, transactionId);
    executeTransaction(transactionId);
  }
  function revokeConfirmation(uint transactionId)
    public
    ownerExists(msg.sender)
    confirmed(transactionId, msg.sender)
    notExecuted(transactionId)
  {
    confirmations[transactionId][msg.sender] = false;
    emit Revocation(msg.sender, transactionId);
  }
  function executeTransaction(uint transactionId)
    public
    notExecuted(transactionId)
  {
    if (isConfirmed(transactionId)) {
      Transaction storage txToExecute = transactions[transactionId];
      txToExecute.executed = true;
      if (txToExecute.destination.call.value(txToExecute.value)(txToExecute.data))
        emit Execution(transactionId);
      else {
        emit ExecutionFailure(transactionId);
        txToExecute.executed = false;
      }
    }
  }
  function isConfirmed(uint transactionId)
    public
    constant
    returns (bool)
  {
    uint count = 0;
    for (uint i=0; i<owners.length; i++) {
      if (confirmations[transactionId][owners[i]])
        count += 1;
      if (count == required)
        return true;
    }
  }
  function addTransaction(address destination, uint value, bytes data)
    internal
    notNull(destination)
    returns (uint transactionId)
  {
    transactionId = transactionCount;
    transactions[transactionId] = Transaction({
        destination: destination,
        value: value,
        data: data,
        executed: false
    });
    transactionCount += 1;
    emit Submission(transactionId);
  }
  function getConfirmationCount(uint transactionId)
    public
    constant
    returns (uint count)
  {
    for (uint i=0; i<owners.length; i++)
      if (confirmations[transactionId][owners[i]])
        count += 1;
  }
  function getTransactionCount(bool pending, bool executed)
    public
    constant
    returns (uint count)
  {
    for (uint i=0; i<transactionCount; i++)
      if (pending && !transactions[i].executed || executed && transactions[i].executed)
        count += 1;
  }
  function getOwners()
    public
    constant
    returns (address[])
  {
    return owners;
  }
  function getConfirmations(uint transactionId)
    public
    constant
    returns (address[] _confirmations)
  {
    address[] memory confirmationsTemp = new address[](owners.length);
    uint count = 0;
    uint i;
    for (i=0; i<owners.length; i++)
      if (confirmations[transactionId][owners[i]]) {
        confirmationsTemp[count] = owners[i];
        count += 1;
      }
      _confirmations = new address[](count);
      for (i=0; i<count; i++)
        _confirmations[i] = confirmationsTemp[i];
  }
  function getTransactionIds(uint from, uint to, bool pending, bool executed)
    public
    constant
    returns (uint[] _transactionIds)
  {
    uint[] memory transactionIdsTemp = new uint[](transactionCount);
    uint count = 0;
    uint i;
    for (i=0; i<transactionCount; i++)
      if (pending && !transactions[i].executed || executed && transactions[i].executed) {
        transactionIdsTemp[count] = i;
        count += 1;
      }
      _transactionIds = new uint[](to - from);
    for (i=from; i<to; i++)
      _transactionIds[i - from] = transactionIdsTemp[i];
  }
  function tokenFallback(address , uint , bytes )
  public
  returns (bool)
  {
    return true;
  }
}
contract ZethrTokenBankrollInterface is ERC223Receiving {
  uint public jackpotBalance;
  function getMaxProfit(address) public view returns (uint);
  function gameTokenResolution(uint _toWinnerAmount, address _winnerAddress, uint _toJackpotAmount, address _jackpotAddress, uint _originalBetSize) external;
  function payJackpotToWinner(address _winnerAddress, uint payoutDivisor) public;
}
contract ZethrBankrollControllerInterface is ERC223Receiving {
  address public jackpotAddress;
  ZethrTokenBankrollInterface[7] public tokenBankrolls; 
  ZethrMultiSigWallet public multiSigWallet;
  mapping(address => bool) public validGameAddresses;
  function gamePayoutResolver(address _resolver, uint _tokenAmount) public;
  function isTokenBankroll(address _address) public view returns (bool);
  function getTokenBankrollAddressFromTier(uint8 _tier) public view returns (address);
  function tokenFallback(address _from, uint _amountOfTokens, bytes _data) public returns (bool);
}
contract ERC721Interface {
  function approve(address _to, uint _tokenId) public;
  function balanceOf(address _owner) public view returns (uint balance);
  function implementsERC721() public pure returns (bool);
  function ownerOf(uint _tokenId) public view returns (address addr);
  function takeOwnership(uint _tokenId) public;
  function totalSupply() public view returns (uint total);
  function transferFrom(address _from, address _to, uint _tokenId) public;
  function transfer(address _to, uint _tokenId) public;
  event Transfer(address indexed from, address indexed to, uint tokenId);
  event Approval(address indexed owner, address indexed approved, uint tokenId);
}
library AddressUtils {
  function isContract(address addr) internal view returns (bool) {
    uint size;
    assembly { size := extcodesize(addr) }  
    return size > 0;
  }
}
contract ZethrDividendCards is ERC721Interface {
    using SafeMath for uint;
  event Birth(uint tokenId, string name, address owner);
  event TokenSold(uint tokenId, uint oldPrice, uint newPrice, address prevOwner, address winner, string name);
  event Transfer(address from, address to, uint tokenId);
  event BankrollDivCardProfit(uint bankrollProfit, uint percentIncrease, address oldOwner);
  event BankrollProfitFailure(uint bankrollProfit, uint percentIncrease, address oldOwner);
  event UserDivCardProfit(uint divCardProfit, uint percentIncrease, address oldOwner);
  event DivCardProfitFailure(uint divCardProfit, uint percentIncrease, address oldOwner);
  event masterCardProfit(uint toMaster, address _masterAddress, uint _divCardId);
  event masterCardProfitFailure(uint toMaster, address _masterAddress, uint _divCardId);
  event regularCardProfit(uint toRegular, address _regularAddress, uint _divCardId);
  event regularCardProfitFailure(uint toRegular, address _regularAddress, uint _divCardId);
  string public constant NAME           = "ZethrDividendCard";
  string public constant SYMBOL         = "ZDC";
  address public         BANKROLL;
  mapping (uint => address) public      divCardIndexToOwner;
  mapping (uint => uint) public         divCardRateToIndex;
  mapping (address => uint) private     ownershipDivCardCount;
  mapping (uint => address) public      divCardIndexToApproved;
  mapping (uint => uint) private        divCardIndexToPrice;
  mapping (address => bool) internal    administrators;
  address public                        creator;
  bool    public                        onSale;
  struct Card {
    string name;
    uint percentIncrease;
  }
  Card[] private divCards;
  modifier onlyCreator() {
    require(msg.sender == creator);
    _;
  }
  constructor (address _bankroll) public {
    creator = msg.sender;
    BANKROLL = _bankroll;
    createDivCard("2%", 1 ether, 2);
    divCardRateToIndex[2] = 0;
    createDivCard("5%", 1 ether, 5);
    divCardRateToIndex[5] = 1;
    createDivCard("10%", 1 ether, 10);
    divCardRateToIndex[10] = 2;
    createDivCard("15%", 1 ether, 15);
    divCardRateToIndex[15] = 3;
    createDivCard("20%", 1 ether, 20);
    divCardRateToIndex[20] = 4;
    createDivCard("25%", 1 ether, 25);
    divCardRateToIndex[25] = 5;
    createDivCard("33%", 1 ether, 33);
    divCardRateToIndex[33] = 6;
    createDivCard("MASTER", 5 ether, 10);
    divCardRateToIndex[999] = 7;
	  onSale = true;
    administrators[0x4F4eBF556CFDc21c3424F85ff6572C77c514Fcae] = true; 
    administrators[0x11e52c75998fe2E7928B191bfc5B25937Ca16741] = true; 
    administrators[0x20C945800de43394F70D789874a4daC9cFA57451] = true; 
    administrators[0xef764BAC8a438E7E498c2E5fcCf0f174c3E3F8dB] = true; 
    administrators[msg.sender] = true; 
  }
  modifier isNotContract()
  {
    require (msg.sender == tx.origin);
    _;
  }
	modifier hasStarted()
  {
		require (onSale == true);
		_;
	}
	modifier isAdmin()
  {
	  require(administrators[msg.sender]);
	  _;
  }
  function setBankroll(address where)
    public
    isAdmin
  {
    BANKROLL = where;
  }
  function approve(address _to, uint _tokenId)
    public
    isNotContract
  {
    require(_owns(msg.sender, _tokenId));
    divCardIndexToApproved[_tokenId] = _to;
    emit Approval(msg.sender, _to, _tokenId);
  }
  function balanceOf(address _owner)
    public
    view
    returns (uint balance)
  {
    return ownershipDivCardCount[_owner];
  }
  function createDivCard(string _name, uint _price, uint _percentIncrease)
    public
    onlyCreator
  {
    _createDivCard(_name, BANKROLL, _price, _percentIncrease);
  }
	function startCardSale()
        public
        isAdmin
  {
		onSale = true;
	}
  function getDivCard(uint _divCardId)
    public
    view
    returns (string divCardName, uint sellingPrice, address owner)
  {
    Card storage divCard = divCards[_divCardId];
    divCardName = divCard.name;
    sellingPrice = divCardIndexToPrice[_divCardId];
    owner = divCardIndexToOwner[_divCardId];
  }
  function implementsERC721()
    public
    pure
    returns (bool)
  {
    return true;
  }
  function name()
    public
    pure
    returns (string)
  {
    return NAME;
  }
  function ownerOf(uint _divCardId)
    public
    view
    returns (address owner)
  {
    owner = divCardIndexToOwner[_divCardId];
    require(owner != address(0));
	return owner;
  }
  function purchase(uint _divCardId)
    public
    payable
    hasStarted
    isNotContract
  {
    address oldOwner  = divCardIndexToOwner[_divCardId];
    address newOwner  = msg.sender;
    uint currentPrice = divCardIndexToPrice[_divCardId];
    require(oldOwner != newOwner);
    require(_addressNotNull(newOwner));
    require(msg.value >= currentPrice);
    uint percentIncrease = divCards[_divCardId].percentIncrease;
    uint previousPrice   = SafeMath.mul(currentPrice, 100).div(100 + percentIncrease);
    uint totalProfit     = SafeMath.sub(currentPrice, previousPrice);
    uint oldOwnerProfit  = SafeMath.div(totalProfit, 2);
    uint bankrollProfit  = SafeMath.sub(totalProfit, oldOwnerProfit);
    oldOwnerProfit       = SafeMath.add(oldOwnerProfit, previousPrice);
    uint purchaseExcess  = SafeMath.sub(msg.value, currentPrice);
    divCardIndexToPrice[_divCardId] = SafeMath.div(SafeMath.mul(currentPrice, (100 + percentIncrease)), 100);
    _transfer(oldOwner, newOwner, _divCardId);
    if(BANKROLL.send(bankrollProfit)) {
      emit BankrollDivCardProfit(bankrollProfit, percentIncrease, oldOwner);
    } else {
      emit BankrollProfitFailure(bankrollProfit, percentIncrease, oldOwner);
    }
    if(oldOwner.send(oldOwnerProfit)) {
      emit UserDivCardProfit(oldOwnerProfit, percentIncrease, oldOwner);
    } else {
      emit DivCardProfitFailure(oldOwnerProfit, percentIncrease, oldOwner);
    }
    msg.sender.transfer(purchaseExcess);
  }
  function priceOf(uint _divCardId)
    public
    view
    returns (uint price)
  {
    return divCardIndexToPrice[_divCardId];
  }
  function setCreator(address _creator)
    public
    onlyCreator
  {
    require(_creator != address(0));
    creator = _creator;
  }
  function symbol()
    public
    pure
    returns (string)
  {
    return SYMBOL;
  }
  function takeOwnership(uint _divCardId)
    public
    isNotContract
  {
    address newOwner = msg.sender;
    address oldOwner = divCardIndexToOwner[_divCardId];
    require(_addressNotNull(newOwner));
    require(_approved(newOwner, _divCardId));
    _transfer(oldOwner, newOwner, _divCardId);
  }
  function totalSupply()
    public
    view
    returns (uint total)
  {
    return divCards.length;
  }
  function transfer(address _to, uint _divCardId)
    public
    isNotContract
  {
    require(_owns(msg.sender, _divCardId));
    require(_addressNotNull(_to));
    _transfer(msg.sender, _to, _divCardId);
  }
  function transferFrom(address _from, address _to, uint _divCardId)
    public
    isNotContract
  {
    require(_owns(_from, _divCardId));
    require(_approved(_to, _divCardId));
    require(_addressNotNull(_to));
    _transfer(_from, _to, _divCardId);
  }
  function receiveDividends(uint _divCardRate)
    public
    payable
  {
    uint _divCardId = divCardRateToIndex[_divCardRate];
    address _regularAddress = divCardIndexToOwner[_divCardId];
    address _masterAddress = divCardIndexToOwner[7];
    uint toMaster = msg.value.div(2);
    uint toRegular = msg.value.sub(toMaster);
    if(_masterAddress.send(toMaster)){
      emit masterCardProfit(toMaster, _masterAddress, _divCardId);
    } else {
      emit masterCardProfitFailure(toMaster, _masterAddress, _divCardId);
    }
    if(_regularAddress.send(toRegular)) {
      emit regularCardProfit(toRegular, _regularAddress, _divCardId);
    } else {
      emit regularCardProfitFailure(toRegular, _regularAddress, _divCardId);
    }
  }
  function _addressNotNull(address _to)
    private
    pure
    returns (bool)
  {
    return _to != address(0);
  }
  function _approved(address _to, uint _divCardId)
    private
    view
    returns (bool)
  {
    return divCardIndexToApproved[_divCardId] == _to;
  }
  function _createDivCard(string _name, address _owner, uint _price, uint _percentIncrease)
    private
  {
    Card memory _divcard = Card({
      name: _name,
      percentIncrease: _percentIncrease
    });
    uint newCardId = divCards.push(_divcard) - 1;
    require(newCardId == uint(uint32(newCardId)));
    emit Birth(newCardId, _name, _owner);
    divCardIndexToPrice[newCardId] = _price;
    _transfer(BANKROLL, _owner, newCardId);
  }
  function _owns(address claimant, uint _divCardId)
    private
    view
    returns (bool)
  {
    return claimant == divCardIndexToOwner[_divCardId];
  }
  function _transfer(address _from, address _to, uint _divCardId)
    private
  {
    ownershipDivCardCount[_to]++;
    divCardIndexToOwner[_divCardId] = _to;
    if (_from != address(0)) {
      ownershipDivCardCount[_from]--;
      delete divCardIndexToApproved[_divCardId];
    }
    emit Transfer(_from, _to, _divCardId);
  }
}
contract Zethr {
  using SafeMath for uint;
  modifier onlyHolders() {
    require(myFrontEndTokens() > 0);
    _;
  }
  modifier dividendHolder() {
    require(myDividends(true) > 0);
    _;
  }
  modifier onlyAdministrator(){
    address _customerAddress = msg.sender;
    require(administrators[_customerAddress]);
    _;
  }
  event onTokenPurchase(
    address indexed customerAddress,
    uint incomingEthereum,
    uint tokensMinted,
    address indexed referredBy
  );
  event UserDividendRate(
    address user,
    uint divRate
  );
  event onTokenSell(
    address indexed customerAddress,
    uint tokensBurned,
    uint ethereumEarned
  );
  event onReinvestment(
    address indexed customerAddress,
    uint ethereumReinvested,
    uint tokensMinted
  );
  event onWithdraw(
    address indexed customerAddress,
    uint ethereumWithdrawn
  );
  event Transfer(
    address indexed from,
    address indexed to,
    uint tokens
  );
  event Approval(
    address indexed tokenOwner,
    address indexed spender,
    uint tokens
  );