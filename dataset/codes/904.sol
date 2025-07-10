pragma solidity ^0.4.19;
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
contract ERC20Basic {
  uint256 public totalSupply;
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}
contract BasicToken is ERC20Basic {
  using SafeMath for uint256;
  mapping(address => uint256) balances;
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
contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender) public view returns (uint256);
  function transferFrom(address from, address to, uint256 value) public returns (bool);
  function approve(address spender, uint256 value) public returns (bool);
  event Approval(address indexed owner, address indexed spender, uint256 value);
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
contract MintableToken is StandardToken, Ownable {
  event Mint(address indexed to, uint256 amount);
  event MintFinished();
  bool public mintingFinished = false;
  modifier canMint() {
    require(!mintingFinished);
    _;
  }
  function mint(address _to, uint256 _amount) onlyOwner canMint public returns (bool) {
    totalSupply = totalSupply.add(_amount);
    balances[_to] = balances[_to].add(_amount);
    Mint(_to, _amount);
    Transfer(address(0), _to, _amount);
    return true;
  }
  function finishMinting() onlyOwner canMint public returns (bool) {
    mintingFinished = true;
    MintFinished();
    return true;
  }
}
contract CogeneroToken is MintableToken {
  function allowTransfer(address _from, address _to) public view returns (bool);
  function allowManager() public view returns (bool);
  function setManager(address _manager, bool _status) onlyOwner public;
  function setAllowTransferGlobal(bool _status) public;
  function setAllowTransferLocal(bool _status) public;
  function setAllowTransferExternal(bool _status) public;
  function setWhitelist(address _address, uint256 _date) public;
  function setLockupList(address _address, uint256 _date) public;
  function setWildcardList(address _address, bool _status) public;
  function burn(address _burner, uint256 _value) onlyOwner public;
}
contract Crowdsale {
  using SafeMath for uint256;
  CogeneroToken public token;
  uint256 public startTime;
  uint256 public endTime;
  address public wallet;
  uint256 public rate;
  uint256 public weiRaised;
  event TokenPurchase(address indexed purchaser, address indexed beneficiary, uint256 value, uint256 amount);
  function Crowdsale(uint256 _startTime, uint256 _endTime, uint256 _rate, address _wallet) public {
    require(_startTime >= now);
    require(_endTime >= _startTime);
    require(_rate > 0);
    require(_wallet != address(0));
    token = createTokenContract();
    startTime = _startTime;
    endTime = _endTime;
    rate = _rate;
    wallet = _wallet;
  }
  function createTokenContract() internal returns (CogeneroToken);
  function () external payable {
    buyTokens(msg.sender);
  }
  function buyTokens(address beneficiary) public payable {
    require(beneficiary != address(0));
    require(validPurchase());
    uint256 weiAmount = msg.value;
    uint256 tokens = weiAmount.mul(rate);
    weiRaised = weiRaised.add(weiAmount);
    token.mint(beneficiary, tokens);
    TokenPurchase(msg.sender, beneficiary, weiAmount, tokens);
    forwardFunds();
  }
  function forwardFunds() internal {
    wallet.transfer(msg.value);
  }
  function validPurchase() internal view returns (bool) {
    bool withinPeriod = now >= startTime && now <= endTime;
    bool nonZeroPurchase = msg.value != 0;
    return withinPeriod && nonZeroPurchase;
  }
  function hasEnded() public view returns (bool) {
    return now > endTime;
  }
}
contract Cogenero is Crowdsale, Ownable {
  uint256 constant CAP =  1000000000000000000000000000;
  uint256 constant CAP_PRE_SALE = 180000000000000000000000000;
  uint256 constant CAP_ICO_SALE = 320000000000000000000000000;
  uint256 public rate8_end_at = 1556524800;
  uint256 public totalSupplyIco;
  function Cogenero (
    uint256 _startTime,
    uint256 _endTime,
    uint256 _rate,
    address _wallet
  ) public
    Crowdsale(_startTime, _endTime, _rate, _wallet)
  {
  }
  function createTokenContract() internal returns (CogeneroToken) {
    return CogeneroToken(0x88218eb0756bCa01a9f6be0c6EfF641e9b4d8101);
  }
  function validPurchase() internal constant returns (bool) {
    if (msg.value < 20000000000000000) {
      return false;
    }
    if (token.totalSupply().add(msg.value.mul(getRate())) >= CAP) {
      return false;
    }
    if (now > 1538208000 && now < 1554105600) {
      return false;
    }
    if (1535788800 >= now && 1538208000 <= now) {
      if (token.totalSupply().add(msg.value.mul(getRate())) >= CAP_PRE_SALE) {
        return false;
      }
    }
    if (1554105600 >= now && 1556524800 <= now) {
      if (totalSupplyIco.add(msg.value.mul(getRate())) >= CAP_ICO_SALE) {
        return false;
      }
    }
    if (getRate() == 0) {
      return false;
    }
    return super.validPurchase();
  }
  function buyTokens(address beneficiary) payable public {
    require(beneficiary != address(0));
    require(validPurchase());
    uint256 weiAmount = msg.value;
    uint256 tokens = weiAmount.mul(getRate());
    weiRaised = weiRaised.add(weiAmount);
    token.mint(beneficiary, tokens);
    TokenPurchase(msg.sender, beneficiary, weiAmount, tokens);
    forwardFunds();
  }
  function getRate() public constant returns (uint256) {
    if (1535788800 <= now && now <= 1536393599) {
      return 30000;
    }
    if (1536393600 <= now && now <= 1536998399) {
      return 25500;
    }
    if (1536998400 <= now && now <= 1537603199) {
      return 22500;
    }
    if (1537603200 <= now && now <= 1538208000) {
      return 20000;
    }
    if (1554105600 <= now && now <= 1554710399) {
      return 8000;
    }
    if (1554710400 <= now && now <= 1555315199) {
      return 7000;
    }
    if (1555315200 <= now && now <= 1555919999) {
      return 6000;
    }
    if (1555920000 <= now && now <= rate8_end_at) {
      return 5000;
    }
    return 0;
  }
  function mintTokens(address walletToMint, uint256 t) onlyOwner payable public {
    require(token.totalSupply().add(t) < CAP);
    token.mint(walletToMint, t);
  }
  function tokenTransferOwnership(address newOwner) onlyOwner payable public {
    token.transferOwnership(newOwner);
  }
  function setAllowTransferGlobal(bool _status) public {
    token.setAllowTransferGlobal(_status);
  }
  function setAllowTransferLocal(bool _status) public {
    token.setAllowTransferLocal(_status);
  }
  function setAllowTransferExternal(bool _status) public {
    token.setAllowTransferExternal(_status);
  }
  function setManager(address _manager, bool _status) public {
    token.setManager(_manager, _status);
  }
  function setWhitelist(address _address, uint256 _date) public {
    token.setWhitelist(_address, _date);
  }
  function setLockupList(address _address, uint256 _date) public {
    token.setLockupList(_address, _date);
  }
  function setWildcardList(address _address, bool _status) public {
    token.setWildcardList(_address, _status);
  }
  function changeEnd(uint256 _end) onlyOwner public {
    endTime = _end;
    rate8_end_at = _end;
  }
  function burn(address _burner, uint256 _value) onlyOwner public {
    token.burn(_burner, _value);
  }
}