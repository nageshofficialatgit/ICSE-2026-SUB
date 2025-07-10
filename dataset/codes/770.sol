pragma solidity ^0.4.21;
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
contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
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
contract ERC20WithData is StandardToken {
    function approveAndCall(address _spender, uint256 _value, bytes _data) public returns (bool) {
        require(_spender != address(this));
        super.approve(_spender, _value);
        require(_spender.call(_data));
        return true;
    }
    function transferAndCall(address _to, uint256 _value, bytes _data) public returns (bool) {
        require(_to != address(this));
        super.transfer(_to, _value);
        require(_to.call(_data));
        return true;
    }
    function transferFromAndCall(
        address _from,
        address _to,
        uint256 _value,
        bytes _data
    ) public returns (bool)
    {
        require(_to != address(this));
        super.transferFrom(_from, _to, _value);
        require(_to.call(_data));
        return true;
    }
    function increaseApprovalAndCall(address _spender, uint _addedValue, bytes _data) public returns (bool) {
        require(_spender != address(this));
        super.increaseApproval(_spender, _addedValue);
        require(_spender.call(_data));
        return true;
    }
    function decreaseApprovalAndCall(address _spender, uint _subtractedValue, bytes _data) public returns (bool) {
        require(_spender != address(this));
        super.decreaseApproval(_spender, _subtractedValue);
        require(_spender.call(_data));
        return true;
    }
}
contract BurnableToken is BasicToken {
  event Burn(address indexed burner, uint256 value);
  function burn(uint256 _value) public {
    require(_value <= balances[msg.sender]);
    address burner = msg.sender;
    balances[burner] = balances[burner].sub(_value);
    totalSupply_ = totalSupply_.sub(_value);
    Burn(burner, _value);
    Transfer(burner, address(0), _value);
  }
}
contract DetailedERC20 is ERC20 {
  string public name;
  string public symbol;
  uint8 public decimals;
  function DetailedERC20(string _name, string _symbol, uint8 _decimals) public {
    name = _name;
    symbol = _symbol;
    decimals = _decimals;
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
contract MintableToken is StandardToken, Ownable {
  event Mint(address indexed to, uint256 amount);
  event MintFinished();
  bool public mintingFinished = false;
  modifier canMint() {
    require(!mintingFinished);
    _;
  }
  function mint(address _to, uint256 _amount) onlyOwner canMint public returns (bool) {
    totalSupply_ = totalSupply_.add(_amount);
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
contract AleaCoin is DetailedERC20, MintableToken, BurnableToken, ERC20WithData {
    modifier canTransfer() {
        require(mintingFinished);
        _;
    }
    function AleaCoin()
    DetailedERC20("Alea Coin", "ALEA", 18) public
    {}
    function transfer(address _to, uint256 _value) canTransfer public returns (bool) {
        return super.transfer(_to, _value);
    }
    function transferFrom(address _from, address _to, uint256 _value) canTransfer public returns (bool) {
        return super.transferFrom(_from, _to, _value);
    }
    function transferAndCall(address _to, uint256 _value, bytes _data) canTransfer public returns (bool) {
        return super.transferAndCall(_to, _value, _data);
    }
    function transferFromAndCall(
        address _from,
        address _to,
        uint256 _value,
        bytes _data)
    canTransfer public returns (bool)
    {
        return super.transferFromAndCall(
            _from,
            _to,
            _value,
            _data
        );
    }
    function transferAnyERC20Token(address _tokenAddress, uint256 _tokens) onlyOwner public returns (bool success) {
        return ERC20Basic(_tokenAddress).transfer(owner, _tokens);
    }
}
contract Crowdsale {
  using SafeMath for uint256;
  ERC20 public token;
  address public wallet;
  uint256 public rate;
  uint256 public weiRaised;
  event TokenPurchase(address indexed purchaser, address indexed beneficiary, uint256 value, uint256 amount);
  function Crowdsale(uint256 _rate, address _wallet, ERC20 _token) public {
    require(_rate > 0);
    require(_wallet != address(0));
    require(_token != address(0));
    rate = _rate;
    wallet = _wallet;
    token = _token;
  }
  function () external payable {
    buyTokens(msg.sender);
  }
  function buyTokens(address _beneficiary) public payable {
    uint256 weiAmount = msg.value;
    _preValidatePurchase(_beneficiary, weiAmount);
    uint256 tokens = _getTokenAmount(weiAmount);
    weiRaised = weiRaised.add(weiAmount);
    _processPurchase(_beneficiary, tokens);
    TokenPurchase(msg.sender, _beneficiary, weiAmount, tokens);
    _updatePurchasingState(_beneficiary, weiAmount);
    _forwardFunds();
    _postValidatePurchase(_beneficiary, weiAmount);
  }
  function _preValidatePurchase(address _beneficiary, uint256 _weiAmount) internal {
    require(_beneficiary != address(0));
    require(_weiAmount != 0);
  }
  function _postValidatePurchase(address _beneficiary, uint256 _weiAmount) internal {
  }
  function _deliverTokens(address _beneficiary, uint256 _tokenAmount) internal {
    token.transfer(_beneficiary, _tokenAmount);
  }
  function _processPurchase(address _beneficiary, uint256 _tokenAmount) internal {
    _deliverTokens(_beneficiary, _tokenAmount);
  }
  function _updatePurchasingState(address _beneficiary, uint256 _weiAmount) internal {
  }
  function _getTokenAmount(uint256 _weiAmount) internal view returns (uint256) {
    return _weiAmount.mul(rate);
  }
  function _forwardFunds() internal {
    wallet.transfer(msg.value);
  }
}
contract MintedCrowdsale is Crowdsale {
  function _deliverTokens(address _beneficiary, uint256 _tokenAmount) internal {
    require(MintableToken(token).mint(_beneficiary, _tokenAmount));
  }
}
contract TimedCrowdsale is Crowdsale {
  using SafeMath for uint256;
  uint256 public openingTime;
  uint256 public closingTime;
  modifier onlyWhileOpen {
    require(now >= openingTime && now <= closingTime);
    _;
  }
  function TimedCrowdsale(uint256 _openingTime, uint256 _closingTime) public {
    require(_openingTime >= now);
    require(_closingTime >= _openingTime);
    openingTime = _openingTime;
    closingTime = _closingTime;
  }
  function hasClosed() public view returns (bool) {
    return now > closingTime;
  }
  function _preValidatePurchase(address _beneficiary, uint256 _weiAmount) internal onlyWhileOpen {
    super._preValidatePurchase(_beneficiary, _weiAmount);
  }
}
library SafeERC20 {
  function safeTransfer(ERC20Basic token, address to, uint256 value) internal {
    assert(token.transfer(to, value));
  }
  function safeTransferFrom(ERC20 token, address from, address to, uint256 value) internal {
    assert(token.transferFrom(from, to, value));
  }
  function safeApprove(ERC20 token, address spender, uint256 value) internal {
    assert(token.approve(spender, value));
  }
}
contract TokenTimelock {
  using SafeERC20 for ERC20Basic;
  ERC20Basic public token;
  address public beneficiary;
  uint256 public releaseTime;
  function TokenTimelock(ERC20Basic _token, address _beneficiary, uint256 _releaseTime) public {
    require(_releaseTime > now);
    token = _token;
    beneficiary = _beneficiary;
    releaseTime = _releaseTime;
  }
  function release() public {
    require(now >= releaseTime);
    uint256 amount = token.balanceOf(this);
    require(amount > 0);
    token.safeTransfer(beneficiary, amount);
  }
}
contract AleaPrivateSale is TimedCrowdsale, MintedCrowdsale, Ownable {
    bool public initiated = false;
    uint256 public cap;
    address public companyWallet;
    address public teamWallet;
    address public advisorWallet;
    address public reserveWallet;
    TokenTimelock public companyTimeLock;
    TokenTimelock public teamTimeLock;
    uint256 public companyTokens 	= 	40000000 * (10 ** 18);
    uint256 public teamTokens 		= 	16000000 * (10 ** 18);
    uint256 public advisorTokens 	= 	20000000 * (10 ** 18);
    uint256 public reserveTokens 	= 	4000000 * (10 ** 18);
    function AleaPrivateSale(
        uint256 _startTime,
        uint256 _endTime,
        uint256 _rate,
        address _wallet,
        uint256 _tokenCap,
        address _token
    )
    TimedCrowdsale(_startTime, _endTime)
    Crowdsale(_rate, _wallet, ERC20(_token))
    public
    {
        require(_tokenCap != 0);
        cap = (_tokenCap * (10 ** 18)).div(_rate);
    }
    function initSale(
        address _companyWallet,
        address _teamWallet,
        address _advisorWallet,
        address _reserveWallet
    ) public onlyOwner
    {
        require(!initiated);
        require(_companyWallet != 0x0);
        require(_teamWallet != 0x0);
        require(_advisorWallet != 0x0);
        require(_reserveWallet != 0x0);
        companyWallet = _companyWallet;
        teamWallet = _teamWallet;
        advisorWallet = _advisorWallet;
        reserveWallet = _reserveWallet;
        _deliverTokens(companyWallet, companyTokens.div(2)); 
        companyTimeLock = new TokenTimelock(token, companyWallet, uint64(1559339940)); 
        _deliverTokens(address(companyTimeLock), companyTokens.div(2));
        teamTimeLock = new TokenTimelock(token, teamWallet, uint64(1577833140)); 
        _deliverTokens(address(teamTimeLock), teamTokens);
        _deliverTokens(advisorWallet, advisorTokens);
        _deliverTokens(reserveWallet, reserveTokens);
        initiated = true;
    }
    function transferTokenOwnership(address _newOwner) public onlyOwner {
        require(ended());
        require(_newOwner != 0x0);
        Ownable(token).transferOwnership(_newOwner);
    }
    function started() public view returns(bool) {
        return now >= openingTime;
    }
    function ended() public view returns(bool) {
        return hasClosed() || capReached();
    }
    function capReached() public view returns (bool) {
        return weiRaised >= cap;
    }
    function transferAnyERC20Token(address _tokenAddress, uint256 _tokens) onlyOwner public returns (bool success) {
        return ERC20Basic(_tokenAddress).transfer(owner, _tokens);
    }
    function _preValidatePurchase(address _beneficiary, uint256 _weiAmount) internal {
        super._preValidatePurchase(_beneficiary, _weiAmount);
        require(weiRaised.add(_weiAmount) <= cap);
    }
}