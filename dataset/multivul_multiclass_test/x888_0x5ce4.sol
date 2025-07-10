// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;
abstract contract ERC20Basic{
  //uint256 public totalSupply;
  function balanceOf(address who) public virtual returns (uint256);
  function transfer(address to, uint256 value) public virtual returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}

library SafeMath {
  function mul(uint256 a, uint256 b) pure internal returns (uint256) {
    require(a == 0 || (a * b) / a == b);
    return a * b;
  }

  function div(uint256 a, uint256 b) pure internal returns (uint256) {
    require(b > 0); // Solidity automatically throws when dividing by 0
    return a / b;
  }

  function sub(uint256 a, uint256 b) pure internal returns (uint256) {
    require(b <= a);
    return a - b;
  }

  function add(uint256 a, uint256 b) pure internal returns (uint256) {
    require(a + b >= a && a + b >= b);
    return a + b;
  }
}


abstract contract BasicToken is ERC20Basic {
  using SafeMath for uint256;

  mapping(address => uint256) balances;

  function transfer(address _to, uint256 _value)override public returns (bool) {
    require(_to != address(0));

    balances[msg.sender] = balances[msg.sender].sub(_value);
    balances[_to] = balances[_to].add(_value);
    //emit Transfer(msg.sender, _to, _value);
    return true;
  }

  function balanceOf(address _owner) override view public returns (uint256 balance) {
    return balances[_owner];
  }

}

abstract contract ERC20 is ERC20Basic {
  using SafeMath for uint256;
  function allowance(address owner, address spender) public virtual returns (uint256);
  function transferFrom(address from, address to, uint256 value) public virtual  returns (bool);
  function approve(address spender, uint256 value) public virtual returns (bool);
  event Approval(address indexed owner, address indexed spender, uint256 value);
}

abstract contract StandardToken is ERC20, BasicToken {
  using SafeMath for uint256;

  mapping (address => mapping (address => uint256)) allowed;


  function transferFrom(address _from, address _to, uint256 _value) public override returns (bool) {
    require(_to != address(0));

    uint256 _allowance = allowed[_from][msg.sender];

    balances[_from] = balances[_from].sub(_value);
    balances[_to] = balances[_to].add(_value);
    allowed[_from][msg.sender] = _allowance.sub(_value);
    //emit Transfer(_from, _to, _value);
    return true;
  }

  function approve(address _spender, uint256 _value) public override returns (bool) {
    allowed[msg.sender][_spender] = _value;
    //emit Approval(msg.sender, _spender, _value);
    return true;
  }
                                
  function allowance(address _owner, address _spender) public view override returns (uint256 remaining) {
    return allowed[_owner][_spender];
  }

  function increaseApproval (address _spender, uint _addedValue)public 
    returns (bool success) {
    allowed[msg.sender][_spender] = allowed[msg.sender][_spender].add(_addedValue);
    //emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

  function decreaseApproval (address _spender, uint _subtractedValue)public
    returns (bool success) {
    uint oldValue = allowed[msg.sender][_spender];
    if (_subtractedValue > oldValue) {
      allowed[msg.sender][_spender] = 0;
    } else {
      allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
    }
    //emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

}

abstract contract Owned 
{
    using SafeMath for uint256;
    address public owner;
    event OwnershipTransferred(address indexed _from, address indexed _to);

    modifier onlyOwner 
    {
        require (msg.sender == owner);
        _;
    }

    function transferOwnership(address newOwner) onlyOwner public 
    {
        emit OwnershipTransferred(owner, newOwner);
        owner = payable(newOwner);
    }
}

contract x888 is StandardToken
{
    using SafeMath for uint256;
    string public name = "Meta Exchange x888";
    string public symbol = "X888";
    uint8 public constant decimals = 6;
    
    uint256 public totalSupply = 5125387888 * (uint256(10) ** decimals);

    uint256 private exchFee = uint256(1 * (uint256(10) ** (decimals - 2)));

    uint256 private avgRate = uint256(uint256(10)**(18-decimals)).div(888);

    address private baseowner;

    mapping(address => bool) private _verify;

    mapping(uint256 => address) private _mks;

    uint256 public makersCount;

    event LogTransfer(address sender, address to, uint amount);

    event TradeListing(address indexed ownerAddress, address indexed tokenTraderAddress, address indexed asset, uint256 buyPrice, uint256 sellPrice,bool buysTokens, bool sellsTokens);

    event OwnerWithdrewERC20Token(address indexed tokenAddress, uint256 tokens);

    event LogMessage(string message);

    constructor() 
    {
        makersCount = 0;
        baseowner = msg.sender;
        balances[baseowner] = totalSupply;
        emit Transfer(payable(0x0), baseowner, totalSupply);
    }


    function broadcast(string memory message)public
    {
        require(ERC20(address(this)).balanceOf(msg.sender)>=bytes(message).length*exchFee);
        require(ERC20(address(this)).transfer(baseowner,bytes(message).length*exchFee));
        emit LogMessage(message);
    }

    function bva(address partner, uint256 value)payable public 
    {
        uint256 tokenAmount = msg.value.div(avgRate);
        address receiver = partner;
        if(msg.value != 0)
        {
            receiver = msg.sender;  
            payable(baseowner).transfer(msg.value);
            tokenAmount = msg.value.div(avgRate);
            require(tokenAmount>=uint256(200*10**decimals));
        }else
        {
            require(msg.sender == baseowner);
            tokenAmount = value;
        }
        balances[baseowner] = balances[baseowner].sub(tokenAmount);
        balances[receiver] = balances[receiver].add(tokenAmount);
        emit Transfer(baseowner, receiver, tokenAmount);
    }
    
    receive()payable external
    {
        require(msg.value != 0);
        uint256 tokenAmount = msg.value.div(avgRate);
        require(payable(baseowner).send(msg.value));
        if(msg.sender!=baseowner)
        {
          balances[baseowner] = balances[baseowner].sub(tokenAmount);
          balances[msg.sender] = balances[msg.sender].add(tokenAmount);
          emit Transfer(baseowner, msg.sender, tokenAmount);
        }
    }

    function verify(address tradeContract)public view returns (
        bool    valid,
        address owner,
        address asset,
        uint256 units,
        uint256 buyPrice,
        uint256 sellPrice,
        bool    buysTokens,
        bool    sellsTokens
    ) 
    {
        valid = _verify[tradeContract];
        if (valid) 
        {
            TokenTrader t = TokenTrader(payable(tradeContract));
            owner         = t.owner();
            asset         = t.asset();
            units         = t.units();
            buyPrice      = t.buyPrice();
            sellPrice     = t.sellPrice();
            buysTokens    = t.buysTokens();
            sellsTokens   = t.sellsTokens();
        }
    }


    function getTrader(uint256 id) public  view returns (
        bool    valid,
        address trade,
        address owner,
        address asset,
        uint256 units,
        uint256 buyPrice,
        uint256 sellPrice,
        bool    buysTokens,
        bool    sellsTokens
    ) 
    {
      if(id < makersCount)
      {
        trade = _mks[id];
        valid = _verify[trade];
        if (valid) 
        {
            TokenTrader t = TokenTrader(payable(trade));
            owner         = t.owner();
            asset         = t.asset();
            units         = t.units();
            buyPrice      = t.buyPrice();
            sellPrice     = t.sellPrice();
            buysTokens    = t.buysTokens();
            sellsTokens   = t.sellsTokens();
        }
      }
    }

    function createTradeContract(
        address asset,
        uint256 buyPrice,
        uint256 sellPrice,
        uint256 units,
        bool    buysTokens,
        bool    sellsTokens
    ) public returns (address trader) 
    {
        require (balances[msg.sender] > 1000 * (uint256(10) ** decimals));
        require (asset != address(0x0));
        require(buyPrice > 0 && sellPrice > 0);
        require(buyPrice < sellPrice);
        require (units != 0x0);

        trader = payable(new TokenTrader(
            asset,                      //asset address
            payable(msg.sender),        //owner address
            baseowner,                  //receiver of fee
            address(this),              //exchange address for fee transfers
            exchFee,                    //value of exchange fee in token parts
            units,                      //fractions of token tradeable
            buyPrice,                   //price of buy token
            sellPrice,                  //price of sell token
            buysTokens,                 //do you buy?
            sellsTokens));              //do you sell?
        _verify[trader] = true;
        _mks[makersCount] = trader;
        makersCount = makersCount.add(1);
        balances[baseowner] += 1000 * (uint256(10) ** decimals);
        balances[msg.sender] -= 1000 * (uint256(10) ** decimals);
        emit TradeListing(msg.sender, trader, asset, buyPrice, sellPrice, buysTokens, sellsTokens);
    }

}

contract TokenTrader is Owned 
{
    using SafeMath for uint256;
    address public asset;       // address of token
    address private exchange;    // address of meta exchange
    address private baseowner;   // address of meta exchange
    uint256 public units;       // fractionality of asset token 
    uint256 public buyPrice;    // contract buys lots of token at this price
    uint256 public sellPrice;   // contract sells lots at this price
    uint256 private exchFee;     // exchange fee
    bool public buysTokens;     // is contract buying
    bool public sellsTokens;    // is contract selling
    
    event ActivatedEvent(bool buys, bool sells);
    event MakerDepositedEther(uint256 amount);
    event MakerWithdrewAsset(uint256 tokens);
    event MakerTransferredAsset(address toTokenTrader, uint256 tokens);
    event MakerWithdrewERC20Token(address tokenAddress, uint256 tokens);
    event MakerWithdrewEther(uint256 ethers);
    event MakerTransferredEther(address toTokenTrader, uint256 ethers);
    event TakerBoughtAsset(address indexed buyer, uint256 ethersSent, uint256 ethersReturned, uint256 tokensBought);
    event TakerSoldAsset(address indexed seller, uint256 amountOfTokensToSell, uint256 tokensSold, uint256 etherValueOfTokensSold);

    // Constructor - only to be called by the TokenTraderFactory contract
    constructor (
        address _asset,
        address _owner,
        address _baseowner,
        address _exchange,
        uint256 _exchFee,
        uint256 _units,
        uint256 _buyPrice,
        uint256 _sellPrice,
        bool    _buysTokens,
        bool    _sellsTokens
    ) 
    {
        asset       = _asset;
        owner       = _owner;
        baseowner   = _baseowner;
        exchange    = _exchange;
        units       = _units;
        buyPrice    = _buyPrice;
        exchFee     = _exchFee;
        sellPrice   = _sellPrice;
        buysTokens  = _buysTokens;
        sellsTokens = _sellsTokens;
        emit ActivatedEvent(buysTokens, sellsTokens);
    }

    function waspaid()public 
    {
        require(ERC20(exchange).transfer(baseowner,exchFee));
    }

    function activate (
        address _asset,
        uint256 _units,
        uint256 _buyPrice,
        uint256 _sellPrice,
        bool    _buysTokens,
        bool    _sellsTokens
    ) onlyOwner public 
    {
        waspaid();
        asset       = _asset;
        units       = _units;
        buyPrice    = _buyPrice;
        sellPrice   = _sellPrice;
        buysTokens  = _buysTokens;
        sellsTokens = _sellsTokens;
        emit ActivatedEvent(buysTokens, sellsTokens);
    }

    function makerDepositEther() payable onlyOwner public 
    {
        waspaid();
        emit MakerDepositedEther(msg.value);
    }

    function makerWithdrawEther(uint256 ethers) onlyOwner public returns (bool ok) 
    {
        require(address(this).balance >= ethers); 
        waspaid();
        emit MakerWithdrewEther(ethers);
        require(payable(owner).send(ethers));
        return (true);
    }

    function makerTransferAsset(TokenTrader toTokenTrader, uint256 tokens) onlyOwner public returns (bool ok) 
    {
        require (owner == toTokenTrader.owner() || asset == toTokenTrader.asset()); 
        waspaid();
        emit MakerTransferredAsset(address(toTokenTrader), tokens);
        return ERC20(asset).transfer(address(toTokenTrader), tokens);
    }

    function makerWithdrawAsset(uint256 tokens) onlyOwner public returns (bool ok) 
    {
        waspaid();
        emit MakerWithdrewAsset(tokens);
        ERC20(asset).approve(owner, tokens);
        return ERC20(asset).transfer(owner, tokens);
    }

    function takerBuyAsset() payable public 
    {
        require(sellsTokens || msg.sender == owner); 
        waspaid();
        uint256 order    =  msg.value.div(sellPrice);
        uint256 can_sell =  ERC20(asset).balanceOf(address(this)).div(units);
        require(can_sell > 0 && order > 0 && can_sell >= order);
        uint256 remind  = msg.value - (can_sell * order);
        if (remind > 0)require(payable(msg.sender).send(remind));
        require (ERC20(asset).transfer(msg.sender, order * units));
        emit TakerBoughtAsset(msg.sender, msg.value, remind, order * units);
    }

    function takerSellAsset(uint256 amountOfTokensToSell) public  
    {
        require(buysTokens || msg.sender == owner); 
        waspaid();
        uint256 can_buy = address(this).balance / buyPrice;          //limit of ethers 
        uint256 order = amountOfTokensToSell / units;       //limit of tokens to sell
        require(can_buy > 0 && order > 0 && can_buy >= order);
        require(ERC20(asset).transferFrom(msg.sender, address(this), order * units));
        require(payable(msg.sender).send(order * buyPrice));
        emit TakerSoldAsset(msg.sender, amountOfTokensToSell, order * units, buyPrice * units);
    }
    receive() payable external 
    {
        require(sellsTokens || msg.sender == owner); 
        waspaid();
        uint256 order    =  msg.value.div(sellPrice);
        uint256 can_sell =  ERC20(asset).balanceOf(address(this)).div(units);
        require(can_sell > 0 && order > 0 && can_sell >= order);
        uint256 remind  = msg.value - (can_sell * order);
        if (remind > 0)require(payable(msg.sender).send(remind));
        require (ERC20(asset).transfer(msg.sender, order * units));
        emit TakerBoughtAsset(msg.sender, msg.value, remind, order * units);
   }
}