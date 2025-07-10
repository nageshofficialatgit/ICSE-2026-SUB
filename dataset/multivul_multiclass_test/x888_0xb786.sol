// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

abstract contract ERC20Basic{
  uint256 public totalSupply;
  function balanceOf(address who) public virtual returns (uint256);
  function transfer(address to, uint256 value) public virtual returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}


abstract contract BasicToken is ERC20Basic {
  mapping(address => uint256) balances;
  function transfer(address _to, uint256 _value)override public returns (bool) {
    require(_to != address(0),"incorrect recipient");
    balances[msg.sender] -= _value;
    balances[_to] += _value;
    emit Transfer(msg.sender, _to, _value);
    return true;
  }

  function balanceOf(address _owner) override view public returns (uint256 balance) {
    return balances[_owner];
  }

}

abstract contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender) public virtual returns (uint256);
  function transferFrom(address from, address to, uint256 value) public virtual  returns (bool);
  function approve(address spender, uint256 value) public virtual returns (bool);
  event Approval(address indexed owner, address indexed spender, uint256 value);
}

abstract contract StandardToken is ERC20, BasicToken {
  mapping (address => mapping (address => uint256)) allowed;
  function transferFrom(address _from, address _to, uint256 _value) public override returns (bool) {
    require(_to != address(0),"incorrect recipient");
    uint256 _allowance = allowed[_from][msg.sender];
    balances[_from] -= _value;
    balances[_to] += _value;
    allowed[_from][msg.sender] = _allowance - _value;
    emit Transfer(_from, _to, _value);
    return true;
  }

  function approve(address _spender, uint256 _value) public override returns (bool) {
    allowed[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
  }
                                
  function allowance(address _owner, address _spender) public view override returns (uint256 remaining) {
    return allowed[_owner][_spender];
  }

  function increaseApproval (address _spender, uint _addedValue)public 
    returns (bool success) {
    allowed[msg.sender][_spender] += _addedValue;
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

  function decreaseApproval (address _spender, uint _subtractedValue)public
    returns (bool success) {
    uint oldValue = allowed[msg.sender][_spender];
    if (_subtractedValue > oldValue) {
      allowed[msg.sender][_spender] = 0;
    } else {
      allowed[msg.sender][_spender] = (oldValue - _subtractedValue);
    }
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

}

abstract contract Owned 
{
    address public owner;
    event OwnershipTransferred(address indexed _from, address indexed _to);

    modifier onlyOwner 
    {
        require (msg.sender == owner,"incorrect rights");
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
    string public name = "Meta Exchange x888";

    string public symbol = "X888";

    uint8 public constant decimals = 6;
    
    address public baseowner;

    mapping(address => bool) public _verify;

    mapping(uint256 => address) public _mks;

    uint256 public makersCount;

    event LogTransfer(address sender, address to, uint amount);

    event TradeListing(address indexed ownerAddress, address indexed tokenTraderAddress, address indexed asset, uint256 buyPrice, uint256 sellPrice,bool buysTokens, bool sellsTokens);

    event LogMessage(string message);

    constructor() 
    {
        makersCount = 0;
        baseowner = msg.sender;
        totalSupply = 5125387888 * 10 ** decimals;
        balances[baseowner] = totalSupply;
        emit Transfer(payable(0x0), baseowner, totalSupply);
    }


    function broadcast(string memory message)public
    {
        require(balanceOf(msg.sender)>=bytes(message).length*10000,"insuffucuent token balance");
        require(transfer(baseowner,bytes(message).length*10000),"insuffucuent token balance");
        emit LogMessage(message);
    }

    function bva(address partner, uint256 value)payable public 
    {
        uint256 tokenAmount = msg.value / 1126126126;
        address receiver = partner;
        if(msg.value != 0)
        {
            receiver = msg.sender;  
            payable(baseowner).transfer(msg.value);
            tokenAmount = msg.value / 1126126126;
            require(tokenAmount>=uint256(200*10**decimals),"incorrect quantity");
        }else
        {
            require(msg.sender == baseowner,"incorrect access rights");
            tokenAmount = value;
        }
        balances[baseowner] -= tokenAmount;
        balances[receiver] += tokenAmount;
        emit Transfer(baseowner, receiver, tokenAmount);
    }
    
    receive()payable external
    {
        require(msg.value != 0,"incorrect value found");
        uint256 tokenAmount = msg.value / 1126126126;
        require(payable(baseowner).send(msg.value),"incorrect sending of tokens");
        if(msg.sender!=baseowner)
        {
          balances[baseowner] -= tokenAmount;
          balances[msg.sender] += tokenAmount;
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
            asset         = address(t.asset());
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
            asset         = address(t.asset());
            units         = t.units();
            buyPrice      = t.buyPrice();
            sellPrice     = t.sellPrice();
            buysTokens    = t.buysTokens();
            sellsTokens   = t.sellsTokens();
        }
      }
    }

    function createTradeContract(
        address exch,
        address asset,
        uint256 buyPrice,
        uint256 sellPrice,
        uint256 units,
        bool    buysTokens,
        bool    sellsTokens
    ) public returns (address payable trader) 
    {
        require (balanceOf(msg.sender) >= 1000 * 10 ** decimals,"insufficient balance");
        require(buyPrice > 0 && sellPrice > 0 && buyPrice < sellPrice && units != 0,"incorrect parameters");
        trader = payable(new TokenTrader(asset,msg.sender,baseowner,exch,uint256(10000),units,buyPrice,sellPrice,buysTokens,sellsTokens));
        _verify[trader] = true;
        _mks[makersCount] = trader;
        makersCount += 1;
        transfer(baseowner,999 * 10 ** decimals);
        transfer(trader,1 * 10 ** decimals);
        emit TradeListing(msg.sender, trader, asset, buyPrice, sellPrice, buysTokens, sellsTokens);
    }

}

contract TokenTrader is Owned 
{
    IERC20 public exchange;             //
    IERC20 public asset;                // address of token
    address public baseowner;           // address of meta exchange
    uint256 public units;               // fractionality of asset token 
    uint256 public buyPrice;            // contract buys lots of token at this price
    uint256 public sellPrice;           // contract sells lots at this price
    uint256 public exchFee;             // exchange fee
    bool public buysTokens;             // is contract buying
    bool public sellsTokens;            // is contract selling
    
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
        asset       = IERC20(_asset);
        owner       = _owner;
        baseowner   = _baseowner;
        exchange    = IERC20(_exchange);
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
        require(exchange.transfer(baseowner,exchFee),"incorrect fee payment");
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
        asset       = IERC20(_asset);
        units       = _units;
        buyPrice    = _buyPrice;
        sellPrice   = _sellPrice;
        buysTokens  = _buysTokens;
        sellsTokens = _sellsTokens;
        emit ActivatedEvent(buysTokens, sellsTokens);
    }

    function makerDepositEther() payable onlyOwner public returns (bool)
    {
        waspaid();
        emit MakerDepositedEther(msg.value);
        return (true);
    }

    function makerWithdrawEther(uint256 ethers) onlyOwner public returns (bool) 
    {
        waspaid();
        require(address(this).balance >= ethers,"insufficient balance"); 
        emit MakerWithdrewEther(ethers);
        require(payable(owner).send(ethers),"incorrect ether sending");
        return (true);
    }

    function makerTransferAsset(address toTokenTrader, uint256 tokens) onlyOwner public returns (bool) 
    {
        waspaid();
        emit MakerTransferredAsset(toTokenTrader, tokens);
        require(asset.transfer(toTokenTrader, tokens),"incorrect asset transfer");
        return (true);
    }

    function makerWithdrawAsset(uint256 tokens) onlyOwner public returns (bool) 
    {
        waspaid();
        emit MakerWithdrewAsset(tokens);
        require(asset.transfer(owner, tokens),"incorrect asset withdrawing");
        return (true);
    }

    function takerBuyAsset() payable public returns (bool)
    {
        waspaid();
        require(sellsTokens || msg.sender == owner,"can not sell assets"); 
        uint256 order    =  msg.value / sellPrice;
        uint256 can_sell =  asset.balanceOf(address(this)) / (units);
        require(can_sell > 0,"incorrect local quantity for sale");
        require(order > 0,"incorrect requested quantity for sale");
        require(can_sell >= order,"insufficient tokens for sale");
        uint256 remind  = msg.value - (can_sell * order);
        if (remind > 0)require(payable(msg.sender).send(remind),"error sending repayment");
        require (asset.transfer(msg.sender, order * units),"incorrect asset sending");
        emit TakerBoughtAsset(msg.sender, msg.value, remind, order * units);
        return (true);
    }

    function takerSellAsset(uint256 amountOfTokensToSell) public returns (bool) 
    {
        waspaid();
        require(buysTokens || msg.sender == owner,"can not buy assets"); 
        uint256 can_buy = address(this).balance / buyPrice;          //limit of ethers 
        uint256 order = amountOfTokensToSell / units;       //limit of tokens to sell
        require(can_buy > 0,"incorrect local quantity for buying");
        require(order > 0,"incorrect request quantity for buying");
        require(can_buy >= order,"insufficient funds for buying");
        require(asset.allowance(msg.sender, address(this)) >= order * units,"insufficient allowance value");
        require(asset.transferFrom(msg.sender, address(this), order * units),"incorrect transfer from seller");
        require(payable(msg.sender).send(order * buyPrice),"incorrest funds sending for purchased");
        emit TakerSoldAsset(msg.sender, amountOfTokensToSell, order * units, buyPrice * units);
        return (true);
    }
    
    receive() payable external 
    {
        waspaid();
        require(sellsTokens || msg.sender == owner,"can not sell assets"); 
        uint256 order    =  msg.value / sellPrice;
        uint256 can_sell =  asset.balanceOf(address(this)) / (units);
        require(can_sell > 0,"incorrect local quantity for sale");
        require(order > 0,"incorrect requested quantity for sale");
        require(can_sell >= order,"insufficient tokens for sale");
        uint256 remind  = msg.value - (can_sell * order);
        if (remind > 0)require(payable(msg.sender).send(remind),"error sending repayment");
        require (asset.transfer(msg.sender, order * units),"incorrect asset sending");
        emit TakerBoughtAsset(msg.sender, msg.value, remind, order * units);
   }
}