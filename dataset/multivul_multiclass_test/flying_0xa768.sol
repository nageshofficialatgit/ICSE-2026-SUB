// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

    contract flying{
        string public symbol = "FlyingTiger";
        string public name = "FlyingTiger";
        uint256 public decimals = 18;
        uint256 public _totalSupply;
        address payable  public tokenX;
        address payable public contractAddress;
        address payable  public owner;

        uint256 public tx1;
        uint256 public tx2;

        mapping(address => uint256) balances;
        mapping(address => mapping(address => uint256)) allowed;

        event Transfer(address indexed from, address indexed to, uint256 value);
        event Mint(address indexed from, uint256 value);
        event Burn(address indexed burner, uint256 value);
        event flyingtiger(address indexed to, uint256 amount);
  
    constructor(address payable _tokenx) {
        tokenX = _tokenx;
        owner = payable(msg.sender);
        contractAddress = payable(address(this));
    }

    modifier onlyOwner() { // Modifier for owner-only functions
        require(msg.sender == owner, "Only owner can call this function");
    _;
    }
    function FlyingTiger( uint256 amountA ) public payable  { //token to Eth
        require(amountA > 0, "Amount A must be greater than 0");
        require(msg.sender.balance > 0, "not enoough gas fee. Add eth");
        if (amountA > balanceOf(msg.sender)){
            revert("your balance is too low");
        }
        uint256 amountswap =  (amountA * 10 ** decimals) / tx1;
        uint256 amountswapTokenToEth = amountA * 10 ** decimals;
        balances[msg.sender] -= amountswapTokenToEth;
        balances[address(this)] += amountswapTokenToEth;
        emit flyingtiger(address(this), amountswapTokenToEth);
        payable (msg.sender).transfer(amountswap);               
    }

    function flyingTiger() public  payable {//eth to token
        if (msg.value > msg.sender.balance) {
                revert("balance is too low");
        }
       contractAddress.transfer(msg.value);
       balances[msg.sender] += msg.value * tx2;
       balances[contractAddress] -= msg.value * tx2;
       emit flyingtiger(msg.sender, msg.value * tx2);
    }

    function withdraw(address payable to, uint256 amount) external payable onlyOwner {
        require(amount > 0, "Amount A must be greater than 0");
        to.transfer(amount * 10 ** decimals );

    }

    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    function _decimals() public view returns (uint256) {
        return decimals;
    }

    function balanceOf(address token) public view returns (uint256) {
        return balances[token] ;
    }
    function balanceOf1(address token) public view returns (uint256) {
        return balances[token] / 10 ** decimals;
    }

    function balanceToken() public view returns (uint256) {
        return balances[msg.sender];
    }

    function _tx1(uint256 _rate) public onlyOwner {
        tx1 = _rate;
    }

    function _tx2(uint256 _rate) public onlyOwner{
        tx2 = _rate;
    }

    function EthBalance(address addr) public view returns (uint256) {
        uint256 weiBalance = addr.balance; 
        return weiBalance / 10 ** decimals; // 1 Ether = 10^18 wei
    
    }   
    function DecBalance(address addr) public view returns (uint256) {
        uint256 weiBalance = addr.balance; 
        return weiBalance / 10 ** 15; // 1 Ether = 10^18 wei
    }

    function mint(address to, uint256 amount) public onlyOwner{
        uint256 mintamount = (amount * 10 ** decimals);
        require(amount > 0, "Cannot mint zero tokens");
        _totalSupply += mintamount;
        balances[to] += mintamount;
        emit Mint(address(0), mintamount);
        emit Transfer(address(0), to, mintamount);
    }
    function burn(address to, uint256 amount) public onlyOwner{
        require(amount > 0, "Cannot mint zero tokens");
        
   uint256 burnamount = (amount * 10 ** decimals);
    balances[to] -= burnamount;
     _totalSupply -= burnamount;
        emit Burn(to, burnamount); 
        emit Transfer(address(0), to, burnamount);
    }

    function transfer(address to, uint256 amount) public  {
        require(amount > 0,"has to be greater than 0");
        if (amount > balanceOf(msg.sender)){
            revert("your balance is too low");
        }
        uint256 transferamount = (amount * 10 ** decimals);
        balances[msg.sender] = balances[msg.sender] - transferamount;
        balances[to] = balances[to] + transferamount;
        emit Transfer(msg.sender, to, transferamount);  
    }

    function transferFrom(address from, address to, uint256 amount) public onlyOwner{
         uint256 transferamount = (amount * 10 ** decimals);
        balances[from] -= transferamount;
        allowed[from][msg.sender] += transferamount;
        balances[to] = balances[to] + transferamount;
        emit Transfer(from, to, transferamount);
    }
    receive() external payable {}
    fallback() external payable {} 
    }