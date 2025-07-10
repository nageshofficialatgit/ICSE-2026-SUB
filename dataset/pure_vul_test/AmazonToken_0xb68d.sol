/**
 * Submitted for verification at Etherscan.io on 2025-02-28
*/

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

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

interface IERC20Metadata is IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

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
        if (a == 0) return 0;
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

contract Ownable is Context {
    address private _owner;
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
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

interface IUniswapV2Router02 {
    function factory() external pure returns (address);
    function WETH() external pure returns (address);
    function addLiquidityETH(
        address token,
        uint256 amountTokenDesired,
        uint256 amountTokenMin,
        uint256 amountETHMin,
        address to,
        uint256 deadline
    ) external payable returns (uint256 amountToken, uint256 amountETH, uint256 liquidity);
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external;
}

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

// Amazon Token (AMZ) contract with 4 decimals
contract AmazonToken is Context, IERC20, IERC20Metadata, Ownable {
    using SafeMath for uint256;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _isExcludedFromFee;
    mapping(address => bool) private _isAutomaticMarketMaker;

    // Team wallet address for Amazon Token
    address public constant teamWallet = 0x9A73D066D18e431a07Fdbcd2C3119ae1e77F4BD3;

    uint256 public MaxSupply;
    uint256 private _totalSupply;
    string private _name;
    string private _symbol;
    uint8 private _decimals;

    // Fees (in basis points, 100 = 1%)
    uint256 public _liquidityFee = 3; // 3% liquidity fee
    uint256 public _teamFee = 1;     // 1% team fee
    uint256 public _burnFee = 3;     // 3% burn fee

    IUniswapV2Router02 public uniswapV2Router;
    address public uniswapV2Pair;
    uint256 public _maxTxAmount = 10 * 10**6 * 10**4; // Max transaction amount (adjusted for 4 decimals)

    event TaxFeeUpdated(uint256 totalFee);
    event MaxTxAmountUpdated(uint256 updatingTxAmount);

    constructor() {
        _name = "Amazon Token";
        _symbol = "AMZ";
        _decimals = 4; // Set to 4 decimal places
        MaxSupply = 1_000_000_000 * 10**_decimals; // Max supply of 1 billion tokens

        // Set Uniswap V2 Router (Ethereum Mainnet address)
        IUniswapV2Router02 _uniswapV2Router = IUniswapV2Router02(
            0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
        );
        uniswapV2Pair = IUniswapV2Factory(_uniswapV2Router.factory())
            .createPair(address(this), _uniswapV2Router.WETH());
        uniswapV2Router = _uniswapV2Router;

        // Exclude from fees
        _isExcludedFromFee[address(this)] = true;
        _isExcludedFromFee[owner()] = true;
        _isExcludedFromFee[teamWallet] = true;
        _isAutomaticMarketMaker[uniswapV2Pair] = true;

        // Initial mint: 200 million tokens to the owner
        _mint(_msgSender(), 200_000_000);
    }

    // IERC20Metadata implementation
    function name() public view virtual override returns (string memory) {
        return _name;
    }

    function symbol() public view virtual override returns (string memory) {
        return _symbol;
    }

    function decimals() public view virtual override returns (uint8) {
        return _decimals;
    }

    // IERC20 implementation
    function totalSupply() public view virtual override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view virtual override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public virtual override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view virtual override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) public virtual override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) external override returns (bool) {
        require(recipient != address(0), "AMZ: Transfer to zero address not allowed");
        if (sender != msg.sender) {
            uint256 allowed = _allowances[sender][msg.sender];
            if (allowed != type(uint256).max) {
                require(allowed >= amount, "AMZ: Request exceeds allowance");
                _allowances[sender][msg.sender] = allowed.sub(amount);
                emit Approval(sender, msg.sender, allowed.sub(amount));
            }
        }
        _transfer(sender, recipient, amount);
        return true;
    }

    function increaseAllowance(address spender, uint256 addedValue) public virtual returns (bool) {
        _approve(_msgSender(), spender, _allowances[_msgSender()][spender].add(addedValue));
        return true;
    }

    function decreaseAllowance(address spender, uint256 subtractedValue) public virtual returns (bool) {
        uint256 currentAllowance = _allowances[_msgSender()][spender];
        require(currentAllowance >= subtractedValue, "AMZ: Decreased allowance below zero");
        _approve(_msgSender(), spender, currentAllowance.sub(subtractedValue));
        return true;
    }

    // Transfer logic with fees
    function _transfer(address sender, address recipient, uint256 amount) internal virtual {
        require(sender != address(0), "AMZ: Transfer from zero address");
        require(recipient != address(0), "AMZ: Transfer to zero address");

        uint256 _teamAmt;
        uint256 _liquidityAmt;
        uint256 _burnAmt;

        // Apply fees only for transactions involving automated market makers
        if (_isAutomaticMarketMaker[sender] || _isAutomaticMarketMaker[recipient]) {
            if (_isExcludedFromFee[sender] || _isExcludedFromFee[recipient]) {
                _teamAmt = 0;
                _liquidityAmt = 0;
                _burnAmt = 0;
            } else {
                require(amount <= _maxTxAmount, "AMZ: Transaction limit exceeded");
                _teamAmt = amount.mul(_teamFee).div(10000);
                _liquidityAmt = amount.mul(_liquidityFee).div(10000);
                _burnAmt = amount.mul(_burnFee).div(10000);
            }
        } else {
            _teamAmt = 0;
            _liquidityAmt = 0;
            _burnAmt = 0;
        }

        uint256 senderBalance = _balances[sender];
        require(senderBalance >= amount, "AMZ: Transfer amount exceeds balance");

        _balances[sender] = senderBalance.sub(amount);
        _balances[teamWallet] = _balances[teamWallet].add(_teamAmt);
        _balances[address(this)] = _balances[address(this)].add(_liquidityAmt);

        if (_burnAmt > 0) {
            _totalSupply = _totalSupply.sub(_burnAmt);
            emit Transfer(sender, address(0), _burnAmt);
        }

        _balances[recipient] = _balances[recipient].add(amount.sub(_teamAmt).sub(_liquidityAmt).sub(_burnAmt));

        if (_teamAmt > 0) {
            emit Transfer(sender, teamWallet, _teamAmt);
        }
        emit Transfer(sender, recipient, amount.sub(_teamAmt).sub(_liquidityAmt).sub(_burnAmt));
    }

    // Update fees (only owner)
    function updateFees(uint256 liquidityFee, uint256 burnFee, uint256 teamFee) external onlyOwner {
        require(liquidityFee.add(burnFee).add(teamFee) <= 2500, "AMZ: Total fee cannot exceed 25%");
        _liquidityFee = liquidityFee;
        _burnFee = burnFee;
        _teamFee = teamFee;
        emit TaxFeeUpdated(liquidityFee.add(burnFee).add(teamFee));
    }

    // Exclude or include accounts from fees
    function excludeOrIncludeFromFee(address account, bool status) external onlyOwner {
        require(account != address(0), "AMZ: Cannot exclude zero address");
        _isExcludedFromFee[account] = status;
    }

    function isExcludedFromFee(address account) external view returns (bool) {
        return _isExcludedFromFee[account];
    }

    // Set automated market maker status
    function setAutomaticMarketMaker(address account, bool status) external onlyOwner {
        require(account != address(0), "AMZ: Cannot set zero address");
        _isAutomaticMarketMaker[account] = status;
    }

    function isAutomaticMarketMaker(address account) external view returns (bool) {
        return _isAutomaticMarketMaker[account];
    }

    // Set maximum transaction percentage
    function setMaxTxPercentage(uint256 maxTxPercentage) external onlyOwner {
        require(maxTxPercentage >= 1, "AMZ: Percentage must be >= 1");
        _maxTxAmount = _totalSupply.mul(maxTxPercentage).div(10**2);
        emit MaxTxAmountUpdated(_maxTxAmount);
    }

    // Mint new tokens (only owner)
    function _mint(address account, uint256 amount) public virtual onlyOwner {
        uint256 amountWithDecimals = amount * 10**_decimals;
        require(_totalSupply.add(amountWithDecimals) <= MaxSupply, "AMZ: Cannot mint beyond MaxSupply");
        require(account != address(0), "AMZ: Mint to zero address");

        _beforeTokenTransfer(address(0), account, amountWithDecimals);
        _totalSupply = _totalSupply.add(amountWithDecimals);
        _balances[account] = _balances[account].add(amountWithDecimals);
        emit Transfer(address(0), account, amountWithDecimals);
    }

    // Burn tokens
    function _burn(uint256 amount) external virtual {
        uint256 amountWithDecimals = amount * 10**_decimals;
        address account = msg.sender;
        require(account != address(0), "AMZ: Burn from zero address");

        _beforeTokenTransfer(account, address(0), amountWithDecimals);
        require(_balances[account] >= amountWithDecimals, "AMZ: Burn amount exceeds balance");

        _balances[account] = _balances[account].sub(amountWithDecimals);
        _totalSupply = _totalSupply.sub(amountWithDecimals);
        emit Transfer(account, address(0), amountWithDecimals);
    }

    // Approve spending
    function _approve(address owner, address spender, uint256 amount) internal virtual {
        require(owner != address(0), "AMZ: Approve from zero address");
        require(spender != address(0), "AMZ: Approve to zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _beforeTokenTransfer(address from, address to, uint256 amount) internal virtual {}
}