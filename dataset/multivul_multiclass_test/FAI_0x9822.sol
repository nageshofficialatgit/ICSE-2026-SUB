/*
Web:       https://floss.ac
X:         https://x.com/Floss_AI
Telegram:  https://t.me/Floss_AI
Insta:     https://www.instagram.com/floss_ai/
TikTok:    https://www.tiktok.com/@flosscoin
*/
// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

abstract contract Ownable is Context {
    address private _owner;

    error OwnableUnauthorizedAccount(address account);
    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(address initialOwner) {
        if (initialOwner == address(0)) revert OwnableInvalidOwner(address(0));
        _transferOwnership(initialOwner);
    }

    modifier onlyOwner() {
        if (owner() != _msgSender()) revert OwnableUnauthorizedAccount(_msgSender());
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) revert OwnableInvalidOwner(address(0));
        _transferOwnership(newOwner);
    }

    function renounceOwnership() public virtual onlyOwner {
        address oldOwner = _owner;
        _owner = address(0);
        emit OwnershipTransferred(oldOwner, address(0));
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        uint256 c = a - b;
        return c;
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
        require(b > 0, "SafeMath: division by zero");
        uint256 c = a / b;
        return c;
    }
}

interface IUniswapV2Router {
    function factory() external pure returns (address);
    function WETH() external pure returns (address);

    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external;

    function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    )
        external
        payable
        returns (uint amountToken, uint amountETH, uint liquidity);
}

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract FAI is Context, IERC20, Ownable {
    using SafeMath for uint256;

    string public constant name     = "Floss AI";
    string public constant symbol   = "FAI";
    uint8  public constant decimals = 9;

    uint256 private constant _tTotal = 10_000_000 * 10**9; // Total supply

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    mapping(address => bool) private _isExcludedFromFee;

    address payable private _treasuryWallet;
    address payable private _devWallet;

    uint256 public _maxTxAmount        = 300_000 * 10**9; // 3%
    uint256 public _maxWalletSize      = 500_000 * 10**9; // 5%
    uint256 public _taxSwapThreshold   = 10_000 * 10**9;
    uint256 public _maxTaxSwap         = 10_000 * 10**9;

    // Launch Taxes
    uint256 private _buyTax  = 3;   // <= 5% cap
    uint256 private _sellTax = 16;  // <= 15% cap

    IUniswapV2Router private uniswapV2Router;
    address private uniswapV2Pair;

    bool private tradingOpen   = false;
    bool private inSwapProcess = false;
    bool private swapEnabled   = false;

    modifier lockSwapProcess {
        inSwapProcess = true;
        _;
        inSwapProcess = false;
    }

    event MaxTxAmountUpdated(uint256 maxTxAmount);
    event TaxesUpdated(uint256 buyTax, uint256 sellTax);

    constructor() Ownable(msg.sender) payable {
        _treasuryWallet = payable(0xaF16f4C4dB0cA09F7af8D90F9367dA968002780f);
        _devWallet      = payable(0x176e706FcDF73DE2FcB56eC7091dc3d65a1F6b33);

        _isExcludedFromFee[_treasuryWallet] = true;
        _isExcludedFromFee[_devWallet]      = true;
        _isExcludedFromFee[address(this)]   = true;
        _isExcludedFromFee[owner()]         = true;

        uint256 contractMint = _tTotal.mul(95).div(100);
        uint256 devMint      = _tTotal.sub(contractMint);

        _balances[address(this)] = contractMint;
        emit Transfer(address(0), address(this), contractMint);

        _balances[_devWallet] = devMint;
        emit Transfer(address(0), _devWallet, devMint);
    }

    function totalSupply() public pure override returns (uint256) {
        return _tTotal;
    }
    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }
    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }
    function allowance(address owner_, address spender) public view override returns (uint256) {
        return _allowances[owner_][spender];
    }
    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }
    function transferFrom(
        address sender, 
        address recipient, 
        uint256 amount
    ) public override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(
            sender,
            _msgSender(),
            _allowances[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance")
        );
        return true;
    }
    function _approve(address owner_, address spender, uint256 amount) private {
        require(owner_  != address(0), "ERC20: approve from zero address");
        require(spender != address(0), "ERC20: approve to zero address");
        _allowances[owner_][spender] = amount;
        emit Approval(owner_, spender, amount);
    }

    function setTaxes(uint256 newBuyTax, uint256 newSellTax) external onlyOwner {
        require(newBuyTax  <= 5,  "Buy tax cannot exceed 5%");
        require(newSellTax <= 15, "Sell tax cannot exceed 15%");
        _buyTax  = newBuyTax;
        _sellTax = newSellTax;
        emit TaxesUpdated(_buyTax, _sellTax);
    }

    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0) && to != address(0), "ERC20: zero address");
        require(amount > 0, "Transfer amount must be > zero");

        if (from != owner() && to != owner()) {
            require(
                tradingOpen || from == address(this) || to == address(this),
                "Trading not enabled"
            );

            if (from == uniswapV2Pair && !_isExcludedFromFee[to]) {
                require(amount <= _maxTxAmount, "Exceeds max transaction amount");
                require(_balances[to].add(amount) <= _maxWalletSize, "Exceeds max wallet size");
            }

            uint256 contractTokenBalance = _balances[address(this)];
            if (
                contractTokenBalance >= _taxSwapThreshold &&
                !inSwapProcess &&
                to == uniswapV2Pair &&
                swapEnabled
            ) {
                swapTokens(_min(amount, _min(contractTokenBalance, _maxTaxSwap)));
                uint256 ethBalance = address(this).balance;
                if (ethBalance > 0) {
                    sendETHToFee(ethBalance);
                }
            }
        }

        uint256 taxAmount = 0;
        if (!_isExcludedFromFee[from] && !_isExcludedFromFee[to]) {
            if (from == uniswapV2Pair) {
          
                taxAmount = amount.mul(_buyTax).div(100);
            } else if (to == uniswapV2Pair) {

                taxAmount = amount.mul(_sellTax).div(100);
            }
        }

        _balances[from] = _balances[from].sub(amount, "ERC20: transfer exceeds balance");
        _balances[to]   = _balances[to].add(amount.sub(taxAmount));

        if (taxAmount > 0) {
            _balances[address(this)] = _balances[address(this)].add(taxAmount);
            emit Transfer(from, address(this), taxAmount);
        }

        emit Transfer(from, to, amount.sub(taxAmount));
    }

    function _min(uint256 a, uint256 b) private pure returns (uint256) {
        return a > b ? b : a;
    }

    function swapTokens(uint256 tokenAmount) private lockSwapProcess {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = uniswapV2Router.WETH();

        _approve(address(this), address(uniswapV2Router), tokenAmount);
        uniswapV2Router.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function sendETHToFee(uint256 amount) private {
        uint256 treasuryShare = amount.mul(50).div(100);
        uint256 devShare      = amount.sub(treasuryShare);
        _treasuryWallet.transfer(treasuryShare);
        _devWallet.transfer(devShare);
    }

    function addLiquidity() external onlyOwner {
        require(address(uniswapV2Router) == address(0), "Liquidity already added");

        uniswapV2Router = IUniswapV2Router(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);

        _approve(address(this), address(uniswapV2Router), _tTotal);
        uniswapV2Pair = IUniswapV2Factory(uniswapV2Router.factory())
            .createPair(address(this), uniswapV2Router.WETH());

        uniswapV2Router.addLiquidityETH{value: address(this).balance}(
            address(this),
            _balances[address(this)],
            0,
            0,
            owner(),
            block.timestamp
        );

        IERC20(uniswapV2Pair).approve(address(uniswapV2Router), type(uint256).max);
    }

    function openTrading() external onlyOwner {
        require(!tradingOpen, "Trading already open");
        tradingOpen = true;
        swapEnabled = true;
    }

    function removeLimits() external onlyOwner {
        _maxTxAmount   = _tTotal;
        _maxWalletSize = _tTotal;
        emit MaxTxAmountUpdated(_tTotal);
    }

    function rescueETH() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }

    function rescueTokens(address tokenAddress, uint256 amount) external onlyOwner {
        require(tokenAddress != address(this), "Cannot rescue native token");
        IERC20(tokenAddress).transfer(owner(), amount);
    }

    receive() external payable {}
    fallback() external payable {}
}