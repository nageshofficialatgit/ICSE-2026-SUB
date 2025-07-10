// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// --- START OF FLATTENED IMPORTS ---

// OpenZeppelin Ownable
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

abstract contract Ownable is Context {
    address private _owner;

    error OwnableUnauthorizedAccount(address account);
    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

// Minimal IERC20
interface IERC20 {
    function totalSupply() external view returns (uint);
    function balanceOf(address account) external view returns (uint);
    function transfer(address recipient, uint amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint);
    function approve(address spender, uint amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint amount) external returns (bool);
}

// Uniswap Interfaces
interface IUniswapV2Router01 {
    function factory() external pure returns (address);
    function WETH() external pure returns (address);

    function swapExactETHForTokens(uint amountOutMin, address[] calldata path, address to, uint deadline)
        external
        payable
        returns (uint[] memory amounts);

    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;

    function swapExactTokensForTokensSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;
}

interface IUniswapV2Router02 is IUniswapV2Router01 {}

interface IUniswapV2Factory {
    function getPair(address tokenA, address tokenB) external view returns (address pair);
}

interface IUniswapV2Pair {
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
    function token0() external view returns (address);
    function token1() external view returns (address);
}

// --- END OF FLATTENED IMPORTS ---

contract AutoSniperUpgraded is Ownable {
    IUniswapV2Router02 public uniswapRouter;
    address public tokenToTrade;
    address public WETH;
    address public USDC;
    uint256 public priceThreshold;
    uint256 public minProfit;
    bool public tradingEnabled = true;
    uint256 public withdrawTimeLock;
    mapping(address => bool) private authorizedUsers;
    bool public useETHBase = true;

    event TradeExecuted(uint256 amountIn, uint256 amountOut);
    event TokensSold(uint256 tokenAmount, uint256 ethReceived);
    event TradingStopped();
    event TradingStarted();
    event FundsWithdrawn(address indexed owner, uint256 amount);
    event SettingsUpdated(uint256 newThreshold, uint256 newProfit);
    event BaseTokenSwitched(bool useETH);

    constructor(address _router, address _tokenToTrade, address _usdc, uint256 _priceThreshold, uint256 _minProfit)
        Ownable(msg.sender)
    {
        uniswapRouter = IUniswapV2Router02(_router);
        WETH = uniswapRouter.WETH();
        USDC = _usdc;
        tokenToTrade = _tokenToTrade;
        priceThreshold = _priceThreshold;
        minProfit = _minProfit;
        withdrawTimeLock = block.timestamp + 1 days;
    }

    modifier onlyAuthorized() {
        require(authorizedUsers[msg.sender] || msg.sender == owner(), "Not authorized");
        _;
    }

    function setAuthorizedUser(address _user, bool _status) external onlyOwner {
        authorizedUsers[_user] = _status;
    }

    function setTradeSettings(uint256 _newThreshold, uint256 _newProfit) external onlyOwner {
        priceThreshold = _newThreshold;
        minProfit = _newProfit;
        emit SettingsUpdated(_newThreshold, _newProfit);
    }

    function switchBaseToken(bool _useETH) external onlyOwner {
        useETHBase = _useETH;
        emit BaseTokenSwitched(_useETH);
    }

    function getTokenPrice() public view returns (uint256 price) {
        address baseToken = useETHBase ? WETH : USDC;
        address pair = IUniswapV2Factory(uniswapRouter.factory()).getPair(tokenToTrade, baseToken);
        require(pair != address(0), "Pair does not exist");
        (uint reserve0, uint reserve1,) = IUniswapV2Pair(pair).getReserves();

        if (IUniswapV2Pair(pair).token0() == tokenToTrade) {
            price = reserve1 == 0 ? 0 : (reserve0 * 1e18) / reserve1;
        } else {
            price = reserve0 == 0 ? 0 : (reserve1 * 1e18) / reserve0;
        }
    }

    function trade(uint256 amountIn) external onlyAuthorized {
        require(tradingEnabled, "Trading is disabled");
        uint256 priceNow = getTokenPrice();
        require(priceNow >= priceThreshold, "Price not met");

        address baseToken = useETHBase ? WETH : USDC;
        address[] memory path = new address[](2);
        path[0] = baseToken;
        path[1] = tokenToTrade;

        uint256 deadline = block.timestamp + 120;

        if (useETHBase) {
            uint256[] memory amounts = uniswapRouter.swapExactETHForTokens{value: amountIn}(
                minProfit, path, address(this), deadline
            );
            emit TradeExecuted(amountIn, amounts[1]);
        } else {
            IERC20(USDC).approve(address(uniswapRouter), amountIn);
            uint256[] memory amounts = uniswapRouter.swapExactTokensForTokens(
                amountIn, minProfit, path, address(this), deadline
            );
            emit TradeExecuted(amountIn, amounts[1]);
        }
    }

    function sellTokens(uint256 tokenAmount) external onlyAuthorized {
        require(tradingEnabled, "Trading disabled");

        address baseToken = useETHBase ? WETH : USDC;
        address[] memory path = new address[](2);
        path[0] = tokenToTrade;
        path[1] = baseToken;

        IERC20(tokenToTrade).approve(address(uniswapRouter), tokenAmount);
        uint256 balanceBefore = useETHBase ? address(this).balance : IERC20(USDC).balanceOf(address(this));

        uint256 deadline = block.timestamp + 120;

        if (useETHBase) {
            uniswapRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
                tokenAmount, 0, path, address(this), deadline
            );
            uint256 balanceAfter = address(this).balance;
            emit TokensSold(tokenAmount, balanceAfter - balanceBefore);
        } else {
            uniswapRouter.swapExactTokensForTokens(
                tokenAmount, 0, path, address(this), deadline
            );
            uint256 balanceAfter = IERC20(USDC).balanceOf(address(this));
            emit TokensSold(tokenAmount, balanceAfter - balanceBefore);
        }
    }

    function withdrawFunds() external onlyOwner {
        require(block.timestamp >= withdrawTimeLock, "Time-lock active");
        if (useETHBase) {
            uint256 balance = address(this).balance;
            payable(owner()).transfer(balance);
            emit FundsWithdrawn(owner(), balance);
        } else {
            uint256 usdcBalance = IERC20(USDC).balanceOf(address(this));
            IERC20(USDC).transfer(owner(), usdcBalance);
            emit FundsWithdrawn(owner(), usdcBalance);
        }
    }

    function toggleTrading(bool _status) external onlyOwner {
        tradingEnabled = _status;
        if (_status) {
            emit TradingStarted();
        } else {
            emit TradingStopped();
        }
    }

    receive() external payable {}
}