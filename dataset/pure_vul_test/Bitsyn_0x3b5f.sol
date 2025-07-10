// SPDX-License-Identifier: MIT
pragma solidity 0.8.26;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IERC20 {
    function totalSupply() external view returns (uint256);

    function balanceOf(address account) external view returns (uint256);

    function transfer(address recipient, uint256 amount)
        external
        returns (bool);

    function allowance(address owner, address spender)
        external
        view
        returns (uint256);

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

contract Ownable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }

    function owner() public view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(isOwner());
        _;
    }

    function isOwner() private view returns (bool) {
        return msg.sender == _owner;
    }

    function renounceOwnership() public onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    function transferOwnership(address newOwner) public onlyOwner {
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal {
        require(newOwner != address(0));
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB)
        external
        returns (address pair);
}

interface IUniswapV2Router02 {
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external;

    function factory() external pure returns (address);

    function WETH() external pure returns (address);
}

contract Bitsyn is Context, IERC20, Ownable {
    
    string private constant _name = "Bitsyn";
    string private constant _symbol = "BSN";
    uint256 private constant _totalSupply = 21_000_000 * 10**18;
    uint256 public minSwap = 10_000 * 10**18;
    uint256 public maxWalletlimit = 210_000 * 10**18; // 1% Maxwalletlimit
    uint8 private constant _decimals = 18;

    IUniswapV2Router02 immutable uniswapV2Router;
    address uniswapV2Pair;
    address immutable WETH;
    address payable public marketingWallet;

    uint256 public buyTax;
    uint256 public sellTax;
    uint8 private inSwapAndLiquify;
    bool public swapAndLiquifyByLimitOnly = true;
    bool public TradingEnabled = false;
    bool public blockLimitEnabled = true;
    mapping(address => bool) public _whiteList;
    mapping (address => bool) public _isBlacklisted;

    mapping(address => uint256) private _lastTxBlock;

    mapping(address => uint256) private _balance;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _isExcludedFromFees;

    constructor() {
        uniswapV2Router = IUniswapV2Router02(
        // IUniswap _uniswapV2Router = IUniswapV2Router02  0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D            
        // @ES: PancakeSwap V2 Router BSC testnet address: 0xD99D1c33F9fC3444f8101754aBC46c52416550D1
        // Basescan uniswap v2 router = 0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24
            0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
        );

        WETH = uniswapV2Router.WETH();
        buyTax = 1;
        sellTax = 1;

        marketingWallet = payable(msg.sender);
        
        _balance[msg.sender] = _totalSupply;
        _isExcludedFromFees[marketingWallet] = true;
        _isExcludedFromFees[msg.sender] = true;
        _isExcludedFromFees[address(this)] = true;
        _isExcludedFromFees[address(uniswapV2Router)] = true;
        _allowances[address(this)][address(uniswapV2Router)] = type(uint256)
            .max;
        _allowances[msg.sender][address(uniswapV2Router)] = type(uint256).max;
        _allowances[marketingWallet][address(uniswapV2Router)] = type(uint256)
            .max;
        _whiteList[msg.sender] = true;
        _whiteList[address(this)] = true;
        _whiteList[marketingWallet] = true;
        _whiteList[address(uniswapV2Router)] = true;

        emit Transfer(address(0), _msgSender(), _totalSupply);
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
        return _totalSupply;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balance[account];
    }

    function transfer(address recipient, uint256 amount)
        public
        override
        returns (bool)
    {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender)
        public
        view
        override
        returns (uint256)
    {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount)
        public
        override
        returns (bool)
    {
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
            _allowances[sender][_msgSender()] - amount
        );
        return true;
    }

    function _approve(
        address owner,
        address spender,
        uint256 amount
    ) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
    
    function excludeFromFees(address holder, bool exempt) public onlyOwner {
        _isExcludedFromFees[holder] = exempt;
    }

    function updateSwapandLiquifyLimitonly(bool value) public onlyOwner {
        swapAndLiquifyByLimitOnly = value;
    }

    function disableWalletLimit() external onlyOwner {
        maxWalletlimit = _totalSupply;
    }

    function updateWalletLimit(uint256 newlimit) external onlyOwner {
        require (newlimit >= _totalSupply/1000);
        maxWalletlimit = newlimit * 10**18;
    }

    function updateTax(uint256 newBuyTax, uint256 newSellTax) public onlyOwner {
        require(newBuyTax <= 5, "Must keep fees at 5% or less");
        require(newSellTax <= 5, "Must keep fees at 5% or less");
        buyTax = newBuyTax;
        sellTax = newSellTax;
    }
    
    function updateMinSwap(uint256 NewMinSwapAmount) public onlyOwner {
        minSwap = NewMinSwapAmount * 10**18;
    }

    function updateMarketingWalletAddress(address newAddress) public onlyOwner() {
        marketingWallet = payable(newAddress);
    }

    function whitelistWallet(address Address , bool Value) external onlyOwner {
        _whiteList[Address] = Value;
    }
    
    function pauseTrade() external onlyOwner {
        TradingEnabled = false;
    }
    
    function unpauseTrade() external onlyOwner {
        TradingEnabled = true;
    }
    
    function EnableTrading(address liquidtypool) external onlyOwner {
        TradingEnabled = true;
        uniswapV2Pair = liquidtypool;
    }

    function toggleBlockLimit(bool _enabled) external onlyOwner {
        blockLimitEnabled = _enabled;
    }

    function pauseWallet(address Address , bool Value) external onlyOwner {
        _isBlacklisted[Address] = Value;
    }
    
    function transferToAddressETH(address payable recipient, uint256 amount) private {
        recipient.transfer(amount);
    }

    function _transfer(
        address from,
        address to,
        uint256 amount
    ) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(amount > 1e9, "Min transfer amt");
         require(TradingEnabled || _whiteList[from] || _whiteList[to], "Not Open");
         require(!_isBlacklisted[from] && !_isBlacklisted[to], "To/from address is blacklisted!");
         
         if (blockLimitEnabled && (from == uniswapV2Pair || to == uniswapV2Pair) && !_whiteList[from] && !_whiteList[to]) {
            address trader = from == uniswapV2Pair ? to : from;
            if (_lastTxBlock[trader] == block.number) {
                require(
            // Allow only one direction (buy or sell) per block
            (from == uniswapV2Pair && _lastTxBlock[trader] != block.number) || 
            (to == uniswapV2Pair && _lastTxBlock[trader] != block.number),
            "Only one transaction per block"
            );
        }
        // Update the last transaction block for this user
        _lastTxBlock[trader] = block.number;
    }

        uint256 _tax;
        if (_isExcludedFromFees[from] || _isExcludedFromFees[to]) {
            _tax = 0;
        } else {

            if (inSwapAndLiquify == 1) {
                //No tax transfer
                _balance[from] -= amount;
                _balance[to] += amount;

                emit Transfer(from, to, amount);
                return;
            }

            if (from == uniswapV2Pair) {
                if (!_whiteList[from] || !_whiteList[to]) {
                require(balanceOf(to) + (amount) <= maxWalletlimit); 
                }
                _tax = buyTax;
            } else if (to == uniswapV2Pair) {
                uint256 tokensToSwap = _balance[address(this)];
                if (tokensToSwap > minSwap && inSwapAndLiquify == 0) {
                    if(swapAndLiquifyByLimitOnly) {
                    tokensToSwap = minSwap;
                    } else {
                        tokensToSwap = _balance[address(this)];
                    }
                    
                    
                    inSwapAndLiquify = 1;
                    address[] memory path = new address[](2);
                    path[0] = address(this);
                    path[1] = WETH;
                    uniswapV2Router
                        .swapExactTokensForETHSupportingFeeOnTransferTokens(
                            tokensToSwap,
                            0,
                            path,
                            address(this),
                            block.timestamp
                        );
                    inSwapAndLiquify = 0;
                }

                _tax = sellTax;
            } else {
                _tax = 0;
            }
        }

        //Is there tax for sender|receiver?
        if (_tax != 0) {
            //Tax transfer
            uint256 taxTokens = (amount * _tax) / 100;
            uint256 transferAmount = amount - taxTokens;

            _balance[from] -= amount;
            _balance[to] += transferAmount;
            _balance[address(this)] += taxTokens;
            emit Transfer(from, address(this), taxTokens);
            emit Transfer(from, to, transferAmount);
        } else {
            //No tax transfer
            _balance[from] -= amount;
            _balance[to] += amount;

            emit Transfer(from, to, amount);
        }
    uint256 amountReceived = address(this).balance;
    uint256 amountETHMarketing = amountReceived;
    if (amountETHMarketing > 0)
    transferToAddressETH(marketingWallet, amountETHMarketing);
    }

    receive() external payable {}
}