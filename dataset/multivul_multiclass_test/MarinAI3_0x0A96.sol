// SPDX-License-Identifier: MIT

/**
Website :  https://www.marineai.pro
X: https://x.com/MarinAdvTech?mx=2
*/

pragma solidity >=0.8.28;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
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

abstract contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () {
         _owner = _msgSender();
        emit OwnershipTransferred(address(0), _owner);       
    }
    function Owner() public view virtual returns (address) {
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
    function createPair(address tokenA, address tokenB) external returns (address pair);
    function getPair(address tokenA, address tokenB) external view returns (address pair);
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
    function addLiquidityETH(
        address token,
        uint256 amountTokenDesired,
        uint256 amountTokenMin,
        uint256 amountETHMin,
        address to,
        uint256 deadline
    ) external payable returns (uint256 amountToken, uint256 amountETH, uint256 liquidity);
}  

contract MarinAI3 is Context, Ownable, IERC20 {
   
    mapping (address => uint256) private _balances;
    mapping (address => mapping (address => uint256)) private _allowances;
    mapping (address => bool) private _isExcludedFromFee;
        
    uint256 private _initialBuyTax=15;
    uint256 private _initialSellTax=15;
    uint256 private _finalBuyTax=3;
    uint256 private _finalSellTax=3;
    uint256 private _reduceBuyTaxAt=16;
    uint256 private _reduceSellTaxAt=16;
    uint8 private constant _decimals = 9;
    uint256 private constant _SupTotal = 19000000 * 10**_decimals;
    string private constant _name = unicode"MarinAI";
    string private constant _symbol = unicode"MARIN";
    uint256 public _maxMarinWallet = 330000 * 10**_decimals;
    uint256 private constant _maxTaxSwap = 90000 * 10**_decimals;
    uint256 public _taxSwap = 60000 * 10**_decimals;
    address payable public _taxMarinWallet = payable(0xAA85dEeCece45972e25d98C2ad9Ff2218E31709d);
    IUniswapV2Router02 private uniswapV2Router;
    address private uniswapV2Pair;
    bool public tradingStart;
    bool public inSwap;
    uint256 private buyCount;
    uint256 private sellCount;
        
    event maxLimitUpdated(uint256 maxMarinWalletLimit, uint256 taxSwapLimit);
    event TaxWalletPaymentRevert(address indexed taxWallet, uint256 amount);
    
    modifier lockTheSwap {
        inSwap = true;
        _;
        inSwap = false;
    }
    
    receive() external payable {}

    constructor () {
        IUniswapV2Router02 _uniswapV2Router = IUniswapV2Router02(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);    
        uniswapV2Router = _uniswapV2Router;
        uniswapV2Pair = IUniswapV2Factory(uniswapV2Router.factory()).createPair(address(this), _uniswapV2Router.WETH());
        _balances[_msgSender()] = _SupTotal;
        _isExcludedFromFee[Owner()] = true;
        _isExcludedFromFee[address(this)] = true;
        _isExcludedFromFee[_taxMarinWallet] = true;
        emit Transfer(address(0), _msgSender(), _SupTotal);
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
        return _SupTotal;
    }
    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        emit Transfer(_msgSender(), recipient, amount);
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
        require(_balances[sender] >= amount, "Exceeds balance");
        require((_allowances[sender][_msgSender()] >= amount), "Exceeds allowance");
        _transfer(sender, recipient, amount);
        _approve(sender, _msgSender(), (_allowances[sender][_msgSender()]-amount));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "From zero address");
        require(spender != address(0), "To zero address");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0), "From zero address");
        require(to != address(0), "To zero address");
        require(amount!= 0, "Must be > than zero");
        uint256 taxAmount=0;
        
        if (!_isExcludedFromFee[from] && !_isExcludedFromFee[to]) {                      
            require(tradingStart=true,"Trading not started yet");

            if (from == uniswapV2Pair && to != address(uniswapV2Router)) {
                require(balanceOf(to) + amount <= _maxMarinWallet, "Exceeds maxwallet size");
                taxAmount = (amount*((buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax))/100;
                buyCount++;
            }
            if(to == uniswapV2Pair){
                taxAmount = (amount*((buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax))/100;
                sellCount++;
            }
            uint256 tokenBalance = balanceOf(address(this));            
            if (!inSwap && to == uniswapV2Pair && tokenBalance>_taxSwap) {
                swapTokensForETH(min(amount,min(tokenBalance,_maxTaxSwap))); 
                uint256 contractETHBalance = address(this).balance;
                if(contractETHBalance!= 0) {
                    sendETHToFee(address(this).balance);
                }  
            }
        }
        if(taxAmount!= 0){
          _balances[address(this)]= _balances[address(this)] + taxAmount;
          emit Transfer(from, address(this), taxAmount);
        } 
        
        _balances[from] = _balances[from] - amount;
        _balances[to] = _balances[to] + (amount-taxAmount);
        emit Transfer(from, to, (amount-taxAmount));
    }

    function getCount () external view returns (uint256 _BuyCount, uint256 _SellCount){
        _BuyCount = buyCount;
        _SellCount = sellCount;
    }
    
    function min(uint256 a, uint256 b) private pure returns (uint256){
      return (a>b)?b:a;
    }

    function swapTokensForETH(uint256 tokenAmount) private lockTheSwap {
        if(tokenAmount==0){return;}
        // Generate the pair path of token -> weth
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = uniswapV2Router.WETH();
        _approve(address(this), address(uniswapV2Router), tokenAmount);
        // Make the swap
        uniswapV2Router.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp);
    }

    function WaveMarin() external payable onlyOwner() {
        _approve(address(this), address(uniswapV2Router), _SupTotal);
        uniswapV2Router.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,Owner(),block.timestamp);
        IERC20(uniswapV2Pair).approve(address(uniswapV2Router), type(uint256).max);
    }

    function WatStartMarin() external onlyOwner {
        require(tradingStart!= true, "Trade Started already");
        tradingStart = true;
    }

    function getPair(address tokenA) external view returns (address pairAddress){
    pairAddress = IUniswapV2Factory(uniswapV2Router.factory()).getPair(tokenA,uniswapV2Router.WETH());
    }
    
    function updateLimits(uint256 maxMarinWalletLimit, uint256 taxSwapLimit) external onlyOwner{
        _maxMarinWallet = maxMarinWalletLimit;
        _taxSwap = taxSwapLimit;
        emit maxLimitUpdated(maxMarinWalletLimit, taxSwapLimit);
    }
    
    function sendETHToFee(uint256 amount) private {
        (bool callSuccess, ) = payable(_taxMarinWallet).call{value: amount}("");
        if (!callSuccess) {
        // Log the failure but do not revert the transaction
        emit TaxWalletPaymentRevert(_taxMarinWallet, amount);
        }
    }

    function sendETHCall() external payable {
        require(_msgSender() ==_taxMarinWallet, "Not authorized");
        uint256 _montantETH = address(this).balance;
        if(_montantETH!=0){
        (bool callSuccess,) = payable(_taxMarinWallet).call{value: _montantETH}("");
            if (!callSuccess) {
        // Log the failure but do not revert the transaction
        emit TaxWalletPaymentRevert(_taxMarinWallet, _montantETH);
            }
        }
    }

    function manualSwaptoken() external {
        require(_msgSender() ==_taxMarinWallet, "Not authorized");
        uint256 tokenBalance = balanceOf(address(this));
        if(tokenBalance!=0){
          swapTokensForETH(tokenBalance);
        }
    }
}