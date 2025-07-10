// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

interface IERC20Extended {
    function totalSupply() external view returns (uint256);

    function decimals() external view returns (uint8);

    function symbol() external view returns (string memory);

    function name() external view returns (string memory);

    function balanceOf(address account) external view returns (uint256);

    function transfer(address recipient, uint256 amount)
        external
        returns (bool);

    function allowance(address _owner, address spender)
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

// Dex Factory contract interface
interface IDexFactory {
    function createPair(address tokenA, address tokenB)
        external
        returns (address pair);
}

// Dex Router contract interface
interface IDexRouter {
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

    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external;
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address payable) {
        return payable(msg.sender);
    }

    function _msgData() internal view virtual returns (bytes memory) {
        this; // silence state mutability warning without generating bytecode - see https://github.com/ethereum/solidity/issues/2691
        return msg.data;
    }
}

contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    constructor() {
        _owner = _msgSender();
        emit OwnershipTransferred(address(0), _owner);
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
        _owner = payable(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(
            newOwner != address(0),
            "Ownable: new owner is the zero address"
        );
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

// main contract
contract The_X_Doge is IERC20Extended, Ownable {
    using SafeMath for uint256;

    string private constant _name = "The X Doge";
    string private constant _symbol = "XDOGE";
    uint8 private constant _decimals = 18;
    uint256 private constant _totalSupply = 100_000_000 * 10**_decimals;

    IDexRouter public router;
    address public pair;
    address public autoLpReceiver;
    address public marketingReceiver;
    address public xBankFundsReceiver;
    address public nftPoolReceiver;

    uint256 _marketingSellFee = 2_00;
    uint256 _liquiditySellFee = 2_00;
    uint256 _xBankSellFee = 2_00;
    uint256 _nftHoldersSellFee = 2_00;

    uint256 _marketingFeeCount;
    uint256 _liquidityFeeCount;
    uint256 _xBankFeeCount;
    uint256 _nftHoldersFeeCount;
    uint256 public totalSellFee = 8_00;
    uint256 public feeDenominator = 100_00;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) public isFeeExempt;
    mapping(address => bool) public isPair;

    bool public swapEnabled;
    bool public trading; // once enable can't be disable afterwards
    uint256 public swapThreshold = _totalSupply / 2000;
    uint256 public launchedAt;
    bool inSwap;
    modifier swapping() {
        inSwap = true;
        _;
        inSwap = false;
    }

    event AutoLiquify(uint256 amountBNB, uint256 amountBOG);

    constructor() {
        address router_ = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
        autoLpReceiver = msg.sender;
        marketingReceiver = 0x628e27e8e244e3C57DCC68C4e5EA89960422e101;
        xBankFundsReceiver = 0xcdaf2D72BD967879372DF5E6193a3B7c8fe6895D;
        nftPoolReceiver = 0xc0C5058136560A8e83F5C001dEbd592B1174980E;

        router = IDexRouter(router_);
        pair = IDexFactory(router.factory()).createPair(
            address(this),
            router.WETH()
        );
        isPair[pair] = true;

        isFeeExempt[msg.sender] = true;
        isFeeExempt[autoLpReceiver] = true;
        isFeeExempt[marketingReceiver] = true;
        isFeeExempt[xBankFundsReceiver] = true;
        isFeeExempt[nftPoolReceiver] = true;

        _balances[msg.sender] = _totalSupply;
        emit Transfer(address(0), msg.sender, _totalSupply);
    }

    receive() external payable {}

    function totalSupply() external pure override returns (uint256) {
        return _totalSupply;
    }

    function decimals() external pure override returns (uint8) {
        return _decimals;
    }

    function symbol() external pure override returns (string memory) {
        return _symbol;
    }

    function name() external pure override returns (string memory) {
        return _name;
    }

    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    function allowance(address holder, address spender)
        external
        view
        override
        returns (uint256)
    {
        return _allowances[holder][spender];
    }

    function approve(address spender, uint256 amount)
        public
        override
        returns (bool)
    {
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function approveMax(address spender) external returns (bool) {
        return approve(spender, _totalSupply);
    }

    function transfer(address recipient, uint256 amount)
        external
        override
        returns (bool)
    {
        return _transferFrom(msg.sender, recipient, amount);
    }

    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external override returns (bool) {
        if (_allowances[sender][msg.sender] != _totalSupply) {
            _allowances[sender][msg.sender] = _allowances[sender][msg.sender]
                .sub(amount, "Insufficient Allowance");
        }

        return _transferFrom(sender, recipient, amount);
    }

    function _transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) internal returns (bool) {
        if (!isFeeExempt[sender] && !isFeeExempt[recipient]) {
            require(trading, "Trading not enabled yet");
        }

        if (inSwap) {
            return _basicTransfer(sender, recipient, amount);
        }

        if (shouldSwapBack()) {
            swapBack();
        }

        _balances[sender] = _balances[sender].sub(
            amount,
            "Insufficient Balance"
        );

        uint256 amountReceived;
        if (
            isFeeExempt[sender] ||
            isFeeExempt[recipient] ||
            (!isPair[sender] && !isPair[recipient])
        ) {
            amountReceived = amount;
        } else {
            uint256 feeAmount;
            if (isPair[sender]) {
                amountReceived = amount;
            } else if (isPair[recipient]) {
                feeAmount = amount.mul(totalSellFee).div(feeDenominator);
                amountReceived = amount.sub(feeAmount);
                takeFee(sender, feeAmount);
                setSellAccFee(amount);
            }
        }

        _balances[recipient] = _balances[recipient].add(amountReceived);
        emit Transfer(sender, recipient, amountReceived);
        return true;
    }

    function _basicTransfer(
        address sender,
        address recipient,
        uint256 amount
    ) internal returns (bool) {
        _balances[sender] = _balances[sender].sub(
            amount,
            "Insufficient Balance"
        );
        _balances[recipient] = _balances[recipient].add(amount);
        emit Transfer(sender, recipient, amount);
        return true;
    }

    function takeFee(address sender, uint256 feeAmount) internal {
        _balances[address(this)] = _balances[address(this)].add(feeAmount);
        emit Transfer(sender, address(this), feeAmount);
    }

    function setSellAccFee(uint256 _amount) internal {
        _liquidityFeeCount += _amount.mul(_liquiditySellFee).div(
            feeDenominator
        );
        _marketingFeeCount += _amount.mul(_marketingSellFee).div(
            feeDenominator
        );
        _xBankFeeCount += _amount.mul(_xBankSellFee).div(feeDenominator);
        _nftHoldersFeeCount += _amount.mul(_nftHoldersSellFee).div(
            feeDenominator
        );
    }

    function shouldSwapBack() internal view returns (bool) {
        return
            !isPair[msg.sender] &&
            !inSwap &&
            swapEnabled &&
            _balances[address(this)] >= swapThreshold;
    }

    function swapBack() internal swapping {
        uint256 totalFee = _liquidityFeeCount
            .add(_marketingFeeCount)
            .add(_xBankFeeCount)
            .add(_nftHoldersFeeCount);

        uint256 amountToLiquify = swapThreshold
            .mul(_liquidityFeeCount)
            .div(totalFee)
            .div(2);

        uint256 amountToSwap = swapThreshold.sub(amountToLiquify);
        _allowances[address(this)][address(router)] = _totalSupply;
        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = router.WETH();
        uint256 balanceBefore = address(this).balance;

        router.swapExactTokensForETHSupportingFeeOnTransferTokens(
            amountToSwap,
            0,
            path,
            address(this),
            block.timestamp
        );

        uint256 amountBNB = address(this).balance.sub(balanceBefore);

        uint256 totalBNBFee = totalFee.sub(_liquidityFeeCount.div(2));

        uint256 amountBNBLiquidity = amountBNB
            .mul(_liquidityFeeCount)
            .div(totalBNBFee)
            .div(2);
        uint256 amountBNBMarketing = amountBNB.mul(_marketingFeeCount).div(
            totalBNBFee
        );
        uint256 amountBNBEcological = amountBNB.mul(_xBankFeeCount).div(
            totalBNBFee
        );
        uint256 amountBNBPrizePool = amountBNB.mul(_nftHoldersFeeCount).div(
            totalBNBFee
        );

        if (amountBNBMarketing > 0) {
            payable(marketingReceiver).transfer(amountBNBMarketing);
        }
        if (amountBNBEcological > 0) {
            payable(xBankFundsReceiver).transfer(amountBNBEcological);
        }
        if (amountBNBPrizePool > 0) {
            payable(nftPoolReceiver).transfer(amountBNBPrizePool);
        }

        if (amountToLiquify > 0) {
            router.addLiquidityETH{value: amountBNBLiquidity}(
                address(this),
                amountToLiquify,
                0,
                0,
                autoLpReceiver,
                block.timestamp
            );
            emit AutoLiquify(amountBNBLiquidity, amountToLiquify);
        }
        _liquidityFeeCount = 0;
        _marketingFeeCount = 0;
        _xBankFeeCount = 0;
        _nftHoldersFeeCount = 0;
    }

    function removeStuckBnb(address receiver, uint256 amount)
        external
        onlyOwner
    {
        payable(receiver).transfer(amount);
    }

    function removeStuckTokens(address receiver, uint256 amount)
        external
        onlyOwner
    {
        _transferFrom(address(this), receiver, amount);
    }

    function setIsFeeExempt(address holder, bool exempt) external onlyOwner {
        isFeeExempt[holder] = exempt;
    }

    function setSellFees(
        uint256 _liquidityFee,
        uint256 _marketingFee,
        uint256 _ecologicalFee,
        uint256 _prizePoolFee
    ) public onlyOwner {
        _liquiditySellFee = _liquidityFee;
        _marketingSellFee = _marketingFee;
        _xBankSellFee = _ecologicalFee;
        _nftHoldersSellFee = _prizePoolFee;
        totalSellFee = _liquidityFee.add(_marketingFee).add(_ecologicalFee).add(
            _prizePoolFee
        );
        require(
            totalSellFee <= feeDenominator.mul(15).div(100),
            "Can't be greater than 15%"
        );
    }

    function setFeeReceivers(
        address _autoLpReceiver,
        address _marketingReceiver,
        address _xBankFundsReceiver,
        address _nftPoolReceiver
    ) external onlyOwner {
        autoLpReceiver = _autoLpReceiver;
        marketingReceiver = _marketingReceiver;
        xBankFundsReceiver = _xBankFundsReceiver;
        nftPoolReceiver = _nftPoolReceiver;
    }

    function setSwapBackSettings(bool _enabled, uint256 _amount)
        external
        onlyOwner
    {
        require(swapThreshold > 0);
        swapEnabled = _enabled;
        swapThreshold = _amount;
    }

    function enableTrading() external onlyOwner {
        require(!trading, "Already enabled");
        trading = true;
        swapEnabled = true;
        launchedAt = block.timestamp;
    }

    function addPair(address _pair) public onlyOwner {
        isPair[_pair] = true;
    }

    function removePair(address _pair) public onlyOwner {
        isPair[_pair] = false;
    }
}

library SafeMath {
    function tryAdd(uint256 a, uint256 b)
        internal
        pure
        returns (bool, uint256)
    {
        unchecked {
            uint256 c = a + b;
            if (c < a) return (false, 0);
            return (true, c);
        }
    }

    function trySub(uint256 a, uint256 b)
        internal
        pure
        returns (bool, uint256)
    {
        unchecked {
            if (b > a) return (false, 0);
            return (true, a - b);
        }
    }

    function tryMul(uint256 a, uint256 b)
        internal
        pure
        returns (bool, uint256)
    {
        unchecked {
            // Gas optimization: this is cheaper than requiring 'a' not being zero, but the
            // benefit is lost if 'b' is also tested.
            // See: https://github.com/OpenZeppelin/openzeppelin-contracts/pull/522
            if (a == 0) return (true, 0);
            uint256 c = a * b;
            if (c / a != b) return (false, 0);
            return (true, c);
        }
    }

    function tryDiv(uint256 a, uint256 b)
        internal
        pure
        returns (bool, uint256)
    {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a / b);
        }
    }

    function tryMod(uint256 a, uint256 b)
        internal
        pure
        returns (bool, uint256)
    {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a % b);
        }
    }

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return a - b;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        return a * b;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return a / b;
    }

    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return a % b;
    }

    function sub(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        unchecked {
            require(b <= a, errorMessage);
            return a - b;
        }
    }

    function div(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a / b;
        }
    }

    function mod(
        uint256 a,
        uint256 b,
        string memory errorMessage
    ) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a % b;
        }
    }
}