// SPDX-License-Identifier: MIT
/**
                                                Termi.Fun
                    Your all-in-one Web3 toolbox for on-chain data analyzation, trading, and alpha.
                                            https://termi.fun
                                         https://x.com/termifun
                                          https://t.me/termifun
                                        https://app.termi.fun
                                        https://docs.termi.fun

Original contract by: BG Team
**/

pragma solidity 0.8.21;
pragma experimental ABIEncoderV2;

abstract contract Ownable {
    address private _owner;

    constructor() {
        _owner = msg.sender;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == msg.sender, "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        _owner = address(0);
    }
}

library SafeERC20 {
    function safeTransfer(address token, address to, uint256 value) internal {
        (bool success, bytes memory data) = token.call(
            abi.encodeWithSelector(IERC20.transfer.selector, to, value)
        );
        require(
            success && (data.length == 0 || abi.decode(data, (bool))),
            "TransferHelper: INTERNAL TRANSFER_FAILED"
        );
    }
}

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external;
}

interface IUniswapV2Factory {
    function createPair(
        address tokenA,
        address tokenB
    ) external returns (address pair);
}

interface IUniswapV2Router02 {
    function factory() external pure returns (address);

    function WETH() external pure returns (address);

    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external;

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
        returns (uint256 amountToken, uint256 amountETH, uint256 liquidity);
}

contract Termi is Ownable {
    string private constant _name = unicode"Termi Fun";
    string private constant _symbol = unicode"TERMI";
    uint256 private constant _totalSupply = 1_000_000_000 * 1e18;

    uint256 public maxTransactionAmount = 5_000_000 * 1e18;
    uint256 public maxWallet = 5_000_000 * 1e18;
    uint256 public swapTokensAtAmount = (_totalSupply * 5) / 10000;

    address private teamWallet = 0x1D0f4771814Fb28657339C9E6800F0cd361dB2b6;
    address private treasuryWallet =
        0xfACe18E33852740de03Bb9a8Be7d0E2bc51046F8;
    address private revWallet = 0x42114860206f384ae51a4A1A435EA3EaC6FEb473;
    address private constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;

    uint8 public buyTotalFees = 100;
    uint8 public sellTotalFees = 100;

    uint8 public revFee = 20;
    uint8 public teamFee = 80;

    bool private swapping;
    bool public limitsInEffect = true;
    bool private launched;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private _isExcludedFromFees;
    mapping(address => bool) private _isExcludedMaxTransactionAmount;
    mapping(address => bool) private automatedMarketMakerPairs;

    event SwapAndLiquify(
        uint256 tokensSwapped,
        uint256 teamETH,
        uint256 revETH
    );
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );

    IUniswapV2Router02 public constant uniswapV2Router =
        IUniswapV2Router02(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);
    address public immutable uniswapV2Pair;

    constructor() {
        uniswapV2Pair = IUniswapV2Factory(uniswapV2Router.factory()).createPair(
            address(this),
            WETH
        );
        automatedMarketMakerPairs[uniswapV2Pair] = true;

        setExcludedFromFees(owner(), true);
        setExcludedFromFees(address(this), true);
        setExcludedFromFees(address(0xdead), true);
        setExcludedFromFees(teamWallet, true);
        setExcludedFromFees(treasuryWallet, true);
        setExcludedFromFees(revWallet, true);
        setExcludedFromFees(0xE2fE530C047f2d85298b07D9333C05737f1435fB, true); 

        setExcludedFromMaxTransaction(owner(), true);
        setExcludedFromMaxTransaction(address(uniswapV2Router), true);
        setExcludedFromMaxTransaction(address(this), true);
        setExcludedFromMaxTransaction(address(0xdead), true);
        setExcludedFromMaxTransaction(address(uniswapV2Pair), true);
        setExcludedFromMaxTransaction(teamWallet, true);
        setExcludedFromMaxTransaction(revWallet, true);
        setExcludedFromMaxTransaction(treasuryWallet, true);
        setExcludedFromMaxTransaction(
            0xE2fE530C047f2d85298b07D9333C05737f1435fB,
            true
        ); 

        _balances[address(this)] = 800_000_000 * 1e18;
        emit Transfer(address(0), address(this), _balances[address(this)]); 
        _balances[msg.sender] = 100_000_000 * 1e18;
        emit Transfer(address(0), msg.sender, _balances[msg.sender]); 
        _balances[treasuryWallet] = 100_000_000 * 1e18;
        emit Transfer(address(0), treasuryWallet, _balances[treasuryWallet]); 

        _approve(address(this), address(uniswapV2Router), type(uint256).max);
    }

    receive() external payable {}

    function name() public pure returns (string memory) {
        return _name;
    }

    function symbol() public pure returns (string memory) {
        return _symbol;
    }

    function decimals() public pure returns (uint8) {
        return 18;
    }

    function totalSupply() public pure returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function allowance(
        address owner,
        address spender
    ) public view returns (uint256) {
        return _allowances[owner][spender];
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    function transfer(
        address recipient,
        uint256 amount
    ) external returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }

    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool) {
        uint256 currentAllowance = _allowances[sender][msg.sender];
        if (currentAllowance != type(uint256).max) {
            require(
                currentAllowance >= amount,
                "ERC20: transfer amount exceeds allowance"
            );
            unchecked {
                _approve(sender, msg.sender, currentAllowance - amount);
            }
        }

        _transfer(sender, recipient, amount);

        return true;
    }

    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(amount > 0, "Transfer amount must be greater than zero");

        if (
            !launched &&
            (from != owner() && from != address(this) && to != owner())
        ) {
            revert("Trading not enabled");
        }

        if (limitsInEffect) {
            if (
                from != owner() &&
                to != owner() &&
                to != address(0) &&
                to != address(0xdead) &&
                !swapping
            ) {
                if (
                    automatedMarketMakerPairs[from] &&
                    !_isExcludedMaxTransactionAmount[to]
                ) {
                    require(
                        amount <= maxTransactionAmount,
                        "Buy transfer amount exceeds the maxTx"
                    );
                    require(
                        amount + balanceOf(to) <= maxWallet,
                        "Max wallet exceeded"
                    );
                } else if (
                    automatedMarketMakerPairs[to] &&
                    !_isExcludedMaxTransactionAmount[from]
                ) {
                    require(
                        amount <= maxTransactionAmount,
                        "Sell transfer amount exceeds the maxTx"
                    );
                } else if (!_isExcludedMaxTransactionAmount[to]) {
                    require(
                        amount + balanceOf(to) <= maxWallet,
                        "Max wallet exceeded"
                    );
                }
            }
        }

        bool canSwap = balanceOf(address(this)) >= swapTokensAtAmount;

        if (
            canSwap &&
            !swapping &&
            !automatedMarketMakerPairs[from] &&
            !_isExcludedFromFees[from] &&
            !_isExcludedFromFees[to]
        ) {
            swapping = true;
            swapBack();
            swapping = false;
        }

        bool takeFee = !swapping;

        if (_isExcludedFromFees[from] || _isExcludedFromFees[to]) {
            takeFee = false;
        }

        uint256 senderBalance = _balances[from];
        require(
            senderBalance >= amount,
            "ERC20: transfer amount exceeds balance"
        );

        uint256 fees = 0;
        if (takeFee) {
            if (automatedMarketMakerPairs[to] && sellTotalFees > 0) {
                fees = (amount * sellTotalFees) / 1000;
            } else if (automatedMarketMakerPairs[from] && buyTotalFees > 0) {
                fees = (amount * buyTotalFees) / 1000;
            }

            if (fees > 0) {
                unchecked {
                    amount = amount - fees;
                    _balances[from] -= fees;
                    _balances[address(this)] += fees;
                }
                emit Transfer(from, address(this), fees);
            }
        }
        unchecked {
            _balances[from] -= amount;
            _balances[to] += amount;
        }
        emit Transfer(from, to, amount);
    }

    /**
     * @notice Removes all transaction and wallet limits
     * @dev Only callable by the contract owner
     * @custom:security This is irreversible, use with caution
     */
    function removeLimits() external onlyOwner {
        limitsInEffect = false;
    }

    /**
     * @notice Sets the distribution percentages for rev and team fees
     * @dev Only callable by the contract owner
     * @param _RevFee Percentage for rev wallet (out of 100)
     * @param _teamFee Percentage for team wallet (out of 100)
     * @custom:security Requires total to be exactly 100%
     */
    function setDistributionFees(
        uint8 _RevFee,
        uint8 _teamFee
    ) external onlyOwner {
        revFee = _RevFee;
        teamFee = _teamFee;
        require(
            (revFee + teamFee) == 100,
            "Distribution have to be equal to 100%"
        );
    }

    /**
     * @notice Sets the buy and sell fees for the token
     * @dev Only callable by the contract owner
     * @param _buyTotalFees New buy fee (in basis points, e.g., 10 = 1%)
     * @param _sellTotalFees New sell fee (in basis points, e.g., 10 = 1%)
     * @custom:security Fees are capped at 3% (300 basis points) for both buy and sell
     */
    function setFees(
        uint8 _buyTotalFees,
        uint8 _sellTotalFees
    ) external onlyOwner {
        require(
            _buyTotalFees <= 30,
            "Buy fees must be less than or equal to 3%"
        );
        require(
            _sellTotalFees <= 30,
            "Sell fees must be less than or equal to 3%"
        );
        buyTotalFees = _buyTotalFees;
        sellTotalFees = _sellTotalFees;
    }

    /**
     * @notice Excludes or includes an address from paying fees
     * @dev Only callable by the contract owner
     * @param account Address to be excluded or included
     * @param excluded True to exclude, false to include
     */
    function setExcludedFromFees(
        address account,
        bool excluded
    ) public onlyOwner {
        _isExcludedFromFees[account] = excluded;
    }

    /**
     * @notice Excludes or includes an address from max transaction limit
     * @dev Only callable by the contract owner
     * @param account Address to be excluded or included
     * @param excluded True to exclude, false to include
     */
    function setExcludedFromMaxTransaction(
        address account,
        bool excluded
    ) public onlyOwner {
        _isExcludedMaxTransactionAmount[account] = excluded;
    }

    /**
     * @notice Enables trading for the token
     * @dev Only callable by the contract owner, can only be called once
     */
    function openTrade() external onlyOwner {
        require(!launched, "Already launched");
        launched = true;
    }

    /**
     * @notice Adds initial liquidity to the Uniswap pair
     * @dev Only callable by the contract owner, can only be called once
     * @custom:security Sends liquidity tokens to the teamWallet
     */
    function unleashTermi() external payable onlyOwner {
        require(!launched, "Already launched");
        uniswapV2Router.addLiquidityETH{value: msg.value}(
            address(this),
            _balances[address(this)],
            0,
            0,
            teamWallet,
            block.timestamp
        );
    }

    /**
     * @notice Sets or unsets an address as an automated market maker pair
     * @dev Only callable by the contract owner
     * @param pair Address of the pair to be set or unset
     * @param value True to set as AMM pair, false to unset
     * @custom:security Cannot unset the main Uniswap pair
     */
    function setAutomatedMarketMakerPair(
        address pair,
        bool value
    ) external onlyOwner {
        require(pair != uniswapV2Pair, "The pair cannot be removed");
        automatedMarketMakerPairs[pair] = value;
    }

    /**
     * @notice Sets the amount of tokens to swap and liquify
     * @dev Only callable by the contract owner
     * @param newSwapAmount New swap amount (in tokens)
     * @custom:security Limited between 0.001% and 0.5% of total supply
     */
    function setSwapAtAmount(uint256 newSwapAmount) external onlyOwner {
        require(
            newSwapAmount >= (totalSupply() * 1) / 100000,
            "Swap amount cannot be lower than 0.001% of the supply"
        );
        require(
            newSwapAmount <= (totalSupply() * 5) / 1000,
            "Swap amount cannot be higher than 0.5% of the supply"
        );
        swapTokensAtAmount = newSwapAmount;
    }

    /**
     * @notice Sets the maximum transaction amount
     * @dev Only callable by the contract owner
     * @param newMaxTx New maximum transaction amount (in tokens)
     * @custom:security Cannot be set lower than 0.1% of total supply
     */
    function setMaxTxnAmount(uint256 newMaxTx) external onlyOwner {
        require(
            newMaxTx >= ((totalSupply() * 1) / 1000) / 1e18,
            "Cannot set max transaction lower than 0.1%"
        );
        maxTransactionAmount = newMaxTx * (10 ** 18);
    }

    /**
     * @notice Sets the maximum wallet amount
     * @dev Only callable by the contract owner
     * @param newMaxWallet New maximum wallet amount (in tokens)
     * @custom:security Cannot be set lower than 1% of total supply
     */
    function setMaxWalletAmount(uint256 newMaxWallet) external onlyOwner {
        require(
            newMaxWallet >= ((totalSupply() * 1) / 100) / 1e18,
            "Cannot set max wallet lower than 1%"
        );
        maxWallet = newMaxWallet * (10 ** 18);
    }

    /**
     * @notice Updates the rev wallet address
     * @dev Only callable by the contract owner
     * @param newAddress New rev wallet address
     */
    function updateRevWallet(address newAddress) external onlyOwner {
        require(newAddress != address(0), "Address cannot be zero");
        revWallet = newAddress;
    }

    /**
     * @notice Updates the team wallet address
     * @dev Only callable by the contract owner
     * @param newAddress New team wallet address
     */
    function updateTeamWallet(address newAddress) external onlyOwner {
        require(newAddress != address(0), "Address cannot be zero");
        teamWallet = newAddress;
    }

    /**
     * @notice Updates the ecosystem wallet address
     * @dev Only callable by the contract owner
     * @param newAddress New ecosystem wallet address
     */
    function updatetreasuryWallet(address newAddress) external onlyOwner {
        require(newAddress != address(0), "Address cannot be zero");
        treasuryWallet = newAddress;
    }

    /**
     * @notice Checks if an address is excluded from fees
     * @dev Public view function
     * @param account Address to check
     * @return True if excluded, false otherwise
     */
    function excludedFromFee(address account) public view returns (bool) {
        return _isExcludedFromFees[account];
    }

    /**
     * @notice Withdraws stuck tokens from the contract
     * @dev Only callable by the contract owner
     * @param token Address of the token to withdraw
     * @param to Address to send the withdrawn tokens
     * @custom:security Use with caution to avoid withdrawing essential contract tokens
     */
    function withdrawStuckToken(address token, address to) external onlyOwner {
        uint256 _contractBalance = IERC20(token).balanceOf(address(this));
        SafeERC20.safeTransfer(token, to, _contractBalance); // Use safeTransfer
    }

    /**
     * @notice Withdraws stuck ETH from the contract
     * @dev Only callable by the contract owner
     * @param addr Address to send the withdrawn ETH
     * @custom:security Use with caution to avoid withdrawing essential contract ETH
     */
    function withdrawStuckETH(address addr) external onlyOwner {
        require(addr != address(0), "Invalid address");

        (bool success, ) = addr.call{value: address(this).balance}("");
        require(success, "Withdrawal failed");
    }

    /**
     * @notice Swaps accumulated tokens for ETH and distributes to wallets
     * @dev Internal function, called automatically during transfers when threshold is met
     * @custom:security Ensures no reentrancy by using a 'swapping' flag
     */
    function swapBack() private {
        uint256 swapThreshold = swapTokensAtAmount;
        bool success;

        if (balanceOf(address(this)) > swapTokensAtAmount * 20) {
            swapThreshold = swapTokensAtAmount * 20;
        }

        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = WETH;

        uniswapV2Router.swapExactTokensForETHSupportingFeeOnTransferTokens(
            swapThreshold,
            0,
            path,
            address(this),
            block.timestamp
        );

        uint256 ethBalance = address(this).balance;
        if (ethBalance > 0) {
            uint256 ethForRev = (ethBalance * revFee) / 100;
            uint256 ethForTeam = ethBalance - ethForRev;

            (success, ) = address(teamWallet).call{value: ethForTeam}("");
            (success, ) = address(revWallet).call{value: ethForRev}("");

            emit SwapAndLiquify(swapThreshold, ethForTeam, ethForRev);
        }
    }
}