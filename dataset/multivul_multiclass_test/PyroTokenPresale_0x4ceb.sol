// SPDX-License-Identifier: MIT
pragma solidity 0.8.21;

abstract contract Ownable {
    address private _owner;

    error OwnableUnauthorizedAccount(address account);

    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function _checkOwner() internal view virtual {
        if (owner() != msg.sender) {
            revert OwnableUnauthorizedAccount(msg.sender);
        }
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );

    function totalSupply() external view returns (uint256);

    function balanceOf(address account) external view returns (uint256);

    function transfer(address to, uint256 value) external returns (bool);

    function allowance(address owner, address spender)
        external
        view
        returns (uint256);

    function approve(address spender, uint256 value) external returns (bool);

    function transferFrom(
        address from,
        address to,
        uint256 value
    ) external returns (bool);

    function name() external view returns (string memory);

    function symbol() external view returns (string memory);

    function decimals() external view returns (uint8);
}

interface IERC20METADATA is IERC20 {
    function decimals() external view returns (uint8);
}

interface IERC165 {
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}

interface IERC1363 is IERC20, IERC165 {
    function transferAndCall(address to, uint256 value) external returns (bool);

    function transferAndCall(
        address to,
        uint256 value,
        bytes calldata data
    ) external returns (bool);

    function transferFromAndCall(
        address from,
        address to,
        uint256 value
    ) external returns (bool);

    function transferFromAndCall(
        address from,
        address to,
        uint256 value,
        bytes calldata data
    ) external returns (bool);

    function approveAndCall(address spender, uint256 value)
        external
        returns (bool);

    function approveAndCall(
        address spender,
        uint256 value,
        bytes calldata data
    ) external returns (bool);
}

library Errors {
    error InsufficientBalance(uint256 balance, uint256 needed);

    error FailedCall();
    error FailedDeployment();
    error MissingPrecompile(address);
}

library SafeERC20 {
    error SafeERC20FailedOperation(address token);

    error SafeERC20FailedDecreaseAllowance(
        address spender,
        uint256 currentAllowance,
        uint256 requestedDecrease
    );

    function safeTransfer(
        IERC20 token,
        address to,
        uint256 value
    ) internal {
        _callOptionalReturn(token, abi.encodeCall(token.transfer, (to, value)));
    }

    function safeTransferFrom(
        IERC20 token,
        address from,
        address to,
        uint256 value
    ) internal {
        _callOptionalReturn(
            token,
            abi.encodeCall(token.transferFrom, (from, to, value))
        );
    }

    function _callOptionalReturn(IERC20 token, bytes memory data) private {
        uint256 returnSize;
        uint256 returnValue;
        assembly ("memory-safe") {
            let success := call(
                gas(),
                token,
                0,
                add(data, 0x20),
                mload(data),
                0,
                0x20
            )
            // bubble errors
            if iszero(success) {
                let ptr := mload(0x40)
                returndatacopy(ptr, 0, returndatasize())
                revert(ptr, returndatasize())
            }
            returnSize := returndatasize()
            returnValue := mload(0)
        }

        if (
            returnSize == 0 ? address(token).code.length == 0 : returnValue != 1
        ) {
            revert SafeERC20FailedOperation(address(token));
        }
    }

    function _callOptionalReturnBool(IERC20 token, bytes memory data)
        private
        returns (bool)
    {
        bool success;
        uint256 returnSize;
        uint256 returnValue;
        assembly ("memory-safe") {
            success := call(
                gas(),
                token,
                0,
                add(data, 0x20),
                mload(data),
                0,
                0x20
            )
            returnSize := returndatasize()
            returnValue := mload(0)
        }
        return
            success &&
            (
                returnSize == 0
                    ? address(token).code.length > 0
                    : returnValue == 1
            );
    }
}

interface AggregatorV3Interface {
    function latestRoundData()
        external
        view
        returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );
}

abstract contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
    }

    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");

        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }
}

contract PyroTokenPresale is Ownable, ReentrancyGuard {
    struct UserContribution {
        uint256 ethAmount;
        uint256 usdtAmount;
        uint256 usdcAmount;
        uint256 tokenAmount;
        uint256 usdValue;
        uint256 timestamp;
        bool hasClaimed;
    }

    struct Round {
        uint256 price; // Price in USD with 6 decimals (e.g., 800 = $0.000800)
        uint256 allocation; // Allocation percentage with 2 decimals (e.g., 500 = 5.00%)
        uint256 startTime;
        uint256 endTime;
        uint256 totalTokens;
        uint256 soldTokens;
        uint256 carriedTokens;
        uint256 ethRaised;
        uint256 usdtRaised;
        uint256 usdcRaised;
        uint256 totalRaisedUSD;
        uint256 participantCount;
        bool isFinished;
        bool isActive;
        mapping(address => UserContribution) contributions;
    }

    IERC20METADATA public pyroToken;
    AggregatorV3Interface private ethUsdPriceFeed;
    IERC20METADATA public usdt;
    IERC20METADATA public usdc;
    address private presaleWallet;

    mapping(uint256 => Round) public rounds;
    uint256 public currentRound;
    uint256 public constant maxRounds = 4; // Rounds 0-4 (5 total rounds)
    uint256 public roundgap = 10 days;

    event TokensPurchased(address indexed buyer, uint256 amount, uint256 round);
    event TokensClaimed(address indexed buyer, uint256 amount, uint256 round);
    event RoundUpdated(uint256 round, uint256 price, uint256 allocation);
    event ManualContributionAdded(
        address indexed buyer,
        uint256 amount,
        uint256 round
    );
    event TokensCarriedOver(uint256 fromRound, uint256 toRound, uint256 amount);
    event TokensAllotted(
        address indexed buyer,
        uint256 indexed round,
        uint256 tokenAmount,
        uint256 ethAmount,
        uint256 usdtAmount,
        uint256 usdcAmount,
        uint256 usdValue,
        uint256 timestamp
    );
    event AllTokensClaimed(
        address indexed user,
        uint256 totalAmount,
        uint256[] rounds
    );
    event TokenSet(bool attached);
    event FundsWithdrawByOwner(
        address ownerAddress,
        address token,
        uint256 amount
    );

    mapping(address => uint256[]) private userRounds;

    AggregatorV3Interface private usdtUsdPriceFeed;
    AggregatorV3Interface private usdcUsdPriceFeed;

    constructor(
        address _usdt,
        address _usdc,
        address _ethUsdPriceFeed,
        address _usdtUsdPriceFeed,
        address _usdcUsdPriceFeed,
        address _presaleOwnerAddress,
        address _presaleWalletAddress
    ) Ownable(_presaleOwnerAddress) {
        usdt = IERC20METADATA(_usdt);
        usdc = IERC20METADATA(_usdc);
        ethUsdPriceFeed = AggregatorV3Interface(_ethUsdPriceFeed);
        usdtUsdPriceFeed = AggregatorV3Interface(_usdtUsdPriceFeed);
        usdcUsdPriceFeed = AggregatorV3Interface(_usdcUsdPriceFeed);
        presaleWallet = _presaleWalletAddress;
        initializeRounds();
    }

    function initializeRounds() private {
        uint256 startTime = 1740834000;

        rounds[0].price = 800;
        rounds[0].allocation = 500; // 5.00%
        rounds[0].startTime = startTime;
        rounds[0].endTime = startTime + roundgap;
        rounds[0].totalTokens = 1_200_000_000 * 1e18;

        rounds[1].price = 1200;
        rounds[1].allocation = 600; // 6.00%
        rounds[1].startTime = rounds[0].endTime;
        rounds[1].endTime = rounds[1].startTime + roundgap;
        rounds[1].totalTokens = 1_440_000_000 * 1e18;

        rounds[2].price = 1700; // $0.001700
        rounds[2].allocation = 1000; // 10.00%
        rounds[2].startTime = rounds[1].endTime;
        rounds[2].endTime = rounds[2].startTime + roundgap;
        rounds[2].totalTokens = 2_400_000_000 * 1e18;

        // Round 3 - Public Sale
        rounds[3].price = 2300; // $0.002300
        rounds[3].allocation = 500; // 5.00%
        rounds[3].startTime = rounds[2].endTime;
        rounds[3].endTime = rounds[3].startTime + roundgap;
        rounds[3].totalTokens = 1_200_000_000 * 1e18;

        // Round 4 - Final Round
        rounds[4].price = 3000; // $0.003000
        rounds[4].allocation = 400; // 4.00%
        rounds[4].startTime = rounds[3].endTime;
        rounds[4].endTime = rounds[4].startTime + roundgap;
        rounds[4].totalTokens = 960_000_000 * 1e18;

        rounds[0].isActive = true;
    }

    function getEthPrice() internal view returns (uint256) {
        (, int256 price, , , ) = ethUsdPriceFeed.latestRoundData();
        return uint256(price); // Returns price with 8 decimals
    }

    function getUsdtPrice() internal view returns (uint256) {
        (, int256 price, , , ) = usdtUsdPriceFeed.latestRoundData();
        return uint256(price); // 8 decimals
    }

    function getUsdcPrice() internal view returns (uint256) {
        (, int256 price, , , ) = usdcUsdPriceFeed.latestRoundData();
        return uint256(price); // 8 decimals
    }

    function calculateTokenAmount(
        uint256 value,
        uint256 price,
        uint8 inputDecimals
    ) private view returns (uint256) {
        // Corrected formula with 1e6 scaling factor
        uint256 usdValue = (value * price) / (10**inputDecimals);
        return (usdValue * 1e16) / rounds[currentRound].price;
    }

    function buyWithEth() external payable nonReentrant {
        manageRounds(); // Check round status first
        require(msg.value > 0, "Invalid amount");
        require(
            rounds[currentRound].isActive &&
                block.timestamp > rounds[currentRound].startTime &&
                block.timestamp < rounds[currentRound].endTime,
            "Round inactive"
        );

        uint256 tokenAmount = calculateTokenAmount(
            msg.value,
            getEthPrice(),
            18
        );
        _processPurchase(tokenAmount, msg.value, 0, 0);
        emit TokensPurchased(msg.sender, tokenAmount, currentRound);
    }

    function buyWithStablecoin(uint256 amount, bool isUsdt)
        external
        nonReentrant
    {
        manageRounds();
        require(amount > 0, "Invalid amount");
        require(
            rounds[currentRound].isActive &&
                block.timestamp > rounds[currentRound].startTime &&
                block.timestamp < rounds[currentRound].endTime,
            "Round inactive"
        );

        IERC20METADATA token = isUsdt ? usdt : usdc;
        uint8 decimals = token.decimals();
        uint256 price = isUsdt ? getUsdtPrice() : getUsdcPrice();

        SafeERC20.safeTransferFrom(
            isUsdt ? usdt : usdc,
            msg.sender,
            address(this),
            amount
        );

        uint256 tokenAmount = calculateTokenAmount(amount, price, decimals);
        _processPurchase(
            tokenAmount,
            0,
            isUsdt ? amount : 0,
            isUsdt ? 0 : amount
        );
        emit TokensPurchased(msg.sender, tokenAmount, currentRound);
    }

    function _processPurchase(
        uint256 tokenAmount,
        uint256 ethAmount,
        uint256 usdtAmount,
        uint256 usdcAmount
    ) private {
        Round storage round = rounds[currentRound];
        require(tokenAmount > 0, "Invalid token amount");
        require(
            round.soldTokens + tokenAmount <= round.totalTokens,
            "Exceeds allocation"
        );

        UserContribution storage contrib = round.contributions[msg.sender];

        require(
            (contrib.tokenAmount + tokenAmount) <
                ((round.totalTokens * 100) / 1000),
            "Cannot buy more than 10% of round volume"
        );
        if (contrib.timestamp == 0) {
            round.participantCount++;
            userRounds[msg.sender].push(currentRound);
        }

        uint256 usdValue = _calculateUsdValue(
            ethAmount,
            usdtAmount,
            usdcAmount
        );

        contrib.ethAmount += ethAmount;
        contrib.usdtAmount += usdtAmount;
        contrib.usdcAmount += usdcAmount;
        contrib.tokenAmount += tokenAmount;
        contrib.timestamp = block.timestamp;
        contrib.usdValue += usdValue;

        round.soldTokens += tokenAmount;
        round.ethRaised += ethAmount;
        round.usdtRaised += usdtAmount;
        round.usdcRaised += usdcAmount;
        round.totalRaisedUSD += usdValue;
    }

    function _calculateUsdValue(
        uint256 ethAmount,
        uint256 usdtAmount,
        uint256 usdcAmount
    ) private view returns (uint256) {
        uint256 ethValue = (ethAmount * getEthPrice()) / 1e18;
        uint256 usdtValue = (usdtAmount * getUsdtPrice()) /
            (10**usdt.decimals());
        uint256 usdcValue = (usdcAmount * getUsdcPrice()) /
            (10**usdc.decimals());
        return ethValue + usdtValue + usdcValue;
    }

    function manageRounds() internal {
        uint256 _currentRound = currentRound;

        while (_currentRound <= maxRounds) {
            Round storage current = rounds[_currentRound];

            if (
                block.timestamp <= current.endTime &&
                current.soldTokens < current.totalTokens
            ) {
                break;
            }

            uint256 remainingTime;
            if (
                block.timestamp < current.endTime &&
                current.soldTokens >= current.totalTokens
            ) {
                remainingTime = current.endTime - block.timestamp;
            }

            current.isActive = false;
            current.isFinished = true;
            current.endTime = block.timestamp;

            if (_currentRound >= maxRounds) {
                break;
            }

            uint256 remaining = current.totalTokens - current.soldTokens;
            uint256 nextRound = _currentRound + 1;

            // Carry over remaining tokens to the next round if any
            if (remaining > 0) {
                require(nextRound <= maxRounds, "Max rounds reached");
                rounds[nextRound].carriedTokens += remaining;
                rounds[nextRound].totalTokens += remaining;
                emit TokensCarriedOver(_currentRound, nextRound, remaining);
            }

            _currentRound = nextRound;
            rounds[_currentRound].startTime;
            rounds[_currentRound].endTime =
                rounds[_currentRound].startTime +
                roundgap +
                remainingTime;
            rounds[_currentRound].isActive = true;
        }
        currentRound = _currentRound;
    }

    function getPrices(
        uint256 amount,
        address token,
        uint256 round
    ) public view returns (uint256) {
        uint256 returnamount;

        if (token == address(0)) {
            uint256 usdValue = (amount * getEthPrice()) / (10**18);
            returnamount = (usdValue * 1e16) / rounds[round].price;
        } else if (token == address(usdt)) {
            uint256 usdValue = (amount * getUsdtPrice()) /
                (10**usdt.decimals());
            returnamount = (usdValue * 1e16) / rounds[round].price;
        } else if (token == address(usdc)) {
            uint256 usdValue = (amount * getUsdcPrice()) /
                (10**usdc.decimals());
            returnamount = (usdValue * 1e16) / rounds[round].price;
        } else {
            returnamount = 0;
        }
        return returnamount;
    }

    function getUserParticipations(address user, uint256 round)
        public
        view
        returns (
            uint256 ethAmount,
            uint256 usdtAmount,
            uint256 usdcAmount,
            uint256 tokenAmount,
            uint256 usdValue,
            uint256 timestamp,
            bool hasClaimed
        )
    {
        UserContribution storage contrib = rounds[round].contributions[user];
        ethAmount = contrib.ethAmount;
        usdtAmount = contrib.usdtAmount;
        usdcAmount = contrib.usdcAmount;
        tokenAmount = contrib.tokenAmount;
        usdValue = contrib.usdValue;
        timestamp = contrib.timestamp;
        hasClaimed = contrib.hasClaimed;
    }

   function getOverallParticipations(address wallet)
    public
    view
    returns (uint256)
{
    uint256 totalTokens;
    uint256[] storage roundsList = userRounds[wallet];

    for (uint256 i = 0; i < roundsList.length; i++) {
        uint256 roundId = roundsList[i];
        UserContribution storage contrib = rounds[roundId].contributions[wallet];

        totalTokens += contrib.tokenAmount;
    }

    return totalTokens;
}

    function getRaisedValues()
        public
        view
        returns (
            uint256 ethRaised,
            uint256 usdcRaised,
            uint256 usdtRaised,
            uint256 usdRaised
        )
    {
        for (uint256 i = 0; i <= currentRound; i++) {
            Round storage round = rounds[i];
            ethRaised += round.ethRaised;
            usdcRaised += round.usdcRaised;
            usdtRaised += round.usdtRaised;
            usdRaised += round.totalRaisedUSD;
        }
    }

 function getNextRoundTime() public view returns (uint256 time) {
    // If all rounds have completed
    if (block.timestamp >= rounds[maxRounds].endTime) {
        return 0;
    }

    // Check if we're before the first round starts
    if (block.timestamp < rounds[0].startTime) {
        return rounds[0].startTime;
    }

    // Find current active round
    for (uint256 i = 0; i <= maxRounds; i++) {
        if (block.timestamp >= rounds[i].startTime && block.timestamp < rounds[i].endTime) {
            // Return next round's start time (current round's end time)
            // Unless it's the final round
            return i < maxRounds ? rounds[i].endTime : 0;
        }
    }

    // If between rounds, find next unstarted round
    for (uint256 i = 0; i <= maxRounds; i++) {
        if (block.timestamp < rounds[i].startTime) {
            return rounds[i].startTime;
        }
    }

    // Fallback (should never reach)
    return 0;
}
    function attachPresaleToken(IERC20METADATA _token) external onlyOwner {
        require(address(pyroToken) == address(0), "Token already set");
        pyroToken = _token;
        emit TokenSet(true);
    }

    function claimAllTokens() external nonReentrant {
        require(block.timestamp > rounds[maxRounds].endTime, "Presale ongoing");

        uint256 totalTokens;
        uint256[] storage roundsList = userRounds[msg.sender];
        require(roundsList.length > 0, "No contributions");

        for (uint256 i = 0; i < roundsList.length; i++) {
            uint256 roundId = roundsList[i];
            UserContribution storage contrib = rounds[roundId].contributions[
                msg.sender
            ];

            if (!contrib.hasClaimed && contrib.tokenAmount > 0) {
                totalTokens += contrib.tokenAmount;
                contrib.hasClaimed = true;
                emit TokensClaimed(msg.sender, contrib.tokenAmount, roundId);
            }
        }

        require(totalTokens > 0, "Nothing to claim");
        SafeERC20.safeTransferFrom(
            pyroToken,
            presaleWallet,
            msg.sender,
            totalTokens
        );
        emit AllTokensClaimed(msg.sender, totalTokens, roundsList);
    }

    function withdrawEth() external onlyOwner nonReentrant {
        require(address(this).balance>0,"Balance is Zero");
        (bool success, ) = payable(presaleWallet).call{
            value: address(this).balance
        }("");
        require(success, "ETH transfer failed");
    }

    function withdrawTokens(address token) external onlyOwner nonReentrant {
        require(token != address(0), "Invalid token address");
        uint256 balance = IERC20(token).balanceOf(address(this));
        require(
            balance > 0,
            "Balance is Zero of this token. Admin already withdrawn funds"
        );
        SafeERC20.safeTransfer(IERC20(token), presaleWallet, balance);
        emit FundsWithdrawByOwner(owner(), token, balance);
    }

    receive() external payable {}
}