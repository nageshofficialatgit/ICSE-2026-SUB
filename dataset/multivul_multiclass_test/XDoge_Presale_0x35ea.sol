/**
 *Submitted for verification at Etherscan.io on 2025-02-04
*/

/**
 *Submitted for verification at Etherscan.io on 2023-10-07
 */

//SPDX-License-Identifier: MIT Licensed
pragma solidity ^0.8.10;

interface IERC20 {
    function name() external view returns (string memory);

    function symbol() external view returns (string memory);

    function decimals() external view returns (uint8);

    function totalSupply() external view returns (uint256);

    function balanceOf(address owner) external view returns (uint256);

    function allowance(address owner, address spender)
        external
        view
        returns (uint256);

    function approve(address spender, uint256 value) external;

    function transfer(address to, uint256 value) external;

    function transferFrom(
        address from,
        address to,
        uint256 value
    ) external;

    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );
    event Transfer(address indexed from, address indexed to, uint256 value);
}

interface AggregatorV3Interface {
    function decimals() external view returns (uint8);

    function description() external view returns (string memory);

    function version() external view returns (uint256);

    function getRoundData(uint80 _roundId)
        external
        view
        returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );

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

contract XDoge_Presale {
    IERC20 public Token;
    IERC20 public USDT = IERC20(0xdAC17F958D2ee523a2206206994597C13D831ec7);
    AggregatorV3Interface public priceFeeD;
    struct Phase {
        uint256 tokensToSell;
        uint256 totalSoldTokens;
        uint256 tokenPerUsdPrice;
    }
    // Stats
    uint8 public immutable tokenDecimals;
    uint256 public totalStages;
    uint256 public totalStaker;
    uint256 public currentStage;
    address payable public owner;
    uint256 public totalUsers;
    uint256 public soldToken;
    uint256 public totalSupply = 42_400_000 ether;
    uint256 public amountRaised;
    uint256 public amountRaisedOverall;
    uint256 public amountRaisedUSDT;
    address payable public fundReceiver;
    uint256 public presalePhase;
    uint256 public totalStakedAmount;

    uint256 public distributedReward;
    uint256 public refPercentToken = 50;
    uint256 public refPercent = 100;
    uint256 constant APY = 760_00;
    uint256 constant stakeDays = 365 days;
    uint256 constant percentDivider = 100_00;

    uint256 public initialClaimPercent = 10_00;
    uint256 public vestingPercentage = 10_00;
    uint256 public totalClaimCycles = 9;
    uint256 public vestingTime = 10 days;
    uint256 public vestingStartTime;
    bool public isVestingStarted;
    bool public isPresaleEnded;
    bool public presaleStatus;
    bool public enableClaim;

    uint256[] public tokenPerUsdPrice = [
        12870012870012870012,
        11454753722794959908,
        10288065843621399176,
        9496676163342830009,
        8992805755395683453,
        8510638297872340425,
        8058017727639000805,
        7627765064836003051,
        7220216606498194945,
        6835269993164730006
    ];
    uint256[] public tokensToSell = [
        5_973_326 * 10**18,
        5_316_465 * 10**18,
        4_774_973 * 10**18,
        4_407_668 * 10**18,
        4_173_808 * 10**18,
        3_950_021 * 10**18,
        3_739_947 * 10**18,
        3_540_255 * 10**18,
        3_351_100 * 10**18,
        3_172_437 * 10**18
    ];

    struct user {
        uint256 native_balance;
        uint256 eth_usdt_balance;
        uint256 usdt_balance;
        uint256 ref_reward;
        uint256 token_bonus;
        uint256 claimedAmount;
        uint256 claimAbleAmount;
        uint256 stake_count;
        uint256 claimCount;
        uint256 activePercentAmount;
        uint256 claimedVestingAmount;
        uint256 lastClaimTime;
    }
    struct bonus {
        uint256 token_bonus;
        uint256 claimed_bonus;
        uint256 level;
    }
    struct StakeData {
        uint256 stakedTokens;
        uint256 claimedTokens;
        uint256 stakeTime;
        uint256 claimedReward;
        bool isUnstake;
        uint256 unstakeTime;
    }
    mapping(uint256 => Phase) public phases;
    mapping(address => uint256) public referralCount;
    mapping(address => bonus) public Bonus;
    mapping(address => user) public users;
    mapping(address => mapping(uint256 => StakeData)) public userStakes;
    mapping(address => uint256) public wallets;

    modifier onlyOwner() {
        require(msg.sender == owner, "PRESALE: Not an owner");
        _;
    }

    event BuyToken(address indexed _user, uint256 indexed _amount);
    event ClaimToken(address indexed _user, uint256 indexed _amount);
    event ClaimBonus(address indexed _user, uint256 indexed _amount);
    event UpdatePrice(uint256 _oldPrice, uint256 _newPrice);
    event UpdateBonusValue(uint256 _oldValue, uint256 _newValue);
    event UpdateRefPercent(uint256 _oldPercent, uint256 _newPercent);

    event BuyTokenETh(
        address indexed _user,
        uint256 buyingAmount,
        uint256 indexed _tokenamount
    );
    event BuyTokenUSDT(
        address indexed _user,
        uint256 buyingAmount,
        uint256 indexed _tokenamount
    );
    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    constructor(address _feeReceiver) {
        fundReceiver = payable(_feeReceiver);
        Token = IERC20(0x0000000000000000000000000000000000000000);
        owner = payable(msg.sender);
        priceFeeD = AggregatorV3Interface(
            0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419
        );
        tokenDecimals = 18;
        for (uint256 i = 0; i < tokensToSell.length; i++) {
            phases[i].tokensToSell = tokensToSell[i];
            phases[i].tokenPerUsdPrice = tokenPerUsdPrice[i];
        }
        totalStages = tokensToSell.length;
    }

    receive() external payable {}

    // update a presale
    function updatePresale(
        uint256 _phaseId,
        uint256 _tokensToSell,
        uint256 _tokenPerUsdPrice
    ) public onlyOwner {
        require(phases[_phaseId].tokensToSell > 0, "presale don't exist");
        phases[_phaseId].tokensToSell = _tokensToSell;
        phases[_phaseId].tokenPerUsdPrice = _tokenPerUsdPrice;
    }

    // to get real time price of Eth
    function getLatestPrice() public view returns (uint256) {
        (, int256 price, , , ) = priceFeeD.latestRoundData();
        return uint256(price);
    }

    // to buy token during preSale time with Eth => for web3 use

    function buyToken(address _refAddress, bool _isStake) public payable {
        require(_refAddress != msg.sender, "You can't ref yourself");
        require(!isPresaleEnded, "Presale ended!");
        require(presaleStatus, " Presale is Paused, check back later");

        uint256 numberOfTokens;
        uint256 equivalentUSDT;
        numberOfTokens = nativeToToken(msg.value, currentStage);
        require(
            phases[currentStage].totalSoldTokens + numberOfTokens <=
                phases[currentStage].tokensToSell,
            "Phase Sold Out!"
        );
        uint256 refReward;
        if (_refAddress != address(0)) {
            referralCount[_refAddress] += 1;
            users[msg.sender].ref_reward +=
                (numberOfTokens * refPercentToken) /
                1000;
            users[msg.sender].claimAbleAmount +=
                (numberOfTokens * refPercentToken) /
                1000;
            refReward = ((msg.value * refPercent) / 1000);
            payable(_refAddress).transfer(refReward);
        }

        fundReceiver.transfer(msg.value - refReward);

        soldToken = soldToken + (numberOfTokens);
        phases[currentStage].totalSoldTokens += numberOfTokens;
        amountRaised = amountRaised + (msg.value);
        equivalentUSDT = nativeToUsd(msg.value);
        amountRaisedOverall = amountRaisedOverall + equivalentUSDT;

        users[msg.sender].eth_usdt_balance += equivalentUSDT;
        users[msg.sender].native_balance += msg.value;
        if (!_isStake) {
            users[msg.sender].claimAbleAmount += numberOfTokens;
        } else {
            stake(numberOfTokens);
            if (users[msg.sender].stake_count == 0) {
                totalStaker++;
            }
        }
        addUserBonus(users[msg.sender].claimAbleAmount, msg.sender);
        emit BuyTokenETh(msg.sender, msg.value, numberOfTokens);
    }

    // to buy token during preSale time with USDT => for web3 use
    function buyTokenUSDT(
        uint256 amount,
        address _refAddress,
        bool _isStake
    ) public {
        require(_refAddress != msg.sender, "You can't ref yourself");
        require(!isPresaleEnded, "Presale ended!");
        require(presaleStatus, " Presale is Paused, check back later");

        uint256 numberOfTokens;
        numberOfTokens = usdtToToken(amount, currentStage);
        require(
            phases[currentStage].totalSoldTokens + numberOfTokens <=
                phases[currentStage].tokensToSell,
            "Phase Sold Out!"
        );
        uint256 refReward;
        if (_refAddress != address(0)) {
            referralCount[_refAddress] += 1;
            users[_refAddress].ref_reward +=
                (numberOfTokens * refPercentToken) /
                1000;
            users[msg.sender].claimAbleAmount +=
                (numberOfTokens * refPercentToken) /
                1000;
            refReward = ((amount * refPercent) / 1000);
            USDT.transferFrom(msg.sender, _refAddress, refReward);
        }

        USDT.transferFrom(msg.sender, fundReceiver, amount - refReward);

        soldToken = soldToken + (numberOfTokens);
        phases[currentStage].totalSoldTokens += numberOfTokens;
        amountRaisedUSDT = amountRaisedUSDT + (amount);
        amountRaisedOverall = amountRaisedOverall + (amount);
        users[msg.sender].usdt_balance += amount;
        if (!_isStake) {
            users[msg.sender].claimAbleAmount += numberOfTokens;
        } else {
            stake(numberOfTokens);
            if (users[msg.sender].stake_count == 0) {
                totalStaker++;
            }
        }
        addUserBonus(users[msg.sender].claimAbleAmount, msg.sender);
        emit BuyTokenUSDT(msg.sender, amount, numberOfTokens);
    }

    function addUserBonus(uint256 currentUsrTokens, address _user) internal {
        if (currentUsrTokens > 1_287 ether && currentUsrTokens <= 3_217 ether) {
            Bonus[_user].token_bonus = (currentUsrTokens * 5) / 100;
            users[_user].claimAbleAmount += Bonus[_user].token_bonus;
            Bonus[_user].level = 1;
        } else if (
            currentUsrTokens > 3_217 ether && currentUsrTokens <= 6_435 ether
        ) {
            Bonus[_user].token_bonus = (currentUsrTokens * 10) / 100;
            users[_user].claimAbleAmount += Bonus[_user].token_bonus;
            Bonus[_user].level = 2;
        } else if (
            currentUsrTokens > 6_435 ether && currentUsrTokens <= 12_870 ether
        ) {
            Bonus[_user].token_bonus = (currentUsrTokens * 15) / 100;
            users[_user].claimAbleAmount += Bonus[_user].token_bonus;
            Bonus[_user].level = 3;
        } else if (
            currentUsrTokens > 12_870 ether && currentUsrTokens <= 64_350 ether
        ) {
            Bonus[_user].token_bonus = (currentUsrTokens * 20) / 100;
            users[_user].claimAbleAmount += Bonus[_user].token_bonus;
            Bonus[_user].level = 4;
        } else if (currentUsrTokens > 64_350 ether) {
            Bonus[_user].token_bonus = (currentUsrTokens * 25) / 100;
            users[_user].claimAbleAmount += Bonus[_user].token_bonus;
            Bonus[_user].level = 5;
        }
    }

    function stake(uint256 _amount) internal {
        user storage _usr = users[msg.sender];
        StakeData storage userStake = userStakes[msg.sender][_usr.stake_count];
        userStake.stakedTokens = _amount;
        userStake.stakeTime = block.timestamp;
        _usr.stake_count++;
        totalStakedAmount += _amount;
    }

    function unStake(uint256 _index) public {
        user storage _usr = users[msg.sender];
        StakeData storage userStake = userStakes[msg.sender][_index];
        require(isPresaleEnded, "Presale has not ended yet");
        require(_usr.stake_count > 0, "there is no stake");
        require(userStake.stakeTime > 0, "No stake on this index");
        require(
            block.timestamp >= userStake.stakeTime + stakeDays,
            "wait for end time"
        );
        require(!userStake.isUnstake, "unstaked already");
        uint256 _reward = calculateReward(msg.sender, _index);
        if (_reward > 0) {
            Token.transfer(msg.sender, _reward);
            userStake.claimedReward += _reward;
            distributedReward += _reward;
        }
        Token.transfer(msg.sender, userStake.stakedTokens);
        userStake.claimedTokens = userStake.stakedTokens;
        userStake.unstakeTime = block.timestamp;
        userStake.isUnstake = true;
        totalStakedAmount -= userStake.stakedTokens;
        emit ClaimToken(msg.sender, userStake.stakedTokens);
        emit ClaimToken(msg.sender, _reward);
    }

    function calculateReward(address _user, uint256 _index)
        public
        view
        returns (uint256 _reward)
    {
        StakeData memory userStake = userStakes[_user][_index];
        uint256 rewardDuration = block.timestamp - (userStake.stakeTime);
        _reward =
            (userStake.stakedTokens * (rewardDuration) * APY) /
            (percentDivider * (stakeDays));
    }

    function claimVesting() public returns (bool) {
        uint256 amount = users[msg.sender].claimAbleAmount;
        require(amount > 0, "No claimable amount");
        require(
            address(Token) != address(0),
            "Presale token address not set"
        );
        require(
            amount <= Token.balanceOf(address(this)),
            "Not enough tokens in the contract"
        );
        require(isPresaleEnded, "Claim is not enable");
        uint256 transferAmount;
        if (users[msg.sender].claimCount == 0) {
            transferAmount = (amount * (initialClaimPercent)) / percentDivider;
            users[msg.sender].activePercentAmount =
                (amount * vestingPercentage) /
                percentDivider;
            Token.transfer(
                msg.sender,
                transferAmount
            );
            users[msg.sender].claimAbleAmount -= transferAmount;
            users[msg.sender].claimedAmount += transferAmount;
            users[msg.sender].claimCount++;
        } else if (
            users[msg.sender].claimAbleAmount >=
            users[msg.sender].activePercentAmount
        ) {
            require(block.timestamp >= vestingStartTime ,"wait for the vesting start time");
            uint256 duration = block.timestamp - vestingStartTime;
            uint256 multiplier = duration / vestingTime;
            if (multiplier > totalClaimCycles) {
                multiplier = totalClaimCycles;
            }
            uint256 _amount = multiplier *
                users[msg.sender].activePercentAmount;
            transferAmount = _amount - users[msg.sender].claimedVestingAmount;
            require(transferAmount > 0, "Please wait till next claim");
             Token.transfer(
                msg.sender,
                transferAmount
            );
            users[msg.sender].claimAbleAmount -= transferAmount;
            users[msg.sender].claimedVestingAmount += transferAmount;
            users[msg.sender].claimedAmount += transferAmount;
            users[msg.sender].claimCount++;
        } else {
            require(block.timestamp >= vestingStartTime ,"wait for the vesting start time");
            uint256 duration = block.timestamp - vestingStartTime;
            uint256 multiplier = duration / vestingTime;
            if (multiplier > totalClaimCycles) {
                transferAmount = users[msg.sender].claimAbleAmount;
                require(transferAmount > 0, "Please wait till next claim");
                Token.transfer(
                    msg.sender,
                    transferAmount
                );
                users[msg.sender].claimAbleAmount -= transferAmount;
                users[msg.sender].claimedAmount += transferAmount;
                users[msg.sender].claimedVestingAmount += transferAmount;
                users[msg.sender].claimCount++;
            } else {
                revert("Wait for next claim");
            }
        }
        users[msg.sender].lastClaimTime = block.timestamp;
        return true;
    }

    function whitelistdAddresses(
        address[] memory _addresses,
        uint256[] memory _tokenAmount
    ) external onlyOwner {
        require(
            _addresses.length == _tokenAmount.length,
            "Addresses and amounts must be equal"
        );
        for (uint256 i = 0; i < _addresses.length; i++) {
            users[_addresses[i]].claimAbleAmount += _tokenAmount[i];
        }
    }

    function endPresale() external onlyOwner {
        isPresaleEnded = true;
    }

    function EnableClaim() external onlyOwner {
        enableClaim = true;
        vestingStartTime = block.timestamp;
        isVestingStarted = true;
    }

    function setPresaleStatus(bool _status) external onlyOwner {
        presaleStatus = _status;
    }

    // Eth to USD
    function nativeToUsd(uint256 _amount) public view returns (uint256) {
        uint256 nativeTousd = (_amount * (getLatestPrice())) / (1e20);
        return nativeTousd;
    }

    // to check number of token for given Eth
    function nativeToToken(uint256 _amount, uint256 phaseId)
        public
        view
        returns (uint256)
    {
        uint256 ethToUsd = (_amount * (getLatestPrice())) / (1 ether);
        uint256 numberOfTokens = (ethToUsd * phases[phaseId].tokenPerUsdPrice) /
            (1e8);
        return numberOfTokens;
    }

    // to check number of token for given usdt
    function usdtToToken(uint256 _amount, uint256 phaseId)
        public
        view
        returns (uint256)
    {
        uint256 numberOfTokens = (_amount * phases[phaseId].tokenPerUsdPrice) /
            (1e6);
        return numberOfTokens;
    }

    // transfer ownership
    function changeOwner(address payable _newOwner) external onlyOwner {
        require(
            _newOwner != address(0),
            "Ownable: new owner is the zero address"
        );
        address _oldOwner = owner;
        owner = _newOwner;

        emit OwnershipTransferred(_oldOwner, _newOwner);
    }

    // funtion is used to change the stage of presale
    function setCurrentStage(uint256 _stageNum) public onlyOwner {
        currentStage = _stageNum;
    }

    // change tokens
    function changeToken(address _token) external onlyOwner {
        Token = IERC20(_token);
    }

    //change USDT
    function changeUSDT(address _USDT) external onlyOwner {
        USDT = IERC20(_USDT);
    }

    // to draw funds for liquidity
    function transferFunds(uint256 _value) external onlyOwner {
        fundReceiver.transfer(_value);
    }

    // to draw out tokens
    function transferTokens(IERC20 token, uint256 _value) external onlyOwner {
        token.transfer(msg.sender, _value);
    }

    // to draw funds for liquidity
    function setRefPercentages(
        uint256 _updatedRefTokenValue,
        uint256 _updatedRefValue
    ) external onlyOwner {
        refPercentToken = _updatedRefTokenValue;
        refPercent = _updatedRefValue;
    }
}