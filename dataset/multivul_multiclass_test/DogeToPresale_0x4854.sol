//SPDX-License-Identifier: MIT Licensed
pragma solidity ^0.8.17;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    /**
     * @dev Initializes the contract setting the deployer as the initial owner.
     */
    constructor() {
        _transferOwnership(_msgSender());
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(
            newOwner != address(0),
            "Ownable: new owner is the zero address"
        );
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

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

contract DogeToPresale is Ownable {
    IERC20 public mainToken;
    IERC20 public USDT = IERC20(0xdAC17F958D2ee523a2206206994597C13D831ec7);
    IERC20 public USDC = IERC20(0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48);

    AggregatorV3Interface public priceFeed;

    uint256 public refPercent = 400;
    uint256 public totalUsers;
    uint256 public soldToken;
    uint256 public amountRaised;
    uint256 public amountRaisedUSDT;
    uint256 public amountRaisedUSDC;
    uint256 public amountRaisedOverall;
    uint256 public uniqueBuyers;
    address payable public fundReceiver;
    uint256 public tokensToSell;
    uint256 public tokenPerUsdPrice;

    bool public presaleStatus;
    bool public isPresaleEnded;
    bool public isClaimEnabled;

    address[] public UsersAddresses;

    struct User {
        uint256 native_balance;
        uint256 eth_reward;
        uint256 usdt_balance;
        uint256 usdc_balance;
        uint256 usdt_reward;
        uint256 usdc_reward;
        uint256 token_balance;
        uint256 claimed_tokens;
        uint256 total_reward;
    }
    struct TopReferrals {
        address referralsAddress;
        uint256 usdAmount;
    }
    TopReferrals[10] public topReferralsData;
    mapping(address => User) public users;
    mapping(address => bool) public isExist;

    event BuyToken(address indexed _user, uint256 indexed _amount);
    event ClaimToken(address indexed _user, uint256 indexed _amount);
    event UpdatePrice(uint256 _oldPrice, uint256 _newPrice);

    constructor(IERC20 _token, address _fundReceiver) {
        mainToken = _token;
        tokensToSell = 3000000000e18;
        tokenPerUsdPrice = 1000000000000000000000;
        fundReceiver = payable(_fundReceiver);
        priceFeed = AggregatorV3Interface(
            0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419
        );
    }

    // to get real time price of ETH
    function getLatestPrice() public view returns (uint256) {
        (, int256 price, , , ) = priceFeed.latestRoundData();
        return uint256(price);
    }

    // to buy token during preSale time with ETH => for web3 use
    function buyToken(address _referral) public payable {
        require(!isPresaleEnded, "Presale ended!");
        require(presaleStatus, " Presale is Paused, check back later");
        if (!isExist[msg.sender]) {
            isExist[msg.sender] = true;
            uniqueBuyers++;
            UsersAddresses.push(msg.sender);
        }
        uint256 refReward;
        uint256 refRewardInUSD;
        if (_referral != address(0) && _referral != msg.sender) {
            refReward = ((msg.value * refPercent) / 1000);
            payable(_referral).transfer(refReward);
            refRewardInUSD = EthToUsd(refReward);
            refRewardInUSD = refRewardInUSD / 1e12;
        }
        fundReceiver.transfer(msg.value - refReward);

        uint256 numberOfTokens;
        uint256 ethToUSDTConverted;
        numberOfTokens = nativeToToken(msg.value);
        require(
            soldToken + numberOfTokens <= tokensToSell,
            "Presale Sold Out"
        );
        soldToken = soldToken + (numberOfTokens);
        amountRaised = amountRaised + msg.value;
        ethToUSDTConverted = EthToUsd(msg.value);
        ethToUSDTConverted = ethToUSDTConverted / 1e12;
        amountRaisedOverall = amountRaisedOverall + ethToUSDTConverted;

        users[msg.sender].native_balance += (msg.value);
        users[msg.sender].token_balance += numberOfTokens;
        users[_referral].eth_reward += refReward;
        users[_referral].total_reward += refRewardInUSD;
        updateTopReferralsData(_referral, users[_referral].total_reward);
    }

// to buy token during preSale time with USDC => for web3 use
    function buyTokenUSDC(address _referral, uint256 amount) public {
        require(!isPresaleEnded, "Presale ended!");
        require(presaleStatus, " Presale is Paused, check back later");
        if (!isExist[msg.sender]) {
            isExist[msg.sender] = true;
            uniqueBuyers++;
            UsersAddresses.push(msg.sender);
        }
        uint256 refReward;
        if (_referral != address(0) && _referral != msg.sender) {
            refReward = ((amount * refPercent) / 1000);
            USDC.transferFrom(msg.sender, _referral, refReward);
        }
        USDC.transferFrom(msg.sender, fundReceiver, amount - refReward);

        uint256 numberOfTokens;
        numberOfTokens = usdtToToken(amount);
        require(
            soldToken + numberOfTokens <= tokensToSell,
            "Presale Sold Out"
        );
        soldToken = soldToken + numberOfTokens;
        amountRaisedUSDC = amountRaisedUSDC + amount;
        amountRaisedOverall = amountRaisedOverall + amount;

        users[msg.sender].usdc_balance += amount;
        users[msg.sender].token_balance += numberOfTokens;
        users[_referral].usdc_reward += refReward;
        users[_referral].total_reward += refReward;
        updateTopReferralsData(_referral, users[_referral].total_reward);
    }

    // to buy token during preSale time with USDT => for web3 use
    function buyTokenUSDT(address _referral, uint256 amount) public {
        require(!isPresaleEnded, "Presale ended!");
        require(presaleStatus, " Presale is Paused, check back later");
        if (!isExist[msg.sender]) {
            isExist[msg.sender] = true;
            uniqueBuyers++;
            UsersAddresses.push(msg.sender);
        }
        uint256 refReward;
        if (_referral != address(0) && _referral != msg.sender) {
            refReward = ((amount * refPercent) / 1000);
            USDT.transferFrom(msg.sender, _referral, refReward);
        }
        USDT.transferFrom(msg.sender, fundReceiver, amount - refReward);

        uint256 numberOfTokens;
        numberOfTokens = usdtToToken(amount);
        require(
            soldToken + numberOfTokens <= tokensToSell,
            "Presale Sold Out"
        );
        soldToken = soldToken + numberOfTokens;
        amountRaisedUSDT = amountRaisedUSDT + amount;
        amountRaisedOverall = amountRaisedOverall + amount;

        users[msg.sender].usdt_balance += amount;
        users[msg.sender].token_balance += numberOfTokens;
        users[_referral].usdt_reward += refReward;
        users[_referral].total_reward += refReward;
        updateTopReferralsData(_referral, users[_referral].total_reward);
    }

    function updateTopReferralsData(address _user, uint256 _usdAmount) internal {
        for (uint256 i = 0; i < topReferralsData.length; i++) {
            if (_usdAmount > topReferralsData[i].usdAmount) {
                for (uint256 j = topReferralsData.length - 1; j > i; j--) {
                    topReferralsData[j] = topReferralsData[j - 1];
                }

                topReferralsData[i] = TopReferrals(_user, _usdAmount);
                break;
            }
        }
    }

    function claimTokens() external {
        require(isPresaleEnded, "Presale has not ended yet");
        require(isClaimEnabled, "Claim has not enabled yet");
        User storage user = users[msg.sender];
        require(user.token_balance > 0, "No tokens purchased");
        uint256 claimableTokens = user.token_balance - user.claimed_tokens;
        require(claimableTokens > 0, "No tokens to claim");
        user.claimed_tokens += claimableTokens;
        mainToken.transfer(msg.sender, claimableTokens);
        emit ClaimToken(msg.sender, claimableTokens);
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
            users[_addresses[i]].token_balance += _tokenAmount[i];
        }
    }
    
    function setPresaleStatus(bool _status) external onlyOwner {
        presaleStatus = _status;
    }

    function endPresale() external onlyOwner {
        isPresaleEnded = true;
    }
    function startClaim() external onlyOwner {
        isClaimEnabled = true;
    }
    // to check number of token for given ETH
    function nativeToToken(uint256 _amount) public view returns (uint256) {
        uint256 ethToUsd = (_amount * (getLatestPrice())) / (1 ether);
        uint256 numberOfTokens = (ethToUsd * tokenPerUsdPrice) / (1e8);
        return numberOfTokens;
    }

    // ETH to USD
    function EthToUsd(uint256 _amount) public view returns (uint256) {
        uint256 ethToUsd = (_amount * (getLatestPrice())) / (1e8);
        return ethToUsd;
    }

    // to check number of token for given usdt
    function usdtToToken(uint256 _amount) public view returns (uint256) {
        uint256 numberOfTokens = (_amount * tokenPerUsdPrice) / (1e6);
        return numberOfTokens;
    }
    // change tokens
    function updateToken(address _token) external onlyOwner {
        mainToken = IERC20(_token);
    }
    //change tokens for buy
    function updateStableTokens(IERC20 _USDT,IERC20 _USDC) external onlyOwner {
        USDT = IERC20(_USDT);
        USDC = IERC20(_USDC);
    }

    // to withdraw funds for liquidity
    function initiateTransfer(uint256 _value) external onlyOwner {
        fundReceiver.transfer(_value);
    }

    function totalUsersCount() external view returns (uint256) {
        return UsersAddresses.length;
    }

    // to withdraw funds for liquidity
    function changeFundReciever(address _addr) external onlyOwner {
        fundReceiver = payable(_addr);
    }

    // to withdraw funds for liquidity
    function updatePriceFeed(AggregatorV3Interface _priceFeed)
        external
        onlyOwner
    {
        priceFeed = _priceFeed;
    }

    // to withdraw out tokens
    function transferTokens(IERC20 token, uint256 _value) external onlyOwner {
        token.transfer(msg.sender, _value);
    }

    function ChangePriceAndTokensTOSell(
        uint256 _tokenPerUsdPrice,
        uint256 _tokensToSell
    ) public onlyOwner {
        tokenPerUsdPrice = _tokenPerUsdPrice;
        tokensToSell = _tokensToSell;
    }

    function setRefPercentage(
        uint256 _updatedRefValue
    ) external onlyOwner {
        refPercent = _updatedRefValue;
    }
}