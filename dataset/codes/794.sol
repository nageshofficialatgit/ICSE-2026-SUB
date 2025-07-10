pragma solidity ^0.4.25;
contract IStdToken {
    function balanceOf(address _owner) public view returns (uint256);
    function transfer(address _to, uint256 _value) public returns (bool);
    function transferFrom(address _from, address _to, uint256 _value) public returns(bool);
}
contract EtheramaCommon {
    mapping(address => bool) private _administrators;
    mapping(address => bool) private _managers;
    modifier onlyAdministrator() {
        require(_administrators[msg.sender]);
        _;
    }
    modifier onlyAdministratorOrManager() {
        require(_administrators[msg.sender] || _managers[msg.sender]);
        _;
    }
    constructor() public {
        _administrators[msg.sender] = true;
    }
    function addAdministator(address addr) onlyAdministrator public {
        _administrators[addr] = true;
    }
    function removeAdministator(address addr) onlyAdministrator public {
        _administrators[addr] = false;
    }
    function isAdministrator(address addr) public view returns (bool) {
        return _administrators[addr];
    }
    function addManager(address addr) onlyAdministrator public {
        _managers[addr] = true;
    }
    function removeManager(address addr) onlyAdministrator public {
        _managers[addr] = false;
    }
    function isManager(address addr) public view returns (bool) {
        return _managers[addr];
    }
}
contract EtheramaGasPriceLimit is EtheramaCommon {
    uint256 public MAX_GAS_PRICE = 0 wei;
    event onSetMaxGasPrice(uint256 val);    
    modifier validGasPrice(uint256 val) {
        require(val > 0);
        _;
    }
    constructor(uint256 maxGasPrice) public validGasPrice(maxGasPrice) {
        setMaxGasPrice(maxGasPrice);
    } 
    function setMaxGasPrice(uint256 val) public validGasPrice(val) onlyAdministratorOrManager {
        MAX_GAS_PRICE = val;
        emit onSetMaxGasPrice(val);
    }
}
contract EtheramaCore is EtheramaGasPriceLimit {
    uint256 constant public MAGNITUDE = 2**64;
    uint256 constant public MIN_TOKEN_DEAL_VAL = 0.1 ether;
    uint256 constant public MAX_TOKEN_DEAL_VAL = 1000000 ether;
    uint256 constant public MIN_ETH_DEAL_VAL = 0.001 ether;
    uint256 constant public MAX_ETH_DEAL_VAL = 200000 ether;
    uint256 public _bigPromoPercent = 5 ether;
    uint256 public _quickPromoPercent = 5 ether;
    uint256 public _devRewardPercent = 15 ether;
    uint256 public _tokenOwnerRewardPercent = 30 ether;
    uint256 public _shareRewardPercent = 25 ether;
    uint256 public _refBonusPercent = 20 ether;
    uint128 public _bigPromoBlockInterval = 9999;
    uint128 public _quickPromoBlockInterval = 100;
    uint256 public _promoMinPurchaseEth = 1 ether;
    uint256 public _minRefEthPurchase = 0.5 ether;
    uint256 public _totalIncomeFeePercent = 100 ether;
    uint256 public _currentBigPromoBonus;
    uint256 public _currentQuickPromoBonus;
    uint256 public _devReward;
    uint256 public _initBlockNum;
    mapping(address => bool) private _controllerContracts;
    mapping(uint256 => address) private _controllerIndexer;
    uint256 private _controllerContractCount;
    mapping(address => mapping(address => uint256)) private _userTokenLocalBalances;
    mapping(address => mapping(address => uint256)) private _rewardPayouts;
    mapping(address => mapping(address => uint256)) private _refBalances;
    mapping(address => mapping(address => uint256)) private _promoQuickBonuses;
    mapping(address => mapping(address => uint256)) private _promoBigBonuses;  
    mapping(address => mapping(address => uint256)) private _userEthVolumeSaldos;  
    mapping(address => uint256) private _bonusesPerShare;
    mapping(address => uint256) private _buyCounts;
    mapping(address => uint256) private _sellCounts;
    mapping(address => uint256) private _totalVolumeEth;
    mapping(address => uint256) private _totalVolumeToken;
    event onWithdrawUserBonus(address indexed userAddress, uint256 ethWithdrawn); 
    modifier onlyController() {
        require(_controllerContracts[msg.sender]);
        _;
    }
    constructor(uint256 maxGasPrice) EtheramaGasPriceLimit(maxGasPrice) public { 
         _initBlockNum = block.number;
    }
    function getInitBlockNum() public view returns (uint256) {
        return _initBlockNum;
    }
    function addControllerContract(address addr) onlyAdministrator public {
        _controllerContracts[addr] = true;
        _controllerIndexer[_controllerContractCount] = addr;
        _controllerContractCount = SafeMath.add(_controllerContractCount, 1);
    }
    function removeControllerContract(address addr) onlyAdministrator public {
        _controllerContracts[addr] = false;
    }
    function changeControllerContract(address oldAddr, address newAddress) onlyAdministrator public {
         _controllerContracts[oldAddr] = false;
         _controllerContracts[newAddress] = true;
    }
    function setBigPromoInterval(uint128 val) onlyAdministrator public {
        _bigPromoBlockInterval = val;
    }
    function setQuickPromoInterval(uint128 val) onlyAdministrator public {
        _quickPromoBlockInterval = val;
    }
    function addBigPromoBonus() onlyController payable public {
        _currentBigPromoBonus = SafeMath.add(_currentBigPromoBonus, msg.value);
    }
    function addQuickPromoBonus() onlyController payable public {
        _currentQuickPromoBonus = SafeMath.add(_currentQuickPromoBonus, msg.value);
    }
    function setPromoMinPurchaseEth(uint256 val) onlyAdministrator public {
        _promoMinPurchaseEth = val;
    }
    function setMinRefEthPurchase(uint256 val) onlyAdministrator public {
        _minRefEthPurchase = val;
    }
    function setTotalIncomeFeePercent(uint256 val) onlyController public {
        require(val > 0 && val <= 100 ether);
        _totalIncomeFeePercent = val;
    }
    function setRewardPercentages(uint256 tokenOwnerRewardPercent, uint256 shareRewardPercent, uint256 refBonusPercent, uint256 bigPromoPercent, uint256 quickPromoPercent) onlyAdministrator public {
        require(tokenOwnerRewardPercent <= 40 ether);
        require(shareRewardPercent <= 100 ether);
        require(refBonusPercent <= 100 ether);
        require(bigPromoPercent <= 100 ether);
        require(quickPromoPercent <= 100 ether);
        require(tokenOwnerRewardPercent + shareRewardPercent + refBonusPercent + _devRewardPercent + _bigPromoPercent + _quickPromoPercent == 100 ether);
        _tokenOwnerRewardPercent = tokenOwnerRewardPercent;
        _shareRewardPercent = shareRewardPercent;
        _refBonusPercent = refBonusPercent;
        _bigPromoPercent = bigPromoPercent;
        _quickPromoPercent = quickPromoPercent;
    }    
    function payoutQuickBonus(address userAddress) onlyController public {
        address dataContractAddress = Etherama(msg.sender).getDataContractAddress();
        _promoQuickBonuses[dataContractAddress][userAddress] = SafeMath.add(_promoQuickBonuses[dataContractAddress][userAddress], _currentQuickPromoBonus);
        _currentQuickPromoBonus = 0;
    }
    function payoutBigBonus(address userAddress) onlyController public {
        address dataContractAddress = Etherama(msg.sender).getDataContractAddress();
        _promoBigBonuses[dataContractAddress][userAddress] = SafeMath.add(_promoBigBonuses[dataContractAddress][userAddress], _currentBigPromoBonus);
        _currentBigPromoBonus = 0;
    }
    function addDevReward() onlyController payable public {
        _devReward = SafeMath.add(_devReward, msg.value);
    }    
    function withdrawDevReward() onlyAdministrator public {
        uint256 reward = _devReward;
        _devReward = 0;
        msg.sender.transfer(reward);
    }
    function getBlockNumSinceInit() public view returns(uint256) {
        return block.number - getInitBlockNum();
    }
    function getQuickPromoRemainingBlocks() public view returns(uint256) {
        uint256 d = getBlockNumSinceInit() % _quickPromoBlockInterval;
        d = d == 0 ? _quickPromoBlockInterval : d;
        return _quickPromoBlockInterval - d;
    }
    function getBigPromoRemainingBlocks() public view returns(uint256) {
        uint256 d = getBlockNumSinceInit() % _bigPromoBlockInterval;
        d = d == 0 ? _bigPromoBlockInterval : d;
        return _bigPromoBlockInterval - d;
    } 
    function getBonusPerShare(address dataContractAddress) public view returns(uint256) {
        return _bonusesPerShare[dataContractAddress];
    }
    function getTotalBonusPerShare() public view returns (uint256 res) {
        for (uint256 i = 0; i < _controllerContractCount; i++) {
            res = SafeMath.add(res, _bonusesPerShare[Etherama(_controllerIndexer[i]).getDataContractAddress()]);
        }          
    }
    function addBonusPerShare() onlyController payable public {
        EtheramaData data = Etherama(msg.sender)._data();
        uint256 shareBonus = (msg.value * MAGNITUDE) / data.getTotalTokenSold();
        _bonusesPerShare[address(data)] = SafeMath.add(_bonusesPerShare[address(data)], shareBonus);
    }        
    function getUserRefBalance(address dataContractAddress, address userAddress) public view returns(uint256) {
        return _refBalances[dataContractAddress][userAddress];
    }
    function getUserRewardPayouts(address dataContractAddress, address userAddress) public view returns(uint256) {
        return _rewardPayouts[dataContractAddress][userAddress];
    }    
    function resetUserRefBalance(address userAddress) onlyController public {
        resetUserRefBalance(Etherama(msg.sender).getDataContractAddress(), userAddress);
    }
    function resetUserRefBalance(address dataContractAddress, address userAddress) internal {
        _refBalances[dataContractAddress][userAddress] = 0;
    }
    function addUserRefBalance(address userAddress) onlyController payable public {
        address dataContractAddress = Etherama(msg.sender).getDataContractAddress();
        _refBalances[dataContractAddress][userAddress] = SafeMath.add(_refBalances[dataContractAddress][userAddress], msg.value);
    }
    function addUserRewardPayouts(address userAddress, uint256 val) onlyController public {
        addUserRewardPayouts(Etherama(msg.sender).getDataContractAddress(), userAddress, val);
    }    
    function addUserRewardPayouts(address dataContractAddress, address userAddress, uint256 val) internal {
        _rewardPayouts[dataContractAddress][userAddress] = SafeMath.add(_rewardPayouts[dataContractAddress][userAddress], val);
    }
    function resetUserPromoBonus(address userAddress) onlyController public {
        resetUserPromoBonus(Etherama(msg.sender).getDataContractAddress(), userAddress);
    }
    function resetUserPromoBonus(address dataContractAddress, address userAddress) internal {
        _promoQuickBonuses[dataContractAddress][userAddress] = 0;
        _promoBigBonuses[dataContractAddress][userAddress] = 0;
    }
    function trackBuy(address userAddress, uint256 volEth, uint256 volToken) onlyController public {
        address dataContractAddress = Etherama(msg.sender).getDataContractAddress();
        _buyCounts[dataContractAddress] = SafeMath.add(_buyCounts[dataContractAddress], 1);
        _userEthVolumeSaldos[dataContractAddress][userAddress] = SafeMath.add(_userEthVolumeSaldos[dataContractAddress][userAddress], volEth);
        trackTotalVolume(dataContractAddress, volEth, volToken);
    }
    function trackSell(address userAddress, uint256 volEth, uint256 volToken) onlyController public {
        address dataContractAddress = Etherama(msg.sender).getDataContractAddress();
        _sellCounts[dataContractAddress] = SafeMath.add(_sellCounts[dataContractAddress], 1);
        _userEthVolumeSaldos[dataContractAddress][userAddress] = SafeMath.sub(_userEthVolumeSaldos[dataContractAddress][userAddress], volEth);
        trackTotalVolume(dataContractAddress, volEth, volToken);
    }
    function trackTotalVolume(address dataContractAddress, uint256 volEth, uint256 volToken) internal {
        _totalVolumeEth[dataContractAddress] = SafeMath.add(_totalVolumeEth[dataContractAddress], volEth);
        _totalVolumeToken[dataContractAddress] = SafeMath.add(_totalVolumeToken[dataContractAddress], volToken);
    }
    function getBuyCount(address dataContractAddress) public view returns (uint256) {
        return _buyCounts[dataContractAddress];
    }
    function getTotalBuyCount() public view returns (uint256 res) {
        for (uint256 i = 0; i < _controllerContractCount; i++) {
            res = SafeMath.add(res, _buyCounts[Etherama(_controllerIndexer[i]).getDataContractAddress()]);
        }         
    }
    function getSellCount(address dataContractAddress) public view returns (uint256) {
        return _sellCounts[dataContractAddress];
    }
    function getTotalSellCount() public view returns (uint256 res) {
        for (uint256 i = 0; i < _controllerContractCount; i++) {
            res = SafeMath.add(res, _sellCounts[Etherama(_controllerIndexer[i]).getDataContractAddress()]);
        }         
    }
    function getTotalVolumeEth(address dataContractAddress) public view returns (uint256) {
        return _totalVolumeEth[dataContractAddress];
    }
    function getTotalVolumeToken(address dataContractAddress) public view returns (uint256) {
        return _totalVolumeToken[dataContractAddress];
    }
    function getUserEthVolumeSaldo(address dataContractAddress, address userAddress) public view returns (uint256) {
        return _userEthVolumeSaldos[dataContractAddress][userAddress];
    }
    function getUserTotalEthVolumeSaldo(address userAddress) public view returns (uint256 res) {
        for (uint256 i = 0; i < _controllerContractCount; i++) {
            res = SafeMath.add(res, _userEthVolumeSaldos[Etherama(_controllerIndexer[i]).getDataContractAddress()][userAddress]);
        } 
    }
    function getTotalCollectedPromoBonus() public view returns (uint256) {
        return SafeMath.add(_currentBigPromoBonus, _currentQuickPromoBonus);
    }
    function getUserTotalPromoBonus(address dataContractAddress, address userAddress) public view returns (uint256) {
        return SafeMath.add(_promoQuickBonuses[dataContractAddress][userAddress], _promoBigBonuses[dataContractAddress][userAddress]);
    }
    function getUserQuickPromoBonus(address dataContractAddress, address userAddress) public view returns (uint256) {
        return _promoQuickBonuses[dataContractAddress][userAddress];
    }
    function getUserBigPromoBonus(address dataContractAddress, address userAddress) public view returns (uint256) {
        return _promoBigBonuses[dataContractAddress][userAddress];
    }
    function getUserTokenLocalBalance(address dataContractAddress, address userAddress) public view returns(uint256) {
        return _userTokenLocalBalances[dataContractAddress][userAddress];
    }
    function addUserTokenLocalBalance(address userAddress, uint256 val) onlyController public {
        address dataContractAddress = Etherama(msg.sender).getDataContractAddress();
        _userTokenLocalBalances[dataContractAddress][userAddress] = SafeMath.add(_userTokenLocalBalances[dataContractAddress][userAddress], val);
    }
    function subUserTokenLocalBalance(address userAddress, uint256 val) onlyController public {
        address dataContractAddress = Etherama(msg.sender).getDataContractAddress();
        _userTokenLocalBalances[dataContractAddress][userAddress] = SafeMath.sub(_userTokenLocalBalances[dataContractAddress][userAddress], val);
    }
    function getUserReward(address dataContractAddress, address userAddress, bool incShareBonus, bool incRefBonus, bool incPromoBonus) public view returns(uint256 reward) {
        EtheramaData data = EtheramaData(dataContractAddress);
        if (incShareBonus) {
            reward = data.getBonusPerShare() * data.getActualUserTokenBalance(userAddress);
            reward = ((reward < data.getUserRewardPayouts(userAddress)) ? 0 : SafeMath.sub(reward, data.getUserRewardPayouts(userAddress))) / MAGNITUDE;
        }
        if (incRefBonus) reward = SafeMath.add(reward, data.getUserRefBalance(userAddress));
        if (incPromoBonus) reward = SafeMath.add(reward, data.getUserTotalPromoBonus(userAddress));
        return reward;
    }
    function getUserTotalReward(address userAddress, bool incShareBonus, bool incRefBonus, bool incPromoBonus) public view returns(uint256 res) {
        for (uint256 i = 0; i < _controllerContractCount; i++) {
            address dataContractAddress = Etherama(_controllerIndexer[i]).getDataContractAddress();
            res = SafeMath.add(res, getUserReward(dataContractAddress, userAddress, incShareBonus, incRefBonus, incPromoBonus));
        }
    }
    function getCurrentUserReward(bool incRefBonus, bool incPromoBonus) public view returns(uint256) {
        return getUserTotalReward(msg.sender, true, incRefBonus, incPromoBonus);
    }
    function getCurrentUserTotalReward() public view returns(uint256) {
        return getUserTotalReward(msg.sender, true, true, true);
    }
    function getCurrentUserShareBonus() public view returns(uint256) {
        return getUserTotalReward(msg.sender, true, false, false);
    }
    function getCurrentUserRefBonus() public view returns(uint256) {
        return getUserTotalReward(msg.sender, false, true, false);
    }
    function getCurrentUserPromoBonus() public view returns(uint256) {
        return getUserTotalReward(msg.sender, false, false, true);
    }
    function isRefAvailable(address refAddress) public view returns(bool) {
        return getUserTotalEthVolumeSaldo(refAddress) >= _minRefEthPurchase;
    }
    function isRefAvailable() public view returns(bool) {
        return isRefAvailable(msg.sender);
    }
    function withdrawUserReward() public {
        uint256 reward = getRewardAndPrepareWithdraw();
        require(reward > 0);
        msg.sender.transfer(reward);
        emit onWithdrawUserBonus(msg.sender, reward);
    }
    function getRewardAndPrepareWithdraw() internal returns(uint256 reward) {
        for (uint256 i = 0; i < _controllerContractCount; i++) {
            address dataContractAddress = Etherama(_controllerIndexer[i]).getDataContractAddress();
            reward = SafeMath.add(reward, getUserReward(dataContractAddress, msg.sender, true, false, false));
            addUserRewardPayouts(dataContractAddress, msg.sender, reward * MAGNITUDE);
            reward = SafeMath.add(reward, getUserRefBalance(dataContractAddress, msg.sender));
            resetUserRefBalance(dataContractAddress, msg.sender);
            reward = SafeMath.add(reward, getUserTotalPromoBonus(dataContractAddress, msg.sender));
            resetUserPromoBonus(dataContractAddress, msg.sender);
        }
        return reward;
    }
    function withdrawRemainingEthAfterAll() onlyAdministrator public {
        for (uint256 i = 0; i < _controllerContractCount; i++) {
            if (Etherama(_controllerIndexer[i]).isActive()) revert();
        }
        msg.sender.transfer(address(this).balance);
    }
    function calcPercent(uint256 amount, uint256 percent) public pure returns(uint256) {
        return SafeMath.div(SafeMath.mul(SafeMath.div(amount, 100), percent), 1 ether);
    }
    function convertRealTo256(int128 realVal) public pure returns(uint256) {
        int128 roundedVal = RealMath.fromReal(RealMath.mul(realVal, RealMath.toReal(1e12)));
        return SafeMath.mul(uint256(roundedVal), uint256(1e6));
    }
    function convert256ToReal(uint256 val) public pure returns(int128) {
        uint256 intVal = SafeMath.div(val, 1e6);
        require(RealMath.isUInt256ValidIn64(intVal));
        return RealMath.fraction(int64(intVal), 1e12);
    }    
}
contract EtheramaData {
    address constant public TOKEN_CONTRACT_ADDRESS = 0x83cee9e086A77e492eE0bB93C2B0437aD6fdECCc;
    uint256 constant public TOKEN_PRICE_INITIAL = 0.0023 ether;
    uint64 constant public PRICE_SPEED_PERCENT = 5;
    uint64 constant public PRICE_SPEED_INTERVAL = 10000;
    uint64 constant public EXP_PERIOD_DAYS = 365;
    mapping(address => bool) private _administrators;
    uint256 private  _administratorCount;
    uint64 public _initTime;
    uint64 public _expirationTime;
    uint256 public _tokenOwnerReward;
    uint256 public _totalSupply;
    int128 public _realTokenPrice;
    address public _controllerAddress = address(0x0);
    EtheramaCore public _core;
    uint256 public _initBlockNum;
    bool public _hasMaxPurchaseLimit = false;
    IStdToken public _token;
    modifier onlyController() {
        require(msg.sender == _controllerAddress);
        _;
    }
    constructor(address coreAddress) public {
        require(coreAddress != address(0x0));
        _core = EtheramaCore(coreAddress);
        _initBlockNum = block.number;
    }
    function init() public {
        require(_controllerAddress == address(0x0));
        require(TOKEN_CONTRACT_ADDRESS != address(0x0));
        require(RealMath.isUInt64ValidIn64(PRICE_SPEED_PERCENT) && PRICE_SPEED_PERCENT > 0);
        require(RealMath.isUInt64ValidIn64(PRICE_SPEED_INTERVAL) && PRICE_SPEED_INTERVAL > 0);
        _controllerAddress = msg.sender;
        _token = IStdToken(TOKEN_CONTRACT_ADDRESS);
        _initTime = uint64(now);
        _expirationTime = _initTime + EXP_PERIOD_DAYS * 1 days;
        _realTokenPrice = _core.convert256ToReal(TOKEN_PRICE_INITIAL);
    }
    function isInited()  public view returns(bool) {
        return (_controllerAddress != address(0x0));
    }
    function getCoreAddress()  public view returns(address) {
        return address(_core);
    }
    function setNewControllerAddress(address newAddress) onlyController public {
        _controllerAddress = newAddress;
    }
    function getPromoMinPurchaseEth() public view returns(uint256) {
        return _core._promoMinPurchaseEth();
    }
    function addAdministator(address addr) onlyController public {
        _administrators[addr] = true;
        _administratorCount = SafeMath.add(_administratorCount, 1);
    }
    function removeAdministator(address addr) onlyController public {
        _administrators[addr] = false;
        _administratorCount = SafeMath.sub(_administratorCount, 1);
    }
    function getAdministratorCount() public view returns(uint256) {
        return _administratorCount;
    }
    function isAdministrator(address addr) public view returns(bool) {
        return _administrators[addr];
    }
    function getCommonInitBlockNum() public view returns (uint256) {
        return _core.getInitBlockNum();
    }
    function resetTokenOwnerReward() onlyController public {
        _tokenOwnerReward = 0;
    }
    function addTokenOwnerReward(uint256 val) onlyController public {
        _tokenOwnerReward = SafeMath.add(_tokenOwnerReward, val);
    }
    function getCurrentBigPromoBonus() public view returns (uint256) {
        return _core._currentBigPromoBonus();
    }        
    function getCurrentQuickPromoBonus() public view returns (uint256) {
        return _core._currentQuickPromoBonus();
    }    
    function getTotalCollectedPromoBonus() public view returns (uint256) {
        return _core.getTotalCollectedPromoBonus();
    }    
    function setTotalSupply(uint256 val) onlyController public {
        _totalSupply = val;
    }
    function setRealTokenPrice(int128 val) onlyController public {
        _realTokenPrice = val;
    }    
    function setHasMaxPurchaseLimit(bool val) onlyController public {
        _hasMaxPurchaseLimit = val;
    }
    function getUserTokenLocalBalance(address userAddress) public view returns(uint256) {
        return _core.getUserTokenLocalBalance(address(this), userAddress);
    }
    function getActualUserTokenBalance(address userAddress) public view returns(uint256) {
        return SafeMath.min(getUserTokenLocalBalance(userAddress), _token.balanceOf(userAddress));
    }  
    function getBonusPerShare() public view returns(uint256) {
        return _core.getBonusPerShare(address(this));
    }
    function getUserRewardPayouts(address userAddress) public view returns(uint256) {
        return _core.getUserRewardPayouts(address(this), userAddress);
    }
    function getUserRefBalance(address userAddress) public view returns(uint256) {
        return _core.getUserRefBalance(address(this), userAddress);
    }
    function getUserReward(address userAddress, bool incRefBonus, bool incPromoBonus) public view returns(uint256) {
        return _core.getUserReward(address(this), userAddress, true, incRefBonus, incPromoBonus);
    }
    function getUserTotalPromoBonus(address userAddress) public view returns(uint256) {
        return _core.getUserTotalPromoBonus(address(this), userAddress);
    }
    function getUserBigPromoBonus(address userAddress) public view returns(uint256) {
        return _core.getUserBigPromoBonus(address(this), userAddress);
    }
    function getUserQuickPromoBonus(address userAddress) public view returns(uint256) {
        return _core.getUserQuickPromoBonus(address(this), userAddress);
    }
    function getRemainingTokenAmount() public view returns(uint256) {
        return _token.balanceOf(_controllerAddress);
    }
    function getTotalTokenSold() public view returns(uint256) {
        return _totalSupply - getRemainingTokenAmount();
    }   
    function getUserEthVolumeSaldo(address userAddress) public view returns(uint256) {
        return _core.getUserEthVolumeSaldo(address(this), userAddress);
    }
}
contract Etherama {
    IStdToken public _token;
    EtheramaData public _data;
    EtheramaCore public _core;
    bool public isActive = false;
    bool public isMigrationToNewControllerInProgress = false;
    bool public isActualContractVer = true;
    address public migrationContractAddress = address(0x0);
    bool public isMigrationApproved = false;
    address private _creator = address(0x0);
    event onTokenPurchase(address indexed userAddress, uint256 incomingEth, uint256 tokensMinted, address indexed referredBy);
    event onTokenSell(address indexed userAddress, uint256 tokensBurned, uint256 ethEarned);
    event onReinvestment(address indexed userAddress, uint256 ethReinvested, uint256 tokensMinted);
    event onWithdrawTokenOwnerReward(address indexed toAddress, uint256 ethWithdrawn); 
    event onWinQuickPromo(address indexed userAddress, uint256 ethWon);    
    event onWinBigPromo(address indexed userAddress, uint256 ethWon);    
    modifier onlyContractUsers() {
        require(getUserLocalTokenBalance(msg.sender) > 0);
        _;
    }
    modifier onlyAdministrator() {
        require(isCurrentUserAdministrator());
        _;
    }
    modifier onlyCoreAdministrator() {
        require(_core.isAdministrator(msg.sender));
        _;
    }
    modifier onlyActive() {
        require(isActive);
        _;
    }
    modifier validGasPrice() {
        require(tx.gasprice <= _core.MAX_GAS_PRICE());
        _;
    }
    modifier validPayableValue() {
        require(msg.value > 0);
        _;
    }
    modifier onlyCoreContract() {
        require(msg.sender == _data.getCoreAddress());
        _;
    }
    constructor(address dataContractAddress) public {
        require(dataContractAddress != address(0x0));
        _data = EtheramaData(dataContractAddress);
        if (!_data.isInited()) {