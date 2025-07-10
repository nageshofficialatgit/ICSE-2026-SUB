pragma solidity ^0.4.24;
contract Star3Devents {
    event onNewName
    (
        uint256 indexed playerID,
        address indexed playerAddress,
        bytes32 indexed playerName,
        bool isNewPlayer,
        uint256 affiliateID,
        address affiliateAddress,
        bytes32 affiliateName,
        uint256 amountPaid,
        uint256 timeStamp
    );
    event onEndTx
    (
        uint256 compressedData,
        uint256 compressedIDs,
        bytes32 playerName,
        address playerAddress,
        uint256 ethIn,
        uint256 keysBought,
        address winnerAddr,
        bytes32 winnerName,
        uint256 amountWon,
        uint256 newPot,
        uint256 genAmount,
        uint256 potAmount
    );
    event onWithdraw
    (
        uint256 indexed playerID,
        address playerAddress,
        bytes32 playerName,
        uint256 ethOut,
        uint256 timeStamp
    );
    event onWithdrawAndDistribute
    (
        address playerAddress,
        bytes32 playerName,
        uint256 ethOut,
        uint256 compressedData,
        uint256 compressedIDs,
        address winnerAddr,
        bytes32 winnerName,
        uint256 amountWon,
        uint256 newPot,
        uint256 genAmount
    );
    event onBuyAndDistribute
    (
        address playerAddress,
        bytes32 playerName,
        uint256 ethIn,
        uint256 compressedData,
        uint256 compressedIDs,
        address winnerAddr,
        bytes32 winnerName,
        uint256 amountWon,
        uint256 newPot,
        uint256 genAmount
    );
    event onReLoadAndDistribute
    (
        address playerAddress,
        bytes32 playerName,
        uint256 compressedData,
        uint256 compressedIDs,
        address winnerAddr,
        bytes32 winnerName,
        uint256 amountWon,
        uint256 newPot,
        uint256 genAmount
    );
    event onAffiliatePayout
    (
        uint256 indexed affiliateID,
        address affiliateAddress,
        bytes32 affiliateName,
        uint256 indexed roundID,
        uint256 indexed buyerID,
        uint256 amount,
        uint256 timeStamp
    );
    event onPotSwapDeposit
    (
        uint256 roundID,
        uint256 amountAddedToPot
    );
}
interface CompanyShareInterface {
    function deposit() external payable;
}
contract modularLong is Star3Devents {}
contract Star3Dlong is modularLong {
    using SafeMath for *;
    using NameFilter for string;
    using Star3DKeysCalcLong for uint256;
    string constant public name = "Save the planet";
    string constant public symbol = "Star";
    CompanyShareInterface constant private CompanyShare = CompanyShareInterface(0x03347ABb58cc3071FDBBa7F7bD7cca03C8E04229);
    uint256 private pID_ = 0;   
	uint256 private rndExtra_ = 0 hours;     
    uint256 private rndGap_ = 10 seconds;         
    uint256 constant private rndInit_ = 10 hours;                     
    uint256 constant private rndInc_ = 30 seconds;               
    uint256 constant private rndMax_ = 24 hours;                     
    uint256 public registrationFee_ = 10 finney;               
    uint256 public rID_;    
    mapping (address => uint256) public pIDxAddr_;          
    mapping (bytes32 => uint256) public pIDxName_;          
    mapping (uint256 => Star3Ddatasets.Player) public plyr_;   
    mapping (uint256 => mapping (uint256 => Star3Ddatasets.PlayerRounds)) public plyrRnds_;    
    mapping (uint256 => mapping (bytes32 => bool)) public plyrNames_; 
    mapping (uint256 => Star3Ddatasets.Round) public round_;   
    mapping (uint256 => mapping(uint256 => uint256)) public rndTmEth_;      
    mapping (uint256 => Star3Ddatasets.TeamFee) public fees_;          
    mapping (uint256 => Star3Ddatasets.PotSplit) public potSplit_;     
    constructor()
        public
    {
        fees_[0] = Star3Ddatasets.TeamFee(32, 45, 10, 3);	
        fees_[1] = Star3Ddatasets.TeamFee(45, 32, 10, 3);	
        fees_[2] = Star3Ddatasets.TeamFee(50, 27, 10, 3);	
        fees_[3] = Star3Ddatasets.TeamFee(40, 37, 10, 3);	
        potSplit_[0] = Star3Ddatasets.PotSplit(20, 30);  
        potSplit_[1] = Star3Ddatasets.PotSplit(15, 35);  
        potSplit_[2] = Star3Ddatasets.PotSplit(25, 25);  
        potSplit_[3] = Star3Ddatasets.PotSplit(30, 20);  
	}
    modifier isActivated() {
        require(activated_ == true, "its not ready yet.  check ?eta in discord");
        _;
    }
    modifier isRegisteredName()
    {
        uint256 _pID = pIDxAddr_[msg.sender];
        require(plyr_[_pID].name == "" || _pID == 0, "already has name");
        _;
    }
    modifier isHuman() {
        address _addr = msg.sender;
        uint256 _codeLength;
        assembly {_codeLength := extcodesize(_addr)}
        require(_codeLength == 0, "sorry humans only");
        _;
    }
    modifier isWithinLimits(uint256 _eth) {
        require(_eth >= 1000000000, "pocket lint: not a valid currency");
        require(_eth <= 100000000000000000000000, "no vitalik, no");
        _;
    }
    function()
        isActivated()
        isHuman()
        isWithinLimits(msg.value)
        public
        payable
    {
        Star3Ddatasets.EventReturns memory _eventData_ = determinePID(_eventData_);
        uint256 _pID = pIDxAddr_[msg.sender];
        buyCore(_pID, plyr_[_pID].laff, 2, _eventData_);
    }
    function buyXid(uint256 _affCode, uint256 _team)
        isActivated()
        isHuman()
        isWithinLimits(msg.value)
        public
        payable
    {
        Star3Ddatasets.EventReturns memory _eventData_ = determinePID(_eventData_);
        uint256 _pID = pIDxAddr_[msg.sender];
        if (_affCode == 0 || _affCode == _pID)
        {
            _affCode = plyr_[_pID].laff;
        } else if (_affCode != plyr_[_pID].laff) {
            plyr_[_pID].laff = _affCode;
        }
        _team = verifyTeam(_team);
        buyCore(_pID, _affCode, _team, _eventData_);
    }
    function buyXaddr(address _affCode, uint256 _team)
        isActivated()
        isHuman()
        isWithinLimits(msg.value)
        public
        payable
    {
        Star3Ddatasets.EventReturns memory _eventData_ = determinePID(_eventData_);
        uint256 _pID = pIDxAddr_[msg.sender];
        _team = verifyTeam(_team);
        uint256 _affID;
        if (_affCode == address(0) || _affCode == msg.sender)
        {
            _affID = plyr_[_pID].laff;
        } else {
            _affID = pIDxAddr_[_affCode];
            if (_affID != plyr_[_pID].laff)
            {
                plyr_[_pID].laff = _affID;
            }
        }
        buyCore(_pID, _affID, _team, _eventData_);
    }
    function buyXname(bytes32 _affCode, uint256 _team)
        isActivated()
        isHuman()
        isWithinLimits(msg.value)
        public
        payable
    {
        Star3Ddatasets.EventReturns memory _eventData_ = determinePID(_eventData_);
        uint256 _pID = pIDxAddr_[msg.sender];
        uint256 _affID;
        if (_affCode == '' || _affCode == plyr_[_pID].name)
        {
            _affID = plyr_[_pID].laff;
        } else {
            _affID = pIDxName_[_affCode];
            if (_affID != plyr_[_pID].laff)
            {
                plyr_[_pID].laff = _affID;
            }
        }
        _team = verifyTeam(_team);
        buyCore(_pID, _affID, _team, _eventData_);
    }
    function reLoadXid(uint256 _affCode, uint256 _team, uint256 _eth)
        isActivated()
        isHuman()
        isWithinLimits(_eth)
        public
    {
        Star3Ddatasets.EventReturns memory _eventData_;
        uint256 _pID = pIDxAddr_[msg.sender];
        if (_affCode == 0 || _affCode == _pID)
        {
            _affCode = plyr_[_pID].laff;
        } else if (_affCode != plyr_[_pID].laff) {
            plyr_[_pID].laff = _affCode;
        }
        _team = verifyTeam(_team);
        reLoadCore(_pID, _affCode, _team, _eth, _eventData_);
    }
    function reLoadXaddr(address _affCode, uint256 _team, uint256 _eth)
        isActivated()
        isHuman()
        isWithinLimits(_eth)
        public
    {
        Star3Ddatasets.EventReturns memory _eventData_;
        uint256 _pID = pIDxAddr_[msg.sender];
        uint256 _affID;
        if (_affCode == address(0) || _affCode == msg.sender)
        {
            _affID = plyr_[_pID].laff;
        } else {
            _affID = pIDxAddr_[_affCode];
            if (_affID != plyr_[_pID].laff)
            {
                plyr_[_pID].laff = _affID;
            }
        }
        _team = verifyTeam(_team);
        reLoadCore(_pID, _affID, _team, _eth, _eventData_);
    }
    function reLoadXname(bytes32 _affCode, uint256 _team, uint256 _eth)
        isActivated()
        isHuman()
        isWithinLimits(_eth)
        public
    {
        Star3Ddatasets.EventReturns memory _eventData_;
        uint256 _pID = pIDxAddr_[msg.sender];
        uint256 _affID;
        if (_affCode == '' || _affCode == plyr_[_pID].name)
        {
            _affID = plyr_[_pID].laff;
        } else {
            _affID = pIDxName_[_affCode];
            if (_affID != plyr_[_pID].laff)
            {
                plyr_[_pID].laff = _affID;
            }
        }
        _team = verifyTeam(_team);
        reLoadCore(_pID, _affID, _team, _eth, _eventData_);
    }
    function withdraw()
        isActivated()
        isHuman()
        public
    {
        uint256 _rID = rID_;
        uint256 _now = now;
        uint256 _pID = pIDxAddr_[msg.sender];
        uint256 _eth;
        if (_now > round_[_rID].end && round_[_rID].ended == false && round_[_rID].plyr != 0)
        {
            Star3Ddatasets.EventReturns memory _eventData_;
			round_[_rID].ended = true;
            _eventData_ = endRound(_eventData_);
            _eth = withdrawEarnings(_pID);
            if (_eth > 0)
                plyr_[_pID].addr.transfer(_eth);
            _eventData_.compressedData = _eventData_.compressedData + (_now * 1000000000000000000);
            _eventData_.compressedIDs = _eventData_.compressedIDs + _pID;
            emit Star3Devents.onWithdrawAndDistribute
            (
                msg.sender,
                plyr_[_pID].name,
                _eth,
                _eventData_.compressedData,
                _eventData_.compressedIDs,
                _eventData_.winnerAddr,
                _eventData_.winnerName,
                _eventData_.amountWon,
                _eventData_.newPot,
                _eventData_.genAmount
            );
        } else {
            _eth = withdrawEarnings(_pID);
            if (_eth > 0)
                plyr_[_pID].addr.transfer(_eth);
            emit Star3Devents.onWithdraw(_pID, msg.sender, plyr_[_pID].name, _eth, _now);
        }
    }
    function registerNameXID(string _nameString, uint256 _affCode)
        isHuman()
        isRegisteredName()
        public
        payable
    {
        bytes32 _name = _nameString.nameFilter();
        address _addr = msg.sender;
        uint256 _paid = msg.value;
        bool _isNewPlayer = isNewPlayer(_addr);
        require (msg.value >= registrationFee_, "umm.....  you have to pay the name fee");
        Star3Ddatasets.EventReturns memory _eventData_ = determinePID(_eventData_);
        uint256 _pID = makePlayerID(msg.sender);
        uint256 _affID = _affCode;
        if (_affID != 0 && _affID != plyr_[_pID].laff && _affID != _pID)
        {
            plyr_[_pID].laff = _affID;
        } else if (_affID == _pID) {
            _affID = 0;
        }
        registerNameCore(_pID, _name);
        emit Star3Devents.onNewName(_pID, _addr, _name, _isNewPlayer, _affID, plyr_[_affID].addr, plyr_[_affID].name, _paid, now);
    }
    function registerNameXaddr(string _nameString, address _affCode)
        isHuman()
        isRegisteredName()
        public
        payable
    {
        bytes32 _name = _nameString.nameFilter();
        address _addr = msg.sender;
        uint256 _paid = msg.value;
        require (msg.value >= registrationFee_, "umm.....  you have to pay the name fee");
        bool _isNewPlayer = isNewPlayer(_addr);
        Star3Ddatasets.EventReturns memory _eventData_ = determinePID(_eventData_);
        uint256 _pID = makePlayerID(msg.sender);
        uint256 _affID;
        if (_affCode != address(0) && _affCode != _addr)
        {
            _affID = pIDxAddr_[_affCode];
            if (_affID != plyr_[_pID].laff)
            {
                plyr_[_pID].laff = _affID;
            }
        }
        registerNameCore(_pID, _name);
        emit Star3Devents.onNewName(_pID, _addr, _name, _isNewPlayer, _affID, plyr_[_affID].addr, plyr_[_affID].name, _paid, now);
    }
    function registerNameXname(string _nameString, bytes32 _affCode)
        isHuman()
        isRegisteredName()
        public
        payable
    {
        bytes32 _name = _nameString.nameFilter();
        address _addr = msg.sender;
        uint256 _paid = msg.value;
        require (msg.value >= registrationFee_, "umm.....  you have to pay the name fee");
        bool _isNewPlayer = isNewPlayer(_addr);
        Star3Ddatasets.EventReturns memory _eventData_ = determinePID(_eventData_);
        uint256 _pID = makePlayerID(msg.sender);
        uint256 _affID;
        if (_affCode != "" && _affCode != _name)
        {
            _affID = pIDxName_[_affCode];
            if (_affID != plyr_[_pID].laff)
            {
                plyr_[_pID].laff = _affID;
            }
        }
        registerNameCore(_pID, _name);
        emit Star3Devents.onNewName(_pID, _addr, _name, _isNewPlayer, _affID, plyr_[_affID].addr, plyr_[_affID].name, _paid, now);
    }
    function registerNameCore(uint256 _pID, bytes32 _name)
        private
    {
        if (pIDxName_[_name] != 0)
            require(plyrNames_[_pID][_name] == true, "sorry that names already taken");
        plyr_[_pID].name = _name;
        pIDxName_[_name] = _pID;
        if (plyrNames_[_pID][_name] == false)
        {
            plyrNames_[_pID][_name] = true;
        }
        CompanyShare.deposit.value(msg.value)();
    }
    function isNewPlayer(address _addr)
    public
    view
    returns (bool)
    {
        if (pIDxAddr_[_addr] == 0)
        {
            return (true);
        } else {
            return (false);
        }
    }
    function getBuyPrice()
        public
        view
        returns(uint256)
    {
        uint256 _rID = rID_;
        uint256 _now = now;
        if (_now > round_[_rID].strt + rndGap_ && (_now <= round_[_rID].end || (_now > round_[_rID].end && round_[_rID].plyr == 0)))
            return ( (round_[_rID].keys.add(1000000000000000000)).ethRec(1000000000000000000) );
        else 
            return ( 75000000000000 ); 
    }
    function getTimeLeft()
        public
        view
        returns(uint256)
    {
        uint256 _rID = rID_;
        uint256 _now = now;
        if (_now < round_[_rID].end)
            if (_now > round_[_rID].strt + rndGap_)
                return( (round_[_rID].end).sub(_now) );
            else
                return( (round_[_rID].strt + rndGap_).sub(_now) );
        else
            return(0);
    }
    function getPlayerVaults(uint256 _pID)
        public
        view
        returns(uint256 ,uint256, uint256)
    {
        uint256 _rID = rID_;
        if (now > round_[_rID].end && round_[_rID].ended == false && round_[_rID].plyr != 0)
        {
            if (round_[_rID].plyr == _pID)
            {
                return
                (
                    (plyr_[_pID].win).add( ((round_[_rID].pot).mul(48)) / 100 ),
                    (plyr_[_pID].gen).add(  getPlayerVaultsHelper(_pID, _rID).sub(plyrRnds_[_pID][_rID].mask)   ),
                    plyr_[_pID].aff
                );
            } else {
                return
                (
                    plyr_[_pID].win,
                    (plyr_[_pID].gen).add(  getPlayerVaultsHelper(_pID, _rID).sub(plyrRnds_[_pID][_rID].mask)  ),
                    plyr_[_pID].aff
                );
            }
        } else {
            return
            (
                plyr_[_pID].win,
                (plyr_[_pID].gen).add(calcUnMaskedEarnings(_pID, plyr_[_pID].lrnd)),
                plyr_[_pID].aff
            );
        }
    }
    function getPlayerVaultsHelper(uint256 _pID, uint25