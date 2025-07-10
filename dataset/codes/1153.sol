pragma solidity ^0.4.24;
contract RSEvents {
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
        uint256 potAmount,
        uint256 airDropPot
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
        uint256 indexed buyerID,
        uint256 amount,
        uint256 timeStamp
    );
}
contract modularRatScam is RSEvents {}
contract RatScam is modularRatScam {
    using SafeMath for *;
    using NameFilter for string;
    using RSKeysCalc for uint256;
    RatBookInterface constant private RatBook = RatBookInterface(0x3257d637B8977781B4f8178365858A474b2A6195);
    string constant public name = "RatScam In One Hour";
    string constant public symbol = "RS";
    uint256 private rndGap_ = 0;
    uint256 private rndInit_ = 1 hours;                
    uint256 private rndInc_ = 30 seconds;              
    uint256 private rndMax_ = 1 hours;                
    uint256 public airDropPot_;             
    uint256 public airDropTracker_ = 0;     
    uint256 public rID_;    
    address private adminAddress;
    mapping (address => uint256) public pIDxAddr_;          
    mapping (bytes32 => uint256) public pIDxName_;          
    mapping (uint256 => RSdatasets.Player) public plyr_;   
    mapping (uint256 => mapping (uint256 => RSdatasets.PlayerRounds)) public plyrRnds_;    
    mapping (uint256 => mapping (bytes32 => bool)) public plyrNames_; 
    mapping (uint256 => RSdatasets.Round) public round_;   
    uint256 public fees_ = 60;          
    uint256 public potSplit_ = 45;     
    constructor()
    public
    {
        adminAddress = msg.sender;
    }
    modifier isActivated() {
        require(activated_ == true, "its not ready yet");
        _;
    }
    modifier isHuman() {
        address _addr = msg.sender;
        uint256 _codeLength;
        assembly {_codeLength := extcodesize(_addr)}
        require(_codeLength == 0, "non smart contract address only");
        _;
    }
    modifier isWithinLimits(uint256 _eth) {
        require(_eth >= 1000000000, "too little money");
        require(_eth <= 100000000000000000000000, "too much money");
        _;
    }
    function()
    isActivated()
    isHuman()
    isWithinLimits(msg.value)
    public
    payable
    {
        RSdatasets.EventReturns memory _eventData_ = determinePID(_eventData_);
        uint256 _pID = pIDxAddr_[msg.sender];
        buyCore(_pID, plyr_[_pID].laff, _eventData_);
    }
    function buyXid(uint256 _affCode)
    isActivated()
    isHuman()
    isWithinLimits(msg.value)
    public
    payable
    {
        RSdatasets.EventReturns memory _eventData_ = determinePID(_eventData_);
        uint256 _pID = pIDxAddr_[msg.sender];
        if (_affCode == 0 || _affCode == _pID)
        {
            _affCode = plyr_[_pID].laff;
        } else if (_affCode != plyr_[_pID].laff) {
            plyr_[_pID].laff = _affCode;
        }
        buyCore(_pID, _affCode, _eventData_);
    }
    function buyXaddr(address _affCode)
    isActivated()
    isHuman()
    isWithinLimits(msg.value)
    public
    payable
    {
        RSdatasets.EventReturns memory _eventData_ = determinePID(_eventData_);
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
        buyCore(_pID, _affID, _eventData_);
    }
    function buyXname(bytes32 _affCode)
    isActivated()
    isHuman()
    isWithinLimits(msg.value)
    public
    payable
    {
        RSdatasets.EventReturns memory _eventData_ = determinePID(_eventData_);
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
        buyCore(_pID, _affID, _eventData_);
    }
    function reLoadXid(uint256 _affCode, uint256 _eth)
    isActivated()
    isHuman()
    isWithinLimits(_eth)
    public
    {
        RSdatasets.EventReturns memory _eventData_;
        uint256 _pID = pIDxAddr_[msg.sender];
        if (_affCode == 0 || _affCode == _pID)
        {
            _affCode = plyr_[_pID].laff;
        } else if (_affCode != plyr_[_pID].laff) {
            plyr_[_pID].laff = _affCode;
        }
        reLoadCore(_pID, _affCode, _eth, _eventData_);
    }
    function reLoadXaddr(address _affCode, uint256 _eth)
    isActivated()
    isHuman()
    isWithinLimits(_eth)
    public
    {
        RSdatasets.EventReturns memory _eventData_;
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
        reLoadCore(_pID, _affID, _eth, _eventData_);
    }
    function reLoadXname(bytes32 _affCode, uint256 _eth)
    isActivated()
    isHuman()
    isWithinLimits(_eth)
    public
    {
        RSdatasets.EventReturns memory _eventData_;
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
        reLoadCore(_pID, _affID, _eth, _eventData_);
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
            RSdatasets.EventReturns memory _eventData_;
            round_[_rID].ended = true;
            _eventData_ = endRound(_eventData_);
            _eth = withdrawEarnings(_pID);
            if (_eth > 0)
                plyr_[_pID].addr.transfer(_eth);
            _eventData_.compressedData = _eventData_.compressedData + (_now * 1000000000000000000);
            _eventData_.compressedIDs = _eventData_.compressedIDs + _pID;
            emit RSEvents.onWithdrawAndDistribute
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
            emit RSEvents.onWithdraw(_pID, msg.sender, plyr_[_pID].name, _eth, _now);
        }
    }
    function registerNameXID(string _nameString, uint256 _affCode, bool _all)
    isHuman()
    public
    payable
    {
        bytes32 _name = _nameString.nameFilter();
        address _addr = msg.sender;
        uint256 _paid = msg.value;
        (bool _isNewPlayer, uint256 _affID) = RatBook.registerNameXIDFromDapp.value(_paid)(_addr, _name, _affCode, _all);
        uint256 _pID = pIDxAddr_[_addr];
        emit RSEvents.onNewName(_pID, _addr, _name, _isNewPlayer, _affID, plyr_[_affID].addr, plyr_[_affID].name, _paid, now);
    }
    function registerNameXaddr(string _nameString, address _affCode, bool _all)
    isHuman()
    public
    payable
    {
        bytes32 _name = _nameString.nameFilter();
        address _addr = msg.sender;
        uint256 _paid = msg.value;
        (bool _isNewPlayer, uint256 _affID) = RatBook.registerNameXaddrFromDapp.value(msg.value)(msg.sender, _name, _affCode, _all);
        uint256 _pID = pIDxAddr_[_addr];
        emit RSEvents.onNewName(_pID, _addr, _name, _isNewPlayer, _affID, plyr_[_affID].addr, plyr_[_affID].name, _paid, now);
    }
    function registerNameXname(string _nameString, bytes32 _affCode, bool _all)
    isHuman()
    public
    payable
    {
        bytes32 _name = _nameString.nameFilter();
        address _addr = msg.sender;
        uint256 _paid = msg.value;
        (bool _isNewPlayer, uint256 _affID) = RatBook.registerNameXnameFromDapp.value(msg.value)(msg.sender, _name, _affCode, _all);
        uint256 _pID = pIDxAddr_[_addr];
        emit RSEvents.onNewName(_pID, _addr, _name, _isNewPlayer, _affID, plyr_[_affID].addr, plyr_[_affID].name, _paid, now);
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
                return( (round_[_rID].strt + rndGap_).sub(_now));
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
    function getPlayerVaultsHelper(uint256 _pID, uint256 _rID)
    private
    view
    returns(uint256)
    {
        return(  ((((round_[_rID].mask).add(((((round_[_rID].pot).mul(potSplit_)) / 100).mul(1000000000000000000)) / (round_[_rID].keys))).mul(plyrRnds_[_pID][_rID].keys)) / 1000000000000000000)  );
    }
    function getCurrentRoundInfo()
    public
    view
    returns(uint256, uint256, uint256, uint256, uint256, address, bytes32, uint256)
    {
        uint256 _rID = rID_;
        return
        (
        round_[rID_].keys,              
        round_[rID_].end,               
        round_[rID_].strt,              
        round_[rID_].pot,               
        round_[rID_].plyr,              
        plyr_[round_[rID_].plyr].addr,  
        plyr_[round_[rID_].plyr].name,  
        airDropTracker_ + (airDropPot_ * 1000)              
        );
    }
    function getPlayerInfoByAddress(address _addr)
    public
    view
    returns(uint256, bytes32, uint256, uint256, uint256, uint256, uint256)
    {
        uint256 _rID = rID_;
        if (_addr == address(0))
        {
            _addr == msg.sender;
        }
        uint256 _pID = pIDxAddr_[_addr];
        return
        (
        _pID,                               
        plyr_[_pID].name,                   
        plyrRnds_[_pID][_rID].keys,         
        plyr_[_pID].win,                    
        (plyr_[_pID].gen).add(calcUnMaskedEarnings(_pID, plyr_[_pID].lrnd)),       
        plyr_[_pID].aff,                    
        plyrRnds_[_pID][_rID].eth           
        );
    }
    function buyCore(uint256 _pID, uint256 _affID, RSdatasets.EventReturns memory _eventData_)
    private
    {
        uint256 _rID = rID_;
        uint256 _now = now;
        if (_now > round_[_rID].end && round_[_rID].ended == false)
        {
            round_[_rID].ended = true;
            _eventData_ = endRound(_eventData_);
            _eventData_.compressedData = _eventData_.compressedData + (_now * 1000000000000000000);
            _eventData_.compressedIDs = _eventData_.compressedIDs + _pID;
            emit RSEvents.onBuyAndDistribute
            (
                msg.sender,
                plyr_[_pID].name,
                msg.value,
                _eventData_.compressedData,
                _eventData_.compressedIDs,
                _eventData_.winnerAddr,
                _eventData_.winnerName,
                _eventData_.amountWon,
                _eventData_.newPot,
                _eventData_.genAmount
            );
        }
        _rID = rID_;
        core(_rID, _pID, msg.value, _affID, _eventData_);
    }
    function reLoadCore(uint256 _pID, uint256 _affID, uint256 _eth, RSdatasets.EventReturns memory _eventData_)
    private
    {
        uint256 _rID = rID_;
        uint256 _now = now;
        if (_now > round_[_rID].end && round_[_rID].ended == false)
        {
            round_[_rID].ended = true;
            _eventData_ = endRound(_eventData_);
            _eventData_.compressedData = _eventData_.compressedData + (_now * 1000000000000000000);
            _eventData_.compressedIDs = _eventData_.compressedIDs + _pID;
            emit RSEvents.onBuyAndDistribute
            (
                msg.sender,
                plyr_[_pID].name,
                msg.value,
                _eventData_.compressedData,
                _eventData_.compressedIDs,
                _eventData_.winnerAddr,
                _eventData_.winnerName,
                _eventData_.amountWon,
                _eventData_.newPot,
                _eventData_.genAmount
            );
        }
        plyr_[_pID].gen = withdrawEarnings(_pID).sub(_eth);
        core( _rID, _pID, _eth, _affID, _eventData_);
    }
    function core(uint256 _rID, uint256 _pID, uint256 _eth, uint256 _affID, RSdatasets.EventReturns memory _eventData_)
    private
    {
        if (plyrRnds_[_pID][_rID].keys == 0)
            _eventData_ = managePlayer(_pID, _eventData_);
        if (round_[_rID].eth < 100000000000000000000 && plyrRnds_[_pID][_rID].eth.add(_eth) > 10000000000000000000)
        {
            uint256 _availableLimit = (10000000000000000000).sub(plyrRnds_[_pID][_rID].eth);
            uint256 _refund = _eth.sub(_availableLimit);
            plyr_[_pID].gen = plyr_[_pID].gen.add(_refund);
            _eth = _availableLimit;
        }
        if (_eth > 1000000000)
        {
            uint256 _keys = (round_[_rID].eth).keysRec(_eth);
            if (_keys >= 1000000000000000000)
            {
                updateTimer(_keys, _rID);
                if (round_[_rID].plyr != _pID)
                    round_[_rID].plyr = _pID;
                _eventData_.compressedData = _eventData_.compressedData + 100;
            }
            if (_eth >= 100000000000000000)
            {
                airDropTracker_++;
                if (airdrop() == true)
                {
                    uint256 _prize;
                    if (_eth >= 10000000000000000000)
                    {
                        _prize = ((airDropPot_).mul(75)) / 100;
                        plyr_[_pID].win = (plyr_[_pID].win).add(_prize);
                        airDropPot_ = (airDropPot_).sub(_prize);
                        _eventData_.compressedData += 300000000000000000000000000000000;
                    } else if (_eth >= 1000000000000000000 && _eth < 10000000000000000000) {
                        _prize = ((airDropPot_).mul(50)) / 100;
                        plyr_[_pID].win = (plyr_[_pID].win).add(_prize);
                        airDropPot_ = (airDropPot_).sub(_prize);
                        _eventData_.compressedData += 200000000000000000000000000000000;
                    } else if (_eth >= 100000000000000000 && _eth < 1000000000000000000) {
                        _prize = ((airDropPot_).mul(25)) / 100;
                        plyr_[_pID].win = (plyr_[_pID].win).add(_prize);
                        airDropPot_ = (airDropPot_).sub(_prize);
                        _eventData_.compressedData += 100000000000000000000000000000000;
                    }
                    _eventData_.compressedData += 10000000000000000000000000000000;
                    _eventData_.compressedData += _prize * 1000000000000000000000000000000000;
                    airDropTracker_ = 0;
                }
            }
            _eventData_.compressedData = _eventData_.compressedData + (airDropTracker_ * 1000);
            plyrRnds_[_pID][_rID].keys = _keys.add(plyrRnds_[_pID][_rID].keys);
            plyrRnds_[_pID][_rID].eth = _eth.add(plyrRnds_[_pID][_rID].eth);
            round_[_rID].keys = _keys.add(round_[_rID].keys);
            round_[_rID].eth = _eth.add(round_[_rID