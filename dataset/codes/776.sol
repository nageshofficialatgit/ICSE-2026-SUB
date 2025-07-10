pragma solidity ^0.4.11;
pragma solidity^0.4.11;
library AttributeStore {
    struct Data {
        mapping(bytes32 => uint) store;
    }
    function getAttribute(Data storage self, bytes32 _UUID, string _attrName)
    public view returns (uint) {
        bytes32 key = keccak256(_UUID, _attrName);
        return self.store[key];
    }
    function setAttribute(Data storage self, bytes32 _UUID, string _attrName, uint _attrVal)
    public {
        bytes32 key = keccak256(_UUID, _attrName);
        self.store[key] = _attrVal;
    }
}
pragma solidity^0.4.11;
library DLL {
  uint constant NULL_NODE_ID = 0;
  struct Node {
    uint next;
    uint prev;
  }
  struct Data {
    mapping(uint => Node) dll;
  }
  function isEmpty(Data storage self) public view returns (bool) {
    return getStart(self) == NULL_NODE_ID;
  }
  function contains(Data storage self, uint _curr) public view returns (bool) {
    if (isEmpty(self) || _curr == NULL_NODE_ID) {
      return false;
    } 
    bool isSingleNode = (getStart(self) == _curr) && (getEnd(self) == _curr);
    bool isNullNode = (getNext(self, _curr) == NULL_NODE_ID) && (getPrev(self, _curr) == NULL_NODE_ID);
    return isSingleNode || !isNullNode;
  }
  function getNext(Data storage self, uint _curr) public view returns (uint) {
    return self.dll[_curr].next;
  }
  function getPrev(Data storage self, uint _curr) public view returns (uint) {
    return self.dll[_curr].prev;
  }
  function getStart(Data storage self) public view returns (uint) {
    return getNext(self, NULL_NODE_ID);
  }
  function getEnd(Data storage self) public view returns (uint) {
    return getPrev(self, NULL_NODE_ID);
  }
  function insert(Data storage self, uint _prev, uint _curr, uint _next) public {
    require(_curr != NULL_NODE_ID);
    remove(self, _curr);
    require(_prev == NULL_NODE_ID || contains(self, _prev));
    require(_next == NULL_NODE_ID || contains(self, _next));
    require(getNext(self, _prev) == _next);
    require(getPrev(self, _next) == _prev);
    self.dll[_curr].prev = _prev;
    self.dll[_curr].next = _next;
    self.dll[_prev].next = _curr;
    self.dll[_next].prev = _curr;
  }
  function remove(Data storage self, uint _curr) public {
    if (!contains(self, _curr)) {
      return;
    }
    uint next = getNext(self, _curr);
    uint prev = getPrev(self, _curr);
    self.dll[next].prev = prev;
    self.dll[prev].next = next;
    delete self.dll[_curr];
  }
}
pragma solidity ^0.4.8;
contract EIP20Interface {
    uint256 public totalSupply;
    function balanceOf(address _owner) public view returns (uint256 balance);
    function transfer(address _to, uint256 _value) public returns (bool success);
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);
    function approve(address _spender, uint256 _value) public returns (bool success);
    function allowance(address _owner, address _spender) public view returns (uint256 remaining);
    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}
library SafeMath {
  function mul(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }
  function div(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a / b;
    return c;
  }
  function sub(uint256 a, uint256 b) internal constant returns (uint256) {
    assert(b <= a);
    return a - b;
  }
  function add(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}
contract PLCRVoting {
    event _VoteCommitted(uint indexed pollID, uint numTokens, address indexed voter);
    event _VoteRevealed(uint indexed pollID, uint numTokens, uint votesFor, uint votesAgainst, uint indexed choice, address indexed voter);
    event _PollCreated(uint voteQuorum, uint commitEndDate, uint revealEndDate, uint indexed pollID, address indexed creator);
    event _VotingRightsGranted(uint numTokens, address indexed voter);
    event _VotingRightsWithdrawn(uint numTokens, address indexed voter);
    event _TokensRescued(uint indexed pollID, address indexed voter);
    using AttributeStore for AttributeStore.Data;
    using DLL for DLL.Data;
    using SafeMath for uint;
    struct Poll {
        uint commitEndDate;     
        uint revealEndDate;     
        uint voteQuorum;	    
        uint votesFor;		    
        uint votesAgainst;      
        mapping(address => bool) didCommit;  
        mapping(address => bool) didReveal;   
    }
    uint constant public INITIAL_POLL_NONCE = 0;
    uint public pollNonce;
    mapping(uint => Poll) public pollMap; 
    mapping(address => uint) public voteTokenBalance; 
    mapping(address => DLL.Data) dllMap;
    AttributeStore.Data store;
    EIP20Interface public token;
    function PLCRVoting(address _tokenAddr) public {
        token = EIP20Interface(_tokenAddr);
        pollNonce = INITIAL_POLL_NONCE;
    }
    function requestVotingRights(uint _numTokens) external {
        require(token.balanceOf(msg.sender) >= _numTokens);
        voteTokenBalance[msg.sender] += _numTokens;
        require(token.transferFrom(msg.sender, this, _numTokens));
        _VotingRightsGranted(_numTokens, msg.sender);
    }
    function withdrawVotingRights(uint _numTokens) external {
        uint availableTokens = voteTokenBalance[msg.sender].sub(getLockedTokens(msg.sender));
        require(availableTokens >= _numTokens);
        voteTokenBalance[msg.sender] -= _numTokens;
        require(token.transfer(msg.sender, _numTokens));
        _VotingRightsWithdrawn(_numTokens, msg.sender);
    }
    function rescueTokens(uint _pollID) external {
        require(isExpired(pollMap[_pollID].revealEndDate));
        require(dllMap[msg.sender].contains(_pollID));
        dllMap[msg.sender].remove(_pollID);
        _TokensRescued(_pollID, msg.sender);
    }
    function commitVote(uint _pollID, bytes32 _secretHash, uint _numTokens, uint _prevPollID) external {
        require(commitPeriodActive(_pollID));
        require(voteTokenBalance[msg.sender] >= _numTokens); 
        require(_pollID != 0);                
        require(_prevPollID == 0 || dllMap[msg.sender].contains(_prevPollID));
        uint nextPollID = dllMap[msg.sender].getNext(_prevPollID);
        nextPollID = (nextPollID == _pollID) ? dllMap[msg.sender].getNext(_pollID) : nextPollID;
        require(validPosition(_prevPollID, nextPollID, msg.sender, _numTokens));
        dllMap[msg.sender].insert(_prevPollID, _pollID, nextPollID);
        bytes32 UUID = attrUUID(msg.sender, _pollID);
        store.setAttribute(UUID, "numTokens", _numTokens);
        store.setAttribute(UUID, "commitHash", uint(_secretHash));
        pollMap[_pollID].didCommit[msg.sender] = true;
        _VoteCommitted(_pollID, _numTokens, msg.sender);
    }
    function validPosition(uint _prevID, uint _nextID, address _voter, uint _numTokens) public constant returns (bool valid) {
        bool prevValid = (_numTokens >= getNumTokens(_voter, _prevID));
        bool nextValid = (_numTokens <= getNumTokens(_voter, _nextID) || _nextID == 0);
        return prevValid && nextValid;
    }
    function revealVote(uint _pollID, uint _voteOption, uint _salt) external {
        require(revealPeriodActive(_pollID));
        require(pollMap[_pollID].didCommit[msg.sender]);                         
        require(!pollMap[_pollID].didReveal[msg.sender]);                        
        require(keccak256(_voteOption, _salt) == getCommitHash(msg.sender, _pollID)); 
        uint numTokens = getNumTokens(msg.sender, _pollID);
        if (_voteOption == 1) {
            pollMap[_pollID].votesFor += numTokens;
        } else {
            pollMap[_pollID].votesAgainst += numTokens;
        }
        dllMap[msg.sender].remove(_pollID); 
        pollMap[_pollID].didReveal[msg.sender] = true;
        _VoteRevealed(_pollID, numTokens, pollMap[_pollID].votesFor, pollMap[_pollID].votesAgainst, _voteOption, msg.sender);
    }
    function getNumPassingTokens(address _voter, uint _pollID, uint _salt) public constant returns (uint correctVotes) {
        require(pollEnded(_pollID));
        require(pollMap[_pollID].didReveal[_voter]);
        uint winningChoice = isPassed(_pollID) ? 1 : 0;
        bytes32 winnerHash = keccak256(winningChoice, _salt);
        bytes32 commitHash = getCommitHash(_voter, _pollID);
        require(winnerHash == commitHash);
        return getNumTokens(_voter, _pollID);
    }
    function startPoll(uint _voteQuorum, uint _commitDuration, uint _revealDuration) public returns (uint pollID) {
        pollNonce = pollNonce + 1;
        uint commitEndDate = block.timestamp.add(_commitDuration);
        uint revealEndDate = commitEndDate.add(_revealDuration);
        pollMap[pollNonce] = Poll({
            voteQuorum: _voteQuorum,
            commitEndDate: commitEndDate,
            revealEndDate: revealEndDate,
            votesFor: 0,
            votesAgainst: 0
        });
        _PollCreated(_voteQuorum, commitEndDate, revealEndDate, pollNonce, msg.sender);
        return pollNonce;
    }
    function isPassed(uint _pollID) constant public returns (bool passed) {
        require(pollEnded(_pollID));
        Poll memory poll = pollMap[_pollID];
        return (100 * poll.votesFor) > (poll.voteQuorum * (poll.votesFor + poll.votesAgainst));
    }
    function getTotalNumberOfTokensForWinningOption(uint _pollID) constant public returns (uint numTokens) {
        require(pollEnded(_pollID));
        if (isPassed(_pollID))
            return pollMap[_pollID].votesFor;
        else
            return pollMap[_pollID].votesAgainst;
    }
    function pollEnded(uint _pollID) constant public returns (bool ended) {
        require(pollExists(_pollID));
        return isExpired(pollMap[_pollID].revealEndDate);
    }
    function commitPeriodActive(uint _pollID) constant public returns (bool active) {
        require(pollExists(_pollID));
        return !isExpired(pollMap[_pollID].commitEndDate);
    }
    function revealPeriodActive(uint _pollID) constant public returns (bool active) {
        require(pollExists(_pollID));
        return !isExpired(pollMap[_pollID].revealEndDate) && !commitPeriodActive(_pollID);
    }
    function didCommit(address _voter, uint _pollID) constant public returns (bool committed) {
        require(pollExists(_pollID));
        return pollMap[_pollID].didCommit[_voter];
    }
    function didReveal(address _voter, uint _pollID) constant public returns (bool revealed) {
        require(pollExists(_pollID));
        return pollMap[_pollID].didReveal[_voter];
    }
    function pollExists(uint _pollID) constant public returns (bool exists) {
        return (_pollID != 0 && _pollID <= pollNonce);
    }
    function getCommitHash(address _voter, uint _pollID) constant public returns (bytes32 commitHash) {
        return bytes32(store.getAttribute(attrUUID(_voter, _pollID), "commitHash"));
    }
    function getNumTokens(address _voter, uint _pollID) constant public returns (uint numTokens) {
        return store.getAttribute(attrUUID(_voter, _pollID), "numTokens");
    }
    function getLastNode(address _voter) constant public returns (uint pollID) {
        return dllMap[_voter].getPrev(0);
    }
    function getLockedTokens(address _voter) constant public returns (uint numTokens) {
        return getNumTokens(_voter, getLastNode(_voter));
    }
    function getInsertPointForNumTokens(address _voter, uint _numTokens, uint _pollID)
    constant public returns (uint prevNode) {
      uint nodeID = getLastNode(_voter);
      uint tokensInNode = getNumTokens(_voter, nodeID);
      while(nodeID != 0) {
        tokensInNode = getNumTokens(_voter, nodeID);
        if(tokensInNode <= _numTokens) { 
          if(nodeID == _pollID) {
            nodeID = dllMap[_voter].getPrev(nodeID);
          }
          return nodeID; 
        }
        nodeID = dllMap[_voter].getPrev(nodeID);
      }
      return nodeID;
    }
    function isExpired(uint _terminationDate) constant public returns (bool expired) {
        return (block.timestamp > _terminationDate);
    }
    function attrUUID(address _user, uint _pollID) public pure returns (bytes32 UUID) {
        return keccak256(_user, _pollID);
    }
}
pragma solidity^0.4.11;
contract Parameterizer {
  event _ReparameterizationProposal(string name, uint value, bytes32 propID, uint deposit, uint appEndDate, address indexed proposer);
  event _NewChallenge(bytes32 indexed propID, uint challengeID, uint commitEndDate, uint revealEndDate, address indexed challenger);
  event _ProposalAccepted(bytes32 indexed propID, string name, uint value);
  event _ProposalExpired(bytes32 indexed propID);
  event _ChallengeSucceeded(bytes32 indexed propID, uint indexed challengeID, uint rewardPool, uint totalTokens);
  event _ChallengeFailed(bytes32 indexed propID, uint indexed challengeID, uint rewardPool, uint totalTokens);
  event _RewardClaimed(uint indexed challengeID, uint reward, address indexed voter);
  using SafeMath for uint;
  struct ParamProposal {
    uint appExpiry;
    uint challengeID;
    uint deposit;
    string name;
    address owner;
    uint processBy;
    uint value;
  }
  struct Challenge {
    uint rewardPool;        
    address challenger;     
    bool resolved;          
    uint stake;             
    uint winningTokens;     
    mapping(address => bool) tokenClaims;
  }
  mapping(bytes32 => uint) public params;
  mapping(uint => Challenge) public challenges;
  mapping(bytes32 => ParamProposal) public proposals;
  EIP20Interface public token;
  PLCRVoting public voting;
  uint public PROCESSBY = 604800; 
  function Parameterizer(
    address _tokenAddr,
    address _plcrAddr,
    uint _minDeposit,
    uint _pMinDeposit,
    uint _applyStageLen,
    uint _pApplyStageLen,
    uint _commitStageLen,
    uint _pCommitStageLen,
    uint _revealStageLen,
    uint _pRevealStageLen,
    uint _dispensationPct,
    uint _pDispensationPct,
    uint _voteQuorum,
    uint _pVoteQuorum
    ) public {
      token = EIP20Interface(_tokenAddr);
      voting = PLCRVoting(_plcrAddr);
      set("minDeposit", _minDeposit);
      set("pMinDeposit", _pMinDeposit);
      set("applyStageLen", _applyStageLen);
      set("pApplyStageLen", _pApplyStageLen);
      set("commitStageLen", _commitStageLen);
      set("pCommitStageLen", _pCommitStageLen);
      set("revealStageLen", _revealStageLen);
      set("pRevealStageLen", _pRevealStageLen);
      set("dispensationPct", _dispensationPct);
      set("pDispensationPct", _pDispensationPct);
      set("voteQuorum", _voteQuorum);
      set("pVoteQuorum", _pVoteQuorum);
  }
  function proposeReparameterization(string _name, uint _value) public returns (bytes32) {
    uint deposit = get("pMinDeposit");
    bytes32 propID = keccak256(_name, _value);
    if (keccak256(_name) == keccak256('dispensationPct') ||
       keccak256(_name) == keccak256('pDispensationPct')) {
        require(_value <= 100);
    }
    require(!propExists(propID)); 
    require(get(_name) != _value); 
    proposals[propID] = ParamProposal({
      appExpiry: now.add(get("pApplyStageLen")),
      challengeID: 0,
      deposit: deposit,
      name: _name,
      owner: msg.sender,
      processBy: now.add(get("pApplyStageLen"))
        .add(get("pCommitStageLen"))
        .add(get("pRevealStageLen"))
        .add(PROCESSBY),
      value: _value
    });
    require(token.transferFrom(msg.sender, this, deposit)); 
    _ReparameterizationProposal(_name, _value, propID, deposit, proposals[propID].appExpiry, msg.sender);
    return propID;
  }
  function challengeReparameterization(bytes32 _propID) public returns (uint challengeID) {
    ParamProposal memory prop = proposals[_propID];
    uint deposit = prop.deposit;
    require(propExists(_propID) && prop.challengeID == 0);
    uint pollID = voting.startPoll(
      get("pVoteQuorum"),
      get("pCommitStageLen"),
      get("pRevealStageLen")
    );
    challenges[pollID] = Challenge({
      challenger: msg.sender,
      rewardPool: SafeMath.sub(100, get("pDispensationPct")).mul(deposit).div(100),
      stake: deposit,
      resolved: false,
      winningTokens: 0
    });
    proposals[_propID].challengeID = pollID;       
    require(token.transferFrom(msg.sender, this, deposit));
    var (commitEndDate, revealEndDate,) = voting.pollMap(pollID);
    _NewChallenge(_propID, pollID, commitEndDate, revealEndDate, msg.sender);
    return pollID;
  }
  function processProposal(bytes32 _propID) public {
    ParamProposal storage prop = proposals[_propID];
    address propOwner = prop.owner;
    uint propDeposit = prop.deposit;
   if (canBeSet(_propID)) {
      set(prop.name, prop.value);
      _ProposalAccepted(_propID, prop.name, prop.value);
      delete proposals[_propID];
      require(token.transfer(propOwner, propDeposit));
    } else if (challengeCanBeResolved(_propID)) {
      resolveChallenge(_propID);
    } else if (now > prop.processBy) {
      _ProposalExpired(_propID);
      delete proposals[_propID];
      require(token.transfer(propOwner, propDeposit));
    } else {
      revert();
    }
    assert(get("dispensationPct") <= 100);
    assert(get("pDispensationPct") <= 100);
    now.add(get("pApplyStageLen"))
      .add(get("pCommitStageLen"))
      .add(get("pRevealStageLen"))
      .add(PROCESSBY);
    delete proposals[_propID];
  }
  /**
  @notice claim the tokens owed for the msg.sender in the provided chall