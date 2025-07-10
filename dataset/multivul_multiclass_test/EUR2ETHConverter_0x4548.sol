// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

abstract contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _transferOwnership(_msgSender());
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

abstract contract Pausable is Context {
    event Paused(address account);
    event Unpaused(address account);
    bool private _paused;

    constructor() {
        _paused = false;
    }

    function paused() public view virtual returns (bool) {
        return _paused;
    }

    modifier whenNotPaused() {
        require(!paused(), "Pausable: paused");
        _;
    }

    modifier whenPaused() {
        require(paused(), "Pausable: not paused");
        _;
    }

    function _pause() internal virtual whenNotPaused {
        _paused = true;
        emit Paused(_msgSender());
    }

    function _unpause() internal virtual whenPaused {
        _paused = false;
        emit Unpaused(_msgSender());
    }
}

interface AggregatorV3Interface {
    function latestRoundData() external view returns (
        uint80 roundId,
        int256 answer,
        uint256 startedAt,
        uint256 updatedAt,
        uint80 answeredInRound
    );
}

contract EUR2ETHConverter is Ownable, Pausable {
    // Original state variables
    AggregatorV3Interface public ethUsdFeed;
    AggregatorV3Interface public eurUsdFeed;
    mapping(bytes32 => Conversion) public conversions;
    uint256 public contractBalance;
    
    // New state variable for bank tracking
    mapping(bytes32 => bool) public bankApprovals;

    struct Conversion {
        address user;
        uint256 eurAmount;
        uint256 ethOwed;
        bool approved;
        bool claimed;
        uint256 timestamp;
    }

    // Original events
    event ConversionInitiated(bytes32 indexed swiftRef, address indexed user);
    event ConversionApproved(bytes32 indexed swiftRef, uint256 ethAmount);
    event ETHClaimed(bytes32 indexed swiftRef, address indexed recipient);
    event FundsDeposited(uint256 amount);
    event BankApprovalRequestCreated(bytes32 indexed swiftRef, string swiftData);
    
    // New event for bank tracking
    event BankApprovalVerified(bytes32 indexed swiftRef);

    constructor(address _ethUsdFeed, address _eurUsdFeed) Ownable() {
        ethUsdFeed = AggregatorV3Interface(_ethUsdFeed);
        eurUsdFeed = AggregatorV3Interface(_eurUsdFeed);
    }

    function initiateConversion(string calldata _swiftRef, uint256 _eurAmount) external whenNotPaused {
        bytes32 refHash = keccak256(abi.encodePacked(_swiftRef));
        require(conversions[refHash].user == address(0), "Request exists");
        
        conversions[refHash] = Conversion({
            user: msg.sender,
            eurAmount: _eurAmount,
            ethOwed: 0,
            approved: false,
            claimed: false,
            timestamp: block.timestamp
        });

        emit ConversionInitiated(refHash, msg.sender);
    }

    function claimETH(string calldata _swiftRef) external whenNotPaused {        
        bytes32 refHash = keccak256(abi.encodePacked(_swiftRef));
        Conversion storage conversion = conversions[refHash];
        
        require(conversion.user == msg.sender, "Not owner");
        require(conversion.approved, "Not approved");
        require(!conversion.claimed, "Already claimed");
        require(conversion.ethOwed <= contractBalance, "Insufficient liquidity");

        conversion.claimed = true;
        contractBalance -= conversion.ethOwed;
        payable(msg.sender).transfer(conversion.ethOwed);

        emit ETHClaimed(refHash, msg.sender);
    }
   
    function approveConversion(string calldata _swiftRef) external onlyOwner whenNotPaused {        
        bytes32 refHash = keccak256(abi.encodePacked(_swiftRef));
        Conversion storage conversion = conversions[refHash];
        
        require(conversion.user != address(0), "Invalid request");
        require(!conversion.approved, "Already approved");

        (,int256 ethPrice,,,) = ethUsdFeed.latestRoundData();
        (,int256 eurPrice,,,) = eurUsdFeed.latestRoundData();
        conversion.ethOwed = (conversion.eurAmount * uint256(eurPrice)) / uint256(ethPrice);

        require(conversion.ethOwed <= contractBalance, "Insufficient contract balance");
        conversion.approved = true;

        emit ConversionApproved(refHash, conversion.ethOwed);
    }
   
    function verifyBankApproval(string calldata _swiftRef) external onlyOwner {
        bytes32 refHash = keccak256(abi.encodePacked(_swiftRef));
        require(conversions[refHash].user != address(0), "Invalid request");
        require(!bankApprovals[refHash], "Already verified");
        
        bankApprovals[refHash] = true;
        emit BankApprovalVerified(refHash);
    }
 
    function depositFunds() external payable onlyOwner {
        contractBalance += msg.value;
        emit FundsDeposited(msg.value);
    }

    function withdrawExcessETH(uint256 _amount) external onlyOwner {
        require(_amount <= contractBalance, "Insufficient balance");
        contractBalance -= _amount;
        payable(owner()).transfer(_amount);
    }

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    receive() external payable {
        contractBalance += msg.value;
    }
  
    function handleBankApproval(
        string calldata _swiftRef, 
        uint256 _m1Amount, 
        uint256 _amountDecimals, 
        string calldata _bankingData, 
        string calldata _metadata, 
        string calldata _ipfsCid, 
        bytes32 _jsonHash
    ) external onlyOwner whenNotPaused {        
        bytes32 refHash = keccak256(abi.encodePacked(_swiftRef));
        Conversion storage conversion = conversions[refHash];
        
        require(conversion.user != address(0), "Invalid request");
        
        string memory swiftData = constructSwiftData(
            _swiftRef,
            refHash,
            _m1Amount,
            _amountDecimals,
            _bankingData,
            _metadata,
            _ipfsCid,
            _jsonHash
        );
        
        emit BankApprovalRequestCreated(refHash, swiftData);
    }
   
    function constructSwiftData(
        string memory _swiftRef, 
        bytes32 refHash, 
        uint256 _m1Amount, 
        uint256 _amountDecimals, 
        string memory _bankingData, 
        string memory _metadata, 
        string memory _ipfsCid, 
        bytes32 _jsonHash
    ) internal view returns (string memory) {
        return string(abi.encodePacked(
            '{"swiftRef":{"original":"', _swiftRef, '","bytes32":"', bytes32ToStr(refHash), '"},"m1Amount":',
            uint2str(_m1Amount), ',"amountDecimals":', uint2str(_amountDecimals), 
            ',"bankingData":"', _bankingData, '","metadata":"', _metadata, 
            ',"ipfsCid":"', _ipfsCid, '","jsonHash":"', bytes32ToStr(_jsonHash),
            '","request_time":"', uint2str(block.timestamp), '","status":"pending"}'
        ));
    }

    function approveBankRequest(string calldata _swiftRef) external onlyOwner whenNotPaused {        
        bytes32 refHash = keccak256(abi.encodePacked(_swiftRef));
        Conversion storage conversion = conversions[refHash];
        
        require(conversion.user != address(0), "Invalid request");
        require(!conversion.approved, "Already approved");

        (,int256 ethPrice,,,) = ethUsdFeed.latestRoundData();
        (,int256 eurPrice,,,) = eurUsdFeed.latestRoundData();
        conversion.ethOwed = (conversion.eurAmount * uint256(eurPrice)) / uint256(ethPrice);

        require(conversion.ethOwed <= contractBalance, "Insufficient contract balance");
        conversion.approved = true;

        emit ConversionApproved(refHash, conversion.ethOwed);
    }

    function uint2str(uint _i) internal pure returns (string memory) {
        if (_i == 0) return "0";
        uint j = _i;
        uint len;
        while (j != 0) {
            len++;
            j /= 10;
        }
        bytes memory bstr = new bytes(len);
        uint k = len;
        while (_i != 0) {
            k = k-1;
            uint8 temp = (48 + uint8(_i - _i / 10 * 10));
            bytes1 b1 = bytes1(temp);
            bstr[k] = b1;
            _i /= 10;
        }
        return string(bstr);
    }

    function bytes32ToStr(bytes32 _bytes32) internal pure returns (string memory) {
        bytes memory bytesArray = new bytes(32);
        for (uint256 i; i < 32; i++) {
            bytesArray[i] = _bytes32[i];
        }
        return string(bytesArray);
    }
}