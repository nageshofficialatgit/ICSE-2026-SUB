// SPDX-License-Identifier: BUSL-1.1
pragma solidity ^0.8.19;




interface ITimelockedCall {
    function initScheduler(address addr, uint256 newTimeLockDuration) external;
    function enableScheduler(address addr) external;
    function disableScheduler(address addr) external;

    function schedule(bytes32 h, address consumerAddr) external;
    function consume(bytes32 h) external;
    function consumeOwnership(bytes32 h, address prevOwnerAddr, address newOwnerAddr) external;
}







/**
 * @notice Defines the interface for whitelisting addresses.
 */
interface IAddressWhitelist {
    /**
     * @notice Whitelists the address specified.
     * @param addr The address to enable
     */
    function enableAddress (address addr) external;

    /**
     * @notice Whitelists the addresses specified.
     * @param arr The addresses to enable
     */
    function enableAddresses (address[] calldata arr) external;

    /**
     * @notice Disables the address specified.
     * @param addr The address to disable
     */
    function disableAddress (address addr) external;

    /**
     * @notice Disables the addresses specified.
     * @param arr The addresses to disable
     */
    function disableAddresses (address[] calldata arr) external;

    /**
     * @notice Indicates if the address is whitelisted or not.
     * @param addr The address to disable
     * @return Returns 1 if the address is whitelisted
     */
    function isWhitelistedAddress (address addr) external view returns (bool);

    /**
     * This event is triggered when a new address is whitelisted.
     * @param addr The address that was whitelisted
     */
    event OnAddressEnabled(address addr);

    /**
     * This event is triggered when an address is disabled.
     * @param addr The address that was disabled
     */
    event OnAddressDisabled(address addr);
}







interface IOwnable {
    function transferOwnership(address newOwner) external;
    function owner() external view returns (address);
}




/**
 * @title Base reentrancy guard. This is constructor-less implementation for both proxies and standalone contracts.
 */
abstract contract BaseReentrancyGuard {
    error ReentrantCall();
    
    uint256 internal constant _REENTRANCY_NOT_ENTERED = 1;
    uint256 internal constant _REENTRANCY_ENTERED = 2;

    uint256 internal _reentrancyStatus;

    /**
     * @dev Prevents a contract from calling itself, directly or indirectly.
     * Calling a `nonReentrant` function from another `nonReentrant`
     * function is not supported. It is possible to prevent this from happening
     * by making the `nonReentrant` function external, and making it call a
     * `private` function that does the actual work.
     */
    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {
        // On the first call to nonReentrant, _status will be _NOT_ENTERED
        if (_reentrancyStatus == _REENTRANCY_ENTERED) revert ReentrantCall();

        // Any calls to nonReentrant after this point will fail
        _reentrancyStatus = _REENTRANCY_ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _reentrancyStatus = _REENTRANCY_NOT_ENTERED;
    }
}




abstract contract BaseOwnable {
    error OwnerOnly();

    address internal _owner;

    /**
     * @notice Triggers when contract ownership changes.
     * @param previousOwner The previous owner of the contract.
     * @param newOwner The new owner of the contract.
     */
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        if (msg.sender != _owner) revert OwnerOnly();
        _;
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}


/**
 * @title Lightweight version of the ownership contract. This contract has a reentrancy guard.
 */
abstract contract LightweightOwnable is IOwnable, BaseReentrancyGuard, BaseOwnable {
    /**
     * @notice Transfers ownership of the contract to the account specified.
     * @param newOwner The address of the new owner.
     */
    function transferOwnership(address newOwner) external virtual nonReentrant onlyOwner {
        _transferOwnership(newOwner);
    }

    /**
     * @notice Gets the owner of the contract.
     * @return address The address who owns the contract.
     */
    function owner() external view virtual returns (address) {
        return _owner;
    }    
}


/**
 * @title Standalone contract for whitelisting addresses.
 */
contract AddressWhitelist is IAddressWhitelist, LightweightOwnable {
    mapping (address => bool) internal _whitelistedAddresses;

    constructor(address ownerAddr) {
        require(ownerAddr != address(0), "Owner required");
        _owner = ownerAddr;
    }

    /**
     * @notice Whitelists the address specified.
     * @param addr The address to enable
     */
    function enableAddress (address addr) external override nonReentrant onlyOwner {
        require(!_whitelistedAddresses[addr], "Already enabled");
        _whitelistedAddresses[addr] = true;
        emit OnAddressEnabled(addr);
    }

    /**
     * @notice Whitelists the addresses specified.
     * @param arr The addresses to enable
     */
    function enableAddresses (address[] calldata arr) external override nonReentrant onlyOwner {
        require(arr.length > 0, "Addresses required");

        for (uint256 i; i < arr.length; i++) {
            require(arr[i] != address(0), "Invalid address");
            require(!_whitelistedAddresses[arr[i]], "Already enabled");
            _whitelistedAddresses[arr[i]] = true;
            emit OnAddressEnabled(arr[i]);
        }
    }

    /**
     * @notice Disables the address specified.
     * @param addr The address to disable
     */
    function disableAddress (address addr) external override nonReentrant onlyOwner {
        require(_whitelistedAddresses[addr], "Already disabled");
        _whitelistedAddresses[addr] = false;
        emit OnAddressDisabled(addr);
    }

    /**
     * @notice Disables the addresses specified.
     * @param arr The addresses to disable
     */
    function disableAddresses (address[] calldata arr) external override nonReentrant onlyOwner {
        for (uint256 i; i < arr.length; i++) {
            require(_whitelistedAddresses[arr[i]], "Already disabled");
            _whitelistedAddresses[arr[i]] = false;
            emit OnAddressDisabled(arr[i]);
        }
    }

    /**
     * @notice Indicates if the address is whitelisted or not.
     * @param addr The address to evaluate.
     * @return Returns true if the address is whitelisted.
     */
    function isWhitelistedAddress (address addr) external view override returns (bool) {
        return _whitelistedAddresses[addr];
    }
}


/**
 * @title Contract for managing time-locked function calls.
 */
contract TimelockedCall is ITimelockedCall, AddressWhitelist {
    struct TimelockedCallInfo {
        uint256 targetEpoch;     // The unix epoch at which the hash can be consumed
        address createdBy;       // The address of the scheduler
        address consumerAddress; // The address of the consumer
    }

    /// @notice The schedulers authorized for a given sender. (sender => scheduler => enabled/disabled)
    mapping (address => mapping(address => bool)) private whitelistedSchedulers;

    /// @notice The time-lock info of a given hash.
    mapping (bytes32 => TimelockedCallInfo) public queue;

    /// @notice The time-lock duration of every consumer address.
    mapping (address => uint256) public timeLockDuration;

    /// @notice Triggers when a hash is scheduled for the address specified.
    event HashScheduled(bytes32 h, address consumerAddress);

    /// @notice Triggers when a hash is consumed by the address specified.
    event HashConsumed(bytes32 h, address consumerAddress);

    /// @notice Triggers when a new scheduler is enabled for the consumer address specified.
    event SchedulerEnabled(address consumerAddress, address schedulerAddress);

    /// @notice Triggers when an existing scheduler is disabled for the consumer address specified.
    event SchedulerDisabled(address consumerAddress, address schedulerAddress);

    constructor(address ownerAddr) AddressWhitelist(ownerAddr) {
    }

    modifier ifSenderWhitelisted() {
        require(_whitelistedAddresses[msg.sender], "Unauthorized sender");
        _;
    }

    modifier ifTimeLockConfigured() {
        require(timeLockDuration[msg.sender] > 0, "Not configured");
        _;
    }

    /**
     * @notice Sets the initial scheduler and time-lock duration for the current message sender.
     * @param addr The address of the initial scheduler. You can add more addresses later.
     * @param newTimeLockDuration The duration of the time-lock for the current message sender.
     */
    function initScheduler(address addr, uint256 newTimeLockDuration) external override nonReentrant ifSenderWhitelisted {
        require(addr != address(0), "Address required");
        require(newTimeLockDuration > 0, "Duration required");
        require(timeLockDuration[msg.sender] == 0, "Already initialized");

        whitelistedSchedulers[msg.sender][addr] = true;
        timeLockDuration[msg.sender] = newTimeLockDuration;

        emit SchedulerEnabled(msg.sender, addr);
    }

    /**
     * @notice Authorizes the address specified to schedule calls. The calls will be consumed by the current message sender.
     * @param addr Specifies the address of the scheduler to authorize.
     */
    function enableScheduler(address addr) external override nonReentrant ifTimeLockConfigured {
        _enableScheduler(addr);
    }

    /**
     * @notice Revokes the address specified from scheduling calls for the current message sender.
     * @param addr Specifies the address of the scheduler to revoke.
     */
    function disableScheduler(address addr) external override nonReentrant ifTimeLockConfigured {
        _disableScheduler(addr);
    }

    /**
     * @notice Schedules a hash to be consumed by the address specified.
     * @param h Specifies the hash.
     * @param consumerAddr Specifies the address of the consumer.
     */
    function schedule(bytes32 h, address consumerAddr) external override nonReentrant {
        require(h != bytes32(0), "Hash required");
        require(whitelistedSchedulers[consumerAddr][msg.sender], "Unauthorized sender");
        require(timeLockDuration[consumerAddr] > 0, "Not configured");

        bytes32 h2 = keccak256(abi.encode(h, consumerAddr));
        require(queue[h2].targetEpoch == 0, "Already enqueued");
        
        queue[h2] = TimelockedCallInfo({
            createdBy: msg.sender,
            consumerAddress: consumerAddr,
            targetEpoch: block.timestamp + timeLockDuration[consumerAddr]
        });

        emit HashScheduled(h, consumerAddr);
    }

    /**
     * @notice Consumes the hash specified.
     * @param h Specifies the hash.
     */
    function consume(bytes32 h) external override nonReentrant ifTimeLockConfigured {
        bytes32 h2 = keccak256(abi.encode(h, msg.sender));
        _consume(h2);
    }

    /**
     * @notice Consumes the hash specified. The hash represents the transferOwnership function.
     * @param h Specifies the hash.
     * @param prevOwnerAddr The current owner of the contract at hand.
     * @param newOwnerAddr The address of the new owner.
     */
    function consumeOwnership(
        bytes32 h,
        address prevOwnerAddr,
        address newOwnerAddr
    ) external override nonReentrant ifTimeLockConfigured {
        bytes32 h2 = keccak256(abi.encode(h, msg.sender));

        _disableScheduler(prevOwnerAddr);
        _consume(h2);
        _enableScheduler(newOwnerAddr);
    }

    function _consume(bytes32 h) internal {
        require(queue[h].targetEpoch > 0, "Hash not enqueued");
        require(msg.sender == queue[h].consumerAddress, "Unauthorized consumer");
        require(block.timestamp > queue[h].targetEpoch, "Timelock in place");

        delete queue[h];

        emit HashConsumed(h, msg.sender);
    }

    function _enableScheduler(address addr) internal {
        require(addr != address(0), "Address required");
        require(!whitelistedSchedulers[msg.sender][addr], "Already enabled");
        whitelistedSchedulers[msg.sender][addr] = true;
        emit SchedulerEnabled(msg.sender, addr);
    }

    function _disableScheduler(address addr) internal {
        require(addr != address(0), "Address required");
        require(whitelistedSchedulers[msg.sender][addr], "Already disabled");
        whitelistedSchedulers[msg.sender][addr] = false;
        emit SchedulerDisabled(msg.sender, addr);
    }
}