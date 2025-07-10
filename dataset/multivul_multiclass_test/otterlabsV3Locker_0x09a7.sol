// SPDX-License-Identifier: MIT
/*
   ___ _____ _____ ___ ___   _      _   ___ ___ 
  / _ \_   _|_   _| __| _ \ | |    /_\ | _ ) __|
 | (_) || |   | | | _||   / | |__ / _ \| _ \__ \
  \___/ |_|   |_| |___|_|_\ |____/_/ \_\___/___/
OTTERLABS UNI V3 Liquidity Locker                 

This contract allows anyone to lock their Uniswap V3 LP positions for a specified duration.
The locked positions can only be withdrawn by their original owner after the lock period expires.

Instructions:
1. First approve this contract to handle your LP position NFT
2. Call lockPosition with your token ID and desired lock duration
3. After lock period expires, call unlockPosition to retrieve your LP

*/

pragma solidity ^0.8.0;

interface IERC165 {
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}

interface IERC721 is IERC165 {
    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);
    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);
    
    function balanceOf(address owner) external view returns (uint256 balance);
    function ownerOf(uint256 tokenId) external view returns (address owner);
    function safeTransferFrom(address from, address to, uint256 tokenId, bytes calldata data) external;
    function safeTransferFrom(address from, address to, uint256 tokenId) external;
    function transferFrom(address from, address to, uint256 tokenId) external;
    function approve(address to, uint256 tokenId) external;
    function setApprovalForAll(address operator, bool _approved) external;
    function getApproved(uint256 tokenId) external view returns (address operator);
    function isApprovedForAll(address owner, address operator) external view returns (bool);
}

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

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function _checkOwner() internal view virtual {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
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

contract otterlabsV3Locker is ReentrancyGuard, Ownable {
    // Uniswap V3 NonfungiblePositionManager
    IERC721 public constant POSITION_MANAGER = IERC721(0xC36442b4a4522E871399CD717aBDD847Ab11FE88);
    
    // Minimum lock duration (1 day)
    uint256 public constant MIN_LOCK_DURATION = 1 days;
    
    // Maximum lock duration (2 years)
    uint256 public constant MAX_LOCK_DURATION = 730 days;
    
    struct Lock {
        uint256 tokenId;
        address owner;
        uint256 unlockTime;
        bool isWithdrawn;
    }
    
    // Mapping from token ID to lock info
    mapping(uint256 => Lock) public locks;
    
    // Events
    event PositionLocked(
        uint256 indexed tokenId, 
        address indexed owner, 
        uint256 unlockTime,
        uint256 lockDuration
    );
    event PositionUnlocked(uint256 indexed tokenId, address indexed owner);
    
    /**
     * @notice Lock a Uniswap V3 LP position
     * @param tokenId The ID of the LP position token
     * @param lockDuration Duration in seconds for which to lock the position
     */
    function lockPosition(uint256 tokenId, uint256 lockDuration) external nonReentrant {
        require(lockDuration >= MIN_LOCK_DURATION, "Lock duration too short");
        require(lockDuration <= MAX_LOCK_DURATION, "Lock duration too long");
        require(locks[tokenId].owner == address(0), "Position already locked");
        
        // Calculate unlock time
        uint256 unlockTime = block.timestamp + lockDuration;
        
        // Transfer the NFT to this contract
        POSITION_MANAGER.transferFrom(msg.sender, address(this), tokenId);
        
        // Create lock
        locks[tokenId] = Lock({
            tokenId: tokenId,
            owner: msg.sender,
            unlockTime: unlockTime,
            isWithdrawn: false
        });
        
        emit PositionLocked(tokenId, msg.sender, unlockTime, lockDuration);
    }
    
    /**
     * @notice Unlock and withdraw a locked position after the lock period has expired
     * @param tokenId The ID of the LP position token to unlock
     */
    function unlockPosition(uint256 tokenId) external nonReentrant {
        Lock storage lock = locks[tokenId];
        require(lock.owner == msg.sender, "Not the owner");
        require(block.timestamp >= lock.unlockTime, "Lock period not expired");
        require(!lock.isWithdrawn, "Position already withdrawn");
        
        lock.isWithdrawn = true;
        
        // Transfer the NFT back to the owner
        POSITION_MANAGER.transferFrom(address(this), msg.sender, tokenId);
        
        emit PositionUnlocked(tokenId, msg.sender);
    }
    
    /**
     * @notice Get the remaining lock time for a position
     * @param tokenId The ID of the LP position token
     * @return Time remaining in seconds, 0 if unlocked or never locked
     */
    function getRemainingLockTime(uint256 tokenId) external view returns (uint256) {
        Lock memory lock = locks[tokenId];
        if (lock.owner == address(0) || lock.isWithdrawn) {
            return 0;
        }
        if (block.timestamp >= lock.unlockTime) {
            return 0;
        }
        return lock.unlockTime - block.timestamp;
    }
    
    /**
     * @notice Get all info about a locked position
     * @param tokenId The ID of the LP position token
     * @return owner The owner of the locked position
     * @return unlockTime The timestamp when the position can be withdrawn
     * @return isWithdrawn Whether the position has been withdrawn
     * @return isLocked Whether the position is currently locked
     */
    function getLockInfo(uint256 tokenId) external view returns (
        address owner,
        uint256 unlockTime,
        bool isWithdrawn,
        bool isLocked
    ) {
        Lock memory lock = locks[tokenId];
        return (
            lock.owner,
            lock.unlockTime,
            lock.isWithdrawn,
            lock.owner != address(0) && !lock.isWithdrawn && block.timestamp < lock.unlockTime
        );
    }
    
    /**
     * @notice Emergency function to recover stuck tokens in case of emergency
     * @dev Only callable by owner after a significant delay period
     * @param tokenId The ID of the LP position token to recover
     */
    function emergencyRecover(uint256 tokenId) external onlyOwner {
        Lock storage lock = locks[tokenId];
        require(lock.unlockTime + 30 days < block.timestamp, "Emergency delay not passed");
        require(!lock.isWithdrawn, "Position already withdrawn");
        
        lock.isWithdrawn = true;
        
        // Transfer the NFT to the original owner
        POSITION_MANAGER.transferFrom(address(this), lock.owner, tokenId);
        
        emit PositionUnlocked(tokenId, lock.owner);
    }
}