// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title USDC Vesting Contract
 * @dev Allows the owner to withdraw fixed USDC amount every 23h or full balance after 30 days
 */

// USDC Interface
interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

// Minimal Reentrancy Guard (no external imports)
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

contract USDCVesting is ReentrancyGuard {
    address public owner;
    address public pendingOwner;

    // Hardcoded USDC contract on Ethereum Mainnet
    IERC20 public constant usdcToken = IERC20(0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48);

    uint256 public constant WITHDRAW_AMOUNT = 300 * 10 ** 6; // 300 USDC
    uint256 public constant INTERVAL = 23 hours;
    uint256 public immutable startTime;
    uint256 public lastWithdrawTime;

    event Withdrawn(address indexed to, uint256 amount, uint256 timestamp);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event OwnershipTransferInitiated(address indexed currentOwner, address indexed pendingOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        startTime = block.timestamp;
    }

    /**
     * @dev Withdraws either fixed 300 USDC every 23h, or full balance after 30 days
     */
    function withdraw() external onlyOwner nonReentrant {
        uint256 currentTime = block.timestamp;

        uint256 balance = usdcToken.balanceOf(address(this));
        require(balance > 0, "Contract balance is empty");

        if (currentTime >= startTime + 30 days) {
            require(usdcToken.transfer(owner, balance), "USDC transfer failed");
            emit Withdrawn(owner, balance, currentTime);
            return;
        }

        require(
            lastWithdrawTime == 0 || currentTime >= lastWithdrawTime + INTERVAL,
            "Must wait 23 hours between withdrawals"
        );

        uint256 amount = balance >= WITHDRAW_AMOUNT ? WITHDRAW_AMOUNT : balance;
        lastWithdrawTime = currentTime;

        require(usdcToken.transfer(owner, amount), "USDC transfer failed");
        emit Withdrawn(owner, amount, currentTime);
    }

    /**
     * @dev View current USDC balance in the contract
     */
    function getContractBalance() external view returns (uint256) {
        return usdcToken.balanceOf(address(this));
    }

    // ---------------- Ownership Security ---------------- //

    /**
     * @dev Initiates ownership transfer. Must be accepted by the new owner.
     */
    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "New owner cannot be zero address");
        pendingOwner = _newOwner;
        emit OwnershipTransferInitiated(owner, _newOwner);
    }

    /**
     * @dev Called by pending owner to finalize the transfer.
     */
    function acceptOwnership() external {
        require(msg.sender == pendingOwner, "Caller is not pending owner");
        address oldOwner = owner;
        owner = pendingOwner;
        pendingOwner = address(0);
        emit OwnershipTransferred(oldOwner, owner);
    }

    /**
     * @dev Allows current owner to renounce ownership.
     */
    function renounceOwnership() external onlyOwner {
        emit OwnershipTransferred(owner, address(0));
        owner = address(0);
    }
}