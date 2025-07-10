// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title WhiteHouseToken
 * @notice A simple ERC-20 token contract for the White House Token.
 *         It implements standard ERC-20 functions, minting, burning, and a
 *         timelocked ownership transfer mechanism.
 */
contract WhiteHouseToken {
    // Token metadata and supply
    string public name = "White House Token";
    string public symbol = "WHT";
    uint8 public decimals = 9;
    uint256 public totalSupply;

    // Balances and allowances mapping
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    // Ownership management
    address public owner;
    uint256 public ownershipChangeTime; // Timestamp when new owner can confirm transfer
    address public pendingOwner;

    // Events as per ERC-20 standard and ownership changes
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event OwnershipTransferRequested(address indexed newOwner);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @notice Constructor sets the total supply and assigns it to the deployer.
     * Total supply is 38,000,000,000,000 tokens, expressed in smallest units (38e21).
     */
    constructor() {
        owner = msg.sender;
        totalSupply = 38e21; // 38,000,000,000,000 * 10^9
        balanceOf[owner] = totalSupply;
        emit Transfer(address(0), owner, totalSupply);
    }

    /**
     * @notice Transfer tokens to a recipient.
     */
    function transfer(address to, uint256 value) public returns (bool success) {
        require(to != address(0), "Invalid recipient");
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }

    /**
     * @notice Approve a spender to use tokens on your behalf.
     */
    function approve(address spender, uint256 value) public returns (bool success) {
        require(spender != address(0), "Invalid spender");
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    /**
     * @notice Transfer tokens from one address to another using an allowance.
     */
    function transferFrom(address from, address to, uint256 value) public returns (bool success) {
        require(to != address(0), "Invalid recipient");
        require(balanceOf[from] >= value, "Insufficient balance");
        require(allowance[from][msg.sender] >= value, "Allowance exceeded");
        balanceOf[from] -= value;
        balanceOf[to] += value;
        allowance[from][msg.sender] -= value;
        emit Transfer(from, to, value);
        return true;
    }

    /**
     * @notice Mint new tokens. Only the owner can mint.
     */
    function mint(uint256 value) public {
        require(msg.sender == owner, "Only owner can mint");
        totalSupply += value;
        balanceOf[owner] += value;
        emit Transfer(address(0), owner, value);
    }

    /**
     * @notice Burn tokens from the owner's balance. Only the owner can burn.
     */
    function burn(uint256 value) public {
        require(msg.sender == owner, "Only owner can burn");
        require(balanceOf[owner] >= value, "Insufficient balance to burn");
        balanceOf[owner] -= value;
        totalSupply -= value;
        emit Transfer(owner, address(0), value);
    }

    /**
     * @notice Request an ownership transfer. The transfer can only be confirmed after 3 days.
     */
    function requestOwnershipTransfer(address newOwner) public {
        require(msg.sender == owner, "Only owner can request transfer");
        require(newOwner != address(0), "Invalid new owner");
        ownershipChangeTime = block.timestamp + 3 days;
        pendingOwner = newOwner;
        emit OwnershipTransferRequested(newOwner);
    }

    /**
     * @notice Confirm the pending ownership transfer after the timelock expires.
     */
    function confirmOwnershipTransfer() public {
        require(msg.sender == pendingOwner, "Only pending owner can confirm");
        require(block.timestamp >= ownershipChangeTime, "Timelock active");
        emit OwnershipTransferred(owner, pendingOwner);
        owner = pendingOwner;
        pendingOwner = address(0);
        ownershipChangeTime = 0;
    }
}