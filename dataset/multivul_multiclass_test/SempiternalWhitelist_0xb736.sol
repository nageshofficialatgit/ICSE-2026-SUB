// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title SempiternalWhitelist
 * @dev Contract for managing the whitelist for the SEMPITERNAL ICO
 */
contract SempiternalWhitelist {
    // Token contract address
    address public tokenContract;
    
    // Owner of the whitelist contract
    address public owner;
    
    // Mapping of whitelisted addresses
    mapping(address => bool) public whitelist;
    
    // Array to keep track of all whitelisted addresses
    address[] public whitelistedAddresses;
    
    // Events
    event AddedToWhitelist(address indexed account);
    event RemovedFromWhitelist(address indexed account);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;
    }
    
    /**
     * @dev Constructor sets the token contract address and the owner
     * @param _tokenContract Address of the SEMPITERNAL token contract
     */
    constructor(address _tokenContract) {
        require(_tokenContract != address(0), "Token contract cannot be zero address");
        tokenContract = _tokenContract;
        owner = msg.sender;
    }
    
    /**
     * @dev Add an address to the whitelist
     * @param _address Address to be added to the whitelist
     */
    function addToWhitelist(address _address) external onlyOwner {
        require(_address != address(0), "Cannot add zero address to whitelist");
        require(!whitelist[_address], "Address already whitelisted");
        
        whitelist[_address] = true;
        whitelistedAddresses.push(_address);
        
        emit AddedToWhitelist(_address);
    }
    
    /**
     * @dev Add multiple addresses to the whitelist
     * @param _addresses Array of addresses to be added to the whitelist
     */
    function addMultipleToWhitelist(address[] calldata _addresses) external onlyOwner {
        for (uint256 i = 0; i < _addresses.length; i++) {
            address _address = _addresses[i];
            
            if (_address != address(0) && !whitelist[_address]) {
                whitelist[_address] = true;
                whitelistedAddresses.push(_address);
                
                emit AddedToWhitelist(_address);
            }
        }
    }
    
    /**
     * @dev Remove an address from the whitelist
     * @param _address Address to be removed from the whitelist
     */
    function removeFromWhitelist(address _address) external onlyOwner {
        require(whitelist[_address], "Address not whitelisted");
        
        whitelist[_address] = false;
        
        // Remove from array (this is gas-intensive, consider if you really need this)
        for (uint256 i = 0; i < whitelistedAddresses.length; i++) {
            if (whitelistedAddresses[i] == _address) {
                whitelistedAddresses[i] = whitelistedAddresses[whitelistedAddresses.length - 1];
                whitelistedAddresses.pop();
                break;
            }
        }
        
        emit RemovedFromWhitelist(_address);
    }
    
    /**
     * @dev Check if an address is whitelisted
     * @param _address Address to check
     * @return bool True if the address is whitelisted, false otherwise
     */
    function isWhitelisted(address _address) external view returns (bool) {
        return whitelist[_address];
    }
    
    /**
     * @dev Get all whitelisted addresses
     * @return address[] Array of all whitelisted addresses
     */
    function getAllWhitelistedAddresses() external view returns (address[] memory) {
        return whitelistedAddresses;
    }
    
    /**
     * @dev Get the total number of whitelisted addresses
     * @return uint256 Total number of whitelisted addresses
     */
    function getWhitelistCount() external view returns (uint256) {
        return whitelistedAddresses.length;
    }
    
    /**
     * @dev Transfer ownership of the contract
     * @param newOwner Address of the new owner
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
    
    /**
     * @dev Allow users to add themselves to the whitelist (if you want to enable this)
     * This function can be enabled or disabled based on your requirements
     */
    function selfRegister() external {
        require(!whitelist[msg.sender], "Address already whitelisted");
        
        whitelist[msg.sender] = true;
        whitelistedAddresses.push(msg.sender);
        
        emit AddedToWhitelist(msg.sender);
    }
}