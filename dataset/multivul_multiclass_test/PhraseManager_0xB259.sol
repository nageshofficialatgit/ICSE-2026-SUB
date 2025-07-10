// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PhraseManager {
    address public owner;
    
    // Store phrases in a fixed array for gas efficiency
    string[] private phrases;
    
    // Track used phrases with a bitmap for gas efficiency
    // Each uint256 can track 256 phrases
    uint256[] private usedPhraseMap;
    
    // Counter for unused phrases
    uint256 public unusedCount;
    
    // Events
    event PhraseAdded(string phrase, uint256 index);
    event PhraseUsed(string phrase, uint256 index);
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    // Add a new phrase
    function addPhrase(string calldata _phrase) external onlyOwner {
        phrases.push(_phrase);
        uint256 index = phrases.length - 1;
        uint256 mapIndex = index / 256;
        
        // Extend usedPhraseMap if needed
        if (mapIndex >= usedPhraseMap.length) {
            usedPhraseMap.push(0);
        }
        
        unusedCount++;
        emit PhraseAdded(_phrase, index);
    }
    
    // Add multiple phrases in one transaction
    function addPhrases(string[] calldata _phrases) external onlyOwner {
        uint256 startIndex = phrases.length;
        for (uint256 i = 0; i < _phrases.length; i++) {
            phrases.push(_phrases[i]);
            uint256 mapIndex = (startIndex + i) / 256;
            if (mapIndex >= usedPhraseMap.length) {
                usedPhraseMap.push(0);
            }
        }
        unusedCount += _phrases.length;
    }
    
    // Pick an unused phrase pseudo-randomly
    function pickPhrase() external returns (string memory) {
        require(unusedCount > 0, "No unused phrases left");
        
        // Generate pseudo-random index using block data and timestamp
        uint256 randomIndex = uint256(keccak256(abi.encodePacked(
            block.timestamp,
            block.prevrandao,
            phrases.length
        ))) % phrases.length;
        
        // Find the next unused phrase starting from randomIndex
        uint256 index = findNextUnusedPhrase(randomIndex);
        
        // Mark phrase as used
        uint256 mapIndex = index / 256;
        uint256 bitIndex = index % 256;
        usedPhraseMap[mapIndex] |= (1 << bitIndex);
        unusedCount--;
        
        emit PhraseUsed(phrases[index], index);
        return phrases[index];
    }
    
    // Helper function to find next unused phrase
    function findNextUnusedPhrase(uint256 startIndex) private view returns (uint256) {
        uint256 index = startIndex;
        uint256 length = phrases.length;
        
        for (uint256 i = 0; i < length; i++) {
            uint256 currentIndex = (index + i) % length;
            uint256 mapIndex = currentIndex / 256;
            uint256 bitIndex = currentIndex % 256;
            
            if ((usedPhraseMap[mapIndex] & (1 << bitIndex)) == 0) {
                return currentIndex;
            }
        }
        
        revert("No unused phrases found");
    }
    
    // View functions
    function getPhraseCount() external view returns (uint256) {
        return phrases.length;
    }
    
    function getUnusedCount() external view returns (uint256) {
        return unusedCount;
    }
    
    function isUsed(uint256 index) external view returns (bool) {
        require(index < phrases.length, "Index out of bounds");
        uint256 mapIndex = index / 256;
        uint256 bitIndex = index % 256;
        return (usedPhraseMap[mapIndex] & (1 << bitIndex)) != 0;
    }
    
     function getPhraseAtIndex(uint256 index) external view returns (string memory) {
        require(index < phrases.length, "Index out of bounds");
        return phrases[index];
    }
    
    // Optional: Reset usage status of all phrases
    function resetUsage() external onlyOwner {
        for (uint256 i = 0; i < usedPhraseMap.length; i++) {
            usedPhraseMap[i] = 0;
        }
        unusedCount = phrases.length;
    }
}