//      ___   __  __________  _________   _______________________________
//     /   | / / / /_  __/ / / / ____/ | / /_  __/  _/ ____/  _/ ____/   |
//    / /| |/ / / / / / / /_/ / __/ /  |/ / / /  / // /_   / // /   / /| |
//   / ___ / /_/ / / / / __  / /___/ /|  / / / _/ // __/ _/ // /___/ ___ |
//  /_/  |_\____/ /_/ /_/ /_/_____/_/ |_/ /_/ /___/_/   /___/\____/_/  |_/
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ShallerAuthentifica {
    // Struct to store product information
    struct Product {
        string name;
        string productType;
        string description;
        uint256 createdAt;  // Timestamp of product registration
        string dimensions;
        string serialNumber;
        string[4] media;  // Array to store media URLs (images/videos)
        address owner;
        string history;  // Concatenated history log
        bool exists;  // Boolean flag to check if product exists
    }

    // Mapping to store products by their unique ID
    mapping(uint256 => Product) private products;
    address public admin;  // Admin address

    // Events to track actions on products
    event ProductRegistered(uint256 indexed productId, string name, string productType, address indexed owner, uint256 timestamp);
    event ProductUpdated(uint256 indexed productId, string field, string newValue, address indexed updatedBy, uint256 timestamp);
    event OwnershipTransferred(uint256 indexed productId, address indexed previousOwner, address indexed newOwner);
    event HistoryUpdated(uint256 indexed productId, string newEntry, address indexed updatedBy, uint256 timestamp);

    // Modifiers to restrict access
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this action");
        _;
    }

    modifier onlyOwnerOrAdmin(uint256 productId) {
        require(msg.sender == products[productId].owner || msg.sender == admin, "Not authorized to update");
        _;
    }

    modifier productExists(uint256 productId) {
        require(products[productId].exists, "Product does not exist");
        _;
    }

    // Constructor: sets the contract deployer as admin
    constructor() {
        admin = msg.sender;
    }

    // Function to register a new product
    function registerProduct(
        uint256 productId,
        string memory name,
        string memory productType,
        string memory description,
        string memory dimensions,
        string memory serialNumber,
        string[4] memory media
    ) public onlyAdmin {
        require(!products[productId].exists, "Product ID already registered");

        products[productId] = Product({
            name: name,
            productType: productType,
            description: description,
            createdAt: block.timestamp,
            dimensions: dimensions,
            serialNumber: serialNumber,
            media: media,
            owner: msg.sender,
            history: "",  // Initialize empty history
            exists: true
        });

        emit ProductRegistered(productId, name, productType, msg.sender, block.timestamp);
    }

    // Functions to update individual product fields
    function updateName(uint256 productId, string memory newName) public onlyOwnerOrAdmin(productId) productExists(productId) {
        products[productId].name = newName;
        emit ProductUpdated(productId, "name", newName, msg.sender, block.timestamp);
    }

    function updateProductType(uint256 productId, string memory newType) public onlyOwnerOrAdmin(productId) productExists(productId) {
        products[productId].productType = newType;
        emit ProductUpdated(productId, "productType", newType, msg.sender, block.timestamp);
    }

    function updateDescription(uint256 productId, string memory newDescription) public onlyOwnerOrAdmin(productId) productExists(productId) {
        products[productId].description = newDescription;
        emit ProductUpdated(productId, "description", newDescription, msg.sender, block.timestamp);
    }

    function updateDimensions(uint256 productId, string memory newDimensions) public onlyOwnerOrAdmin(productId) productExists(productId) {
        products[productId].dimensions = newDimensions;
        emit ProductUpdated(productId, "dimensions", newDimensions, msg.sender, block.timestamp);
    }

    function updateSerialNumber(uint256 productId, string memory newSerialNumber) public onlyOwnerOrAdmin(productId) productExists(productId) {
        products[productId].serialNumber = newSerialNumber;
        emit ProductUpdated(productId, "serialNumber", newSerialNumber, msg.sender, block.timestamp);
    }

    // Function to update a specific media file
    function updateMedia(uint256 productId, uint8 index, string memory newMedia) public onlyOwnerOrAdmin(productId) productExists(productId) {
        require(index < 4, "Invalid media index"); // Ensure the index is within bounds
        products[productId].media[index] = newMedia;
        emit ProductUpdated(productId, string(abi.encodePacked("media", index)), newMedia, msg.sender, block.timestamp);
    }

    // Function to transfer product ownership to a new address
    function transferOwnership(uint256 productId, address newOwner) public productExists(productId) {
        require(msg.sender == products[productId].owner, "Only owner can transfer ownership");

        address previousOwner = products[productId].owner;
        products[productId].owner = newOwner;

        emit OwnershipTransferred(productId, previousOwner, newOwner);
    }

    // Function to add an entry to the product's history
    function addToHistory(uint256 productId, string memory newEntry) public onlyOwnerOrAdmin(productId) productExists(productId) {
        products[productId].history = string(abi.encodePacked(products[productId].history, " | ", newEntry));
        emit HistoryUpdated(productId, newEntry, msg.sender, block.timestamp);
    }

    // Getter functions to retrieve product details
    function getProductBasicInfo(uint256 productId) public view productExists(productId) returns (
        string memory name,
        string memory productType,
        string memory description,
        uint256 createdAt
    ) {
        Product memory p = products[productId];
        return (p.name, p.productType, p.description, p.createdAt);
    }

    function getProductDetails(uint256 productId) public view productExists(productId) returns (
        string memory dimensions,
        string memory serialNumber,
        address owner,
        string memory history
    ) {
        Product memory p = products[productId];
        return (p.dimensions, p.serialNumber, p.owner, p.history);
    }

    function getProductMedia(uint256 productId) public view productExists(productId) returns (
        string memory media1,
        string memory media2,
        string memory media3,
        string memory media4
    ) {
        Product memory p = products[productId];
        return (p.media[0], p.media[1], p.media[2], p.media[3]);
    }
}


//    / ___|| | | |  / \  | |   | |   | ____|  _ \
//    \___ \| |_| | / _ \ | |   | |   |  _| | |_) |
//     ___) |  _  |/ ___ \| |___| |___| |___|  _ <
//    |____/|_| |_/_/   \_\_____|_____|_____|_| \_\