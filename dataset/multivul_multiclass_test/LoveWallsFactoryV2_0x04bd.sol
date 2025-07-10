// SPDX-License-Identifier: MIT
// File: @openzeppelin/contracts/utils/introspection/IERC165.sol


// OpenZeppelin Contracts (last updated v5.1.0) (utils/introspection/IERC165.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC-165 standard, as defined in the
 * https://eips.ethereum.org/EIPS/eip-165[ERC].
 *
 * Implementers can declare support of contract interfaces, which can then be
 * queried by others ({ERC165Checker}).
 *
 * For an implementation, see {ERC165}.
 */
interface IERC165 {
    /**
     * @dev Returns true if this contract implements the interface defined by
     * `interfaceId`. See the corresponding
     * https://eips.ethereum.org/EIPS/eip-165#how-interfaces-are-identified[ERC section]
     * to learn more about how these ids are created.
     *
     * This function call must use less than 30 000 gas.
     */
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}

// File: @openzeppelin/contracts/token/ERC721/IERC721.sol


// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC721/IERC721.sol)

pragma solidity ^0.8.20;


/**
 * @dev Required interface of an ERC-721 compliant contract.
 */
interface IERC721 is IERC165 {
    /**
     * @dev Emitted when `tokenId` token is transferred from `from` to `to`.
     */
    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);

    /**
     * @dev Emitted when `owner` enables `approved` to manage the `tokenId` token.
     */
    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);

    /**
     * @dev Emitted when `owner` enables or disables (`approved`) `operator` to manage all of its assets.
     */
    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);

    /**
     * @dev Returns the number of tokens in ``owner``'s account.
     */
    function balanceOf(address owner) external view returns (uint256 balance);

    /**
     * @dev Returns the owner of the `tokenId` token.
     *
     * Requirements:
     *
     * - `tokenId` must exist.
     */
    function ownerOf(uint256 tokenId) external view returns (address owner);

    /**
     * @dev Safely transfers `tokenId` token from `from` to `to`.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `tokenId` token must exist and be owned by `from`.
     * - If the caller is not `from`, it must be approved to move this token by either {approve} or {setApprovalForAll}.
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon
     *   a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId, bytes calldata data) external;

    /**
     * @dev Safely transfers `tokenId` token from `from` to `to`, checking first that contract recipients
     * are aware of the ERC-721 protocol to prevent tokens from being forever locked.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `tokenId` token must exist and be owned by `from`.
     * - If the caller is not `from`, it must have been allowed to move this token by either {approve} or
     *   {setApprovalForAll}.
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon
     *   a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId) external;

    /**
     * @dev Transfers `tokenId` token from `from` to `to`.
     *
     * WARNING: Note that the caller is responsible to confirm that the recipient is capable of receiving ERC-721
     * or else they may be permanently lost. Usage of {safeTransferFrom} prevents loss, though the caller must
     * understand this adds an external call which potentially creates a reentrancy vulnerability.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `tokenId` token must be owned by `from`.
     * - If the caller is not `from`, it must be approved to move this token by either {approve} or {setApprovalForAll}.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 tokenId) external;

    /**
     * @dev Gives permission to `to` to transfer `tokenId` token to another account.
     * The approval is cleared when the token is transferred.
     *
     * Only a single account can be approved at a time, so approving the zero address clears previous approvals.
     *
     * Requirements:
     *
     * - The caller must own the token or be an approved operator.
     * - `tokenId` must exist.
     *
     * Emits an {Approval} event.
     */
    function approve(address to, uint256 tokenId) external;

    /**
     * @dev Approve or remove `operator` as an operator for the caller.
     * Operators can call {transferFrom} or {safeTransferFrom} for any token owned by the caller.
     *
     * Requirements:
     *
     * - The `operator` cannot be the address zero.
     *
     * Emits an {ApprovalForAll} event.
     */
    function setApprovalForAll(address operator, bool approved) external;

    /**
     * @dev Returns the account approved for `tokenId` token.
     *
     * Requirements:
     *
     * - `tokenId` must exist.
     */
    function getApproved(uint256 tokenId) external view returns (address operator);

    /**
     * @dev Returns if the `operator` is allowed to manage all of the assets of `owner`.
     *
     * See {setApprovalForAll}
     */
    function isApprovedForAll(address owner, address operator) external view returns (bool);
}

// File: @openzeppelin/contracts/utils/Context.sol


// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

pragma solidity ^0.8.20;

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}

// File: @openzeppelin/contracts/access/Ownable.sol


// OpenZeppelin Contracts (last updated v5.0.0) (access/Ownable.sol)

pragma solidity ^0.8.28;


/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * The initial owner is set to the address provided by the deployer. This can
 * later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
    address private _owner;

    /**
     * @dev The caller account is not authorized to perform an operation.
     */
    error OwnableUnauthorizedAccount(address account);

    /**
     * @dev The owner is not a valid owner account. (eg. `address(0)`)
     */
    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Initializes the contract setting the address provided by the deployer as the initial owner.
     */
    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if the sender is not the owner.
     */
    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby disabling any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

// File: @openzeppelin/contracts/utils/Base64.sol


// OpenZeppelin Contracts (last updated v5.1.0) (utils/Base64.sol)

pragma solidity ^0.8.20;

/**
 * @dev Provides a set of functions to operate with Base64 strings.
 */
library Base64 {
    /**
     * @dev Base64 Encoding/Decoding Table
     * See sections 4 and 5 of https://datatracker.ietf.org/doc/html/rfc4648
     */
    string internal constant _TABLE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    string internal constant _TABLE_URL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

    /**
     * @dev Converts a `bytes` to its Bytes64 `string` representation.
     */
    function encode(bytes memory data) internal pure returns (string memory) {
        return _encode(data, _TABLE, true);
    }

    /**
     * @dev Converts a `bytes` to its Bytes64Url `string` representation.
     * Output is not padded with `=` as specified in https://www.rfc-editor.org/rfc/rfc4648[rfc4648].
     */
    function encodeURL(bytes memory data) internal pure returns (string memory) {
        return _encode(data, _TABLE_URL, false);
    }

    /**
     * @dev Internal table-agnostic conversion
     */
    function _encode(bytes memory data, string memory table, bool withPadding) private pure returns (string memory) {
        /**
         * Inspired by Brecht Devos (Brechtpd) implementation - MIT licence
         * https://github.com/Brechtpd/base64/blob/e78d9fd951e7b0977ddca77d92dc85183770daf4/base64.sol
         */
        if (data.length == 0) return "";

        // If padding is enabled, the final length should be `bytes` data length divided by 3 rounded up and then
        // multiplied by 4 so that it leaves room for padding the last chunk
        // - `data.length + 2`  -> Prepare for division rounding up
        // - `/ 3`              -> Number of 3-bytes chunks (rounded up)
        // - `4 *`              -> 4 characters for each chunk
        // This is equivalent to: 4 * Math.ceil(data.length / 3)
        //
        // If padding is disabled, the final length should be `bytes` data length multiplied by 4/3 rounded up as
        // opposed to when padding is required to fill the last chunk.
        // - `4 * data.length`  -> 4 characters for each chunk
        // - ` + 2`             -> Prepare for division rounding up
        // - `/ 3`              -> Number of 3-bytes chunks (rounded up)
        // This is equivalent to: Math.ceil((4 * data.length) / 3)
        uint256 resultLength = withPadding ? 4 * ((data.length + 2) / 3) : (4 * data.length + 2) / 3;

        string memory result = new string(resultLength);

        assembly ("memory-safe") {
            // Prepare the lookup table (skip the first "length" byte)
            let tablePtr := add(table, 1)

            // Prepare result pointer, jump over length
            let resultPtr := add(result, 0x20)
            let dataPtr := data
            let endPtr := add(data, mload(data))

            // In some cases, the last iteration will read bytes after the end of the data. We cache the value, and
            // set it to zero to make sure no dirty bytes are read in that section.
            let afterPtr := add(endPtr, 0x20)
            let afterCache := mload(afterPtr)
            mstore(afterPtr, 0x00)

            // Run over the input, 3 bytes at a time
            for {

            } lt(dataPtr, endPtr) {

            } {
                // Advance 3 bytes
                dataPtr := add(dataPtr, 3)
                let input := mload(dataPtr)

                // To write each character, shift the 3 byte (24 bits) chunk
                // 4 times in blocks of 6 bits for each character (18, 12, 6, 0)
                // and apply logical AND with 0x3F to bitmask the least significant 6 bits.
                // Use this as an index into the lookup table, mload an entire word
                // so the desired character is in the least significant byte, and
                // mstore8 this least significant byte into the result and continue.

                mstore8(resultPtr, mload(add(tablePtr, and(shr(18, input), 0x3F))))
                resultPtr := add(resultPtr, 1) // Advance

                mstore8(resultPtr, mload(add(tablePtr, and(shr(12, input), 0x3F))))
                resultPtr := add(resultPtr, 1) // Advance

                mstore8(resultPtr, mload(add(tablePtr, and(shr(6, input), 0x3F))))
                resultPtr := add(resultPtr, 1) // Advance

                mstore8(resultPtr, mload(add(tablePtr, and(input, 0x3F))))
                resultPtr := add(resultPtr, 1) // Advance
            }

            // Reset the value that was cached
            mstore(afterPtr, afterCache)

            if withPadding {
                // When data `bytes` is not exactly 3 bytes long
                // it is padded with `=` characters at the end
                switch mod(mload(data), 3)
                case 1 {
                    mstore8(sub(resultPtr, 1), 0x3d)
                    mstore8(sub(resultPtr, 2), 0x3d)
                }
                case 2 {
                    mstore8(sub(resultPtr, 1), 0x3d)
                }
            }
        }

        return result;
    }
}

// File: contracts/ILoveMessageFactory.sol


pragma solidity ^0.8.28;



// Interface déclarée en dehors du contrat LoveWallsFactory
interface ILoveMessageFactory {
     struct LoveMessage {
        string lm_event;
        uint256 amount;
        address sender;
        address recipient;
        bool claimed;
        bool refund;
        bool read;
        uint256 creationTime;
    }

    // Fonction pour lire un LoveMessage
    function getLoveMessage(uint256 id) external view returns (LoveMessage memory);
    function loveMessageExists(uint256 id) external  view returns (bool);
}

// File: contracts/LoveWallFactoryV2.sol


pragma solidity ^0.8.28;





contract LoveWallsFactoryV2 is  Ownable {


 struct Wall {
    uint256 wallId; 
    string name;
    uint256 permanentPlacePrice;  
    uint256 nonPermanentPlacePrice;  
    uint256 permanentPlacesSold; 
    uint256 nonPermanentPlacesSold;
    mapping(uint256 => address) permanentPlaceOwners;
    mapping(uint256 => bool) permanentPlaceForSale; 
    mapping(uint256 => address) nonPermanentPlaceOwners;
    mapping(uint256 => uint256)  placeToLoveMessage;   
    mapping(uint256 => bool) placeExists;
    uint256[] permanentPlaceIds;
    uint256[] nonPermanentPlaceIds;
}

    address public LoveMessageFactoryAddress =  0x350D6C447Dcf43c859DD78CC88A380a1F04b6103; 
    address public LoveMessageFactoryStagingAddress = address(0);

   
 
    

    uint256 public  permanentPlacesLimit = 18;
    IERC721 private _LoveMessageContract = IERC721(LoveMessageFactoryAddress);
    uint256 public  nonPermanentPlacesLimit = 360;
    uint256 public wallLenght = 5;
    
    mapping(uint256 => Wall) public walls; 
 


    event PermanentPlacePurchased(uint256 indexed wallId, uint256 placeId, address owner, uint256 price);
    event NonPermanentPlacePurchased(uint256 indexed wallId, uint256 placeId, address owner, uint256 price);
    event LoveMessageBanned(uint256 loveMessageTokenId, uint256 placeId, address bannedBy);
    event LoveMessageFactoryUpdated(address newAddress);
     event LoveMessageFactoryStaged(address newAddress);
    event FundsTransferred(address recipient, uint256 amount);

    constructor()  Ownable(msg.sender) {
       
        initializeWalls();
    }



    function initializeWalls() private {
        string[5] memory wallNames = ["MAYA", "MAMU", "ENNN", "BIII", "KOUBI KOUBI"];
        for (uint256 i = 0; i < wallNames.length; i++) {
            walls[i].wallId = i;
            walls[i].name = wallNames[i];
            walls[i].permanentPlacePrice = 1 ether;
            walls[i].nonPermanentPlacePrice = 0.0025 ether;
           
        }
    }
  

 function getLoveMessage(uint256 tokenId) public view returns (ILoveMessageFactory.LoveMessage memory) {
        ILoveMessageFactory loveMessageFactory = ILoveMessageFactory(LoveMessageFactoryAddress);

      
        return loveMessageFactory.getLoveMessage(tokenId);
    }
 function loveMessageExists(uint256 tokenId) public view returns (bool) {
        ILoveMessageFactory loveMessageFactory = ILoveMessageFactory(LoveMessageFactoryAddress);

      
        return loveMessageFactory.loveMessageExists(tokenId);
    }


function putPermanentPlaceForSale(uint256 wallId, uint256 placeId) external {
    Wall storage wall = walls[wallId];
    require(!wall.permanentPlaceForSale[placeId] , "Place already for sale");
    require(wall.permanentPlaceOwners[placeId] == msg.sender, "You are not the owner of this place");
    require(wall.permanentPlacesSold == permanentPlacesLimit, "All permanent places must be sold before putting a place for sale");
    wall.permanentPlaceForSale[placeId] = true; 
}

function removePermanentPlaceFromSale(uint256 wallId, uint256 placeId) external {
    Wall storage wall = walls[wallId];
    require(wall.permanentPlaceForSale[placeId] , "Place not for sale");
    require(wall.permanentPlaceOwners[placeId] == msg.sender, "You are not the owner of this place");
    require(wall.permanentPlacesSold == permanentPlacesLimit, "All permanent places must be sold before putting a place for sale");
    wall.permanentPlaceForSale[placeId] = false; 
}
// Fonction pour acheter une place permanente
function buyPermanentPlace(uint256 wallId, uint256 placeId, uint256 loveMessageTokenId) external payable {
    Wall storage wall = walls[wallId];
      require(loveMessageExists(loveMessageTokenId),"Love message does not exist");
     ILoveMessageFactory.LoveMessage memory loveMessage = getLoveMessage(loveMessageTokenId);
  
    require(msg.value >= wall.permanentPlacePrice, "Incorrect amount sent");
   
    require((_LoveMessageContract.ownerOf(loveMessageTokenId) == msg.sender || loveMessage.sender == msg.sender ) , "You do not own this LoveMessage");
    require(loveMessage.claimed, "This LoveMessage has already been claimed");
    require(wall.permanentPlaceOwners[placeId] != msg.sender, "You already own this place");
    require((wall.placeExists[placeId] && wall.permanentPlaceForSale[placeId]) ||(!wall.placeExists[placeId] && !wall.permanentPlaceForSale[placeId])  , "Place already taken and not for sale");
  
      for (uint256 w = 0; w < wallLenght; w++) { 
        Wall storage currentWall = walls[w];
        for (uint256 i = 0; i < currentWall.permanentPlaceIds.length; i++) { 
            uint256 existingPlaceId = currentWall.permanentPlaceIds[i];
            if (currentWall.placeToLoveMessage[existingPlaceId] == loveMessageTokenId) {
                revert("This LoveMessage has already been used to buy a place");
            }
        }
        
    }

    if (wall.permanentPlacesSold < permanentPlacesLimit) { 
        
        // Phase initiale : vente des 100 premières places
        require(placeId == wall.permanentPlacesSold, "Invalid placeId for initial sale");
        wall.permanentPlaceOwners[placeId] = msg.sender;
        wall.permanentPlaceIds.push(placeId);
        wall.permanentPlacesSold++;
        payable(owner()).transfer(msg.value); 
        if (wall.permanentPlacesSold == permanentPlacesLimit) {
            uint256 newPrice = (wall.permanentPlacePrice * 11402) / 10000;
            wall.permanentPlacePrice = newPrice;
        }
    } else {
        // Phase secondaire : revente des places
        require(wall.permanentPlaceForSale[placeId], "This place is not for sale");

        address previousOwner = wall.permanentPlaceOwners[placeId];
        wall.permanentPlaceOwners[placeId] = msg.sender; 
        wall.permanentPlaceForSale[placeId] = false; 

        payable(previousOwner).transfer(msg.value); 


        uint256 newPrice = (wall.permanentPlacePrice * 11402) / 10000;
        wall.permanentPlacePrice = newPrice;
    }

    // Lier le LoveMessage à la place achetée
    wall.placeToLoveMessage[placeId] = loveMessageTokenId;
    wall.placeExists[placeId] = true;
    emit PermanentPlacePurchased(wallId, placeId, msg.sender, wall.permanentPlacePrice);
    emit FundsTransferred(owner(), msg.value);
}

function buyNonPermanentPlace(uint256 wallId, uint256 loveMessageTokenId) external payable {
    Wall storage wall = walls[wallId];
    require(msg.value >= wall.nonPermanentPlacePrice, "Incorrect amount sent");
    ILoveMessageFactory.LoveMessage memory loveMessage = this.getLoveMessage(loveMessageTokenId);
    require((_LoveMessageContract.ownerOf(loveMessageTokenId) == msg.sender || loveMessage.sender == msg.sender), "You do not own this LoveMessage");
    require(loveMessage.claimed, "This LoveMessage has not been claimed");
  
  
    uint256 placeId;

    if (wall.nonPermanentPlacesSold < nonPermanentPlacesLimit) {
     
        placeId = (permanentPlacesLimit + wall.nonPermanentPlacesSold);
        wall.nonPermanentPlaceOwners[placeId] = msg.sender;
        wall.nonPermanentPlaceIds.push(placeId);
        wall.nonPermanentPlacesSold++;
    } else {
    
        placeId = wall.nonPermanentPlaceIds[0];
       
        
        wall.nonPermanentPlaceOwners[placeId] = msg.sender; 

        // Déplacer la place remplacée à la fin de la liste
        for (uint256 i = 0; i < wall.nonPermanentPlaceIds.length - 1; i++) {
            wall.nonPermanentPlaceIds[i] = wall.nonPermanentPlaceIds[i + 1];
        }
        wall.nonPermanentPlaceIds[wall.nonPermanentPlaceIds.length - 1] = placeId;
      
    }

  
    if (wall.nonPermanentPlacesSold == nonPermanentPlacesLimit) {
        uint256 newPrice = (wall.nonPermanentPlacePrice * 102904) / 100000;
        wall.nonPermanentPlacePrice = newPrice;
        if(wall.nonPermanentPlacePrice >= wall.permanentPlacePrice){
           wall.permanentPlacePrice = (wall.permanentPlacePrice + wall.nonPermanentPlacePrice);
        }
    }

    // Lier le LoveMessage à la place achetée
    wall.placeToLoveMessage[placeId] = loveMessageTokenId;
    wall.placeExists[placeId] = true;
       payable(owner()).transfer(msg.value); 
         emit FundsTransferred(owner(), msg.value);
    emit NonPermanentPlacePurchased(wallId, placeId, msg.sender, wall.nonPermanentPlacePrice);
}


  function getPermanentPlaceOwner(uint256 wallId, uint256 placeId) public view returns (address) {
        return walls[wallId].permanentPlaceOwners[placeId];
    }

    // Getter pour permanentPlaceForSale
    function getPermanentPlaceForSale(uint256 wallId, uint256 placeId) public view returns (bool) {
        return walls[wallId].permanentPlaceForSale[placeId];
    }

    // Getter pour nonPermanentPlaceOwners
    function getNonPermanentPlaceOwner(uint256 wallId, uint256 placeId) public view returns (address) {
        return walls[wallId].nonPermanentPlaceOwners[placeId];
    }

    

    // Getter pour placeToLoveMessage
    function getPlaceToLoveMessage(uint256 wallId, uint256 placeId) public view returns (uint256, bool) {

        if(!walls[wallId].placeExists[placeId]){
            return (0,false);
        }
        return (walls[wallId].placeToLoveMessage[placeId],true);


    }


function findWallAndPlaceIdByLoveMessage(uint256 loveMessageTokenId) public view returns (uint256 wallId, uint256 placeId, bool found) {
    // Parcourir tous les murs
    for (uint256 w = 0; w < wallLenght; w++) {
        Wall storage wall = walls[w];

        // Parcourir toutes les places permanentes du mur
        for (uint256 i = 0; i < wall.permanentPlaceIds.length; i++) {
            uint256 currentPlaceId = wall.permanentPlaceIds[i];
            if (wall.placeToLoveMessage[currentPlaceId] == loveMessageTokenId) {
                return (w, currentPlaceId, true); // Retourner le wallId, le placeId et true
            }
        }
 
        // Parcourir toutes les places non permanentes du mur
        for (uint256 i = 0; i < wall.nonPermanentPlaceIds.length; i++) {
            uint256 currentPlaceId = wall.nonPermanentPlaceIds[i];
            if (wall.placeToLoveMessage[currentPlaceId] == loveMessageTokenId) {
                return (w, currentPlaceId, true); // Retourner le wallId, le placeId et true
            }
        }
    }

    return (0, 0, false); // Retourner (0, 0, false) si le LoveMessage n'est trouvé sur aucun mur
}

 function updateLoveMessageFactoryAddress() onlyOwner external  {
  require(msg.sender == owner(), "Only admin can update excute this action");
  LoveMessageFactoryAddress = LoveMessageFactoryStagingAddress;
  emit LoveMessageFactoryUpdated(LoveMessageFactoryAddress);
 }


 function stagingLoveMessageFactoryAddress(address loveMessageFactoryStagedAddress) onlyOwner external  {
  require( loveMessageFactoryStagedAddress != address(0), "Invalid address");
  require(msg.sender == owner(), "Only admin can update excute this action");
  LoveMessageFactoryStagingAddress =  loveMessageFactoryStagedAddress;
  emit LoveMessageFactoryStaged(loveMessageFactoryStagedAddress);

 }




function getPermanentPlacesForSale(uint256 wallId) public view returns (uint256[] memory) {
        Wall storage wall = walls[wallId];
        uint256[] memory placesForSale = new uint256[](wall.permanentPlaceIds.length);
        uint256 count = 0;

        for (uint256 i = 0; i < wall.permanentPlaceIds.length; i++) {
            uint256 placeId = wall.permanentPlaceIds[i];
            if (wall.permanentPlaceForSale[placeId]) {
                placesForSale[count] = placeId;
                count++;
            }
        }

        // Redimensionner le tableau pour qu'il corresponde au nombre réel de places en vente
        uint256[] memory result = new uint256[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = placesForSale[i];
        }

     return result;
}

}