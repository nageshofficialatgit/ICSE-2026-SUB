// SPDX-License-Identifier: MIT
pragma solidity ^ 0.8.20;
library RLPReader {
    uint8 constant STRING_SHORT_START = 0x80;
    uint8 constant STRING_LONG_START = 0xb8;
    uint8 constant LIST_SHORT_START = 0xc0;
    uint8 constant LIST_LONG_START = 0xf8;
    uint8 constant WORD_SIZE = 32;
    struct RLPItem {
        uint256 len;
        uint256 memPtr;
    }
    struct Iterator {
        RLPItem item; // Item that's being iterated over.
        uint256 nextPtr; // Position of the next item in the list.
    }
    /*
     * @dev Returns the next element in the iteration. Reverts if it has not next element.
     * @param self The iterator.
     * @return The next element in the iteration.
     */
    function next(Iterator memory self) internal pure returns(RLPItem memory) {
        require(hasNext(self));
        uint256 ptr = self.nextPtr;
        uint256 itemLength = _itemLength(ptr);
        self.nextPtr = ptr + itemLength;
        return RLPItem(itemLength, ptr);
    }
    /*
     * @dev Returns true if the iteration has more elements.
     * @param self The iterator.
     * @return true if the iteration has more elements.
     */
    function hasNext(Iterator memory self) internal pure returns(bool) {
        RLPItem memory item = self.item;
        return self.nextPtr < item.memPtr + item.len;
    }
    /*
     * @param item RLP encoded bytes
     */
    function toRlpItem(bytes memory item) internal pure returns(RLPItem memory) {
        uint256 memPtr;
        assembly {
            memPtr:= add(item, 0x20)
        }
        return RLPItem(item.length, memPtr);
    }
    /*
     * @dev Create an iterator. Reverts if item is not a list.
     * @param self The RLP item.
     * @return An 'Iterator' over the item.
     */
    function iterator(RLPItem memory self) internal pure returns(Iterator memory) {
        require(isList(self));
        uint256 ptr = self.memPtr + _payloadOffset(self.memPtr);
        return Iterator(self, ptr);
    }
    /*
     * @param the RLP item.
     */
    function rlpLen(RLPItem memory item) internal pure returns(uint256) {
        return item.len;
    }
    /*
     * @param the RLP item.
     * @return (memPtr, len) pair: location of the item's payload in memory.
     */
    function payloadLocation(RLPItem memory item) internal pure returns(uint256, uint256) {
        uint256 offset = _payloadOffset(item.memPtr);
        uint256 memPtr = item.memPtr + offset;
        uint256 len = item.len - offset; // data length
        return (memPtr, len);
    }
    /*
     * @param the RLP item.
     */
    function payloadLen(RLPItem memory item) internal pure returns(uint256) {
        (, uint256 len) = payloadLocation(item);
        return len;
    }
    /*
     * @param the RLP item containing the encoded list.
     */
    function toList(RLPItem memory item) internal pure returns(RLPItem[] memory) {
        require(isList(item));
        uint256 items = numItems(item);
        RLPItem[] memory result = new RLPItem[](items);
        uint256 memPtr = item.memPtr + _payloadOffset(item.memPtr);
        uint256 dataLen;
        for (uint256 i = 0; i < items; i++) {
            dataLen = _itemLength(memPtr);
            result[i] = RLPItem(dataLen, memPtr);
            memPtr = memPtr + dataLen;
        }
        return result;
    }
    // @return indicator whether encoded payload is a list. negate this function call for isData.
    function isList(RLPItem memory item) internal pure returns(bool) {
        if (item.len == 0) return false;
        uint8 byte0;
        uint256 memPtr = item.memPtr;
        assembly {
            byte0:= byte(0, mload(memPtr))
        }
        if (byte0 < LIST_SHORT_START) return false;
        return true;
    }
    /*
     * @dev A cheaper version of keccak256(toRlpBytes(item)) that avoids copying memory.
     * @return keccak256 hash of RLP encoded bytes.
     */
    function rlpBytesKeccak256(RLPItem memory item) internal pure returns(bytes32) {
        uint256 ptr = item.memPtr;
        uint256 len = item.len;
        bytes32 result;
        assembly {
            result:= keccak256(ptr, len)
        }
        return result;
    }
    /*
     * @dev A cheaper version of keccak256(toBytes(item)) that avoids copying memory.
     * @return keccak256 hash of the item payload.
     */
    function payloadKeccak256(RLPItem memory item) internal pure returns(bytes32) {
        (uint256 memPtr, uint256 len) = payloadLocation(item);
        bytes32 result;
        assembly {
            result:= keccak256(memPtr, len)
        }
        return result;
    }
    /** RLPItem conversions into data types **/
    // @returns raw rlp encoding in bytes
    function toRlpBytes(RLPItem memory item) internal pure returns(bytes memory) {
        bytes memory result = new bytes(item.len);
        if (result.length == 0) return result;
        uint256 ptr;
        assembly {
            ptr:= add(0x20, result)
        }
        copy(item.memPtr, ptr, item.len);
        return result;
    }
    // any non-zero byte except "0x80" is considered true
    function toBoolean(RLPItem memory item) internal pure returns(bool) {
        require(item.len == 1);
        uint256 result;
        uint256 memPtr = item.memPtr;
        assembly {
            result:= byte(0, mload(memPtr))
        }
        // SEE Github Issue #5.
        // Summary: Most commonly used RLP libraries (i.e Geth) will encode
        // "0" as "0x80" instead of as "0". We handle this edge case explicitly
        // here.
        if (result == 0 || result == STRING_SHORT_START) {
            return false;
        } else {
            return true;
        }
    }

    function toAddress(RLPItem memory item) internal pure returns(address) {
        // 1 byte for the length prefix
        require(item.len == 21);
        return address(uint160(toUint(item)));
    }

    function toUint(RLPItem memory item) internal pure returns(uint256) {
        require(item.len > 0 && item.len <= 33);
        (uint256 memPtr, uint256 len) = payloadLocation(item);
        uint256 result;
        assembly {
            result:= mload(memPtr)
            // shift to the correct location if neccesary
            if lt(len, 32) {
                result:= div(result, exp(256, sub(32, len)))
            }
        }
        return result;
    }
    // enforces 32 byte length
    function toUintStrict(RLPItem memory item) internal pure returns(uint256) {
        // one byte prefix
        require(item.len == 33);
        uint256 result;
        uint256 memPtr = item.memPtr + 1;
        assembly {
            result:= mload(memPtr)
        }
        return result;
    }

    function toBytes(RLPItem memory item) internal pure returns(bytes memory) {
        require(item.len > 0);
        (uint256 memPtr, uint256 len) = payloadLocation(item);
        bytes memory result = new bytes(len);
        uint256 destPtr;
        assembly {
            destPtr:= add(0x20, result)
        }
        copy(memPtr, destPtr, len);
        return result;
    }
    /*
     * Private Helpers
     */
    // @return number of payload items inside an encoded list.
    function numItems(RLPItem memory item) private pure returns(uint256) {
        if (item.len == 0) return 0;
        uint256 count = 0;
        uint256 currPtr = item.memPtr + _payloadOffset(item.memPtr);
        uint256 endPtr = item.memPtr + item.len;
        while (currPtr < endPtr) {
            currPtr = currPtr + _itemLength(currPtr); // skip over an item
            count++;
        }
        return count;
    }
    // @return entire rlp item byte length
    function _itemLength(uint256 memPtr) private pure returns(uint256) {
        uint256 itemLen;
        uint256 byte0;
        assembly {
            byte0:= byte(0, mload(memPtr))
        }
        if (byte0 < STRING_SHORT_START) {
            itemLen = 1;
        } else if (byte0 < STRING_LONG_START) {
            itemLen = byte0 - STRING_SHORT_START + 1;
        } else if (byte0 < LIST_SHORT_START) {
            assembly {
                let byteLen:= sub(byte0, 0xb7) // # of bytes the actual length is
                memPtr:= add(memPtr, 1) // skip over the first byte
                /* 32 byte word size */
                let dataLen:= div(mload(memPtr), exp(256, sub(32, byteLen))) // right shifting to get the len
                itemLen:= add(dataLen, add(byteLen, 1))
            }
        } else if (byte0 < LIST_LONG_START) {
            itemLen = byte0 - LIST_SHORT_START + 1;
        } else {
            assembly {
                let byteLen:= sub(byte0, 0xf7)
                memPtr:= add(memPtr, 1)
                let dataLen:= div(mload(memPtr), exp(256, sub(32, byteLen))) // right shifting to the correct length
                itemLen:= add(dataLen, add(byteLen, 1))
            }
        }
        return itemLen;
    }
    // @return number of bytes until the data
    function _payloadOffset(uint256 memPtr) private pure returns(uint256) {
        uint256 byte0;
        assembly {
            byte0:= byte(0, mload(memPtr))
        }
        if (byte0 < STRING_SHORT_START) {
            return 0;
        } else if (byte0 < STRING_LONG_START || (byte0 >= LIST_SHORT_START && byte0 < LIST_LONG_START)) {
            return 1;
        } else if (byte0 < LIST_SHORT_START) {
            // being explicit
            return byte0 - (STRING_LONG_START - 1) + 1;
        } else {
            return byte0 - (LIST_LONG_START - 1) + 1;
        }
    }
    /*
     * @param src Pointer to source
     * @param dest Pointer to destination
     * @param len Amount of memory to copy from the source
     */
    function copy(uint256 src, uint256 dest, uint256 len) private pure {
        if (len == 0) return;
        // copy as many word sizes as possible
        for (; len >= WORD_SIZE; len -= WORD_SIZE) {
            assembly {
                mstore(dest, mload(src))
            }
            src += WORD_SIZE;
            dest += WORD_SIZE;
        }
        if (len > 0) {
            // left over bytes. Mask is used to remove unwanted bytes from the word
            uint256 mask = 256 ** (WORD_SIZE - len) - 1;
            assembly {
                let srcpart:= and(mload(src), not(mask)) // zero out src
                let destpart:= and(mload(dest), mask) // retrieve the bytes
                mstore(dest, or(destpart, srcpart))
            }
        }
    }
}

library MerklePatriciaProofVerifier {
    using RLPReader
    for RLPReader.RLPItem;
    using RLPReader
    for bytes;
    /// @dev Validates a Merkle-Patricia-Trie proof.
    ///      If the proof proves the inclusion of some key-value pair in the
    ///      trie, the value is returned. Otherwise, i.e. if the proof proves
    ///      the exclusion of a key from the trie, an empty byte array is
    ///      returned.
    /// @param rootHash is the Keccak-256 hash of the root node of the MPT.
    /// @param path is the key of the node whose inclusion/exclusion we are
    ///        proving.
    /// @param stack is the stack of MPT nodes (starting with the root) that
    ///        need to be traversed during verification.
    /// @return value whose inclusion is proved or an empty byte array for
    ///         a proof of exclusion
    function extractProofValue(bytes32 rootHash, bytes memory path, RLPReader.RLPItem[] memory stack) internal pure returns(bytes memory value) {
        bytes memory mptKey = _decodeNibbles(path, 0);
        uint256 mptKeyOffset = 0;
        bytes32 nodeHashHash;
        RLPReader.RLPItem[] memory node;
        RLPReader.RLPItem memory rlpValue;
        if (stack.length == 0) {
            // Root hash of empty Merkle-Patricia-Trie
            require(rootHash == 0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421);
            return new bytes(0);
        }
        // Traverse stack of nodes starting at root.
        for (uint256 i = 0; i < stack.length; i++) {
            // We use the fact that an rlp encoded list consists of some
            // encoding of its length plus the concatenation of its
            // *rlp-encoded* items.
            // The root node is hashed with Keccak-256 ...
            if (i == 0 && rootHash != stack[i].rlpBytesKeccak256()) {
                revert('1');
            }
            // ... whereas all other nodes are hashed with the MPT
            // hash function.
            if (i != 0 && nodeHashHash != _mptHashHash(stack[i])) {
                revert('2');
            }
            // We verified that stack[i] has the correct hash, so we
            // may safely decode it.
            node = stack[i].toList();
            if (node.length == 2) {
                // Extension or Leaf node
                bool isLeaf;
                bytes memory nodeKey;
                (isLeaf, nodeKey) = _merklePatriciaCompactDecode(node[0].toBytes());
                uint256 prefixLength = _sharedPrefixLength(mptKeyOffset, mptKey, nodeKey);
                mptKeyOffset += prefixLength;
                if (prefixLength < nodeKey.length) {
                    // Proof claims divergent extension or leaf. (Only
                    // relevant for proofs of exclusion.)
                    // An Extension/Leaf node is divergent iff it "skips" over
                    // the point at which a Branch node should have been had the
                    // excluded key been included in the trie.
                    // Example: Imagine a proof of exclusion for path [1, 4],
                    // where the current node is a Leaf node with
                    // path [1, 3, 3, 7]. For [1, 4] to be included, there
                    // should have been a Branch node at [1] with a child
                    // at 3 and a child at 4.
                    // Sanity check
                    if (i < stack.length - 1) {
                        // divergent node must come last in proof
                        revert('3');
                    }
                    return new bytes(0);
                }
                if (isLeaf) {
                    // Sanity check
                    if (i < stack.length - 1) {
                        // leaf node must come last in proof
                        revert('4');
                    }
                    if (mptKeyOffset < mptKey.length) {
                        return new bytes(0);
                    }
                    rlpValue = node[1];
                    return rlpValue.toBytes();
                } else {
                    // extension
                    // Sanity check
                    if (i == stack.length - 1) {
                        // shouldn't be at last level
                        revert('5');
                    }
                    if (!node[1].isList()) {
                        // rlp(child) was at least 32 bytes. node[1] contains
                        // Keccak256(rlp(child)).
                        nodeHashHash = node[1].payloadKeccak256();
                    } else {
                        // rlp(child) was less than 32 bytes. node[1] contains
                        // rlp(child).
                        nodeHashHash = node[1].rlpBytesKeccak256();
                    }
                }
            } else if (node.length == 17) {
                // Branch node
                if (mptKeyOffset != mptKey.length) {
                    // we haven't consumed the entire path, so we need to look at a child
                    uint8 nibble = uint8(mptKey[mptKeyOffset]);
                    mptKeyOffset += 1;
                    if (nibble >= 16) {
                        // each element of the path has to be a nibble
                        revert('6');
                    }
                    if (_isEmptyBytesequence(node[nibble])) {
                        // Sanity
                        if (i != stack.length - 1) {
                            // leaf node should be at last level
                            revert('7');
                        }
                        return new bytes(0);
                    } else if (!node[nibble].isList()) {
                        nodeHashHash = node[nibble].payloadKeccak256();
                    } else {
                        nodeHashHash = node[nibble].rlpBytesKeccak256();
                    }
                } else {
                    // we have consumed the entire mptKey, so we need to look at what's contained in this node.
                    // Sanity
                    if (i != stack.length - 1) {
                        // should be at last level
                        revert('8');
                    }
                    return node[16].toBytes();
                }
            }
        }
    }
    /// @dev Computes the hash of the Merkle-Patricia-Trie hash of the RLP item.
    ///      Merkle-Patricia-Tries use a weird "hash function" that outputs
    ///      *variable-length* hashes: If the item is shorter than 32 bytes,
    ///      the MPT hash is the item. Otherwise, the MPT hash is the
    ///      Keccak-256 hash of the item.
    ///      The easiest way to compare variable-length byte sequences is
    ///      to compare their Keccak-256 hashes.
    /// @param item The RLP item to be hashed.
    /// @return Keccak-256(MPT-hash(item))
    function _mptHashHash(RLPReader.RLPItem memory item)
    private
    pure
    returns(bytes32) {
        if (item.len < 32) {
            return item.rlpBytesKeccak256();
        } else {
            return keccak256(abi.encodePacked(item.rlpBytesKeccak256()));
        }
    }

    function _isEmptyBytesequence(RLPReader.RLPItem memory item)
    private
    pure
    returns(bool) {
        if (item.len != 1) {
            return false;
        }
        uint8 b;
        uint256 memPtr = item.memPtr;
        assembly {
            b:= byte(0, mload(memPtr))
        }
        return b == 0x80; /* empty byte string */
    }

    function _merklePatriciaCompactDecode(bytes memory compact)
    private
    pure
    returns(bool isLeaf, bytes memory nibbles) {
        require(compact.length > 0);
        uint256 first_nibble = (uint8(compact[0]) >> 4) & 0xF;
        uint256 skipNibbles;
        if (first_nibble == 0) {
            skipNibbles = 2;
            isLeaf = false;
        } else if (first_nibble == 1) {
            skipNibbles = 1;
            isLeaf = false;
        } else if (first_nibble == 2) {
            skipNibbles = 2;
            isLeaf = true;
        } else if (first_nibble == 3) {
            skipNibbles = 1;
            isLeaf = true;
        } else {
            // Not supposed to happen!
            revert('9');
        }
        return (isLeaf, _decodeNibbles(compact, skipNibbles));
    }

    function _decodeNibbles(bytes memory compact, uint256 skipNibbles)
    private
    pure
    returns(bytes memory nibbles) {
        require(compact.length > 0);
        uint256 length = compact.length * 2;
        require(skipNibbles <= length);
        length -= skipNibbles;
        nibbles = new bytes(length);
        uint256 nibblesLength = 0;
        for (uint256 i = skipNibbles; i < skipNibbles + length; i += 1) {
            if (i % 2 == 0) {
                nibbles[nibblesLength] = bytes1(
                    (uint8(compact[i / 2]) >> 4) & 0xF);
            } else {
                nibbles[nibblesLength] = bytes1(
                    (uint8(compact[i / 2]) >> 0) & 0xF);
            }
            nibblesLength += 1;
        }
        assert(nibblesLength == nibbles.length);
    }

    function _sharedPrefixLength(uint256 xsOffset, bytes memory xs, bytes memory ys) private pure returns(uint256) {
        uint256 i;
        for (i = 0; i + xsOffset < xs.length && i < ys.length; i++) {
            if (xs[i + xsOffset] != ys[i]) {
                return i;
            }
        }
        return i;
    }
}

library StateProofVerifier {
    using RLPReader
    for RLPReader.RLPItem;
    using RLPReader
    for bytes;
    uint256 constant HEADER_STATE_ROOT_INDEX = 3;
    uint256 constant HEADER_NUMBER_INDEX = 8;
    uint256 constant HEADER_TIMESTAMP_INDEX = 11;
    struct Account {
        bool exists;
        uint256 nonce;
        uint256 balance;
        bytes32 storageRoot;
        bytes32 codeHash;
    }
    struct SlotValue {
        bool exists;
        uint256 value;
    }
    /**
     * @notice Verifies Merkle Patricia proof of an account and extracts the account fields.
     *
     * @param _addressHash Keccak256 hash of the address corresponding to the account.
     * @param _stateRootHash MPT root hash of the Ethereum state trie.
     */
    function extractAccountFromProof(bytes32 _addressHash, // keccak256(abi.encodePacked(address))
        bytes32 _stateRootHash, RLPReader.RLPItem[] memory _proof) internal pure returns(Account memory) {
        bytes memory acctRlpBytes = MerklePatriciaProofVerifier.extractProofValue(_stateRootHash, abi.encodePacked(_addressHash), _proof);
        Account memory account;
        if (acctRlpBytes.length == 0) {
            return account;
        }
        RLPReader.RLPItem[] memory acctFields = acctRlpBytes.toRlpItem().toList();
        require(acctFields.length == 4);
        account.exists = true;
        account.nonce = acctFields[0].toUint();
        account.balance = acctFields[1].toUint();
        account.storageRoot = bytes32(acctFields[2].toUint());
        account.codeHash = bytes32(acctFields[3].toUint());
        return account;
    }
    /**
     * @notice Verifies Merkle Patricia proof of a slot and extracts the slot's value.
     *
     * @param _slotHash Keccak256 hash of the slot position.
     * @param _storageRootHash MPT root hash of the account's storage trie.
     */
    function extractSlotValueFromProof(bytes32 _slotHash, bytes32 _storageRootHash, RLPReader.RLPItem[] memory _proof) internal pure returns(SlotValue memory) {
        bytes memory valueRlpBytes = MerklePatriciaProofVerifier.extractProofValue(_storageRootHash, abi.encodePacked(_slotHash), _proof);
        SlotValue memory value;
        if (valueRlpBytes.length != 0) {
            value.exists = true;
            value.value = valueRlpBytes.toRlpItem().toUint();
        }
        return value;
    }
}



interface INameWrapper {
    function ownerOf(uint256 id) external view returns(address);
}
interface ENS {
    function owner(bytes32 node) external view returns(address);
}
interface IERC20 {
    function transfer(address to, uint256 value) external returns(bool);
}

interface Dispute {
    function gameCount() external view returns(uint256);
    function gameAtIndex(uint256 index) external view returns(uint32, uint64, address);  
}
interface Game {
    function rootClaim() external view returns(bytes32);
    function l2BlockNumber() external view returns (uint256);
}

contract Resolver {
    using RLPReader
    for bytes;
    using RLPReader
    for RLPReader.RLPItem;

    string public url;
    address public deployedRegistry;
    mapping(address => bool) public signers;
    event NewSigners(address[] signers);
    error OffchainLookup(address sender, string[] urls, bytes callData, bytes4 callbackFunction, bytes extraData);

    ENS immutable ens = ENS(0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e); //ENS registry address
    INameWrapper immutable nameWrapper = INameWrapper(0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401); //ENS namewrapper address (mainnet)
    constructor(string memory _url, address[] memory _signers, address l2RegistryAddr) {
        deployedRegistry = l2RegistryAddr;
        url = _url;
        for (uint i = 0; i < _signers.length; i++) {
            signers[_signers[i]] = true;
        }
        emit NewSigners(_signers);
    }

    struct l2Oracle {address disputeOracle;}
    mapping(uint256 => l2Oracle) public addrOf;

    modifier onlySigner() {
        require(signers[msg.sender], "Not a signer");
        _;
    }

    function addChain(uint256 chainId, address oracle) external onlySigner {
        addrOf[chainId].disputeOracle = oracle;
    }


    function setURL(string calldata _url) external onlySigner{
        url = _url;
    }

    function setSigners(address[] calldata _signers) external onlySigner{
        for (uint i = 0; i < _signers.length; i++) {
            signers[_signers[i]] = true;
        }
        emit NewSigners(_signers);
    }

    function decodeData(bytes calldata callData) public pure returns(bytes4 functionSelector, bytes calldata callDataWithoutSelector, bytes32 node) {
        
        functionSelector = bytes4(callData[:4]);
        callDataWithoutSelector = callData[4:];

        // Decode node based on function selector
    if (functionSelector == 0x3b3b57de) {  
        node = abi.decode(callDataWithoutSelector, (bytes32));
    } 
    else if (functionSelector == 0xbc1c58d1) { 
        node = abi.decode(callDataWithoutSelector, (bytes32));
    } 
    else if (functionSelector == 0xf1cb7e06) {  
        (node, ) = abi.decode(callDataWithoutSelector, (bytes32, uint256));
    } 
    else if (functionSelector == 0x59d1d43c) {
        (node, ) = abi.decode(callDataWithoutSelector, (bytes32, string));
    } 
    else {
        revert("Record type not supported");
    }
    
    return (functionSelector, callDataWithoutSelector, node);
    }

    function decodeNameAndComputeNamehashes(bytes memory input) public pure returns(bytes32[] memory) {
        uint pos = 0;
        uint labelsCount = 0;
        // First pass to count the number of labels
        while (pos < input.length) {
            uint8 length = uint8(input[pos]);
            if (length == 0) {
                break;
            }
            pos += length + 1;
            labelsCount++;
        }
        // Reset position for the second pass
        pos = 0;
        // We need an array of labelsCount - 1 for the upper-level domains
        string[] memory labels = new string[](labelsCount);
        // Extract all labels
        for (uint labelIndex = 0; pos < input.length; labelIndex++) {
            uint8 length = uint8(input[pos]);
            if (length == 0) {
                break;
            }
            require(length > 0 && length <= 63, "Invalid length");
            bytes memory labelBytes = new bytes(length);
            for (uint i = 0; i < length; i++) {
                labelBytes[i] = input[pos + i + 1];
            }
            labels[labelIndex] = string(labelBytes);
            pos += length + 1;
        }
        // Compute the number of upper-level domains excluding the TLD and the first label
        uint upperDomainCount = labelsCount > 1 ? labelsCount - 2 : 0;
        bytes32[] memory namehashes = new bytes32[](upperDomainCount);
        // Compute namehashes for each upper-level domain, skipping the first label and the TLD
        for (uint i = 1; i < labelsCount - 1; i++) {
            string memory domain = labels[i];
            for (uint j = i + 1; j < labelsCount; j++) {
                domain = string(abi.encodePacked(labels[j], ".", domain));
            }
            namehashes[i - 1] = namehash(domain);
        }
        return namehashes;
    }

    function namehash(string memory name) public pure returns(bytes32) {
        bytes32 node = 0x0000000000000000000000000000000000000000000000000000000000000000;
        if (bytes(name).length > 0) {
            string[] memory labels = splitLabels(name);
            for (uint i = labels.length; i > 0; i--) {
                node = keccak256(abi.encodePacked(node, keccak256(abi.encodePacked(labels[i - 1]))));
            }
        }
        return node;
    }

    function splitLabels(string memory name) public pure returns(string[] memory) {
        uint labelCount = 1;
        for (uint i = 0; i < bytes(name).length; i++) {
            if (bytes(name)[i] == '.') {
                labelCount++;
            }
        }
        string[] memory labels = new string[](labelCount);
        uint labelIndex = 0;
        uint start = 0;
        for (uint i = 0; i <= bytes(name).length; i++) {
            if (i == bytes(name).length || bytes(name)[i] == '.') {
                bytes memory labelBytes = new bytes(i - start);
                for (uint j = 0; j < labelBytes.length; j++) {
                    labelBytes[j] = bytes(name)[start + j];
                }
                labels[labelIndex++] = string(labelBytes);
                start = i + 1;
            }
        }
        return labels;
    }

    function findAuthorizedAddress(bytes memory input) public view returns(address, bytes32) {
        bytes32[] memory namehashes = decodeNameAndComputeNamehashes(input);
        for (uint i = 0; i < namehashes.length; i++) {
            address authorized = ens.owner(namehashes[i]);
            if (authorized != address(0)) {
                return (authorized, namehashes[i]);
            }
        }
        return (address(0), 0x0000000000000000000000000000000000000000000000000000000000000000);
    }

    function getValueFromStateProof(bytes32 stateRoot, address target, bytes32 slotPosition, bytes memory proofsBlob) internal pure returns(bytes32) {
        RLPReader.RLPItem[] memory proofs = proofsBlob.toRlpItem().toList();
        require(proofs.length == 2, "total proofs");
        StateProofVerifier.Account memory account = StateProofVerifier.extractAccountFromProof(keccak256(abi.encodePacked(target)), stateRoot, proofs[0].toList());
        require(account.exists, "Account does not exist or proof is incorrect");
        StateProofVerifier.SlotValue memory storageValue = StateProofVerifier.extractSlotValueFromProof(keccak256(abi.encodePacked(slotPosition)), account.storageRoot, proofs[1].toList());
        require(storageValue.exists, "Storage Value not found");
        return bytes32(storageValue.value);
    }

    

    function getOutputRoot(address oracle) internal view returns(bytes32){
        uint256 index = (Dispute(oracle).gameCount()) - 1;
        uint32 gameType;
        uint64 timestamp;
        address gameAddr;
        (gameType, timestamp, gameAddr) = Dispute(oracle).gameAtIndex(index);
        bytes32 outputRoot = Game(gameAddr).rootClaim();
        return (outputRoot);

    }

    struct ProofData {
        bytes32 stateRoot;
        bytes32 slotPosition;
        bytes proofsBlob;
    }

    function compareOutputRoot(address oracle, bytes32 stateRoot, bytes32 withdrawalStorageRoot, bytes32 latestBlockhash) internal view returns(bool) {
        (bytes32 outputRoot) = getOutputRoot(oracle);
        bytes32 calculatedOutputRoot = keccak256(abi.encode(0, stateRoot, withdrawalStorageRoot, latestBlockhash)); 
        return (outputRoot == calculatedOutputRoot);
    }
    
    function getL2Blocks() internal view returns (uint256, uint256 ){
        address opAddr = addrOf[10].disputeOracle;
        address baseAddr = addrOf[8453].disputeOracle;
        uint256 opIndex = (Dispute(opAddr).gameCount()) - 1;
        uint256 baseIndex = (Dispute(baseAddr).gameCount()) - 1;
        (,,address gameAddrOp) = Dispute(opAddr).gameAtIndex(opIndex);
        (,,address gameAddrBase) = Dispute(baseAddr).gameAtIndex(baseIndex);
        uint256 opBlock = Game(gameAddrOp).l2BlockNumber();
        uint256 baseBlock = Game(gameAddrBase).l2BlockNumber();
        return(opBlock, baseBlock);
        
    }

    function resolve(bytes calldata name, bytes calldata data) external view returns(bytes memory) {
        (bytes4 functionSelector, bytes calldata callDataWithoutSelector, bytes32 node) = decodeData(data);
        address authorized = ens.owner(node);
        if (authorized == address(0)) {
            (authorized, node) = findAuthorizedAddress(name);
        }
        if (authorized == address(nameWrapper)) {
            authorized = nameWrapper.ownerOf(uint256(node));
        }
        (uint256 opBlock, uint256 baseBlock) = getL2Blocks();
        bytes memory callData = abi.encode(functionSelector, callDataWithoutSelector, authorized, opBlock, baseBlock);
        string[] memory urls = new string[](1);
        urls[0] = url;
        revert OffchainLookup(address(this), urls, callData, Resolver.resolveWithProof.selector, callData);
    }

    function resolve(bytes calldata name, bytes calldata data, address context) external view returns(bytes memory) {
        (bytes4 functionSelector, bytes calldata callDataWithoutSelector,) = decodeData(data);
        address authorized = context;
        (uint256 opBlock, uint256 baseBlock) = getL2Blocks();
        bytes memory callData = abi.encode(functionSelector, callDataWithoutSelector, authorized, opBlock, baseBlock);
        string[] memory urls = new string[](1);
        urls[0] = url;
        revert OffchainLookup(address(this), urls, callData, Resolver.resolveWithProof.selector, callData);
    }

    function resolveWithProof(bytes calldata response, bytes calldata extraData) external view returns(bytes memory result) {
        uint256 chainId;
        bytes memory encodedResult;
        ProofData memory proof;
        bytes32 withdrawalStorageRoot;
        bytes32 latestBlockhash;
        bytes4 functionSelector;
        (functionSelector, , ) = abi.decode(extraData, (bytes4, bytes, address));
        (chainId, encodedResult, proof.slotPosition, proof.proofsBlob, proof.stateRoot, withdrawalStorageRoot, latestBlockhash) = abi.decode(response, (uint256, bytes, bytes32, bytes, bytes32, bytes32, bytes32));
        address oracle = addrOf[chainId].disputeOracle;
        require(compareOutputRoot(oracle, proof.stateRoot, withdrawalStorageRoot, latestBlockhash) == true, "Output root comparison failed");
        
        if (functionSelector == 0xf1cb7e06 || functionSelector == 0xbc1c58d1) {
            require(getValueFromStateProof(proof.stateRoot, deployedRegistry, proof.slotPosition, proof.proofsBlob) == keccak256(abi.encodePacked(abi.decode(encodedResult, (bytes)))), "StorageProof Value Mismatch");
            return encodedResult;
        }
        if (functionSelector == 0x3b3b57de) {
            require(getValueFromStateProof(proof.stateRoot, deployedRegistry, proof.slotPosition, proof.proofsBlob) == keccak256(abi.encodePacked(abi.decode(encodedResult, (address)))), "StorageProof Value Mismatch");
            return encodedResult;
        }
        if (functionSelector == 0x59d1d43c) {
            require(getValueFromStateProof(proof.stateRoot, deployedRegistry, proof.slotPosition, proof.proofsBlob) == keccak256(abi.encodePacked(abi.decode(encodedResult, (string)))), "StorageProof Value Mismatch");
            return encodedResult;
        }


    }

    function supportsInterface(bytes4 interfaceID) public pure returns(bool) {
        return interfaceID == 0x9061b923 || interfaceID == 0xa0b7b54e || interfaceID == 0x01ffc9a7;
    }

    function withdrawERC20(address payable _to, address tokenAddr, uint256 amount) external onlySigner {
        IERC20(tokenAddr).transfer(_to, amount);
    }

    function withdraw(address payable _to) external onlySigner {
        _to.transfer(address(this).balance);
    }

    error OperationHandledOnchain(
        uint256 chainId,
        address contractAddress
    );

    error FunctionNotSupported();

    function getOperationHandler(bytes calldata encodedFunction) external view {

        bytes4 selector = bytes4(encodedFunction[:4]);

        if (selector == 0xd5fa2b00 || selector == 0x8b95dd71 || selector == 0x304e6ade || 
            selector == 0x10f13a8c || selector == 0xac9650d8){
        
        revert OperationHandledOnchain(
                10,
                deployedRegistry
            );
        }
        else{
        revert FunctionNotSupported();
        }
    }
}