pragma solidity ^0.4.19;
library SafeMath {
  function mul(uint256 _a, uint256 _b) internal pure returns (uint256 c) {
    if (_a == 0) {
      return 0;
    }
    c = _a * _b;
    assert(c / _a == _b);
    return c;
  }
  function div(uint256 _a, uint256 _b) internal pure returns (uint256) {
    return _a / _b;
  }
  function sub(uint256 _a, uint256 _b) internal pure returns (uint256) {
    assert(_b <= _a);
    return _a - _b;
  }
  function add(uint256 _a, uint256 _b) internal pure returns (uint256 c) {
    c = _a + _b;
    assert(c >= _a);
    return c;
  }
}
contract SyscoinDepositsManager {
    using SafeMath for uint;
    mapping(address => uint) public deposits;
    event DepositMade(address who, uint amount);
    event DepositWithdrawn(address who, uint amount);
    function() public payable {
        makeDeposit();
    }
    function getDeposit(address who) constant public returns (uint) {
        return deposits[who];
    }
    function makeDeposit() public payable returns (uint) {
        increaseDeposit(msg.sender, msg.value);
        return deposits[msg.sender];
    }
    function increaseDeposit(address who, uint amount) internal {
        deposits[who] = deposits[who].add(amount);
        require(deposits[who] <= address(this).balance);
        emit DepositMade(who, amount);
    }
    function withdrawDeposit(uint amount) public returns (uint) {
        require(deposits[msg.sender] >= amount);
        deposits[msg.sender] = deposits[msg.sender].sub(amount);
        msg.sender.transfer(amount);
        emit DepositWithdrawn(msg.sender, amount);
        return deposits[msg.sender];
    }
}
contract SyscoinTransactionProcessor {
    function processTransaction(uint txHash, uint value, address destinationAddress, uint32 _assetGUID, address superblockSubmitterAddress) public returns (uint);
    function burn(uint _value, uint32 _assetGUID, bytes syscoinWitnessProgram) payable public returns (bool success);
}
library SyscoinMessageLibrary {
    uint constant p = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f;  
    uint constant q = (p + 1) / 4;
    uint constant ERR_INVALID_HEADER = 10050;
    uint constant ERR_COINBASE_INDEX = 10060; 
    uint constant ERR_NOT_MERGE_MINED = 10070; 
    uint constant ERR_FOUND_TWICE = 10080; 
    uint constant ERR_NO_MERGE_HEADER = 10090; 
    uint constant ERR_NOT_IN_FIRST_20 = 10100; 
    uint constant ERR_CHAIN_MERKLE = 10110;
    uint constant ERR_PARENT_MERKLE = 10120;
    uint constant ERR_PROOF_OF_WORK = 10130;
    uint constant ERR_INVALID_HEADER_HASH = 10140;
    uint constant ERR_PROOF_OF_WORK_AUXPOW = 10150;
    uint constant ERR_PARSE_TX_OUTPUT_LENGTH = 10160;
    uint constant ERR_PARSE_TX_SYS = 10170;
    enum Network { MAINNET, TESTNET, REGTEST }
    uint32 constant SYSCOIN_TX_VERSION_ASSET_ALLOCATION_BURN = 0x7407;
    uint32 constant SYSCOIN_TX_VERSION_BURN = 0x7401;
    struct AuxPoW {
        uint blockHash;
        uint txHash;
        uint coinbaseMerkleRoot; 
        uint[] chainMerkleProof; 
        uint syscoinHashIndex; 
        uint coinbaseMerkleRootCode; 
        uint parentMerkleRoot; 
        uint[] parentMerkleProof; 
        uint coinbaseTxIndex; 
        uint parentNonce;
    }
    struct BlockHeader {
        uint32 bits;
        uint blockHash;
    }
    function parseVarInt(bytes memory txBytes, uint pos) private pure returns (uint, uint) {
        uint8 ibit = uint8(txBytes[pos]);
        pos += 1;  
        if (ibit < 0xfd) {
            return (ibit, pos);
        } else if (ibit == 0xfd) {
            return (getBytesLE(txBytes, pos, 16), pos + 2);
        } else if (ibit == 0xfe) {
            return (getBytesLE(txBytes, pos, 32), pos + 4);
        } else if (ibit == 0xff) {
            return (getBytesLE(txBytes, pos, 64), pos + 8);
        }
    }
    function getBytesLE(bytes memory data, uint pos, uint bits) internal pure returns (uint) {
        if (bits == 8) {
            return uint8(data[pos]);
        } else if (bits == 16) {
            return uint16(data[pos])
                 + uint16(data[pos + 1]) * 2 ** 8;
        } else if (bits == 32) {
            return uint32(data[pos])
                 + uint32(data[pos + 1]) * 2 ** 8
                 + uint32(data[pos + 2]) * 2 ** 16
                 + uint32(data[pos + 3]) * 2 ** 24;
        } else if (bits == 64) {
            return uint64(data[pos])
                 + uint64(data[pos + 1]) * 2 ** 8
                 + uint64(data[pos + 2]) * 2 ** 16
                 + uint64(data[pos + 3]) * 2 ** 24
                 + uint64(data[pos + 4]) * 2 ** 32
                 + uint64(data[pos + 5]) * 2 ** 40
                 + uint64(data[pos + 6]) * 2 ** 48
                 + uint64(data[pos + 7]) * 2 ** 56;
        }
    }
    function parseTransaction(bytes memory txBytes) internal pure
             returns (uint, uint, address, uint32)
    {
        uint output_value;
        uint32 assetGUID;
        address destinationAddress;
        uint32 version;
        uint pos = 0;
        version = bytesToUint32Flipped(txBytes, pos);
        if(version != SYSCOIN_TX_VERSION_ASSET_ALLOCATION_BURN && version != SYSCOIN_TX_VERSION_BURN){
            return (ERR_PARSE_TX_SYS, output_value, destinationAddress, assetGUID);
        }
        pos = skipInputs(txBytes, 4);
        (output_value, destinationAddress, assetGUID) = scanBurns(txBytes, version, pos);
        return (0, output_value, destinationAddress, assetGUID);
    }
    function skipWitnesses(bytes memory txBytes, uint pos, uint n_inputs) private pure
             returns (uint)
    {
        uint n_stack;
        (n_stack, pos) = parseVarInt(txBytes, pos);
        uint script_len;
        for (uint i = 0; i < n_inputs; i++) {
            for (uint j = 0; j < n_stack; j++) {
                (script_len, pos) = parseVarInt(txBytes, pos);
                pos += script_len;
            }
        }
        return n_stack;
    }    
    function skipInputs(bytes memory txBytes, uint pos) private pure
             returns (uint)
    {
        uint n_inputs;
        uint script_len;
        (n_inputs, pos) = parseVarInt(txBytes, pos);
        if(n_inputs == 0x00){
            (n_inputs, pos) = parseVarInt(txBytes, pos); 
            assert(n_inputs != 0x00);
            (n_inputs, pos) = parseVarInt(txBytes, pos);
        }
        require(n_inputs < 100);
        for (uint i = 0; i < n_inputs; i++) {
            pos += 36;  
            (script_len, pos) = parseVarInt(txBytes, pos);
            pos += script_len + 4;  
        }
        return pos;
    }
    function scanBurns(bytes memory txBytes, uint32 version, uint pos) private pure
             returns (uint, address, uint32)
    {
        uint script_len;
        uint output_value;
        uint32 assetGUID = 0;
        address destinationAddress;
        uint n_outputs;
        (n_outputs, pos) = parseVarInt(txBytes, pos);
        require(n_outputs < 10);
        for (uint i = 0; i < n_outputs; i++) {
            if(version == SYSCOIN_TX_VERSION_BURN){
                output_value = getBytesLE(txBytes, pos, 64);
            }
            pos += 8;
            (script_len, pos) = parseVarInt(txBytes, pos);
            if(!isOpReturn(txBytes, pos)){
                pos += script_len;
                output_value = 0;
                continue;
            }
            pos += 1;
            if(version == SYSCOIN_TX_VERSION_ASSET_ALLOCATION_BURN){
                (output_value, destinationAddress, assetGUID) = scanAssetDetails(txBytes, pos);
            }
            else if(version == SYSCOIN_TX_VERSION_BURN){                
                destinationAddress = scanSyscoinDetails(txBytes, pos);   
            }
            break;
        }
        return (output_value, destinationAddress, assetGUID);
    }
    function skipOutputs(bytes memory txBytes, uint pos) private pure
             returns (uint)
    {
        uint n_outputs;
        uint script_len;
        (n_outputs, pos) = parseVarInt(txBytes, pos);
        require(n_outputs < 10);
        for (uint i = 0; i < n_outputs; i++) {
            pos += 8;
            (script_len, pos) = parseVarInt(txBytes, pos);
            pos += script_len;
        }
        return pos;
    }
    function getSlicePos(bytes memory txBytes, uint pos) private pure
             returns (uint slicePos)
    {
        slicePos = skipInputs(txBytes, pos + 4);
        slicePos = skipOutputs(txBytes, slicePos);
        slicePos += 4; 
    }
    function scanMerkleBranch(bytes memory txBytes, uint pos, uint stop) private pure
             returns (uint[], uint)
    {
        uint n_siblings;
        uint halt;
        (n_siblings, pos) = parseVarInt(txBytes, pos);
        if (stop == 0 || stop > n_siblings) {
            halt = n_siblings;
        } else {
            halt = stop;
        }
        uint[] memory sibling_values = new uint[](halt);
        for (uint i = 0; i < halt; i++) {
            sibling_values[i] = flip32Bytes(sliceBytes32Int(txBytes, pos));
            pos += 32;
        }
        return (sibling_values, pos);
    }   
    function sliceBytes20(bytes memory data, uint start) private pure returns (bytes20) {
        uint160 slice = 0;
        for (uint i = 0; i < 20; i++) {
            slice += uint160(data[i + start]) << (8 * (19 - i));
        }
        return bytes20(slice);
    }
    function sliceBytes32Int(bytes memory data, uint start) private pure returns (uint slice) {
        for (uint i = 0; i < 32; i++) {
            if (i + start < data.length) {
                slice += uint(data[i + start]) << (8 * (31 - i));
            }
        }
    }
    function sliceArray(bytes memory _rawBytes, uint offset, uint _endIndex) internal view returns (bytes) {
        uint len = _endIndex - offset;
        bytes memory result = new bytes(len);
        assembly {
            if iszero(staticcall(gas, 0x04, add(add(_rawBytes, 0x20), offset), len, add(result, 0x20), len)) {
                revert(0, 0)
            }
        }
        return result;
    }
    function isOpReturn(bytes memory txBytes, uint pos) private pure
             returns (bool) {
        return 
            txBytes[pos] == byte(0x6a);
    }
    function scanSyscoinDetails(bytes memory txBytes, uint pos) private pure
             returns (address) {      
        uint8 op;
        (op, pos) = getOpcode(txBytes, pos);
        require(op == 0x14);
        return readEthereumAddress(txBytes, pos);
    }    
    function scanAssetDetails(bytes memory txBytes, uint pos) private pure
             returns (uint, address, uint32) {
        uint32 assetGUID;
        address destinationAddress;
        uint output_value;
        uint8 op;
        (op, pos) = getOpcode(txBytes, pos);
        require(op == 0x04);
        assetGUID = bytesToUint32(txBytes, pos);
        pos += op;
        (op, pos) = getOpcode(txBytes, pos);
        require(op == 0x08);
        output_value = bytesToUint64(txBytes, pos);
        pos += op;
        (op, pos) = getOpcode(txBytes, pos);
        require(op == 0x14);
        destinationAddress = readEthereumAddress(txBytes, pos);       
        return (output_value, destinationAddress, assetGUID);
    }         
    function readEthereumAddress(bytes memory txBytes, uint pos) private pure
             returns (address) {
        uint256 data;
        assembly {
            data := mload(add(add(txBytes, 20), pos))
        }
        return address(uint160(data));
    }
    function getOpcode(bytes memory txBytes, uint pos) private pure
             returns (uint8, uint)
    {
        require(pos < txBytes.length);
        return (uint8(txBytes[pos]), pos + 1);
    }
    function flip32Bytes(uint _input) internal pure returns (uint result) {
        assembly {
            let pos := mload(0x40)
            for { let i := 0 } lt(i, 32) { i := add(i, 1) } {
                mstore8(add(pos, i), byte(sub(31, i), _input))
            }
            result := mload(pos)
        }
    }
    struct UintWrapper {
        uint value;
    }
    function ptr(UintWrapper memory uw) private pure returns (uint addr) {
        assembly {
            addr := uw
        }
    }
    function parseAuxPoW(bytes memory rawBytes, uint pos) internal view
             returns (AuxPoW memory auxpow)
    {
        pos += 80; 
        uint slicePos;
        (slicePos) = getSlicePos(rawBytes, pos);
        auxpow.txHash = dblShaFlipMem(rawBytes, pos, slicePos - pos);
        pos = slicePos;
        pos += 32;
        (auxpow.parentMerkleProof, pos) = scanMerkleBranch(rawBytes, pos, 0);
        auxpow.coinbaseTxIndex = getBytesLE(rawBytes, pos, 32);
        pos += 4;
        (auxpow.chainMerkleProof, pos) = scanMerkleBranch(rawBytes, pos, 0);
        auxpow.syscoinHashIndex = getBytesLE(rawBytes, pos, 32);
        pos += 4;
        auxpow.blockHash = dblShaFlipMem(rawBytes, pos, 80);
        pos += 36; 
        auxpow.parentMerkleRoot = sliceBytes32Int(rawBytes, pos);
        pos += 40; 
        auxpow.parentNonce = getBytesLE(rawBytes, pos, 32);
        uint coinbaseMerkleRootPosition;
        (auxpow.coinbaseMerkleRoot, coinbaseMerkleRootPosition, auxpow.coinbaseMerkleRootCode) = findCoinbaseMerkleRoot(rawBytes);
    }
    function findCoinbaseMerkleRoot(bytes memory rawBytes) private pure
             returns (uint, uint, uint)
    {
        uint position;
        bool found = false;
        for (uint i = 0; i < rawBytes.length; ++i) {
            if (rawBytes[i] == 0xfa && rawBytes[i+1] == 0xbe && rawBytes[i+2] == 0x6d && rawBytes[i+3] == 0x6d) {
                if (found) { 
                    return (0, position - 4, ERR_FOUND_TWICE);
                } else {
                    found = true;
                    position = i + 4;
                }
            }
        }
        if (!found) { 
            return (0, position - 4, ERR_NO_MERGE_HEADER);
        } else {
            return (sliceBytes32Int(rawBytes, position), position - 4, 1);
        }
    }
    function makeMerkle(bytes32[] hashes2) external pure returns (bytes32) {
        bytes32[] memory hashes = hashes2;
        uint length = hashes.length;
        if (length == 1) return hashes[0];
        require(length > 0);
        uint i;
        uint j;
        uint k;
        k = 0;
        while (length > 1) {
            k = 0;
            for (i = 0; i < length; i += 2) {
                j = i+1<length ? i+1 : length-1;
                hashes[k] = bytes32(concatHash(uint(hashes[i]), uint(hashes[j])));
                k += 1;
            }
            length = k;
        }
        return hashes[0];
    }
    function computeMerkle(uint _txHash, uint _txIndex, uint[] memory _siblings) internal pure returns (uint) {
        uint resultHash = _txHash;
        uint i = 0;
        while (i < _siblings.length) {
            uint proofHex = _siblings[i];
            uint sideOfSiblings = _txIndex % 2;  
            uint left;
            uint right;
            if (sideOfSiblings == 1) {
                left = proofHex;
                right = resultHash;
            } else if (sideOfSiblings == 0) {
                left = resultHash;
                right = proofHex;
            }
            resultHash = concatHash(left, right);
            _txIndex /= 2;
            i += 1;
        }
        return resultHash;
    }
    function computeParentMerkle(AuxPoW memory _ap) internal pure returns (uint) {
        return flip32Bytes(computeMerkle(_ap.txHash,
                                         _ap.coinbaseTxIndex,
                                         _ap.parentMerkleProof));
    }
    function computeChainMerkle(uint _blockHash, AuxPoW memory _ap) internal pure returns (uint) {
        return computeMerkle(_blockHash,
                             _ap.syscoinHashIndex,
                             _ap.chainMerkleProof);
    }
    function concatHash(uint _tx1, uint _tx2) internal pure returns (uint) {
        return flip32Bytes(uint(sha256(abi.encodePacked(sha256(abi.encodePacked(flip32Bytes(_tx1), flip32Bytes(_tx2)))))));
    }
    function checkAuxPoW(uint _blockHash, AuxPoW memory _ap) internal pure returns (uint) {
        if (_ap.coinbaseTxIndex != 0) {
            return ERR_COINBASE_INDEX;
        }
        if (_ap.coinbaseMerkleRootCode != 1) {
            return _ap.coinbaseMerkleRootCode;
        }
        if (computeChainMerkle(_blockHash, _ap) != _ap.coinbaseMerkleRoot) {
            return ERR_CHAIN_MERKLE;
        }
        if (computeParentMerkle(_ap) != _ap.parentMerkleRoot) {
            return ERR_PARENT_MERKLE;
        }
        return 1;
    }
    function sha256mem(bytes memory _rawBytes, uint offset, uint len) internal view returns (bytes32 result) {
        assembly {
            let ptr := mload(0x40)
            if iszero(staticcall(gas, 0x02, add(add(_rawBytes, 0x20), offset), len, ptr, 0x20)) {
                revert(0, 0)
            }
            result := mload(ptr)
        }
    }
    function dblShaFlip(bytes _dataBytes) internal pure returns (uint) {
        return flip32Bytes(uint(sha256(abi.encodePacked(sha256(abi.encodePacked(_dataBytes))))));
    }
    function dblShaFlipMem(bytes memory _rawBytes, uint offset, uint len) internal view returns (uint) {
        return flip32Bytes(uint(sha256(abi.encodePacked(sha256mem(_rawBytes, offset, len)))));
    }
    function readBytes32(bytes memory data, uint offset) internal pure returns (bytes32) {
        bytes32 result;
        assembly {
            result := mload(add(add(data, 0x20), offset))
        }
        return result;
    }
    function readUint32(bytes memory data, uint offset) internal pure returns (uint32) {
        uint32 result;
        assembly {
            result := mload(add(add(data, 0x20), offset))
        }
        return result;
    }
    function targetFromBits(uint32 _bits) internal pure returns (uint) {
        uint exp = _bits / 0x1000000;  
        uint mant = _bits & 0xffffff;
        return mant * 256**(exp - 3);
    }
    uint constant SYSCOIN_DIFFICULTY_ONE = 0xFFFFF * 256**(0x1e - 3);
    function targetToDiff(uint target) internal pure returns (uint) {
        return SYSCOIN_DIFFICULTY_ONE / target;
    }