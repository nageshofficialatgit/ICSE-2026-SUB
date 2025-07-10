// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

contract SignatureVerifier {
    function verifySignature(
        string memory message,
        bytes memory signature,
        address signer
    ) public pure returns (bool) {
        bytes32 messageHash = getEthSignedMessageHash(message);
        address recoveredSigner = recoverSigner(messageHash, signature);
        return recoveredSigner == signer;
    }

    function getEthSignedMessageHash(string memory message) private pure returns (bytes32) {
        bytes memory messageBytes = bytes(message);
        string memory prefix = string(abi.encodePacked("\x19Ethereum Signed Message:\n", uintToString(messageBytes.length)));
        return keccak256(abi.encodePacked(prefix, message));
    }

    function recoverSigner(bytes32 messageHash, bytes memory signature) private pure returns (address) {
        (uint8 v, bytes32 r, bytes32 s) = splitSignature(signature);
        return ecrecover(messageHash, v, r, s);
    }

    function splitSignature(bytes memory sig) private pure returns (uint8 v, bytes32 r, bytes32 s) {
        require(sig.length == 65, "Invalid signature length");
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
        if (v < 27) {
            v += 27;
        }
        return (v, r, s);
    }

    function uintToString(uint v) private pure returns (string memory) {
        if (v == 0) return "0";
        uint maxlength = 100;
        bytes memory reversed = new bytes(maxlength);
        uint i = 0;
        while (v != 0) {
            uint remainder = v % 10;
            v = v / 10;
            reversed[i++] = bytes1(uint8(48 + remainder));
        }
        bytes memory s = new bytes(i);
        for (uint j = 0; j < i; j++) {
            s[j] = reversed[i - 1 - j];
        }
        return string(s);
    }
}