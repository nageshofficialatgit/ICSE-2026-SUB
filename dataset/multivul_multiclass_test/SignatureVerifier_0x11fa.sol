// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

contract SignatureVerifier {
    function verifySignature(
        string memory message,
        bytes memory signature,
        address signer
    ) public pure returns (bool) {
        bytes32 messageHash = keccak256(abi.encodePacked(message));
        
        bytes32 ethSignedMessageHash = getEthSignedMessageHash(messageHash);
        
        address recoveredSigner = recoverSigner(ethSignedMessageHash, signature);
        
        return recoveredSigner == signer;
    }
    

    // Fonction pour obtenir le hash du message signé Ethereum

    function getEthSignedMessageHash(bytes32 messageHash) private pure returns (bytes32) {
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", messageHash));
    }
    

    // Fonction pour récupérer l'adresse du signataire à partir de la signature

    function recoverSigner(bytes32 ethSignedMessageHash, bytes memory signature) private pure returns (address) {
        (uint8 v, bytes32 r, bytes32 s) = splitSignature(signature);
        return ecrecover(ethSignedMessageHash, v, r, s);
    }
    
    // Fonction pour séparer la signature en ses composants v, r, s

    function splitSignature(bytes memory sig) private pure returns (uint8 v, bytes32 r, bytes32 s) {
        require(sig.length == 65, "Invalid signature length");
        
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
        
        // Ajuster v si nécessaire (selon la convention Ethereum)
        if (v < 27) {
            v += 27;
        }
        
        return (v, r, s);
    }
}