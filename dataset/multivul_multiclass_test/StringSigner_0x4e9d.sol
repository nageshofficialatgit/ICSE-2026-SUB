// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

contract StringSigner {
    address public signer;
    address public owner;

    event StringSigned(
        address indexed signer,
        string message,
        bytes signature
    );

    event SignerUpdated(address indexed oldSigner, address indexed newSigner);

    constructor() {
        signer = msg.sender; // Le déployeur est le signataire initial
        owner = msg.sender;  // Le déployeur est le propriétaire
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Seul le proprietaire peut executer cette fonction");
        _;
    }


    // SIGN

    function signString(string memory _message, bytes memory _signature) public returns (bool) {
        require(msg.sender == signer, "Seul le signataire autorise peut signer");

        bytes32 messageHash = getMessageHash(_message);
        bytes32 ethSignedMessageHash = getEthSignedMessageHash(messageHash);

        address recoveredSigner = recoverSigner(ethSignedMessageHash, _signature);
        require(recoveredSigner == signer, "Signature invalide");

        emit StringSigned(signer, _message, _signature);
        return true;
    }


    // VERIFY

    function verifySignature(
        string memory _message,
        bytes memory _signature
    ) public view returns (bool) {
        bytes32 messageHash = getMessageHash(_message);
        bytes32 ethSignedMessageHash = getEthSignedMessageHash(messageHash);
        address recoveredSigner = recoverSigner(ethSignedMessageHash, _signature);
        return recoveredSigner == signer;
    }


    // UD SIGNER

    function updateSigner(address _newSigner) public onlyOwner {
        require(_newSigner != address(0), "Adresse invalide");
        require(_newSigner != signer, "Nouveau signataire identique a l actuel");
        
        address oldSigner = signer;
        signer = _newSigner;
        
        emit SignerUpdated(oldSigner, _newSigner);
    }


    // Fonction helper pour calculer le hash du message

    function getMessageHash(string memory _message) private pure returns (bytes32) {
        return keccak256(abi.encodePacked(_message));
    }

    // Fonction helper pour ajouter le préfixe Ethereum Signed Message

    function getEthSignedMessageHash(bytes32 _messageHash) private pure returns (bytes32) {
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", _messageHash));
    }

    // Fonction helper pour récupérer le signataire à partir de la signature

    function recoverSigner(bytes32 _ethSignedMessageHash, bytes memory _signature) private pure returns (address) {
        (uint8 v, bytes32 r, bytes32 s) = splitSignature(_signature);
        return ecrecover(_ethSignedMessageHash, v, r, s);
    }

    // Fonction helper pour décomposer la signature en r, s, v

    function splitSignature(bytes memory sig) private pure returns (uint8 v, bytes32 r, bytes32 s) {
        require(sig.length == 65, "Signature invalide");

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
}