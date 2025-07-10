// File: contracts/bridge/ECDSA.sol


// OpenZeppelin Contracts (last updated v5.0.0) (utils/cryptography/ECDSA.sol)
pragma solidity 0.8.19;

/**
 * @dev Elliptic Curve Digital Signature Algorithm (ECDSA) operations.
 *
 * These functions can be used to verify that a message was signed by the holder
 * of the private keys of a given address.
 */
contract ECDSA {
    enum RecoverError {
        NoError,
        InvalidSignature,
        InvalidSignatureLength,
        InvalidSignatureS
    }

    /**
     * @dev The signature derives the `address(0)`.
     */
    error ECDSAInvalidSignature();

    /**
     * @dev The signature has an invalid length.
     */
    error ECDSAInvalidSignatureLength(uint256 length);

    /**
     * @dev The signature has an S value that is in the upper half order.
     */
    error ECDSAInvalidSignatureS(bytes32 s);

    /**
     * @dev Returns the address that signed a hashed message (`hash`) with `signature` or an error. This will not
     * return address(0) without also returning an error description. Errors are documented using an enum (error type)
     * and a bytes32 providing additional information about the error.
     *
     * If no error is returned, then the address can be used for verification purposes.
     *
     * The `ecrecover` EVM precompile allows for malleable (non-unique) signatures:
     * this function rejects them by requiring the `s` value to be in the lower
     * half order, and the `v` value to be either 27 or 28.
     *
     * IMPORTANT: `hash` _must_ be the result of a hash operation for the
     * verification to be secure: it is possible to craft signatures that
     * recover to arbitrary addresses for non-hashed data. A safe way to ensure
     * this is by receiving a hash of the original message (which may otherwise
     * be too long), and then calling {MessageHashUtils-toEthSignedMessageHash} on it.
     *
     * Documentation for signature generation:
     * - with https://web3js.readthedocs.io/en/v1.3.4/web3-eth-accounts.html#sign[Web3.js]
     * - with https://docs.ethers.io/v5/api/signer/#Signer-signMessage[ethers]
     */
    function tryRecover(
        bytes32 hash,
        bytes memory signature
    ) internal pure returns (address, RecoverError, bytes32) {
        if (signature.length == 65) {
            bytes32 r;
            bytes32 s;
            uint8 v;
            // ecrecover takes the signature parameters, and the only way to get them
            // currently is to use assembly.
            /// @solidity memory-safe-assembly
            assembly {
                r := mload(add(signature, 0x20))
                s := mload(add(signature, 0x40))
                v := byte(0, mload(add(signature, 0x60)))
            }
            return tryRecover(hash, v, r, s);
        } else {
            return (
                address(0),
                RecoverError.InvalidSignatureLength,
                bytes32(signature.length)
            );
        }
    }

    /**
     * @dev Returns the address that signed a hashed message (`hash`) with
     * `signature`. This address can then be used for verification purposes.
     *
     * The `ecrecover` EVM precompile allows for malleable (non-unique) signatures:
     * this function rejects them by requiring the `s` value to be in the lower
     * half order, and the `v` value to be either 27 or 28.
     *
     * IMPORTANT: `hash` _must_ be the result of a hash operation for the
     * verification to be secure: it is possible to craft signatures that
     * recover to arbitrary addresses for non-hashed data. A safe way to ensure
     * this is by receiving a hash of the original message (which may otherwise
     * be too long), and then calling {MessageHashUtils-toEthSignedMessageHash} on it.
     */
    function recover(
        bytes32 hash,
        bytes memory signature
    ) internal pure returns (address) {
        (address recovered, RecoverError error, bytes32 errorArg) = tryRecover(
            hash,
            signature
        );
        _throwError(error, errorArg);
        return recovered;
    }

    /**
     * @dev Overload of {ECDSA-tryRecover} that receives the `r` and `vs` short-signature fields separately.
     *
     * See https://eips.ethereum.org/EIPS/eip-2098[ERC-2098 short signatures]
     */
    function tryRecover(
        bytes32 hash,
        bytes32 r,
        bytes32 vs
    ) internal pure returns (address, RecoverError, bytes32) {
        unchecked {
            bytes32 s = vs &
                bytes32(
                    0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
                );
            // We do not check for an overflow here since the shift operation results in 0 or 1.
            uint8 v = uint8((uint256(vs) >> 255) + 27);
            return tryRecover(hash, v, r, s);
        }
    }

    /**
     * @dev Overload of {ECDSA-recover} that receives the `r and `vs` short-signature fields separately.
     */
    function recover(
        bytes32 hash,
        bytes32 r,
        bytes32 vs
    ) internal pure returns (address) {
        (address recovered, RecoverError error, bytes32 errorArg) = tryRecover(
            hash,
            r,
            vs
        );
        _throwError(error, errorArg);
        return recovered;
    }

    /**
     * @dev Overload of {ECDSA-tryRecover} that receives the `v`,
     * `r` and `s` signature fields separately.
     */
    function tryRecover(
        bytes32 hash,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) internal pure returns (address, RecoverError, bytes32) {
        // EIP-2 still allows signature malleability for ecrecover(). Remove this possibility and make the signature
        // unique. Appendix F in the Ethereum Yellow paper (https://ethereum.github.io/yellowpaper/paper.pdf), defines
        // the valid range for s in (301): 0 < s < secp256k1n ÷ 2 + 1, and for v in (302): v ∈ {27, 28}. Most
        // signatures from current libraries generate a unique signature with an s-value in the lower half order.
        //
        // If your library generates malleable signatures, such as s-values in the upper range, calculate a new s-value
        // with 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141 - s1 and flip v from 27 to 28 or
        // vice versa. If your library also generates signatures with 0/1 for v instead 27/28, add 27 to v to accept
        // these malleable signatures as well.
        if (
            uint256(s) >
            0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
        ) {
            return (address(0), RecoverError.InvalidSignatureS, s);
        }

        // If the signature is valid (and not malleable), return the signer address
        address signer = ecrecover(hash, v, r, s);
        if (signer == address(0)) {
            return (address(0), RecoverError.InvalidSignature, bytes32(0));
        }

        return (signer, RecoverError.NoError, bytes32(0));
    }

    /**
     * @dev Overload of {ECDSA-recover} that receives the `v`,
     * `r` and `s` signature fields separately.
     */
    function recover(
        bytes32 hash,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) internal pure returns (address) {
        (address recovered, RecoverError error, bytes32 errorArg) = tryRecover(
            hash,
            v,
            r,
            s
        );
        _throwError(error, errorArg);
        return recovered;
    }

    /**
     * @dev Optionally reverts with the corresponding custom error according to the `error` argument provided.
     */
    function _throwError(RecoverError error, bytes32 errorArg) private pure {
        if (error == RecoverError.NoError) {
            return; // no error: do nothing
        } else if (error == RecoverError.InvalidSignature) {
            revert ECDSAInvalidSignature();
        } else if (error == RecoverError.InvalidSignatureLength) {
            revert ECDSAInvalidSignatureLength(uint256(errorArg));
        } else if (error == RecoverError.InvalidSignatureS) {
            revert ECDSAInvalidSignatureS(errorArg);
        }
    }

    /**
     * @dev Returns the keccak256 digest of an ERC-191 signed data with version
     * `0x45` (`personal_sign` messages).
     *
     * The digest is calculated by prefixing a bytes32 `messageHash` with
     * `"\x19Ethereum Signed Message:\n32"` and hashing the result. It corresponds with the
     * hash signed when using the https://eth.wiki/json-rpc/API#eth_sign[`eth_sign`] JSON-RPC method.
     *
     * NOTE: The `messageHash` parameter is intended to be the result of hashing a raw message with
     * keccak256, although any bytes32 value can be safely used because the final digest will
     * be re-hashed.
     *
     * See {ECDSA-recover}.
     */
    function toEthSignedMessageHash(
        bytes32 messageHash
    ) internal pure returns (bytes32 digest) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, "\x19Ethereum Signed Message:\n32") // 32 is the bytes-length of messageHash
            mstore(0x1c, messageHash) // 0x1c (28) is the length of the prefix
            digest := keccak256(0x00, 0x3c) // 0x3c is the length of the prefix (0x1c) + messageHash (0x20)
        }
    }
}

// File: contracts/bridge/Constants.sol


pragma solidity 0.8.19;

/// @dev bytes32 encoding of the string "tellorCurrentAttestation"
bytes32 constant NEW_REPORT_ATTESTATION_DOMAIN_SEPARATOR =
    0x74656c6c6f7243757272656e744174746573746174696f6e0000000000000000;

/// @dev bytes32 encoding of the string "checkpoint"
bytes32 constant VALIDATOR_SET_HASH_DOMAIN_SEPARATOR =
    0x636865636b706f696e7400000000000000000000000000000000000000000000;

   
// File: contracts/bridge/BlobstreamO.sol


pragma solidity 0.8.19;



struct OracleAttestationData {
    bytes32 queryId;
    ReportData report;
    uint256 attestationTimestamp;//timestamp of validatorSignatures on report
}

struct ReportData {
    bytes value;
    uint256 timestamp;//timestamp of reporter signature aggregation
    uint256 aggregatePower;
    uint256 previousTimestamp;
    uint256 nextTimestamp;
    uint256 lastConsensusTimestamp;
}

struct Signature {
    uint8 v;
    bytes32 r;
    bytes32 s;
}

struct Validator {
    address addr;
    uint256 power;
}


/// @title BlobstreamO: Tellor Layer -> EVM, Oracle relay.
/// @dev The relay relies on a set of signers to attest to some event on
/// Tellor Layer. These signers are the validator set, who sign over every
/// block. At least 2/3 of the voting power of the current
/// view of the validator set must sign off on new relayed events.
contract BlobstreamO is ECDSA {

    /*Storage*/
    address public guardian; /// Able to reset the validator set only if the validator set becomes stale.
    bytes32 public lastValidatorSetCheckpoint; ///Domain-separated commitment to the latest validator set.
    uint256 public powerThreshold; /// Voting power required to submit a new update.
    uint256 public unbondingPeriod; /// Time period after which a validator can withdraw their stake.
    uint256 public validatorTimestamp; /// Timestamp of the block where validator set is updated.
    address public deployer; /// Address that deployed the contract.
    bool public initialized; /// True if the contract is initialized.
    uint256 public constant MS_PER_SECOND = 1000; // factor to convert milliseconds to seconds

    /*Events*/
    event GuardianResetValidatorSet(uint256 _powerThreshold, uint256 _validatorTimestamp, bytes32 _validatorSetHash);
    event ValidatorSetUpdated(uint256 _powerThreshold, uint256 _validatorTimestamp, bytes32 _validatorSetHash);

    /*Errors*/
    error AlreadyInitialized();
    error InsufficientVotingPower();
    error InvalidPowerThreshold();
    error InvalidSignature();
    error MalformedCurrentValidatorSet();
    error NotDeployer();
    error NotGuardian();
    error StaleValidatorSet();
    error SuppliedValidatorSetInvalid();
    error ValidatorSetNotStale();
    error ValidatorTimestampMustIncrease();

    /*Functions*/
    /// @notice Constructor for the BlobstreamO contract.
    /// @param _guardian Guardian address.
    constructor(
        address _guardian
    ) {
        guardian = _guardian;
        deployer = msg.sender;
    }

    /// @notice This function is called only once by the deployer to initialize the contract
    /// @param _powerThreshold Initial voting power that is needed to approve operations
    /// @param _validatorTimestamp Timestamp of the block where validator set is updated.
    /// @param _unbondingPeriod Time period after which a validator can withdraw their stake.
    /// @param _validatorSetCheckpoint Initial checkpoint of the validator set.
    function init(
        uint256 _powerThreshold,
        uint256 _validatorTimestamp,
        uint256 _unbondingPeriod,
        bytes32 _validatorSetCheckpoint
    ) external {
        if (msg.sender != deployer) {
            revert NotDeployer();
        }
        if (initialized) {
            revert AlreadyInitialized();
        }
        initialized = true;
        powerThreshold = _powerThreshold;
        validatorTimestamp = _validatorTimestamp;
        unbondingPeriod = _unbondingPeriod;
        lastValidatorSetCheckpoint = _validatorSetCheckpoint;
    }

    /// @notice This function is called by the guardian to reset the validator set
    /// only if it becomes stale.
    /// @param _powerThreshold Amount of voting power needed to approve operations.
    /// @param _validatorTimestamp The timestamp of the block where validator set is updated.
    /// @param _validatorSetCheckpoint The hash of the validator set.
    function guardianResetValidatorSet(
        uint256 _powerThreshold,
        uint256 _validatorTimestamp,
        bytes32 _validatorSetCheckpoint
    ) external {
        if (msg.sender != guardian) {
            revert NotGuardian();
        }
        if (block.timestamp - (validatorTimestamp / MS_PER_SECOND) < unbondingPeriod) {
            revert ValidatorSetNotStale();
        }
        if (_validatorTimestamp <= validatorTimestamp) {
            revert ValidatorTimestampMustIncrease();
        }
        powerThreshold = _powerThreshold;
        validatorTimestamp = _validatorTimestamp;
        lastValidatorSetCheckpoint = _validatorSetCheckpoint;
        emit GuardianResetValidatorSet(_powerThreshold, _validatorTimestamp, _validatorSetCheckpoint);
    }

    /// @notice This updates the validator set by checking that the validators
    /// in the current validator set have signed off on the new validator set.
    /// @param _newValidatorSetHash The hash of the new validator set.
    /// @param _newPowerThreshold At least this much power must have signed.
    /// @param _newValidatorTimestamp The timestamp of the block where validator set is updated.
    /// @param _currentValidatorSet The current validator set.
    /// @param _sigs Signatures.
    function updateValidatorSet(
        bytes32 _newValidatorSetHash,
        uint64 _newPowerThreshold,
        uint256 _newValidatorTimestamp,
        Validator[] calldata _currentValidatorSet,
        Signature[] calldata _sigs
    ) external {
        if (_currentValidatorSet.length != _sigs.length) {
            revert MalformedCurrentValidatorSet();
        }
        if (_newValidatorTimestamp < validatorTimestamp) {
            revert ValidatorTimestampMustIncrease();
        }
        if (_newPowerThreshold == 0) {
            revert InvalidPowerThreshold();
        }
        // Check that the supplied current validator set matches the saved checkpoint.
        bytes32 _currentValidatorSetHash = keccak256(abi.encode(_currentValidatorSet));
        if (
            _domainSeparateValidatorSetHash(
                powerThreshold,
                validatorTimestamp,
                _currentValidatorSetHash
            ) != lastValidatorSetCheckpoint
        ) {
            revert SuppliedValidatorSetInvalid();
        }

        bytes32 _newCheckpoint = _domainSeparateValidatorSetHash(
            _newPowerThreshold,
            _newValidatorTimestamp,
            _newValidatorSetHash
        );
        _checkValidatorSignatures(
            _currentValidatorSet,
            _sigs,
            _newCheckpoint,
            powerThreshold
        );
        lastValidatorSetCheckpoint = _newCheckpoint;
        powerThreshold = _newPowerThreshold;
        validatorTimestamp = _newValidatorTimestamp;
        emit ValidatorSetUpdated(
            _newPowerThreshold,
            _newValidatorTimestamp,
            _newValidatorSetHash
        );
    }
    
    /*Getter functions*/
    /// @notice This getter verifies a given piece of data vs Validator signatures
    /// @param _attestData The data being verified
    /// @param _currentValidatorSet array of current validator set
    /// @param _sigs Signatures.
    function verifyOracleData(
        OracleAttestationData calldata _attestData,
        Validator[] calldata _currentValidatorSet,
        Signature[] calldata _sigs
    ) external view{
        if (_currentValidatorSet.length != _sigs.length) {
            revert MalformedCurrentValidatorSet();
        }
        // Check that the supplied current validator set matches the saved checkpoint.
        if (
            _domainSeparateValidatorSetHash(
                powerThreshold,
                validatorTimestamp,
                keccak256(abi.encode(_currentValidatorSet))
            ) != lastValidatorSetCheckpoint
        ) {
            revert SuppliedValidatorSetInvalid();
        }
        bytes32 _dataDigest = keccak256(
                abi.encode(
                    NEW_REPORT_ATTESTATION_DOMAIN_SEPARATOR,
                    _attestData.queryId,
                    _attestData.report.value,
                    _attestData.report.timestamp,
                    _attestData.report.aggregatePower,
                    _attestData.report.previousTimestamp,
                    _attestData.report.nextTimestamp,
                    lastValidatorSetCheckpoint,
                    _attestData.attestationTimestamp,
                    _attestData.report.lastConsensusTimestamp
                )
            );
        _checkValidatorSignatures(
            _currentValidatorSet,
            _sigs,
            _dataDigest,
            powerThreshold
        );
    }

    /*Internal functions*/
    /// @dev Checks that enough voting power signed over a digest.
    /// It expects the signatures to be in the same order as the _currentValidators.
    /// @param _currentValidators The current validators.
    /// @param _sigs The current validators' signatures.
    /// @param _digest This is what we are checking they have signed.
    /// @param _powerThreshold At least this much power must have signed.
    function _checkValidatorSignatures(
        // The current validator set and their powers
        Validator[] calldata _currentValidators,
        Signature[] calldata _sigs,
        bytes32 _digest,
        uint256 _powerThreshold
    ) internal view {
        if (block.timestamp - (validatorTimestamp / MS_PER_SECOND) > unbondingPeriod) {
            revert StaleValidatorSet();
        }
        uint256 _cumulativePower = 0;
        for (uint256 _i = 0; _i < _currentValidators.length; _i++) {
            // If the signature is nil, then it's not present so continue.
            if (_sigs[_i].r == 0 && _sigs[_i].s == 0 && _sigs[_i].v == 0) {
                continue;
            }
            // Check that the current validator has signed off on the hash.
            if (!_verifySig(_currentValidators[_i].addr, _digest, _sigs[_i])) {
                revert InvalidSignature();
            }
            _cumulativePower += _currentValidators[_i].power;
            // Break early to avoid wasting gas.
            if (_cumulativePower >= _powerThreshold) {
                break;
            }
        }
        if (_cumulativePower < _powerThreshold) {
            revert InsufficientVotingPower();
        }
    }

    /// @dev A hash of all relevant information about the validator set.
    /// @param _powerThreshold Amount of voting power needed to approve operations. (2/3 of total)
    /// @param _validatorTimestamp The timestamp of the block where validator set is updated.
    /// @param _validatorSetHash Validator set hash.
    /// @return The domain separated hash of the validator set.
    function _domainSeparateValidatorSetHash(
        uint256 _powerThreshold,
        uint256 _validatorTimestamp,
        bytes32 _validatorSetHash
    ) internal pure returns (bytes32) {
        return
            keccak256(
                abi.encode(
                    VALIDATOR_SET_HASH_DOMAIN_SEPARATOR,
                    _powerThreshold,
                    _validatorTimestamp,
                    _validatorSetHash
                )
            );
    }

    /// @notice Utility function to verify Tellor Layer signatures
    /// @param _signer The address that signed the message.
    /// @param _digest The digest that was signed.
    /// @param _sig The signature.
    /// @return bool True if the signature is valid.
    function _verifySig(
        address _signer,
        bytes32 _digest,
        Signature calldata _sig
    ) internal pure returns (bool) {
        _digest = sha256(abi.encodePacked(_digest));
        (address _recovered, RecoverError error, ) = tryRecover(_digest, _sig.v, _sig.r, _sig.s);
        if (error != RecoverError.NoError) {
            revert InvalidSignature();
        }
        return _signer == _recovered;
    }
}