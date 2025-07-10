// SPDX-License-Identifier: MIT

// File: Context.sol


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
// File: Ownable.sol


// OpenZeppelin Contracts (last updated v5.0.0) (access/Ownable.sol)

pragma solidity ^0.8.20;


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
// File: Ownable2Step.sol


// OpenZeppelin Contracts (last updated v5.0.0) (access/Ownable2Step.sol)

pragma solidity ^0.8.20;


/**
 * @dev Contract module which provides access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * The initial owner is specified at deployment time in the constructor for `Ownable`. This
 * can later be changed with {transferOwnership} and {acceptOwnership}.
 *
 * This module is used through inheritance. It will make available all functions
 * from parent (Ownable).
 */
abstract contract Ownable2Step is Ownable {
    address private _pendingOwner;

    event OwnershipTransferStarted(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Returns the address of the pending owner.
     */
    function pendingOwner() public view virtual returns (address) {
        return _pendingOwner;
    }

    /**
     * @dev Starts the ownership transfer of the contract to a new account. Replaces the pending transfer if there is one.
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual override onlyOwner {
        _pendingOwner = newOwner;
        emit OwnershipTransferStarted(owner(), newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`) and deletes any pending owner.
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual override {
        delete _pendingOwner;
        super._transferOwnership(newOwner);
    }

    /**
     * @dev The new owner accepts the ownership transfer.
     */
    function acceptOwnership() public virtual {
        address sender = _msgSender();
        if (pendingOwner() != sender) {
            revert OwnableUnauthorizedAccount(sender);
        }
        _transferOwnership(sender);
    }
}
// File: Util.sol

// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//


pragma solidity ^0.8.9;

/// @notice reverse the byte order of the uint256 value.
/// @dev Solidity uses a big-endian ABI encoding. Reversing the byte order before encoding
/// ensure that the encoded value will be little-endian.
/// Written by k06a. https://ethereum.stackexchange.com/a/83627
function reverseByteOrderUint256(uint256 input) pure returns (uint256 v) {
    v = input;

    // swap bytes
    v = ((v & 0xFF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00) >> 8)
        | ((v & 0x00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF00FF) << 8);

    // swap 2-byte long pairs
    v = ((v & 0xFFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000) >> 16)
        | ((v & 0x0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF0000FFFF) << 16);

    // swap 4-byte long pairs
    v = ((v & 0xFFFFFFFF00000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF00000000) >> 32)
        | ((v & 0x00000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF00000000FFFFFFFF) << 32);

    // swap 8-byte long pairs
    v = ((v & 0xFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0000000000000000) >> 64)
        | ((v & 0x0000000000000000FFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF) << 64);

    // swap 16-byte long pairs
    v = (v >> 128) | (v << 128);
}

/// @notice reverse the byte order of the uint32 value.
/// @dev Solidity uses a big-endian ABI encoding. Reversing the byte order before encoding
/// ensure that the encoded value will be little-endian.
/// Written by k06a. https://ethereum.stackexchange.com/a/83627
function reverseByteOrderUint32(uint32 input) pure returns (uint32 v) {
    v = input;

    // swap bytes
    v = ((v & 0xFF00FF00) >> 8) | ((v & 0x00FF00FF) << 8);

    // swap 2-byte long pairs
    v = (v >> 16) | (v << 16);
}

/// @notice reverse the byte order of the uint16 value.
/// @dev Solidity uses a big-endian ABI encoding. Reversing the byte order before encoding
/// ensure that the encoded value will be little-endian.
/// Written by k06a. https://ethereum.stackexchange.com/a/83627
function reverseByteOrderUint16(uint16 input) pure returns (uint16 v) {
    v = input;

    // swap bytes
    v = (v >> 8) | ((v & 0x00FF) << 8);
}
// File: IRiscZeroVerifier.sol

// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//


pragma solidity ^0.8.9;


/// @notice A receipt attesting to the execution of a guest program.
/// @dev A receipt contains two parts: a seal and a claim. The seal is a zero-knowledge proof
/// attesting to knowledge of a zkVM execution resulting in the claim. The claim is a set of public
/// outputs for the execution. Crucially, the claim includes the journal and the image ID. The
/// image ID identifies the program that was executed, and the journal is the public data written
/// by the program. Note that this struct only contains the claim digest, as can be obtained with
/// the `digest()` function on `ReceiptClaimLib`.
struct Receipt {
    bytes seal;
    bytes32 claimDigest;
}

/// @notice Public claims about a zkVM guest execution, such as the journal committed to by the guest.
/// @dev Also includes important information such as the exit code and the starting and ending system
/// state (i.e. the state of memory). `ReceiptClaim` is a "Merkle-ized struct" supporting
/// partial openings of the underlying fields from a hash commitment to the full structure.
struct ReceiptClaim {
    /// @notice Digest of the SystemState just before execution has begun.
    bytes32 preStateDigest;
    /// @notice Digest of the SystemState just after execution has completed.
    bytes32 postStateDigest;
    /// @notice The exit code for the execution.
    ExitCode exitCode;
    /// @notice A digest of the input to the guest.
    /// @dev This field is currently unused and must be set to the zero digest.
    bytes32 input;
    /// @notice Digest of the Output of the guest, including the journal
    /// and assumptions set during execution.
    bytes32 output;
}

library ReceiptClaimLib {
    using OutputLib for Output;
    using SystemStateLib for SystemState;

    bytes32 constant TAG_DIGEST = sha256("risc0.ReceiptClaim");

    // Define a constant to ensure hashing is done at compile time. Can't use the
    // SystemStateLib.digest method here because the Solidity compiler complains.
    bytes32 constant SYSTEM_STATE_ZERO_DIGEST = 0xa3acc27117418996340b84e5a90f3ef4c49d22c79e44aad822ec9c313e1eb8e2;

    /// @notice Construct a ReceiptClaim from the given imageId and journalDigest.
    ///         Returned ReceiptClaim will represent a successful execution of the zkVM, running
    ///         the program committed by imageId and resulting in the journal specified by
    ///         journalDigest.
    /// @param imageId The identifier for the guest program.
    /// @param journalDigest The SHA-256 digest of the journal bytes.
    /// @dev Input hash and postStateDigest are set to all-zeros (i.e. no committed input, or
    ///      final memory state), the exit code is (Halted, 0), and there are no assumptions
    ///      (i.e. the receipt is unconditional).
    function ok(bytes32 imageId, bytes32 journalDigest) internal pure returns (ReceiptClaim memory) {
        return ReceiptClaim(
            imageId,
            SYSTEM_STATE_ZERO_DIGEST,
            ExitCode(SystemExitCode.Halted, 0),
            bytes32(0),
            Output(journalDigest, bytes32(0)).digest()
        );
    }

    function digest(ReceiptClaim memory claim) internal pure returns (bytes32) {
        return sha256(
            abi.encodePacked(
                TAG_DIGEST,
                // down
                claim.input,
                claim.preStateDigest,
                claim.postStateDigest,
                claim.output,
                // data
                uint32(claim.exitCode.system) << 24,
                uint32(claim.exitCode.user) << 24,
                // down.length
                uint16(4) << 8
            )
        );
    }
}

/// @notice Commitment to the memory state and program counter (pc) of the zkVM.
/// @dev The "pre" and "post" fields of the ReceiptClaim are digests of the system state at the
///      start are stop of execution. Programs are loaded into the zkVM by creating a memory image
///      of the loaded program, and creating a system state for initializing the zkVM. This is
///      known as the "image ID".
struct SystemState {
    /// @notice Program counter.
    uint32 pc;
    /// @notice Root hash of a merkle tree which confirms the integrity of the memory image.
    bytes32 merkle_root;
}

library SystemStateLib {
    bytes32 constant TAG_DIGEST = sha256("risc0.SystemState");

    function digest(SystemState memory state) internal pure returns (bytes32) {
        return sha256(
            abi.encodePacked(
                TAG_DIGEST,
                // down
                state.merkle_root,
                // data
                reverseByteOrderUint32(state.pc),
                // down.length
                uint16(1) << 8
            )
        );
    }
}

/// @notice Exit condition indicated by the zkVM at the end of the guest execution.
/// @dev Exit codes have a "system" part and a "user" part. Semantically, the system part is set to
/// indicate the type of exit (e.g. halt, pause, or system split) and is directly controlled by the
/// zkVM. The user part is an exit code, similar to exit codes used in Linux, chosen by the guest
/// program to indicate additional information (e.g. 0 to indicate success or 1 to indicate an
/// error).
struct ExitCode {
    SystemExitCode system;
    uint8 user;
}

/// @notice Exit condition indicated by the zkVM at the end of the execution covered by this proof.
/// @dev
/// `Halted` indicates normal termination of a program with an interior exit code returned from the
/// guest program. A halted program cannot be resumed.
///
/// `Paused` indicates the execution ended in a paused state with an interior exit code set by the
/// guest program. A paused program can be resumed such that execution picks up where it left
/// of, with the same memory state.
///
/// `SystemSplit` indicates the execution ended on a host-initiated system split. System split is
/// mechanism by which the host can temporarily stop execution of the execution ended in a system
/// split has no output and no conclusions can be drawn about whether the program will eventually
/// halt. System split is used in continuations to split execution into individually provable segments.
enum SystemExitCode {
    Halted,
    Paused,
    SystemSplit
}

/// @notice Output field in the `ReceiptClaim`, committing to a claimed journal and assumptions list.
struct Output {
    /// @notice Digest of the journal committed to by the guest execution.
    bytes32 journalDigest;
    /// @notice Digest of the ordered list of `ReceiptClaim` digests corresponding to the
    /// calls to `env::verify` and `env::verify_integrity`.
    /// @dev Verifying the integrity of a `Receipt` corresponding to a `ReceiptClaim` with a
    /// non-empty assumptions list does not guarantee unconditionally any of the claims over the
    /// guest execution (i.e. if the assumptions list is non-empty, then the journal digest cannot
    /// be trusted to correspond to a genuine execution). The claims can be checked by additional
    /// verifying a `Receipt` for every digest in the assumptions list.
    bytes32 assumptionsDigest;
}

library OutputLib {
    bytes32 constant TAG_DIGEST = sha256("risc0.Output");

    function digest(Output memory output) internal pure returns (bytes32) {
        return sha256(
            abi.encodePacked(
                TAG_DIGEST,
                // down
                output.journalDigest,
                output.assumptionsDigest,
                // down.length
                uint16(2) << 8
            )
        );
    }
}

/// @notice Error raised when cryptographic verification of the zero-knowledge proof fails.
error VerificationFailed();

/// @notice Verifier interface for RISC Zero receipts of execution.
interface IRiscZeroVerifier {
    /// @notice Verify that the given seal is a valid RISC Zero proof of execution with the
    ///     given image ID and journal digest. Reverts on failure.
    /// @dev This method additionally ensures that the input hash is all-zeros (i.e. no
    /// committed input), the exit code is (Halted, 0), and there are no assumptions (i.e. the
    /// receipt is unconditional).
    /// @param seal The encoded cryptographic proof (i.e. SNARK).
    /// @param imageId The identifier for the guest program.
    /// @param journalDigest The SHA-256 digest of the journal bytes.
    function verify(bytes calldata seal, bytes32 imageId, bytes32 journalDigest) external view;

    /// @notice Verify that the given receipt is a valid RISC Zero receipt, ensuring the `seal` is
    /// valid a cryptographic proof of the execution with the given `claim`. Reverts on failure.
    /// @param receipt The receipt to be verified.
    function verifyIntegrity(Receipt calldata receipt) external view;
}
// File: RiscZeroVerifierRouter.sol

// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//


pragma solidity ^0.8.9;



/// @notice Router for IRiscZeroVerifier, allowing multiple implementations to be accessible behind a single address.
contract RiscZeroVerifierRouter is IRiscZeroVerifier, Ownable2Step {
    /// @notice Mapping from 4-byte verifier selector to verifier contracts.
    ///         Used to route receipts to verifiers that are able to check the receipt.
    mapping(bytes4 => IRiscZeroVerifier) public verifiers;

    /// @notice Value of an entry that has never been set.
    IRiscZeroVerifier internal constant UNSET = IRiscZeroVerifier(address(0));
    /// @notice A "tombstone" value used to mark verifier entries that have been removed from the mapping.
    IRiscZeroVerifier internal constant TOMBSTONE = IRiscZeroVerifier(address(1));

    /// @notice Error raised when attempting to verify a receipt with a selector that is not
    ///         registered on this router. Generally, this indicates a version mismatch where the
    ///         prover generated a receipt with version of the zkVM that does not match any
    ///         registered version on this router contract.
    error SelectorUnknown(bytes4 selector);
    /// @notice Error raised when attempting to add a verifier for a selector that is already registered.
    error SelectorInUse(bytes4 selector);
    /// @notice Error raised when attempting to verify a receipt with a selector that has been
    ///         removed, or attempting to add a new verifier with a selector that was previously
    ///         registered and then removed.
    error SelectorRemoved(bytes4 selector);

    constructor(address admin) Ownable(admin) {}

    /// @notice Adds a verifier to the router, such that it can receive receipt verification calls.
    function addVerifier(bytes4 selector, IRiscZeroVerifier verifier) external onlyOwner {
        if (verifiers[selector] == TOMBSTONE) {
            revert SelectorRemoved({selector: selector});
        }
        if (verifiers[selector] != UNSET) {
            revert SelectorInUse({selector: selector});
        }
        verifiers[selector] = verifier;
    }

    /// @notice Removes verifier from the router, such that it can not receive verification calls.
    ///         Removing a selector sets it to the tombstone value. It can never be set to any
    ///         other value, and can never be reused for a new verifier, in order to enforce the
    ///         property that each selector maps to at most one implementation across time.
    function removeVerifier(bytes4 selector) external onlyOwner {
        // Simple check to reduce the chance of accidents.
        // NOTE: If there ever _is_ a reason to remove a selector that has never been set, the owner
        // can call addVerifier with the tombstone address.
        if (verifiers[selector] == UNSET) {
            revert SelectorUnknown({selector: selector});
        }
        verifiers[selector] = TOMBSTONE;
    }

    /// @notice Get the associatied verifier, reverting if the selector is unknown or removed.
    function getVerifier(bytes4 selector) public view returns (IRiscZeroVerifier) {
        IRiscZeroVerifier verifier = verifiers[selector];
        if (verifier == UNSET) {
            revert SelectorUnknown({selector: selector});
        }
        if (verifier == TOMBSTONE) {
            revert SelectorRemoved({selector: selector});
        }
        return verifier;
    }

    /// @notice Get the associatied verifier, reverting if the selector is unknown or removed.
    function getVerifier(bytes calldata seal) public view returns (IRiscZeroVerifier) {
        // Use the first 4 bytes of the seal at the selector to look up in the mapping.
        return getVerifier(bytes4(seal[0:4]));
    }

    /// @inheritdoc IRiscZeroVerifier
    function verify(bytes calldata seal, bytes32 imageId, bytes32 journalDigest) external view {
        getVerifier(seal).verify(seal, imageId, journalDigest);
    }
    
    function _verify(bytes calldata seal, bytes32 imageId, bytes32 journalDigest) internal view {
        getVerifier(seal).verify(seal, imageId, journalDigest);
    }
    
    function verifyWithJournal(bytes calldata seal, bytes32 imageId, bytes calldata journal) external view {
        getVerifier(seal).verify(seal, imageId, sha256(journal));
    }

    /// @inheritdoc IRiscZeroVerifier
    function verifyIntegrity(Receipt calldata receipt) external view {
        getVerifier(receipt.seal).verifyIntegrity(receipt);
    }
}
// File: QTOVToken.sol


pragma solidity ^0.8.20;

contract QTOVToken {
    string public constant name = "Quantova";
    string public constant symbol = "QTOV";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;

    address public owner;
    address public bridge;
    
    bool public bridgeInitialized; // Track if bridge has been set
    
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event BridgeUpdated(address indexed newBridge);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        owner = msg.sender;
        bridgeInitialized = false; // Bridge not set at deployment
        emit OwnershipTransferred(address(0), owner);
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier onlyBridge() {
        require(msg.sender == bridge, "Only bridge");
        _;
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function setBridge(address _bridge) public onlyOwner {
        require(_bridge != address(0), "Zero address");
        bridge = _bridge;
        bridgeInitialized = true; // Mark bridge as initialized
        emit BridgeUpdated(_bridge);
    }

    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }

    function allowance(address _owner, address spender) public view returns (uint256) {
        return _allowances[_owner][spender];
    }

    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public returns (bool) {
        _spendAllowance(from, msg.sender, amount);
        _transfer(from, to, amount);
        return true;
    }

    function mint(address to, uint256 amount) public onlyBridge {
        require(bridgeInitialized, "Bridge not initialized");
        _mint(to, amount);
    }

    function burn(address from, uint256 amount) external {
        require(msg.sender == bridge, "Unauthorized");
        _burn(from, amount);
    }

    function burnFrom(address account, uint256 amount) public {
        _spendAllowance(account, msg.sender, amount);
        _burn(account, amount);
    }

    function _transfer(address from, address to, uint256 amount) internal {
        require(from != address(0), "From zero address");
        require(to != address(0), "To zero address");

        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "Insufficient balance");
        unchecked {
            _balances[from] = fromBalance - amount;
        }
        _balances[to] += amount;

        emit Transfer(from, to, amount);
    }

    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "Mint to zero address");
        totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }

    function _burn(address account, uint256 amount) internal {
        require(account != address(0), "Burn from zero address");
        uint256 accountBalance = _balances[account];
        require(accountBalance >= amount, "Burn amount exceeds balance");
        unchecked {
            _balances[account] = accountBalance - amount;
        }
        totalSupply -= amount;
        emit Transfer(account, address(0), amount);
    }

    function _approve(address _owner, address spender, uint256 amount) internal {
        require(_owner != address(0), "Approve from zero address");
        require(spender != address(0), "Approve to zero address");
        _allowances[_owner][spender] = amount;
        emit Approval(_owner, spender, amount);
    }

    function _spendAllowance(address _owner, address spender, uint256 amount) internal {
        uint256 currentAllowance = allowance(_owner, spender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "Insufficient allowance");
            unchecked {
                _approve(_owner, spender, currentAllowance - amount);
            }
        }
    }
}
// File: Bridge.sol


pragma solidity ^0.8.9;



contract Bridge {
    // Constant verifier image ID
    bytes32 public immutable IMAGE_ID;
    
    RiscZeroVerifierRouter public verifier;
    QTOVToken public token;
    address public owner;
    
    // State variables
    bytes32 public latestFinalizedHash;
    mapping(uint256 => bool) public processedNonces;

    // Sequential nonce tracking
    uint64 public inwardNonce;  // Last processed incoming nonce (Gradium → BSC)
    uint64 public outwardNonce; // Next outgoing nonce (BSC → Gradium)

    // Separate storage for each direction
    struct OutwardRemittance {
        address beneficiary;
        uint256 amount;
        uint64 nonce;
    }

    struct InwardRemittance {
        address beneficiary;
        uint256 amount;
        uint64 nonce;
    }

    // Storage - Single bytes value per transfer
    mapping(uint64 => bytes) public outwardTransfersEncoded;

    // Mapped storage for Merkle proofs
    mapping(uint64 => OutwardRemittance) public outwardTransfers;
    mapping(uint64 => InwardRemittance) public inwardTransfers;


    // Nonce status tracking
    struct NonceStatus {
        uint256 nonce;
        bool isClaimed;
        uint256 timestamp;
        address beneficiary;
        uint256 amount;
        bool isInbound;
    }
    NonceStatus[] public nonceStatuses;

    // Transfer data structure
    struct Transfer {
        uint256 nonce;
        address beneficiary;
        uint256 amount;
    }

    // Events (unchanged)
    event VerificationSuccess(bytes32 indexed imageId, bytes32 journalHash);
    event VerificationFailed(bytes32 indexed imageId, string reason);
    event TransfersProcessed(bytes32 indexed lastFinalizedHash, bytes32 indexed latestFinalizedHash, uint256 count);
    event VerifierChanged(address oldVerifier, address newVerifier);
    event NonceRecorded(uint256 indexed nonce, address beneficiary, uint256 amount, uint256 timestamp);
    event NonceClaimed(uint256 indexed nonce, address claimer, uint256 amount, uint256 timestamp);
    event TokenContractChanged(address oldToken, address newToken);
    event OutboundTransferInitiated(uint64 indexed nonce, address sender, address beneficiary, uint256 amount);
    event InwardNonceAdvanced(uint64 newNonce);
    event OutwardNonceIncremented(uint64 newNonce);

    constructor(address verifierAddress, address tokenAddress, bytes32 _imageId) {
        require(verifierAddress != address(0), "Invalid verifier address");
        require(tokenAddress != address(0), "Invalid token address");
        verifier = RiscZeroVerifierRouter(verifierAddress);
        token = QTOVToken(tokenAddress);
        owner = msg.sender;
        IMAGE_ID = _imageId;
        
        // Initialize nonces
        inwardNonce = 0;
        outwardNonce = 0;
    }

    function transferOut(address beneficiary, uint256 amount) external {
        require(amount > 0, "Amount must be positive");
        require(beneficiary != address(0), "Invalid beneficiary");
        
        token.burn(msg.sender, amount);
        
        outwardNonce++;
        uint64 currentNonce = outwardNonce;
        outwardTransfers[currentNonce] = OutwardRemittance(beneficiary, amount, currentNonce);

        // ABI-encode all fields into single bytes
        outwardTransfersEncoded[currentNonce] = abi.encode(
            beneficiary,
            amount,
            currentNonce
        );

        nonceStatuses.push(NonceStatus({
            nonce: currentNonce,
            isClaimed: false,
            timestamp: block.timestamp,
            beneficiary: beneficiary,
            amount: amount,
            isInbound: false
        }));
        
        emit OutboundTransferInitiated(currentNonce, msg.sender, beneficiary, amount);
        emit OutwardNonceIncremented(outwardNonce);
    }

    function bridgeWithVerification(bytes calldata seal, bytes calldata journal) external {
        try verifier.verifyWithJournal(seal, IMAGE_ID, journal) {
            _processVerifiedTransaction(journal);
            emit VerificationSuccess(IMAGE_ID, sha256(journal));
        } catch Error(string memory reason) {
            emit VerificationFailed(IMAGE_ID, reason);
        } catch (bytes memory) {
            emit VerificationFailed(IMAGE_ID, "Low-level verification error");
        }
    }

    function bridgeWithVerificationDebug(bytes calldata seal, bytes calldata journal, bytes calldata postStateJournal) external {
        try verifier.verifyWithJournal(seal, IMAGE_ID, journal) {
            _processVerifiedTransaction(postStateJournal);
            emit VerificationSuccess(IMAGE_ID, sha256(journal));
        } catch Error(string memory reason) {
            emit VerificationFailed(IMAGE_ID, reason);
        } catch (bytes memory) {
            emit VerificationFailed(IMAGE_ID, "Low-level verification error");
        }
    }

    function _processVerifiedTransaction(bytes memory journal) internal {
        (bytes memory lastFinalizedHashBytes, Transfer[] memory transfers, bytes memory latestFinalizedHashBytes) = 
            abi.decode(journal, (bytes, Transfer[], bytes));
        
        require(lastFinalizedHashBytes.length == 32, "Invalid lastFinalizedHash length");
        require(latestFinalizedHashBytes.length == 32, "Invalid latestFinalizedHash length");
        
        bytes32 lastFinalizedHash = abi.decode(lastFinalizedHashBytes, (bytes32));
        bytes32 newLatestFinalizedHash = abi.decode(latestFinalizedHashBytes, (bytes32));

        require(newLatestFinalizedHash != latestFinalizedHash, "This batch was already processed");
        
        if (latestFinalizedHash != bytes32(0)) {
            require(
                lastFinalizedHash == latestFinalizedHash,
                "Invalid hash sequence: lastFinalizedHash doesn't match stored latestFinalizedHash"
            );
        }

        if (transfers.length > 0) {
            require(uint64(transfers[0].nonce) == inwardNonce + 1, "Nonce sequence violation");

            for (uint256 i = 0; i < transfers.length; i++) {
                Transfer memory transfer = transfers[i];
                
                if (i > 0) {
                    require(transfer.nonce == transfers[i-1].nonce + 1, "Nonce sequence violation");
                }

                require(!processedNonces[transfer.nonce], "Transfer nonce already processed");
                processedNonces[transfer.nonce] = true;
                
                inwardTransfers[uint64(transfer.nonce)] = InwardRemittance(
                    transfer.beneficiary,
                    transfer.amount,
                    uint64(transfer.nonce)
                );

                nonceStatuses.push(NonceStatus({
                    nonce: transfer.nonce,
                    isClaimed: false,
                    timestamp: block.timestamp,
                    beneficiary: transfer.beneficiary,
                    amount: transfer.amount,
                    isInbound: true
                }));
                
                emit NonceRecorded(transfer.nonce, transfer.beneficiary, transfer.amount, block.timestamp);
            }

            inwardNonce = uint64(transfers[transfers.length - 1].nonce);
            emit InwardNonceAdvanced(inwardNonce);
        }

        latestFinalizedHash = newLatestFinalizedHash;
        emit TransfersProcessed(lastFinalizedHash, latestFinalizedHash, transfers.length);
    }

    function claim(uint256 nonce) external {
        require(processedNonces[nonce], "Nonce does not exist");
        
        bool found;
        uint256 index;
        address beneficiary;
        uint256 amount;
        
        for (uint256 i = 0; i < nonceStatuses.length; i++) {
            if (nonceStatuses[i].nonce == nonce) {
                require(!nonceStatuses[i].isClaimed, "Nonce already claimed");
                
                nonceStatuses[i].isClaimed = true;
                found = true;
                index = i;
                beneficiary = nonceStatuses[i].beneficiary;
                amount = nonceStatuses[i].amount;
                break;
            }
        }
        require(found, "Nonce not found in records");
        
        token.mint(beneficiary, amount);
        emit NonceClaimed(nonce, beneficiary, amount, block.timestamp);
    }

    function setVerifier(address newVerifier) external {
        require(msg.sender == owner, "Only owner");
        require(newVerifier != address(0), "Invalid verifier address");
        emit VerifierChanged(address(verifier), newVerifier);
        verifier = RiscZeroVerifierRouter(newVerifier);
    }

    function setTokenContract(address newToken) external {
        require(msg.sender == owner, "Only owner");
        require(newToken != address(0), "Invalid token address");
        emit TokenContractChanged(address(token), newToken);
        token = QTOVToken(newToken);
    }


    /**
    * @notice Returns the total count of processed transfers (both directions)
    * @dev Calculated using nonce counters since we're using mapped storage now
    * @return Total count of inward and outward transfers
    */
    function getTransferCount() external view returns (uint256) {
        // Outward: outwardNonce starts at 1 and increments after each transfer
        // Inward: inwardNonce starts at 0 and updates to last processed nonce
        return (outwardNonce - 1) + inwardNonce;
    }

    /**
    * @notice Returns the count of processed inbound transfers (Gradium → BSC)
    * @dev Uses inwardNonce which tracks the last processed nonce
    * @return Count of verified inbound transfers
    */
    function getInwardTransferCount() external view returns (uint256) {
        // inwardNonce is 0 initially, equals last processed nonce
        return inwardNonce;
    }

    /**
    * @notice Returns the count of processed outbound transfers (BSC → Gradium)
    * @dev Uses outwardNonce which is pre-incremented (starts at 1)
    * @return Count of verified outbound transfers
    */
    function getOutwardTransferCount() external view returns (uint256) {
        // outwardNonce starts at 1, is incremented after each transfer
        return outwardNonce - 1;
    }

    function getNonceStatusCount() external view returns (uint256) {
        return nonceStatuses.length;
    }

    function getNonceStatus(uint256 index) external view returns (
        uint256 nonce, bool isClaimed, uint256 timestamp, address beneficiary, uint256 amount
    ) {
        require(index < nonceStatuses.length, "Index out of bounds");
        NonceStatus memory status = nonceStatuses[index];
        return (status.nonce, status.isClaimed, status.timestamp, status.beneficiary, status.amount);
    }

    function getNonceStatusByNonce(uint256 nonce) external view returns (
        bool isClaimed, uint256 timestamp, address beneficiary, uint256 amount, bool exists
    ) {
        if (!processedNonces[nonce]) return (false, 0, address(0), 0, false);
        
        for (uint256 i = 0; i < nonceStatuses.length; i++) {
            if (nonceStatuses[i].nonce == nonce) {
                return (
                    nonceStatuses[i].isClaimed,
                    nonceStatuses[i].timestamp,
                    nonceStatuses[i].beneficiary,
                    nonceStatuses[i].amount,
                    true
                );
            }
        }
        return (false, 0, address(0), 0, false);
    }
    
    function getInwardNonce() external view returns (uint64) {
        return inwardNonce;
    }

    function getOutwardNonce() external view returns (uint64) {
        return outwardNonce;
    }

    /**
    * @notice Gets an inbound transfer by its sequential index (0-based)
    * @dev Uses nonce sequencing to emulate array-like access
    * @param index The sequential index (0 to getInwardTransferCount()-1)
    * @return Transfer details (nonce, beneficiary, amount)
    */
    function getInwardTransfer(uint256 index) external view returns (Transfer memory) {
        require(index < inwardNonce, "Invalid index"); // inwardNonce = last processed nonce = count
        uint64 nonce = uint64(index + 1); // inwardNonce starts counting from 1
        InwardRemittance memory remittance = inwardTransfers[nonce];
        require(remittance.nonce == nonce, "Transfer not found");
        return Transfer(nonce, remittance.beneficiary, remittance.amount);
    }

    /**
    * @notice Gets an outbound transfer by its sequential index (0-based)
    * @dev Uses nonce sequencing to emulate array-like access
    * @param index The sequential index (0 to getOutwardTransferCount()-1)
    * @return Transfer details (nonce, beneficiary, amount)
    */
    function getOutwardTransfer(uint256 index) external view returns (Transfer memory) {
        require(index < outwardNonce - 1, "Invalid index"); // outwardNonce-1 = count
        uint64 nonce = uint64(index + 1); // outwardNonce starts at 1
        OutwardRemittance memory remittance = outwardTransfers[nonce];
        require(remittance.nonce == nonce, "Transfer not found");
        return Transfer(nonce, remittance.beneficiary, remittance.amount);
    }
}