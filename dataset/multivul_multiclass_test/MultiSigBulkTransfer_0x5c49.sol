// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * Minimal interface for an ERC20 token.
 */
interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract MultiSigBulkTransfer {
    /// ---------------------
    /// Roles & Signer State
    /// ---------------------
    address public admin;          // Default admin, set at deployment
    address[3] public signers;     // The three designated signers
    mapping(address => bool) public hasSigned; // Track signer's approval
    uint256 public signCount;      // Number of signers who signed

    /// Events
    event BulkTransferSigned(address indexed signer, uint256 currentSignCount);
    event BulkTransferExecuted(address indexed executor);
    event AdminTransferExecuted(address indexed adminExecutor, address token, address to, uint256 amount);
    event SignerUpdated(uint256 indexed index, address indexed oldSigner, address indexed newSigner);

    constructor(address[3] memory _signers) {
        admin = msg.sender;
        signers = _signers;
    }

    /**
     * @notice Allows one of the three signers to sign off on a bulk transfer.
     */
    function signBulkTransfer() external {
        require(isSigner(msg.sender), "Not a designated signer");
        require(!hasSigned[msg.sender], "Already signed");

        hasSigned[msg.sender] = true;
        signCount += 1;

        emit BulkTransferSigned(msg.sender, signCount);
    }

    /**
     * @notice Execute the bulk transfer once all three signers have signed.
     * @param tokens     The array of ERC20 token addresses to transfer.
     * @param recipients The array of recipient addresses.
     * @param amounts    The array of amounts for each recipient.
     */
    function executeBulkTransfer(
        address[] calldata tokens,
        address[] calldata recipients,
        uint256[] calldata amounts
    ) external {
        require(signCount == 3, "All three signers must approve");
        require(
            tokens.length == recipients.length && recipients.length == amounts.length,
            "Array length mismatch"
        );

        // Reset signers for the next operation
        _resetSignatures();

        // Perform each token transfer
        for (uint256 i = 0; i < tokens.length; i++) {
            IERC20 token = IERC20(tokens[i]);
            require(token.transfer(recipients[i], amounts[i]), "Transfer failed");
        }

        emit BulkTransferExecuted(msg.sender);
    }

    /**
     * @notice Admin override to execute any transfer from this contract.
     * @dev    Bypasses the multi-sig requirement.
     * @param token  Address of the token to transfer.
     * @param to     Recipient address.
     * @param amount Amount to transfer.
     */
    function adminExecuteBulkTransfer(
        address token,
        address to,
        uint256 amount
    ) external {
        require(msg.sender == admin, "Caller is not admin");
        require(IERC20(token).transfer(to, amount), "Transfer failed");
        emit AdminTransferExecuted(msg.sender, token, to, amount);
    }

    /**
     * @notice Admin can update any of the three designated signers.
     * @param index The index of the signer to be replaced (0, 1, or 2).
     * @param newSigner The new signer's address.
     */
    function updateSigner(uint256 index, address newSigner) external {
        require(msg.sender == admin, "Caller is not admin");
        require(index < 3, "Invalid index");

        address oldSigner = signers[index];
        signers[index] = newSigner;

        // Reset signatures after changing signers to avoid partial approvals.
        _resetSignatures();

        emit SignerUpdated(index, oldSigner, newSigner);
    }

    /**
     * @notice Clears all signers' approvals for a fresh start.
     */
    function _resetSignatures() internal {
        for (uint256 i = 0; i < signers.length; i++) {
            hasSigned[signers[i]] = false;
        }
        signCount = 0;
    }

    /**
     * @notice Checks if an address is one of the three signers.
     */
    function isSigner(address account) public view returns (bool) {
        for (uint256 i = 0; i < signers.length; i++) {
            if (signers[i] == account) {
                return true;
            }
        }
        return false;
    }
}