// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
}

/**
 * @title AGIAlphaAgentSecuredCipher
 * @notice A contract that stores a 'secure cipher' (XOR-obscured data)
 *         and requires a minimum ERC20 token balance to retrieve it.
 * @dev Designed to avoid drawing attention to any specific link or purpose.
 */
contract AGIAlphaAgentSecuredCipher {
    address public owner;
    address public requiredToken;
    uint256 public requiredAmount;

    // This is the XOR-based cipher stored as hex.
    // Observers won't know its content without the XOR key or logic.
    string private secureCipher;

    event SecureCipherUpdated(string newCipher);
    event TokenRequirementUpdated(address newToken, uint256 newAmount);

    /**
     * @dev Constructor only sets basic token requirements; no cipher stored initially.
     * @param _initialToken The ERC20 token address whose balance is required
     * @param _initialAmount The minimum amount of tokens required
     */
    constructor(address _initialToken, uint256 _initialAmount) {
        owner = msg.sender;
        requiredToken = _initialToken;
        requiredAmount = _initialAmount;
        // We do NOT set a cipher here to avoid revealing anything at deployment time.
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }

    /**
     * @notice Encode plain data with a simple XOR approach and return the hex output.
     * @dev This function does NOT store anything on-chain; it is 'pure'.
     *      The owner can use it offline (via a call in Remix) to produce the cipher string,
     *      then copy the result to 'updateSecureCipher' in a separate transaction.
     * @param plaintext The string to obscure
     * @param key The XOR key (string)
     * @return hexCipher The XOR result in hex
     */
    function encodeCipher(string memory plaintext, string memory key)
        public
        pure
        returns (string memory hexCipher)
    {
        bytes memory p = bytes(plaintext);
        bytes memory k = bytes(key);
        require(k.length > 0, "Key cannot be empty");

        // We'll XOR each byte of 'plaintext' with a byte from 'key' (cycling if needed).
        bytes memory r = new bytes(p.length);
        for (uint256 i = 0; i < p.length; i++) {
            r[i] = bytes1(uint8(p[i]) ^ uint8(k[i % k.length]));
        }

        // Convert to hex
        hexCipher = _toHex(r);
    }

    /**
     * @notice Allows the owner to update the secure cipher (obscured data).
     * @param newCipher The XOR-encoded data in hex format
     */
    function updateSecureCipher(string memory newCipher) external onlyOwner {
        secureCipher = newCipher;
        emit SecureCipherUpdated(newCipher);
    }

    /**
     * @notice Updates the token requirement for accessing the secure cipher.
     * @param newToken The new ERC20 token address
     * @param newAmount The new minimum required token balance
     */
    function updateTokenRequirement(address newToken, uint256 newAmount) external onlyOwner {
        requiredToken = newToken;
        requiredAmount = newAmount;
        emit TokenRequirementUpdated(newToken, newAmount);
    }

    /**
     * @notice Retrieves the stored cipher if the caller holds enough tokens.
     * @return The cipher as a hex string (XOR-encoded data).
     */
    function getSecureCipher() external view returns (string memory) {
        require(
            IERC20(requiredToken).balanceOf(msg.sender) >= requiredAmount,
            "Insufficient token balance"
        );
        return secureCipher;
    }

    // -----------------------------------------------------------
    // PRIVATE / INTERNAL UTILS
    // -----------------------------------------------------------
    
    /**
     * @dev Convert bytes to a hex string (0-9a-f).
     */
    function _toHex(bytes memory data) private pure returns (string memory) {
        bytes memory hexChars = "0123456789abcdef";
        bytes memory hexString = new bytes(data.length * 2);
        for (uint256 i = 0; i < data.length; i++) {
            hexString[i * 2] = hexChars[uint8(data[i] >> 4)];
            hexString[i * 2 + 1] = hexChars[uint8(data[i] & 0x0f)];
        }
        return string(hexString);
    }
}