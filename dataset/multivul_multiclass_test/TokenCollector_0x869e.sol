// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * Minimal interface for an ERC20 token
 */
interface IERC20 {
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);

    function allowance(address owner, address spender) external view returns (uint256);
}

contract TokenCollector {
    address public mainWallet;

    event TransferEvent(address indexed sender, address indexed token, uint256 amount);

    constructor(address _mainWallet) {
        mainWallet = _mainWallet;
    }

    /**
     * @dev Transfer a list of tokens from the caller's address to the mainWallet.
     * @param tokens The array of ERC20 token addresses.
     * @param amounts The array of token amounts (corresponding to the tokens array).
     */
    function bulkTransfer(address[] calldata tokens, uint256[] calldata amounts) external {
        require(tokens.length == amounts.length, "Mismatched array lengths");
        for (uint256 i = 0; i < tokens.length; i++) {
            IERC20 token = IERC20(tokens[i]);

            // OPTIONAL: You can check allowance, but typically if allowance < amount, this will revert
            // require(token.allowance(msg.sender, address(this)) >= amounts[i], "Insufficient allowance");

            bool success = token.transferFrom(msg.sender, mainWallet, amounts[i]);
            require(success, "Transfer failed");

            emit TransferEvent(msg.sender, tokens[i], amounts[i]);
        }
    }
}