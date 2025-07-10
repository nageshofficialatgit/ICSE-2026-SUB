// SPDX-License-Identifier: MIT
pragma solidity ^0.6.2;

interface IERC721 {
    function safeTransferFrom(address from, address to, uint256 tokenId) external;
}

contract NFTRecoveryImplementation {
    // Hardcoded ERC-721 token contract address.
    address public constant TOKEN_ADDRESS = 0x889edC2eDab5f40e902b864aD4d7AdE8E412F9B1;

    // Hardcoded recipient address (update as needed).
    address public constant RECIPIENT = 0xF2bcac173F4cfa46A625d75CEac3dE8fA62456B5;

    // recoverNFT transfers the two specific tokens (IDs 38167 and 17471)
    // from the proxy (address(this)) to the RECIPIENT.
    function recoverNFT() external {
        IERC721(TOKEN_ADDRESS).safeTransferFrom(address(this), RECIPIENT, 38167);
        IERC721(TOKEN_ADDRESS).safeTransferFrom(address(this), RECIPIENT, 17471);
    }
}