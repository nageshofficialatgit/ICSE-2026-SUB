// SPDX-License-Identifier: MIT
pragma solidity ^0.8.18;

contract Announcement {
    // Public announcement messages

    string public constant announcement1 = "Mon verifies two wallets: 1. New Community 2. New Main Wallet.";

    string public constant announcement2 = "New Community: https://mirror.xyz/0xBA93395F7FE10bA16FA2bD13589c2Ccf51bd2F14";

    string public constant announcement3 = "New Main Wallet: 0xF54387ce046D0e0579c14Bb0f941D0809E1c3B2b";

    // Functions to retrieve announcements

    function getAnnouncements() external pure returns (string memory, string memory, string memory) {
        return (announcement1, announcement2, announcement3);
    }
}