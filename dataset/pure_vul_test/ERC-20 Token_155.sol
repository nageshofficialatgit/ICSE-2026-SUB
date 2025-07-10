// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token155 is ERC20, Ownable {
    constructor() ERC20("Token155", "T155") {
        _mint(msg.sender, 1232698 * 10**decimals());
    }
}
