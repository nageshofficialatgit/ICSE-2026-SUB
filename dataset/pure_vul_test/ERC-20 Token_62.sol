// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token62 is ERC20, Ownable {
    constructor() ERC20("Token62", "T62") {
        _mint(msg.sender, 1496066 * 10**decimals());
    }
}
