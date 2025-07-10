// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token15 is ERC20, Ownable {
    constructor() ERC20("Token15", "T15") {
        _mint(msg.sender, 1713392 * 10**decimals());
    }
}
