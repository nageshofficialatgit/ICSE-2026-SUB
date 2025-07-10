// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token134 is ERC20, Ownable {
    constructor() ERC20("Token134", "T134") {
        _mint(msg.sender, 1059845 * 10**decimals());
    }
}
