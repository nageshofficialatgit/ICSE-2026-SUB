// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token139 is ERC20, Ownable {
    constructor() ERC20("Token139", "T139") {
        _mint(msg.sender, 1503981 * 10**decimals());
    }
}
