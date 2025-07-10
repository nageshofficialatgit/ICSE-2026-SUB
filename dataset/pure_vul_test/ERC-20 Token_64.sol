// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token64 is ERC20, Ownable {
    constructor() ERC20("Token64", "T64") {
        _mint(msg.sender, 1749480 * 10**decimals());
    }
}
