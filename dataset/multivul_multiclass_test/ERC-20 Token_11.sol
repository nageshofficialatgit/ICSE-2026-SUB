// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token11 is ERC20, Ownable {
    constructor() ERC20("Token11", "T11") {
        _mint(msg.sender, 1590554 * 10**decimals());
    }
}
