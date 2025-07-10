// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token117 is ERC20, Ownable {
    constructor() ERC20("Token117", "T117") {
        _mint(msg.sender, 1053784 * 10**decimals());
    }
}
