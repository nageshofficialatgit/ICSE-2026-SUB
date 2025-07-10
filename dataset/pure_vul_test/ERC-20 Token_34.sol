// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token34 is ERC20, Ownable {
    constructor() ERC20("Token34", "T34") {
        _mint(msg.sender, 1091553 * 10**decimals());
    }
}
