// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token76 is ERC20, Ownable {
    constructor() ERC20("Token76", "T76") {
        _mint(msg.sender, 860156 * 10**decimals());
    }
}
