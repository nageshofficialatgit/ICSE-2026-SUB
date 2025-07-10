// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token71 is ERC20, Ownable {
    constructor() ERC20("Token71", "T71") {
        _mint(msg.sender, 996396 * 10**decimals());
    }
}
