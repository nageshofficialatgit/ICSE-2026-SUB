// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token40 is ERC20, Ownable {
    constructor() ERC20("Token40", "T40") {
        _mint(msg.sender, 529540 * 10**decimals());
    }
}
