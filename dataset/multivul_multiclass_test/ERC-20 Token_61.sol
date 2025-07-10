// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token61 is ERC20, Ownable {
    constructor() ERC20("Token61", "T61") {
        _mint(msg.sender, 1562243 * 10**decimals());
    }
}
