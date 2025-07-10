// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token95 is ERC20, Ownable {
    constructor() ERC20("Token95", "T95") {
        _mint(msg.sender, 897374 * 10**decimals());
    }
}
