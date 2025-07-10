// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token54 is ERC20, Ownable {
    constructor() ERC20("Token54", "T54") {
        _mint(msg.sender, 575648 * 10**decimals());
    }
}
