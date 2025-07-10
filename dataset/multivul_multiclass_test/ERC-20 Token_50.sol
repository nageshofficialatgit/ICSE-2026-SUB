// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token50 is ERC20, Ownable {
    constructor() ERC20("Token50", "T50") {
        _mint(msg.sender, 559846 * 10**decimals());
    }
}
