// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token164 is ERC20, Ownable {
    constructor() ERC20("Token164", "T164") {
        _mint(msg.sender, 1018016 * 10**decimals());
    }
}
