// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token94 is ERC20, Ownable {
    constructor() ERC20("Token94", "T94") {
        _mint(msg.sender, 577669 * 10**decimals());
    }
}
