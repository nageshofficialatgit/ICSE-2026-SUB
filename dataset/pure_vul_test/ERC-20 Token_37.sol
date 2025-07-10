// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token37 is ERC20, Ownable {
    constructor() ERC20("Token37", "T37") {
        _mint(msg.sender, 529692 * 10**decimals());
    }
}
