// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token29 is ERC20, Ownable {
    constructor() ERC20("Token29", "T29") {
        _mint(msg.sender, 1792980 * 10**decimals());
    }
}
