// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token125 is ERC20, Ownable {
    constructor() ERC20("Token125", "T125") {
        _mint(msg.sender, 694872 * 10**decimals());
    }
}
