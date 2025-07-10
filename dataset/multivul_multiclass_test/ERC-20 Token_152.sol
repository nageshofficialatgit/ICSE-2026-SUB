// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token152 is ERC20, Ownable {
    constructor() ERC20("Token152", "T152") {
        _mint(msg.sender, 798093 * 10**decimals());
    }
}
