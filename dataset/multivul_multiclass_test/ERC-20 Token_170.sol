// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token170 is ERC20, Ownable {
    constructor() ERC20("Token170", "T170") {
        _mint(msg.sender, 569159 * 10**decimals());
    }
}
