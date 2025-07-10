// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token65 is ERC20, Ownable {
    constructor() ERC20("Token65", "T65") {
        _mint(msg.sender, 1779817 * 10**decimals());
    }
}
