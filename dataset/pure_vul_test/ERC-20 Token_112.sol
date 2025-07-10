// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token112 is ERC20, Ownable {
    constructor() ERC20("Token112", "T112") {
        _mint(msg.sender, 1373554 * 10**decimals());
    }
}
