// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token111 is ERC20, Ownable {
    constructor() ERC20("Token111", "T111") {
        _mint(msg.sender, 930436 * 10**decimals());
    }
}
