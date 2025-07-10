// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token146 is ERC20, Ownable {
    constructor() ERC20("Token146", "T146") {
        _mint(msg.sender, 1977509 * 10**decimals());
    }
}
