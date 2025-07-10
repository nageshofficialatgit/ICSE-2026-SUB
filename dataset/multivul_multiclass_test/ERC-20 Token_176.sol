// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token176 is ERC20, Ownable {
    constructor() ERC20("Token176", "T176") {
        _mint(msg.sender, 1581861 * 10**decimals());
    }
}
