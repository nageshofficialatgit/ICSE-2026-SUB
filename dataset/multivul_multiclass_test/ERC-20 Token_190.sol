// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token190 is ERC20, Ownable {
    constructor() ERC20("Token190", "T190") {
        _mint(msg.sender, 1786734 * 10**decimals());
    }
}
