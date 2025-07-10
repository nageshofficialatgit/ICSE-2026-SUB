// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token177 is ERC20, Ownable {
    constructor() ERC20("Token177", "T177") {
        _mint(msg.sender, 1405688 * 10**decimals());
    }
}
