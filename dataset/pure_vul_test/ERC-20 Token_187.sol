// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token187 is ERC20, Ownable {
    constructor() ERC20("Token187", "T187") {
        _mint(msg.sender, 506904 * 10**decimals());
    }
}
