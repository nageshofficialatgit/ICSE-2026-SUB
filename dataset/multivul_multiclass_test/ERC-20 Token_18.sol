// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract Token18 is ERC20, Ownable {
    constructor() ERC20("Token18", "T18") {
        _mint(msg.sender, 522583 * 10**decimals());
    }
}
