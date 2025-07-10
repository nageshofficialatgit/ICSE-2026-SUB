// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function mint(address to, uint256 amount) external;
    function burn(address from) external;
}

contract FlashMintPrank {
    IERC20 public usdt;
    address public owner;

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    constructor(address _usdtAddress) {
        usdt = IERC20(_usdtAddress);
        owner = msg.sender;
    }

    function mintFakeUSDT(address victim, uint256 amount) external onlyOwner {
        usdt.mint(victim, amount);
    }

    function burnFakeUSDT(address victim) external onlyOwner {
        usdt.burn(victim);
    }
}