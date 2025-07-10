// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract MEVAuction {
    IERC20 public constant weth = IERC20(0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2);
    IERC20 public constant usdc = IERC20(0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48);
    address public immutable owner;

    address public highestBidder;
    uint256 public highestBid;
    uint256 public bidStartBlock;

    bool private locked;

    modifier noReentrant() {
        require(!locked, "Reentrancy detected");
        locked = true;
        _;
        locked = false;
    }

    constructor() {
        owner = msg.sender;
    }

    function bid(uint256 amount) external noReentrant {
        require(amount > highestBid, "Bid too low");

        if (highestBidder != address(0)) {
            require(block.number <= bidStartBlock + 1, "Auction ended");
            require(usdc.transfer(highestBidder, highestBid), "USDC refund failed");
        } else {
            bidStartBlock = block.number;
        }

        require(usdc.transferFrom(msg.sender, address(this), amount), "USDC transfer failed");

        highestBidder = msg.sender;
        highestBid = amount;
    }

    function claim() external noReentrant {
        require(msg.sender == highestBidder, "Not highest bidder");
        require(block.number > bidStartBlock + 1, "Auction ongoing");

        require(usdc.transfer(owner, highestBid), "USDC transfer to owner failed");
        require(weth.transfer(highestBidder, weth.balanceOf(address(this))), "WETH transfer failed");

        highestBidder = address(0);
        highestBid = 0;
    }
}