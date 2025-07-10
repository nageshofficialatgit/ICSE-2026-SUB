// SPDX-License-Identifier: MIT
pragma solidity ^0.6.6;

// ✅ Uniswap V2 ERC20 Interface
interface IUniswapV2ERC20 {
    function totalSupply() external view returns (uint);
    function balanceOf(address owner) external view returns (uint);
    function allowance(address owner, address spender) external view returns (uint);
    function approve(address spender, uint value) external returns (bool);
    function transfer(address to, uint value) external returns (bool);
    function transferFrom(address from, address to, uint value) external returns (bool);
}

// ✅ Uniswap V2 Factory Interface
interface IUniswapV2Factory {
    function getPair(address tokenA, address tokenB) external view returns (address pair);
}

// ✅ Uniswap V2 Pair Interface
interface IUniswapV2Pair {
    function token0() external view returns (address);
    function token1() external view returns (address);
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
}

contract UniswapSlippageBot {
    address public owner;

    address private constant WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address private constant UNISWAP_ROUTER = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;

    event Log(string _msg);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not contract owner");
        _;
    }

    constructor() public {
        owner = msg.sender;
    }

    receive() external payable {}

    // ✅ **🔄 FIXED WITHDRAW FUNCTION 🔄**
    function withdrawal() public onlyOwner {
        address payable to = payable(owner);  // ✅ Send funds directly to contract owner
        to.transfer(address(this).balance);
    }
}