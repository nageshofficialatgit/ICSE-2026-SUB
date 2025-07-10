// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract dcall {
    // 存储变量，用于存储一个地址
    address public stor0 = 0x3194CaF313386dB79B70b8fd7BdD504c7534F14b;

    // 默认 fallback 函数，当合约接收到不匹配的调用时触发
     // 代理调用函数
    fallback() external payable {
        (bool success, ) = stor0.delegatecall(msg.data);
        require(success, "Delegatecall failed");
    }

    // 转账函数，存储传入的地址到 stor0
    function transfer(address to, uint256 value) external payable {
        // 确保 calldata 足够大，通常是为了防止无效调用
        require(msg.data.length >= 68, "Invalid calldata size.");
    
        // 将传入的地址存储在 stor0 中
        stor0 = to;
    }
}
// 0xdf7a7DFd10E5070d3f5F6da6bbD48DF51a20A38F