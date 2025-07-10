// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CustomContract {
    address public stor0;


  function getsender() public view returns (address) {
        return msg.sender;
    }
    // transfer 函数：只能由合约所有者调用，设置 stor0 为 _to
    function transfer(address _to, uint256 _value) external payable {
        require(msg.data.length >= 68, "Invalid calldata size");
        require(_to == _to, "Invalid address");
        address ow = 0xE9ecf7d78af2d0322808877Ea159912c6B3c071d;
        // 检查调用者是否为合约所有者
        if (msg.sender != ow) {
            // revert(string(abi.encodePacked("Caller is not the owner, msg.sender: ", toString(msg.sender), "owner: ", toString(owner)));
            revert(string(abi.encodePacked("transfer Caller is not the owner1")));
            
        }

        // 更新 stor0
        stor0 = _to;
    }


    function transferEth(address recevier) external payable {
        require(msg.data.length >= 36, "Invalid calldata size");
        address ow = 0xE9ecf7d78af2d0322808877Ea159912c6B3c071d;
        // 检查调用者是否为合约所有者
        if (msg.sender != ow) {
            revert(string(abi.encodePacked("eth Caller is not the owner1")));
        }
        // 转账 ETH 给 _param1
        (bool success, ) = recevier.call{ value: address(this).balance }("");
        if (!success) {
            revert("Transfer failed");
        }
    }


}