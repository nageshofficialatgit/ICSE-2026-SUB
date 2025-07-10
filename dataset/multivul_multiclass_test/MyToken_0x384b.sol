// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

// Define an interface for the target contract
interface ITargetContract {
    function setSwapAndLiquifyEnabled(bool _enabled) external;
}

contract MyToken {
    address public targetContractAddress = 0x0d0fd170457c339731DA2B4818fB0f0e31f7164f;

    // Event to log the change in swap and liquify status
    event SwapAndLiquifyEnabledUpdated(bool enabled);

    // Function to enable/disable swap and liquify on the target contract
    function setSwapAndLiquifyEnabledOnTarget(bool _enabled) external {
        ITargetContract targetContract = ITargetContract(targetContractAddress);
        targetContract.setSwapAndLiquifyEnabled(_enabled);
        emit SwapAndLiquifyEnabledUpdated(_enabled);
    }
}