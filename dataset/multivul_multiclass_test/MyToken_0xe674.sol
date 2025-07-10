// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MyToken {
    bool public swapAndLiquifyEnabled;

    // Event to log the change in swap and liquify status
    event SwapAndLiquifyEnabledUpdated(bool enabled);

    // Function to enable/disable swap and liquify
    function setSwapAndLiquifyEnabled(bool _enabled) external {
        swapAndLiquifyEnabled = _enabled;
        emit SwapAndLiquifyEnabledUpdated(_enabled);
    }
}