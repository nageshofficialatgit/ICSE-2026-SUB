// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

contract LiquidityMultisig {
    address[] public owners;
    uint256 public requiredSignatures;
    mapping(address => bool) public isOwner;
    mapping(bytes32 => bool) private executedTxs;

    address public immutable lpToken;

    event LiquidityLocked(uint256 amount);
    event LiquidityRecovered(uint256 amount);

    modifier onlyOwners() {
        require(isOwner[msg.sender], "Not authorized");
        _;
    }

    constructor(address[] memory _owners, uint256 _requiredSignatures, address _lpToken) {
        require(_owners.length >= _requiredSignatures, "Owners < Required Signatures");

        for (uint256 i = 0; i < _owners.length; i++) {
            isOwner[_owners[i]] = true; 
            owners.push(_owners[i]);
        }
        requiredSignatures = _requiredSignatures;
        lpToken = _lpToken;
    }

    function lockLiquidity(uint256 amount) external onlyOwners {
        require(IERC20(lpToken).transferFrom(msg.sender, address(this), amount), "Transfer failed");
        emit LiquidityLocked(amount);
    }

    function requestLiquidityRecovery(address to, uint256 amount, bytes32 txHash) external onlyOwners {
        require(!executedTxs[txHash], "Transaction already executed");
        executedTxs[txHash] = true;

        require(IERC20(lpToken).transfer(to, amount), "Transfer failed");
        emit LiquidityRecovered(amount);
    }
}