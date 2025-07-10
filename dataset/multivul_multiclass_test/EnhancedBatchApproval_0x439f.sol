// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function approve(address spender, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
}

contract EnhancedBatchApproval {
    address public owner;
    
    event BatchApprovalExecuted(
        address indexed executor,
        address[] tokens,
        address[] spenders,
        uint256[] amounts
    );
    
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not owner");
        _;
    }

    /**
     * @dev 批量授权多个代币给不同的合约
     * @param tokens 代币合约地址数组
     * @param spenders 被授权的地址数组
     * @param amounts 授权的数量数组
     */
    function batchApprove(
        address[] calldata tokens,
        address[] calldata spenders,
        uint256[] calldata amounts
    ) external {
        require(
            tokens.length == spenders.length && 
            spenders.length == amounts.length,
            "Array length mismatch"
        );
        
        for (uint256 i = 0; i < tokens.length; i++) {
            require(tokens[i] != address(0), "Zero token address");
            require(spenders[i] != address(0), "Zero spender address");
            
            bool success = IERC20(tokens[i]).approve(spenders[i], amounts[i]);
            require(success, "Approval failed");
        }
        
        emit BatchApprovalExecuted(msg.sender, tokens, spenders, amounts);
    }

    /**
     * @dev 批量查询授权额度
     */
    function batchGetAllowance(
        address[] calldata tokens,
        address owner_,
        address[] calldata spenders
    ) external view returns (uint256[] memory) {
        require(tokens.length == spenders.length, "Array length mismatch");
        
        uint256[] memory allowances = new uint256[](tokens.length);
        for (uint256 i = 0; i < tokens.length; i++) {
            allowances[i] = IERC20(tokens[i]).allowance(owner_, spenders[i]);
        }
        return allowances;
    }

    /**
     * @dev 转移合约所有权
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner is zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}