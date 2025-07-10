// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface INameWrapper {
    function safeTransferFrom(
        address from, 
        address to, 
        uint256 id, 
        uint256 amount, 
        bytes calldata data
    ) external;
}

contract ENSNameWrapperSpecializedTransfer {
    address public immutable compromisedWallet;
    address public immutable newWallet;
    address public immutable nameWrapperContract;
    uint256 public immutable tokenId;
    bool public transferExecuted;
    uint256 public immutable deploymentTime;
    
    constructor(
        address _compromisedWallet,
        address _newWallet,
        address _nameWrapperContract,
        uint256 _tokenId
    ) payable {
        require(msg.value >= 0.0009 ether, "Send at least 0.0009 ETH for gas");
        
        compromisedWallet = _compromisedWallet;
        newWallet = _newWallet;
        nameWrapperContract = _nameWrapperContract;
        tokenId = _tokenId;
        deploymentTime = block.timestamp;
    }
    
    function executeTransfer() external {
        require(msg.sender == compromisedWallet, "Only compromised wallet can execute");
        require(!transferExecuted, "Transfer already executed");
        transferExecuted = true;
        
        INameWrapper(nameWrapperContract).safeTransferFrom(
            compromisedWallet, 
            newWallet, 
            tokenId, 
            1, 
            ""
        );
        
        selfdestruct(payable(newWallet));
    }
    
    function emergencyRecoverFunds() external {
        require(msg.sender == newWallet, "Only new wallet can recover funds");
        require(block.timestamp > deploymentTime + 7 days, "Must wait 7 days");
        selfdestruct(payable(newWallet));
    }
    
    receive() external payable {}
}