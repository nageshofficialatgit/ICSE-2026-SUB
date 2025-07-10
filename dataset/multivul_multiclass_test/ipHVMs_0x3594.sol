// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

/* ————————————————————————————— File: sources of ipx/ipHVMs-6-sig.sol

    IPHVMS by IPCS > Intel Port Contract Security (EVM Design).
    @title ipHVMs contract> Intel Port Horo-Vault Multi-Sig Contract (A secure contract vault for securing your Assets with a time-lock).
        @author Ann Mandriana - <ann@intelport.org>
        @author Nouval Safaat - <nvlsft@intelport.org>
        @author Ryan Oktanda - <ryan@intelport.org>
        @author Dimas Fachri - <dimskuy@intelport.org>
        @cisa   Hazib
        @matter > private use only.
        @param > ether / erc20 / erc721.

——————————————————————————————— The procedure to time-lock your Assets!

    Caution! <Please read and fully understand everything before proceeding. Use it at your own risk!>
        1. Before deploying, insert 4 owner addresses in the [constructor] where it specifies '4 owners.' These must be [ethereum personal wallet addresses (EOA)]. Do not use a smart-contract address, as we cannot guarantee functionality if a smart-contract address is used.
        
        2. After it is deployed, you can send ETH, ERC20 tokens, ERC721 NFTs directly to the ipHVMs contract from anywhere you want <3

        3. Use the setTL function to set the time-lock for your assets. You must enter the duration in seconds (e.g., for 2 minutes, you must enter 120).
           WARNING!: After setting the time-lock, there is no way to withdraw or retrieve your assets from the ipHVMs contract. You must wait until the time lock expires.
        
        4. Only the wallet address you register can withdraw anything from the ipHVMs contract, so be careful and make sure not to lose any of the personal ethereum wallet addresses (EOAs) that are registered.
           For withdrawals, you need at least three owner addresses: two for approval and one for execution. Withdrawals can only be made to Owner 1, 2, 3, or 4.
        
        5. To withdraw, two different owners must first call [approve01] and [approve02]. Then, a third owner must execute the withdrawal. This process must be repeated for each withdrawal, as approvals are valid for only one transaction.
           In the [ownerOption] column, simply input 1, 2, 3, or 4, depending on which owner address you want to withdraw to.

——————————————————————————————— IPCS
*/

    abstract contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
        }

    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
        }
    }

    interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    }

    library SafeERC20 {
    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        require(token.transfer(to, value), "SafeERC20: Transfer failed");
        }
    }

    interface IERC721 {
    function safeTransferFrom(address from, address to, uint256 tokenId) external;
    function ownerOf(uint256 tokenId) external view returns (address);
    }

    interface IERC721Receiver {
    function onERC721Received(address operator, address from, uint256 tokenId, bytes calldata data) external returns (bytes4);
    }

    interface IERC1155Receiver {
    function onERC1155Received(address operator, address from, uint256 id, uint256 value, bytes calldata data) external returns (bytes4);
    function onERC1155BatchReceived(address operator, address from, uint256[] calldata ids, uint256[] calldata values, bytes calldata data) external returns (bytes4);
    }

    interface IERC1155 {
    function safeTransferFrom(address from, address to, uint256 id, uint256 amount, bytes calldata data) external;
    function balanceOf(address account, uint256 id) external view returns (uint256);
    }

    contract ipHVMs is ReentrancyGuard, IERC721Receiver, IERC1155Receiver {        
    using SafeERC20 for IERC20;

    address private owner1;
    address private owner2;
    address private owner3;
    address private owner4;
    uint256 public unlockTime;
    
    // IPCS Service
    address private IPCSService = 0x00060E123Fa9b8e33345b745626D1E2078992741; // <EOA only!> [Contract address are not allowed!]

    // IPHVMS Version
    string public Version = "v8.9.2";

    event TimeLockSet(uint256 unlockTime);
    event WithdrawalApproved(address indexed approver);
    event EtherWithdrawn(address indexed recipient, uint256 amount);
    event TokensWithdrawn(address indexed token, address indexed recipient, uint256 amount);
    event NFTWithdrawn(address indexed nftContract, address indexed recipient, uint256 tokenId);
    event ERC1155Withdrawn(address indexed nftContract, address indexed recipient, uint256 tokenId, uint256 amount);
    event EtherReceived(address indexed sender, uint256 amount);

    modifier onlyOwner() {
    require(
        msg.sender == owner1 || msg.sender == owner2 || msg.sender == owner3 || msg.sender == owner4,
        "Only the owners can perform this action"
                );
            _;
        }

    modifier whenUnlocked() {
        require(block.timestamp >= unlockTime, "Assets are still locked!");
            _;
        }

        bool private withdrawalApproved;
        address private lastApprover;
        uint256 private approvalTimestamp;

        bool private withdrawalApproved2;
        address private secondApprover;

    constructor() {
        owner1 = 0x00060E123Fa9b8e33345b745626D1E2078992741; // [O_EOA] 1
        owner2 = 0x2220804674fc9557229e077A12Ea98B40eb2403B; // [O_EOA] 2
        owner3 = 0x77702b30a0276A4436BB688586147Ff75d64E97B; // [O_EOA] 3
        owner4 = 0x999Ba2936EC9c5aa6F33c8358d6ca7fB0890d12B; // [O_EOA] 4
        }

    function setTL(uint256 durationAsSec) external onlyOwner {
        require(block.timestamp >= unlockTime, "Cannot set a new time-lock until the current one expires");
        require(durationAsSec >= 600, "Minimum time-lock is 10 minutes");
        require(durationAsSec <= 157680000, "Invalid duration (max 5 years)");

        unlockTime = block.timestamp + durationAsSec;
        emit TimeLockSet(unlockTime);
        }

    function approve01() external onlyOwner whenUnlocked {
        require(!withdrawalApproved, "Already approved once");
        withdrawalApproved = true;
        lastApprover = msg.sender;
        approvalTimestamp = block.timestamp;
        emit WithdrawalApproved(msg.sender);
        }

        function resetApproval() internal {
        withdrawalApproved = false;
        withdrawalApproved2 = false;
        lastApprover = address(0);
        secondApprover = address(0);
        }

    function approve02() external onlyOwner whenUnlocked {
        require(withdrawalApproved, "First approval required");
        require(msg.sender != lastApprover, "Second approval must be from a different owner");
        require(!withdrawalApproved2, "Already approved twice");
        
        withdrawalApproved2 = true;
        secondApprover = msg.sender;
        emit WithdrawalApproved(msg.sender);
        }


    function wdETH(uint8 ownerOption) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two approvals required");

        if (block.timestamp >= approvalTimestamp + 600) {
        resetApproval();
        revert("Approval expired, please approve again");
        }

        require(msg.sender != lastApprover && msg.sender != secondApprover, "Approvers cannot execute withdrawal");

        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner option");
        address payable recipient = ownerOption == 1 ? payable(owner1) :
                                    ownerOption == 2 ? payable(owner2) :
                                    ownerOption == 3 ? payable(owner3) : payable(owner4);

        uint256 amount = address(this).balance;
        require(amount > 0, "No Ether to Withdraw");

        uint256 fee = (amount * 2) / 100;
        uint256 finalAmount = amount - fee;

        (bool successFee, ) = IPCSService.call{value: fee}("");
        require(successFee, "IPCSS transfer failed");

        (bool success, ) = recipient.call{value: finalAmount}("");
        require(success, "IPCS transfer failed");

        resetApproval();
        emit EtherWithdrawn(recipient, amount);
        }

    function wdERC20(address tokenContract, uint8 ownerOption) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two approvals required");

        if (block.timestamp >= approvalTimestamp + 600) {
        resetApproval();
        revert("Approval expired, please approve again");
        }

        require(msg.sender != lastApprover && msg.sender != secondApprover, "Approvers cannot execute withdrawal");

        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner option");
        address recipient = ownerOption == 1 ? owner1 :
                            ownerOption == 2 ? owner2 :
                            ownerOption == 3 ? owner3 : owner4;

        IERC20 token = IERC20(tokenContract);
        uint256 amount = token.balanceOf(address(this));
        require(amount > 0, "No Tokens to Withdraw");

        uint256 fee = (amount * 2) / 100;
        uint256 finalAmount = amount - fee;

        token.safeTransfer(IPCSService, fee);
        token.safeTransfer(recipient, finalAmount);

        resetApproval();
        emit TokensWithdrawn(tokenContract, recipient, amount);
        }

    function wdERC1155(address nftContract, uint8 ownerOption, uint256 tokenId, uint256 amount) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two approvals required");

        if (block.timestamp >= approvalTimestamp + 600) {
        resetApproval();
        revert("Approval expired, please approve again");
        }

        require(msg.sender != lastApprover && msg.sender != secondApprover, "Approvers cannot execute withdrawal");

        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner option");
        address recipient = ownerOption == 1 ? owner1 :
                            ownerOption == 2 ? owner2 :
                            ownerOption == 3 ? owner3 : owner4;

        IERC1155 token = IERC1155(nftContract);
        require(token.balanceOf(address(this), tokenId) >= amount, "No ERC1155 tokens to withdraw");

        token.safeTransferFrom(address(this), recipient, tokenId, amount, "");

        resetApproval();
        emit ERC1155Withdrawn(nftContract, recipient, tokenId, amount);
        }

    function wdERC721(address nftContract, uint8 ownerOption, uint256 tokenId) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two approvals required");

        if (block.timestamp >= approvalTimestamp + 600) {
        resetApproval();
        revert("Approval expired, please approve again");
        }

        require(msg.sender != lastApprover && msg.sender != secondApprover, "Approvers cannot execute withdrawal");

        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner option");
        address recipient = ownerOption == 1 ? owner1 :
                            ownerOption == 2 ? owner2 :
                            ownerOption == 3 ? owner3 : owner4;

        require(IERC721(nftContract).ownerOf(tokenId) == address(this), "No NFT to Withdraw");

        IERC721(nftContract).safeTransferFrom(address(this), recipient, tokenId);

        resetApproval();
        emit NFTWithdrawn(nftContract, recipient, tokenId);
        }


        function _onERC721Received(address, address, uint256, bytes calldata) internal pure returns (bytes4) {
        return 0x150b7a02;
        }

        function _onERC1155Received(address, address, uint256, uint256, bytes calldata) internal pure returns (bytes4) {
        return 0xf23a6e61;
        }

        function _onERC1155BatchReceived(address, address, uint256[] calldata, uint256[] calldata, bytes calldata) internal pure returns (bytes4) {
        return 0xbc197c81;
        }

        function onERC721Received(address operator, address from, uint256 tokenId, bytes calldata data) external pure override returns (bytes4) {
        return _onERC721Received(operator, from, tokenId, data);
        }

        function onERC1155Received(address operator, address from, uint256 id, uint256 value, bytes calldata data) external pure override returns (bytes4) {
        return _onERC1155Received(operator, from, id, value, data);
        }

        function onERC1155BatchReceived(address operator, address from, uint256[] calldata ids, uint256[] calldata values, bytes calldata data) external pure override returns (bytes4) {
        return _onERC1155BatchReceived(operator, from, ids, values, data);
        }

        receive() external payable {
            emit EtherReceived(msg.sender, msg.value);
            }
        }
/* ——————————————————————————————— EOC

    .___  __________    ___ ___   ____   ____    _____      _________
    |   | \______   \  /   |   \  \   \ /   /   /     \    /   _____/
    |   |  |     ___/ /    ~    \  \   Y   /   /  \ /  \   \_____  \ 
    |   |  |    |     \    Y    /   \     /   /    Y    \  /        \
    |___|  |____|      \___|_  /     \___/    \____|__  / /_______  /
                         \/                       \/          \/
    ©2025 intel port contract security - @prog <G4X39G9LEW9X9X>
    IPCSS charges a 2% fee for every withdrawal.
*/