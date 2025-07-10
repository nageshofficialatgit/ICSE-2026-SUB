// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

/* ————————————————————————————— File: sources of ipcs/ipHVMs-4mt-sig.sol

    ██▓ ███▄    █ ▄▄▄█████▓▓█████  ██▓        ██▓███   ▒█████   ██▀███  ▄▄▄▓████▓
   ▓██▒ ██ ▀█   █ ▓  ██▒ ▓▒▓█   ▀ ▓██▒       ▓██░  ██▒▒██▒  ██▒▓██ ▒ ██▒▓  ██▒ ▓▒
   ▒██▒▓██  ▀█ ██▒▒ ▓██░ ▒░▒███   ▒██░       ▓██░ ██▓▒▒██░  ██▒▓██ ░▄█ ▒▒ ▓██░ ▒░
   ░██░▓██▒  ▐▌██▒░ ▓██▓ ░ ▒▓█  ▄ ▒██░       ▒██▄█▓▒ ▒▒██   ██░▒██▀▀█▄  ░ ▓██▓ ░ 
   ░██░▒██░   ▓██░  ▒██▒ ░ ░▒████▒░██████▒   ▒██▒ ░  ░░ ████▓▒░░██▓ ▒██▒  ▒██▒ ░ 
   ░▓  ░ ▒░   ▒ ▒   ▒ ░░   ░░ ▒░ ░░ ▒░▓  ░   ▒▓▒░ ░  ░░ ▒░▒░▒░ ░ ▒▓ ░▒▓░  ▒ ░░   
    ▒ ░░ ░░   ░ ▒░    ░     ░ ░  ░░ ░ ▒  ░   ░▒ ░       ░ ▒ ▒░   ░▒ ░ ▒░    ░    
    ▒ ░   ░   ░ ░   ░         ░     ░ ░      ░░       ░ ░ ░ ▒    ░░        ░      
    ░           ░             ░  ░    ░  ░                ░ ░                   
    IPHVMS by IPCS > Intel Port Contract Security [EVM Design].
    @title ipHVMs contract> Intel Port Horo-Vault Multi-Sig Contract [A contract vault designed to secure your assets with a time-lock].
        @author Ann Mandriana - <ann@intelport.org>
        @author Nouval Safaat - <nvlsft@intelport.org>
        @author Ryan Oktanda - <ryan@intelport.org>
        @author Dimas Fachri - <dimskuy@intelport.org>
        @author Winni Ismail - <wienzki@intelport.org>
        @cisa   Hazib - <hazib@intelport.org>
        @matter > personal use.
        @matter > multi-sig with time-lock.
        @param > eth / erc20 / erc721 / erc1155.
        ▒ ░       ░ ░   ▒         ░     ░ ▒     ░    
        ░           ░             ▒  ░    ░  ░         
    ——————————————————————————————— The procedure to time-lock your Assets!
    
     ██▓     ▒█████   ▄████▄   ██ ▄█▀    ██▓▄▄▄▓████▓ ▐██▌ 
    ▓██▒    ▒██▒  ██▒▒██▀ ▀█   ██▄█▒    ▓██▒▓  ██▒ ▓▒ ▐██▌ 
    ▒██░    ▒██░  ██▒▒▓█    ▄ ▓███▄░    ▒██▒▒ ▓██░ ▒░ ▐██▌ 
    ▒██░    ▒██   ██░▒▓▓▄ ▄██▒▓██ █▄    ░██░░ ▓██▓ ░  ▓██▒ 
    ░██████▒░ ████▓▒░▒ ▓███▀ ░▒██▒ █▄   ░██░  ▒██▒ ░  ▒▄▄  
    ░ ▒░▓  ░░ ▒░▒░▒░ ░ ░▒ ▒  ░▒ ▒▒ ▓▒   ░▓    ▒ ░░    ░▀▀▒ 
    ░ ░ ▒  ░  ░ ▒ ▒░   ░  ▒   ░ ░▒ ▒░    ▒ ░    ░     ░  ░ 
      ░ ░   ░ ░ ░ ▒  ░        ░ ░░ ░     ▒ ░  ░          ░ 
        ░  ░    ░ ░  ░ ░      ░  ░       ░            ░    
                  ░                                     
    Caution! <Please read and fully understand everything before proceeding. Use it at your own risk!>
        1. Before deploying, insert 4 owner addresses in the [constructor] where it specifies '4 owners.' These must be [ethereum personal wallet addresses (EOA)]. Do not use a smart-contract address, as we cannot guarantee functionality if a smart-contract address is used.
        
        2. After it is deployed, you can send ETH, ERC20 tokens, ERC721 NFTs, and ERC1155 directly to the ipHVMs contract from anywhere you want <3

        3. Use the [set_TL] function to set the time-lock for your assets. You must enter the duration in seconds (e.g., for 10 minutes, you must enter 600).
           WARNING!: After setting the time-lock, there is no way to withdraw or retrieve your assets from the ipHVMs contract. You must wait until the time lock expires.
        
        4. Only the wallet address you register can withdraw anything from the ipHVMs contract, so be careful and make sure not to lose any of the personal ethereum wallet addresses (EOAs) that are registered.
           For withdrawals, you need at least 3 owner addresses: two for approval and one for execution. Withdrawals can only be made to Owner 1, 2, 3, or 4.
        
        5. To withdraw, two different owners must first call [Appr_I] and [Appr_II]. Then, a third owner must execute the withdrawal. This process must be repeated for each withdrawal, as approvals are valid for only one transaction.
           In the [ownerOption] column, simply input 1, 2, 3, or 4, depending on which owner address you want to withdraw to.
           ▒ ░       ░ ░   ░         ░ ▒   
           ░           ░             ░             
    ——————————————————————————————— BOC.*/    
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

    /* ——————————————————————————————— IPCS
    
        ██▓ ██▓███   ██░ ██  ██▒   █▓ ███▄ ▄███▓  ██████ 
       ▓██▒▓██░  ██▒▓██░ ██▒▓██░   █▒▓██▒▀█▀ ██▒▒██    ▒ 
       ▒██▒▓██░ ██▓▒▒██▀▀██░ ▓██  █▒░▓██    ▓██░░ ▓██▄   
       ░██░▒██▄█▓▒ ▒░▓█ ░██   ▒██ █░░▒██    ▒██   ▒   ██▒
       ░██░▒██▒ ░  ░░▓█▒░██▓   ▒▀█░  ▒██▒   ░██▒▒██████▒▒
       ░▓  ▒▓▒░ ░  ░ ▒ ░░▒░▒   ░ ▐░  ░ ▒░   ░  ░▒ ▒▓▒ ▒ ░
        ▒ ░░▒ ░      ▒ ░▒░ ░   ░ ░░  ░  ░      ░░ ░▒  ░ ░
        ▒ ░░░        ░  ░░ ░     ░░  ░      ░   ░  ░  ░  
        ░            ░  ░  ░      ░         ░         ░  
                                  ░                   */
    contract ipHVMs is ReentrancyGuard, IERC721Receiver, IERC1155Receiver {        
    using SafeERC20 for IERC20;

    address private owner1;
    address private owner2;
    address private owner3;
    address private owner4;
    uint256 public unlockTime;
    
    // IPCS Service [EOA]
    address private IPCSService = 0x77702b30a0276A4436BB688586147Ff75d64E97B;

    // IPHVMS Version
    string public Version = "v3.4.2";

    function checkGovern() external view returns (uint8[4] memory, address[4] memory) {
        return ([1, 2, 3, 4], [owner1, owner2, owner3, owner4]);
        }

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
        bool private withdrawalApproved2;
        address private secondApprover;
        address private proposedNewOwner; 
        uint8 private ownerToReplace;   
        bool private replacementProposed;   
        bool private replacementApproved1; 
        bool private replacementApproved2;   
        address private replacementApprover1; 
        address private replacementApprover2;  

    /* ——————————————————————————————— GOV
    
         ▄████  ▒█████   ██▒   █▓▓█████  ██▀███   ███▄    █ 
        ██▒ ▀█▒▒██▒  ██▒▓██░   █▒▓█   ▀ ▓██ ▒ ██▒ ██ ▀█   █ 
       ▒██░▄▄▄░▒██░  ██▒ ▓██  █▒░▒███   ▓██ ░▄█ ▒▓██  ▀█ ██▒
       ░▓█  ██▓▒██   ██░  ▒██ █░░▒▓█  ▄ ▒██▀▀█▄  ▓██▒  ▐▌██▒
       ░▒▓███▀▒░ ████▓▒░   ▒▀█░  ░▒████▒░██▓ ▒██▒▒██░   ▓██░
        ░▒   ▒ ░ ▒░▒░▒░    ░ ▐░  ░░ ▒░ ░░ ▒▓ ░▒▓░░ ▒░   ▒ ▒ 
         ░   ░   ░ ▒ ▒░    ░ ░░   ░ ░  ░  ░▒ ░ ▒░░ ░░   ░ ▒░
         ░   ░ ░ ░ ░ ▒       ░░     ░     ░░   ░    ░   ░ ░ 
             ░     ░ ░        ░     ░  ░   ░              ░ 
                              ░                           */
    constructor(address _owner1, address _owner2, address _owner3, address _owner4) {
        require(_owner1 != address(0) && _owner2 != address(0) && _owner3 != address(0) && _owner4 != address(0), "Owners cannot be zero address");
        require(_owner1 != _owner2 && _owner1 != _owner3 && _owner1 != _owner4 && 
            _owner2 != _owner3 && _owner2 != _owner4 && _owner3 != _owner4, "Owners must be unique");
            owner1 = _owner1;
            owner2 = _owner2;
            owner3 = _owner3;
            owner4 = _owner4;
            }

    function set_TL(uint256 durationAsSec) external onlyOwner {
        require(block.timestamp >= unlockTime, "Cannot set a new time-lock until the current one expires");
        require(durationAsSec >= 600, "Minimum time-lock is 10 minutes");
        require(durationAsSec <= 157680000, "Invalid duration (max 5 years)");

        unlockTime = block.timestamp + durationAsSec;
        emit TimeLockSet(unlockTime);
        }

    function Appr_I() external onlyOwner whenUnlocked {
        require(!withdrawalApproved, "Already approved once");
        withdrawalApproved = true;
        lastApprover = msg.sender;
        emit WithdrawalApproved(msg.sender);
        }

        function resetApproval() internal {
        withdrawalApproved = false;
        withdrawalApproved2 = false;
        lastApprover = address(0);
        secondApprover = address(0);
        }

    function Appr_II() external onlyOwner whenUnlocked {
        require(withdrawalApproved, "First approval required");
        require(msg.sender != lastApprover, "Second approval must be from a different owner");
        require(!withdrawalApproved2, "Already approved twice");
        
        withdrawalApproved2 = true;
        secondApprover = msg.sender;
        emit WithdrawalApproved(msg.sender);
        }

    function _proposeGovern(address newOwner, uint8 ownerNumber) external onlyOwner {
        require(!replacementProposed, "A replacement is already proposed");
        require(ownerNumber >= 1 && ownerNumber <= 4, "Invalid owner number (must be 1-4)");
        require(newOwner != address(0), "New owner cannot be zero address");
        require(newOwner != owner1 && newOwner != owner2 && newOwner != owner3 && newOwner != owner4, 
            "New owner must not already be an owner");

        proposedNewOwner = newOwner;
        ownerToReplace = ownerNumber;
        replacementProposed = true;

        replacementApproved1 = false;
        replacementApproved2 = false;
        replacementApprover1 = address(0);
        replacementApprover2 = address(0);
        }    

    function _apprGovern_I() external onlyOwner {
        require(replacementProposed, "No replacement proposed yet");
        require(!replacementApproved1, "First approval already done");
    
        replacementApproved1 = true;
        replacementApprover1 = msg.sender;
        emit WithdrawalApproved(msg.sender);
        }

    function _apprGovern_II() external onlyOwner {
        require(replacementProposed, "No replacement proposed yet");
        require(replacementApproved1, "First approval required");
        require(!replacementApproved2, "Second approval already done");
        require(msg.sender != replacementApprover1, "Second approval must be from a different owner");

        replacementApproved2 = true;
        replacementApprover2 = msg.sender;
        emit WithdrawalApproved(msg.sender);
        }

    function _repGovern() external onlyOwner {
        require(replacementProposed, "No replacement proposed yet");
        require(replacementApproved1 && replacementApproved2, "Two approvals required");
        require(msg.sender != replacementApprover1 && msg.sender != replacementApprover2, 
            "Executor must be different from approvers");

        if (ownerToReplace == 1) {
            owner1 = proposedNewOwner;
        } else if (ownerToReplace == 2) {
            owner2 = proposedNewOwner;
        } else if (ownerToReplace == 3) {
            owner3 = proposedNewOwner;
        } else if (ownerToReplace == 4) {
            owner4 = proposedNewOwner;
        }

        replacementProposed = false;
        replacementApproved1 = false;
        replacementApproved2 = false;
        proposedNewOwner = address(0);
        ownerToReplace = 0;
        replacementApprover1 = address(0);
        replacementApprover2 = address(0);
        }

    /* ——————————————————————————————— ETH
    
       ▓█████ ▄▄▄▓████▓ ██░ ██ ▓█████  ██▀███  
       ▓█   ▀ ▓  ██▒ ▓▒▓██░ ██▒▓█   ▀ ▓██ ▒ ██▒
       ▒███   ▒ ▓██░ ▒░▒██▀▀██░▒███   ▓██ ░▄█ ▒
       ▒▓█  ▄ ░ ▓██▓ ░ ░▓█ ░██ ▒▓█  ▄ ▒██▀▀█▄  
       ░▒████▒  ▒██▒ ░ ░▓█▒░██▓░▒████▒░██▓ ▒██▒
       ░░ ▒░ ░  ▒ ░░    ▒ ░░▒░▒░░ ▒░ ░░ ▒▓ ░▒▓░
        ░ ░  ░    ░     ▒ ░▒░ ░ ░ ░  ░  ░▒ ░ ▒░
          ░     ░       ░  ░░ ░   ░     ░░   ░ 
          ░  ░          ░  ░  ░   ░  ░   ░   */
    function wd_ETH(uint8 ownerOption) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two approvals required");

        require(msg.sender != lastApprover && msg.sender != secondApprover, "Approvers cannot execute withdrawal");

        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner option");
        address payable recipient = ownerOption == 1 ? payable(owner1) :
                                    ownerOption == 2 ? payable(owner2) :
                                    ownerOption == 3 ? payable(owner3) : payable(owner4);

        uint256 amount = address(this).balance;
        require(amount > 0, "No Ether to Withdraw");

        uint256 fee = (amount * 1) / 100;
        uint256 finalAmount = amount - fee;

        (bool successFee, ) = IPCSService.call{value: fee}("");
        require(successFee, "IPCSS transfer failed");

        (bool success, ) = recipient.call{value: finalAmount}("");
        require(success, "IPCS transfer failed");

        resetApproval();
        emit EtherWithdrawn(recipient, amount);
        }

    /* ——————————————————————————————— ERC-20 Tokens

       ▒▄▄▓████▓ ▒█████   ██ ▄█▀▓█████  ███▄    █   ██████ 
       ▓  ██▒ ▓▒▒██▒  ██▒ ██▄█▒ ▓█   ▀  ██ ▀█   █ ▒██    ▒ 
       ▒ ▓██░ ▒░▒██░  ██▒▓███▄░ ▒███   ▓██  ▀█ ██▒░ ▓██▄   
       ░ ▓██▓ ░ ▒██   ██░▓██ █▄ ▒▓█  ▄ ▓██▒  ▐▌██▒  ▒   ██▒
         ▒██▒ ░ ░ ████▓▒░▒██▒ █▄░▒████▒▒██░   ▓██░▒██████▒▒
         ▒ ░░   ░ ▒░▒░▒░ ▒ ▒▒ ▓▒░░ ▒░ ░░ ▒░   ▒ ▒ ▒ ▒▓▒ ▒ ░
           ░      ░ ▒ ▒░ ░ ░▒ ▒░ ░ ░  ░░ ░░   ░ ▒░░ ░▒  ░ ░
         ░      ░ ░ ░ ▒  ░ ░░ ░    ░      ░   ░ ░ ░  ░  ░  
                    ░ ░  ░  ░      ░  ░         ░       ░ */
    function wd_ERC20(address tokenContract, uint8 ownerOption) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two approvals required");

        require(msg.sender != lastApprover && msg.sender != secondApprover, "Approvers cannot execute withdrawal");

        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner option");
        address recipient = ownerOption == 1 ? owner1 :
                            ownerOption == 2 ? owner2 :
                            ownerOption == 3 ? owner3 : owner4;

        IERC20 token = IERC20(tokenContract);
        uint256 amount = token.balanceOf(address(this));
        require(amount > 0, "No Tokens to Withdraw");

        uint256 fee = (amount * 1) / 100;
        uint256 finalAmount = amount - fee;

        token.safeTransfer(IPCSService, fee);
        token.safeTransfer(recipient, finalAmount);

        resetApproval();
        emit TokensWithdrawn(tokenContract, recipient, amount);
        }

    /* ——————————————————————————————— ERC-1155 Tokens

        ███▄ ▄███▓ █    ██  ██▓    ▄▄▄▓████▓ ██▓   ▄▄█████▓ ▒█████   ██ ▄█▀▓█████  ███▄    ██
       ▓██▒▀█▀ ██▒ ██  ▓██▒▓██▒    ▓  ██▒ ▓▒▓██▒   ▓  ██▒ ▓▒▒██▒  ██▒ ██▄█▒ ▓█   ▀  ██ ▀█   █
       ▓██    ▓██░▓██  ▒██░▒██░    ▒ ▓██░ ▒░▒██▒   ▒ ▓██░ ▒░▒██░  ██▒▓███▄░ ▒███   ▓██  ▀█ ██▒░ 
       ▒██    ▒██ ▓▓█  ░██░▒██░    ░ ▓██▓ ░ ░██░███░ ▓██▓ ░ ▒██   ██░▓██ █▄ ▒▓█  ▄ ▓██▒  ▐▌██▒ 
       ▒██▒   ░██▒▒▒█████▓ ░██████▒  ▒██▒ ░ ░██░     ▒██▒ ░ ░ ████▓▒░▒██▒ █▄░▒████▒▒██░   ▓██░
       ░ ▒░   ░  ░░▒▓▒ ▒ ▒ ░ ▒░▓  ░  ▒ ░░   ░▓       ▒ ░░   ░ ▒░▒░▒░ ▒ ▒▒ ▓▒░░ ▒░ ░░ ▒░   ▒ ▒ 
       ░  ░      ░░░▒░ ░ ░ ░ ░ ▒  ░    ░     ▒ ░       ░      ░ ▒ ▒░ ░ ░▒ ▒░ ░ ░  ░░ ░░   ░ ▒
       ░      ░    ░░░ ░ ░   ░ ░     ░       ▒ ░     ░      ░ ░ ░ ▒  ░ ░░ ░    ░      ░   ░ ░
              ░      ░         ░  ░          ░                  ░ ░  ░  ░      ░  ░         */
    function wd_ERC1155(address nftContract, uint8 ownerOption, uint256 tokenId, uint256 amount) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two approvals required");
        
        require(msg.sender != lastApprover && msg.sender != secondApprover, "Approvers cannot execute withdrawal");
        
        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner option");
        address recipient = ownerOption == 1 ? owner1 :
                            ownerOption == 2 ? owner2 :
                            ownerOption == 3 ? owner3 : owner4;

        IERC1155 token = IERC1155(nftContract);
        uint256 contractBalance = token.balanceOf(address(this), tokenId);
        require(contractBalance >= amount, 
        string(abi.encodePacked("Insufficient ERC1155 balance. Available: ", uint2str(contractBalance), " Requested: ", uint2str(amount)))
        );
        uint256 beforeBalance = token.balanceOf(recipient, tokenId);

        token.safeTransferFrom(address(this), recipient, tokenId, amount, "");

        uint256 afterBalance = token.balanceOf(recipient, tokenId);

        require(afterBalance == beforeBalance + amount, "Transfer failed");

        resetApproval();
        emit ERC1155Withdrawn(nftContract, recipient, tokenId, amount);
        }

    /* ——————————————————————————————— ERC-721 Tokens

        ███▄    █   █████▒▄▄▓█████▓ █  ██████ 
        ██ ▀█   █ ▓██   ▒ ▓  ██▒ ▓▒▒ ▒█    ▒ 
       ▓██  ▀█ ██▒▒████ ░ ▒ ▓██░ ▒░░  ▓██▄   
       ▓██▒  ▐▌██▒░▓█▒  ░ ░ ▓██▓ ░    ▒   ██▒
       ▒██░   ▓██░░▒█░      ▒██▒ ░   ▒██████▒▒
       ░ ▒░   ▒ ▒  ▒ ░      ▒ ░░     ▒ ▒▓▒ ▒ ░
       ░ ░░   ░ ▒░ ░          ░      ░ ░▒  ░ ░
          ░   ░ ░  ░ ░      ░        ░  ░  ░  
                ░                          ░*/
    function wd_ERC721(address nftContract, uint8 ownerOption, uint256 tokenId) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two approvals required");
        
        require(msg.sender != lastApprover && msg.sender != secondApprover, "Approvers cannot execute withdrawal");
        
        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner option");
        address recipient = ownerOption == 1 ? owner1 :
                            ownerOption == 2 ? owner2 :
                            ownerOption == 3 ? owner3 : owner4;

        require(IERC721(nftContract).ownerOf(tokenId) == address(this), "No NFT to Withdraw");

        IERC721(nftContract).safeTransferFrom(address(this), recipient, tokenId);

        require(IERC721(nftContract).ownerOf(tokenId) == recipient, "Transfer failed");

        resetApproval();
        emit NFTWithdrawn(nftContract, recipient, tokenId);
        }

    /* ——————————————————————————————— CON */

        function onERC721Received(address, address, uint256, bytes calldata) external pure override returns (bytes4) {
        return 0x150b7a02;
        }

        function onERC1155Received(address, address, uint256, uint256, bytes calldata) external pure override returns (bytes4) {
        return 0xf23a6e61;
        }

        function onERC1155BatchReceived(address, address, uint256[] calldata, uint256[] calldata, bytes calldata) external pure override returns (bytes4) {
        return 0xbc197c81;
        }

        function uint2str(uint256 _i) internal pure returns (string memory) {
        if (_i == 0) {
        return "0";
        }
        uint256 j = _i;
        uint256 length;
            while (j != 0) {
            length++;
            j /= 10;
            }
        bytes memory bstr = new bytes(length);
        uint256 k = length;
            while (_i != 0) {
            k = k - 1;
            bstr[k] = bytes1(uint8(48 + _i % 10));
            _i /= 10;
            }
        return string(bstr);
        }

        receive() external payable {
            emit EtherReceived(msg.sender, msg.value);
            }
        }
/* ——————————————————————————————— EOC.

     ██░ ██▓███   ▄████▄    ██████      
    ▓██▒▓██░  ██▒▒██▀ ▀█  ▒██    ▒      
    ▒██░░██░ ██▓▒▒▓█    ▄ ░ ▓██▄        
    ░██░▓██▄█▓▒ ▒▒▓▓▄ ▄██▒  ▒   ██▒     
    ░██░▒██▒ ░  ░▒ ▓███▀ ░▒██████▒▒ ██▓ 
    ░▓  ░▓▒░ ░  ░░ ░▒ ▒  ░▒ ▒▓▒ ▒ ░ ▒▓▒ 
     ▒ ░░▒ ░       ░  ▒   ░ ░▒  ░ ░ ░▒  
     ▒ ░░░       ░        ░  ░  ░   ░   
     ░           ░ ░            ░    ░  
                 ░                   ░         
    ©2025 intel port contract security - @prog <G7X7S1L3NT99X8Z>
    IPCSS charges a 1% fee on every withdrawal of ETH or ERC-20 Tokens. */