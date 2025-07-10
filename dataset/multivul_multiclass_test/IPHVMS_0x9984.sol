// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

/* ————————————————————————————— File: sources of ipcs/ip-4m-sig-ts.sol

    ██████ ▄▄▄█████▓ ▄▄▄       ███▄    █ ▓█████▄  ██▓     ▒█████   ███▄    █ ▓█████ 
  ▒██    ▒ ▓  ██▒ ▓▒▒████▄     ██ ▀█   █ ▒██▀ ██▌▓██▒    ▒██▒  ██▒ ██ ▀█   █ ▓█   ▀ 
  ░ ▓██▄   ▒ ▓██░ ▒░▒██  ▀█▄  ▓██  ▀█ ██▒░██   █▌▒██░    ▒██░  ██▒▓██  ▀█ ██▒▒███   
    ▒   ██▒░ ▓██▓ ░ ░██▄▄▄▄██ ▓██▒  ▐▌██▒░▓█▄   ▌▒██░    ▒██   ██░▓██▒  ▐▌██▒▒▓█  ▄ 
  ▒██████▒▒  ▒██▒ ░  ▓█   ▓██▒▒██░   ▓██░░▒████▓ ░██████▒░ ████▓▒░▒██░   ▓██░░▒████▒
  ▒ ▒▓▒ ▒ ░  ▒ ░░    ▒▒   ▓▒█░░ ▒░   ▒ ▒  ▒▒▓  ▒ ░ ▒░▓  ░░ ▒░▒░▒░ ░ ▒░   ▒ ▒ ░░ ▒░ ░
  ░ ░▒  ░ ░    ░      ▒   ▒▒ ░░ ░░   ░ ▒░ ░ ▒  ▒ ░ ░ ▒  ░  ░ ▒ ▒░ ░ ░░   ░ ▒░ ░ ░  ░
     ░  ░    ░        ░   ▒      ░   ░ ░  ░ ░  ░   ░ ░   ░ ░ ░ ▒     ░   ░ ░    ░   
        ░                 ░  ░         ░    ░        ░  ░    ░             ░    ░  
                                          ░                                           
    IPHVMS by IPCS > Intel Port Contract Security [EVM Design].
    @title ipHVMs contract> Intel Port Horo-Vault Multi-Sig Contract, 
    [A contract vault designed to secure your assets with a time-lock].
      - @author Ann Mandriana - <ann@intelport.org>
      - @type > standlone.
      - @case > multi-sig, time-lock.
      - @supports > eth / erc-20 / erc-721 / erc-1155.
        ▒ ░       ░ ░   ▒         ░     ░ ▒         
        ░           ░             ▒  ░    ░          
    ——————————————————————————————— The procedure to time-lock your Assets!
    
  █▒░██▓     ▒█████   ▄████▄   ██ ▄█▀    ██▓▄▄▄▓████▓ ▐██▌ 
    ▓██▒    ▒██▒  ██▒▒██▀ ▀█   ██▄█▒    ▓██▒▓  ██▒ ▓▒ ▐██▌ 
    ▒██░    ▒██░  ██▒▒▓█    ▄ ▓███▄░    ▒██▒▒ ▓██░ ▒░ ▐██▌ 
    ▒██░    ▒██   ██░▒▓▓▄ ▄██▒▓██ █▄    ░██░░ ▓██▓ ░  ▓██▒ 
    ░██████▒░ ████▓▒░▒ ▓███▀ ░▒██▒ █▄   ░██░  ▒██▒ ░  ▒▄▄  
    ░ ▒░▓  ░░ ▒░▒░▒░ ░ ░▒ ▒  ░▒ ▒▒ ▓▒   ░▓    ▒ ░░    ░▀▀▒ 
    ░ ░ ▒  ░  ░ ▒ ▒░   ░  ▒   ░ ░▒ ▒░    ▒ ░    ░     ░  ░ 
      ░ ░   ░ ░ ░ ▒  ░        ░ ░░ ░     ▒ ░  ░          ░ 
        ░  ░    ░ ░  ░ ░      ░  ░       ░            ░    
                  ░                                   ▒  
   Caution! Please read and fully understand all details before proceeding.  
   Use at your own risk!  
   
   1. The ipHVMs contract is managed by four owners who registered it upon  
      deployment; it is not controlled by IPCS or IPCSS.  
      - ipHVMs has no backdoor access for IPCS or IPCSS, meaning governance  
        is entirely in the hands of the four original owners.  
      - Certain functions require at least two owners to execute, while others  
        require three. The fourth owner acts as a backup.  
      - ipHVMs operates as a multi-signature contract with a built-in time-lock  
        mechanism.  
      - WARNING! Always use an Ethereum personal wallet address (EOA) where you 
        control the private key. Contract addresses are not guaranteed to 
        function properly with the ipHVMs contract.  
    
   2. Use the [tlSet] function to configure the time-lock for your assets.  
      - You must enter the duration in seconds (e.g., 600 for 10 minutes).  
      - Another owner must first initiate the request by signing the [tlProp]  
        function before you can set the time-lock.  
      - WARNING! Once the time-lock is set, assets cannot be withdrawn or  
        retrieved until it expires.  
    
   3. Only registered wallet addresses can interact with your ipHVMs contract.  
      - Ensure you do not lose any of the registered ethereum personal wallets  
        address (EOAs), as they are required for all operations.  
    
   4. Withdrawals require at least three owner addresses:  
      - Two owners must first sign the withdrawal proposals using  
        [wdSignProp1] and [wdSignProp2].  
      - A third owner must then initiate the withdrawal.  
      - Withdrawals can only be sent to one of the four owner addresses.  
      - Each withdrawal proposal is valid for a single transaction.  
      - In the "ownerOption" field, input 1, 2, 3, or 4 to select the receiving  
        owner address.  
      - In the "amount" field:  
        - For [wdETH] and [wdERC20], amounts must be specified in wei/decimal.  
        - 1 ETH = 10^18 wei (e.g., 0.1 ETH = 100000000000000000).  
        - For ERC-20 tokens, decimals vary per token (e.g., IPCS Token has 6  
          decimals, so 100 IPCS = 100000000).
            ▒ ░         ░   ░         ░ ▒
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
    contract IPHVMS is ReentrancyGuard, IERC721Receiver, IERC1155Receiver {        
    using SafeERC20 for IERC20;

    address private owner1;
    address private owner2;
    address private owner3;
    address private owner4;
    uint256 public unlockTime;
    
    /// @notice IPHVMS Version
    string public constant vIPHVMS = "v0.0.8 Standlone";

    /// @notice Tip Jar Address <3
    address public tipJar = 0x77702b30a0276A4436BB688586147Ff75d64E97B;

    mapping(address => uint256) private lastGovResetTimestamp;
    uint256 private constant COOLDOWN_TIME = 2 hours;

    mapping(address => uint256) private lastGovProp0Timestamp;
    uint256 private constant GOVPROP0_COOLDOWN = 12 hours;

    function governList() external view returns (uint8[4] memory, address[4] memory) {
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
    bool private timeLockApproved;
    address private timeLockApprover;
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

    /// @notice Another owner must first propose > tlProp.
    /// @param durationAsSec The duration must be entered in seconds!.
    function tlSet(uint256 durationAsSec) external onlyOwner {
        require(timeLockApproved, "Time-lock proposal required, use another owner to call tlProp");
        require(msg.sender != timeLockApprover, "Operator must be different from proposal");
        require(block.timestamp >= unlockTime, "Cannot set a new time-lock until the current one expires");
        require(durationAsSec >= 600, "Minimum time-lock is 10 minutes");
        require(durationAsSec <= 157680000, "Invalid duration (max 5 years)");

        unlockTime = block.timestamp + durationAsSec;
    
        timeLockApproved = false;
        timeLockApprover = address(0);

        emit TimeLockSet(unlockTime);
        }    

    /// @notice Propose this action before setting the time-lock > tlSet.
    function tlProp() external onlyOwner {
        require(!timeLockApproved, "Time-lock already proposed");
        timeLockApproved = true;
        timeLockApprover = msg.sender;
        emit WithdrawalApproved(msg.sender);
        }    

    function govReset() external onlyOwner {
        require(block.timestamp >= lastGovResetTimestamp[msg.sender] + COOLDOWN_TIME, "Cooldown period not met"); 
        require(replacementProposed, "No active governance proposal to reset");

        replacementProposed = false;
        proposedNewOwner = address(0);
        ownerToReplace = 0;

        replacementApproved1 = false;
        replacementApprover1 = address(0);
        lastGovResetTimestamp[msg.sender] = block.timestamp;
        }

    /// @notice Sign this proposal and > wdSignProp2 to allow withdrawals. The signer proposals must come from different owners.
    function wdSignProp1() external onlyOwner whenUnlocked {
        require(!withdrawalApproved, "Already proposed once");
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

    /// @notice Another owner must first sign proposal > wdSignProp1 before you can sign this proposal.
    function wdSignProp2() external onlyOwner whenUnlocked {
        require(withdrawalApproved, "First proposal required");
        require(msg.sender != lastApprover, "Second proposal must be from a different owner");
        require(!withdrawalApproved2, "Already proposed twice");
        
        withdrawalApproved2 = true;
        secondApprover = msg.sender;
        emit WithdrawalApproved(msg.sender);
        }

    /// @notice Propose a change to replace an owner address.
    /// @param newOwner Input the new owner > ethereum personal wallet address.
    /// @param ownerNumber Enter the owner number that you wish to replace.
    function govProp0(address newOwner, uint8 ownerNumber) external onlyOwner {
        require(block.timestamp >= lastGovProp0Timestamp[msg.sender] + GOVPROP0_COOLDOWN, "Must wait 12 hours before proposing again");
        require(!replacementProposed, "A replacement is already proposed");
        require(ownerNumber >= 1 && ownerNumber <= 4, "Invalid owner number (must be 1-4)");
        require(newOwner != address(0), "New owner cannot be zero address");
        require(newOwner != owner1 && newOwner != owner2 && newOwner != owner3 && newOwner != owner4, "New owner must not already be an owner");

        proposedNewOwner = newOwner;
        ownerToReplace = ownerNumber;
        replacementProposed = true;

        replacementApproved1 = false;
        replacementApproved2 = false;
        replacementApprover1 = address(0);
        replacementApprover2 = address(0);
        lastGovProp0Timestamp[msg.sender] = block.timestamp;
        } 

    /// @notice You must first submit proposal > govProp0 before you can sign this proposal.
    function govSignProp1() external onlyOwner {
        require(replacementProposed, "No replacement proposed yet");
        require(!replacementApproved1, "First proposal already done");
    
        replacementApproved1 = true;
        replacementApprover1 = msg.sender;
        emit WithdrawalApproved(msg.sender);
        }

    /// @notice Another owner must first sign proposal > govSignProp1.
    function govSignProp2() external onlyOwner {
        require(replacementProposed, "No replacement proposed yet");
        require(replacementApproved1, "First proposal required");
        require(!replacementApproved2, "Second proposal already done");
        require(msg.sender != replacementApprover1, "Second proposal must be from a different owner");

        replacementApproved2 = true;
        replacementApprover2 = msg.sender;
        emit WithdrawalApproved(msg.sender);
        }

    /// @notice If another owner has proposed > govProp0, govSignProp1, and govSignProp2, you can then operate this action to replace the owner.
    function govPropSet() external onlyOwner {
        require(replacementProposed, "No replacement proposed yet");
        require(replacementApproved1 && replacementApproved2, "Two proposals required");
        require(msg.sender != replacementApprover1 && msg.sender != replacementApprover2, "Operator must be different from proposers");

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
       ▒███   ▒ ▓██▓ ▒░▒██▀▀██░▒███   ▓██ ░▄█ ▒
       ▒▓█  ▄ ░ ▓██▓ ░ ░▓█ ░██ ▒▓█  ▄ ▒██▀▀█▄  
       ░▒████▒  ▒██▒ ░ ░▓█▒░██▓░▒████▒░██▓ ▒██▒
       ░░ ▒░ ░  ▒ ░░    ▒ ░░▒░▒░░ ▒░ ░░ ▒▓ ░▒▓░
        ░ ░  ░    ░     ▒ ░▒░ ░ ░ ░  ░  ░▒ ░ ▒░
          ░     ░       ░  ░░ ░   ░     ░░   ░ 
          ░  ░          ░  ░  ░   ░  ░   ░   */
    /// @notice To withdraw your ETH, two other owners must first sign the withdrawal proposal by calling > wdSignProp1 and > wdSignProp2.
    /// @param ownerOption Enter the owner number who will receive the funds.
    /// @param amount Enter the amount in decimal wei (10^18). For example, 1000000000000000000 represents 1 ETH, as ETH has 18 decimal places.
    function wdETH(uint8 ownerOption, uint256 amount) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two proposals required");

        require(msg.sender != lastApprover && msg.sender != secondApprover, "Proposers cannot operate withdrawal");

        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner number (must be 1-4)");
        address payable recipient = ownerOption == 1 ? payable(owner1) :
                                    ownerOption == 2 ? payable(owner2) :
                                    ownerOption == 3 ? payable(owner3) : payable(owner4);

        require(address(this).balance >= amount, "Not enough ETH");
        require(amount > 0, "No Ether to Withdraw");

        (bool success, ) = recipient.call{value: amount}("");
        require(success, "ETH transfer failed");

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
                      ░     ░      ░  ░         ░       ░ */
    /// @notice To withdraw your ERC-20 tokens, two other owners must first sign the withdrawal proposal by calling > wdSignProp1 and > wdSignProp2.
    /// @param tokenContract Input the ERC-20 token contract address you wish to withdraw from.
    /// @param ownerOption Enter the owner number who will receive the funds.
    /// @param amount Enter the amount in decimal format. For example, 100000000 represents 100 IPCS Token, as IPCS Token has 6 decimal places.
    function wdERC20(address tokenContract, uint8 ownerOption, uint256 amount) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two proposals required");

        require(msg.sender != lastApprover && msg.sender != secondApprover, "Proposers cannot operate withdrawal");

        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner number (must be 1-4)");
        address recipient = ownerOption == 1 ? owner1 :
                            ownerOption == 2 ? owner2 :
                            ownerOption == 3 ? owner3 : owner4;

        IERC20 token = IERC20(tokenContract);
        require(token.balanceOf(address(this)) >= amount, "Not enough tokens");
        require(amount > 0, "No Tokens to Withdraw");

        token.safeTransfer(recipient, amount);

        resetApproval();
        emit TokensWithdrawn(tokenContract, recipient, amount);
        }

    /* ——————————————————————————————— ERC-1155 Multi-Token

        ███▄ ▄███▓ █    ██  ██▓    ▄▄▄▓████▓ ██▓   ▄▄█████▓ ▒█████   ██ ▄█▀▓█████  ███▄    ██
       ▓██▒▀█▀ ██▒ ██  ▓██▒▓██▒    ▓  ██▒ ▓▒▓██▒   ▓  ██▒ ▓▒▒██▒  ██▒ ██▄█▒ ▓█   ▀  ██ ▀█   █
       ▓██    ▓██░▓██  ▒██░▒██░    ▒ ▓██░ ▒░▒██▒   ▒ ▓██░ ▒░▒██░  ██▒▓███▄░ ▒███   ▓██  ▀█ ██▒░ 
       ▒██    ▒██ ▓▓█  ░██░▒██░    ░ ▓██▓ ░ ░██░███░ ▓██▓ ░ ▒██   ██░▓██ █▄ ▒▓█  ▄ ▓██▒  ▐▌██▒ 
       ▒██▒   ░██▒▒▒█████▓ ░██████▒  ▒██▒ ░ ░██░     ▒██▒ ░ ░ ████▓▒░▒██▒ █▄░▒████▒▒██░   ▓██░
       ░ ▒░   ░  ░░▒▓▒ ▒ ▒ ░ ▒░▓  ░  ▒ ░░   ░▓       ▒ ░░   ░ ▒░▒░▒░ ▒ ▒▒ ▓▒░░ ▒░ ░░ ▒░   ▒ ▒ 
       ░  ░      ░░░▒░ ░ ░ ░ ░ ▒  ░    ░     ▒ ░       ░      ░ ▒ ▒░ ░ ░▒ ▒░ ░ ░  ░░ ░░   ░ ▒
       ░      ░    ░░░ ░ ░   ░ ░     ░       ▒ ░     ░      ░ ░ ░ ▒  ░ ░░ ░    ░      ░   ░ ░
              ░      ░         ░  ░          ░                  ░ ░  ░  ░      ░  ░         */
    /// @notice To withdraw your ERC-1155 multi-token, two other owners must first sign the withdrawal proposal by calling > wdSignProp1 and > wdSignProp2.
    /// @param nftContract Input the ERC-1155 token contract address you wish to withdraw from.
    /// @param ownerOption Enter the owner number who will receive the ERC-1155 Multi-Token.
    /// @param tokenId Input the tokenId number you wish to withdraw.
    /// @param amount Enter the amount of tokens you wish to withdraw.
    function wdERC1155(address nftContract, uint8 ownerOption, uint256 tokenId, uint256 amount) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two proposals required");
        
        require(msg.sender != lastApprover && msg.sender != secondApprover, "Proposers cannot operate withdrawal");
        
        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner number (must be 1-4)");
        address recipient = ownerOption == 1 ? owner1 :
                            ownerOption == 2 ? owner2 :
                            ownerOption == 3 ? owner3 : owner4;

        IERC1155 token = IERC1155(nftContract);
        uint256 contractBalance = token.balanceOf(address(this), tokenId);
        require(contractBalance >= amount, 
        string(abi.encodePacked("Insufficient ERC1155 balance. Available: ", uint2Str(contractBalance), " Requested: ", uint2Str(amount)))
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
       ░ ░░   ░ ▒░ ░          ░        ░▒  ░ ░
          ░   ░ ░  ░ ░      ░        ░  ░  ░  
                ░                          ░*/
    /// @notice To withdraw your ERC-721 NFTs, two other owners must first sign the withdrawal proposal by calling > wdSignProp1 and > wdSignProp2.
    /// @param nftContract Input the ERC-721 NFT contract address you wish to withdraw from.
    /// @param ownerOption Enter the owner number who will receive the NFTs.
    /// @param tokenId Input the tokenId number you wish to withdraw.
    function wdERC721(address nftContract, uint8 ownerOption, uint256 tokenId) external onlyOwner whenUnlocked nonReentrant {
        require(withdrawalApproved && withdrawalApproved2, "Two proposals required");
        
        require(msg.sender != lastApprover && msg.sender != secondApprover, "Proposers cannot operate withdrawal");
        
        require(ownerOption >= 1 && ownerOption <= 4, "Invalid owner number (must be 1-4)");
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

        function uint2Str(uint256 _i) internal pure returns (string memory) {
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
     ▒ ░░▒ ░       ░  ▒   ░ ░▒  ░ ░ ░  
     ▒   ░       ░        ░  ░    ▒ ░   
     ░           ░ ░            ░ ░  
                 ░                ░
      ██████  ██▓ ██▓    ▓█████  ███▄    █ ▄▄▄█████▓     ▄████  ███▄    █  ██▓ ███▄    █ ▓█████ 
    ▒██    ▒ ▓██▒▓██▒    ▓█   ▀  ██ ▀█   █ ▓  ██▒ ▓▒    ██▒ ▀█▒ ██ ▀█   █ ▓██▒ ██ ▀█   █ ▓█   ▀ 
    ░ ▓██▄   ▒██▒▒██░    ▒███   ▓██  ▀█ ██▒▒ ▓██░ ▒░   ▒██░▄▄▄░▓██  ▀█ ██▒▒██▒▓██  ▀█ ██▒▒███   
      ▒   ██▒░██░▒██░    ▒▓█  ▄ ▓██▒  ▐▌██▒░ ▓██▓ ░    ░▓█  ██▓▓██▒  ▐▌██▒░██░▓██▒  ▐▌██▒▒▓█  ▄ 
    ▒██████▒▒░██░░██████▒░▒████▒▒██░   ▓██░  ▒██▒ ░    ░▒▓███▀▒▒██░   ▓██░░██░▒██░   ▓██░░▒████▒
    ▒ ▒▓▒ ▒ ░░▓  ░ ▒░▓  ░░░ ▒░ ░░ ▒░   ▒ ▒   ▒ ░░       ░▒   ▒ ░ ▒░   ▒ ▒ ░▓  ░ ▒░   ▒ ▒ ░░ ▒░ ░
    ░ ░▒  ░ ░ ▒ ░░ ░ ▒  ░ ░ ░  ░░ ░░   ░ ▒░    ░         ░   ░ ░ ░░   ░ ▒░ ▒ ░░ ░░   ░ ▒░ ░ ░  ░
    ░  ░  ░   ▒ ░  ░ ░      ░      ░   ░ ░   ░         ░ ░   ░    ░   ░ ░  ▒ ░   ░   ░ ░    ░   
          ░   ░      ░  ░   ░  ░         ░                   ░          ░  ░           ░    ░ 
    intel port contract security - @prog <G9S1L3NT9900910000> */