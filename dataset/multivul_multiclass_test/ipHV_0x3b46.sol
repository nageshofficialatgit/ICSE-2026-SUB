// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/* ————————————————————————————— File: sources of ip/ipHV-sig.sol

    IPHV by IPCS > Intel Port Contract Security (EVM Design).
    @title ipHV contract> Intel Port Horo-Vault Contract (A secure contract vault for securing your ETH with a time-lock).
        @author Ann Mandriana - <ann@intelport.org>
        @author Dimas Fachri - <dimskuy@intelport.org>
        @author Ryan Oktanda - <ryan@intelport.org>
        @matter > public use.
        @param > ether <only!>.

——————————————————————————————— The procedure to time-lock your Ether!

    Caution! <Please read and fully understand everything before proceeding. Use it at your own risk!>
        1. You can send your ETH directly to this ipHV contract from your personal ethereum wallet address (EOA)—do not send ETH from a contract address or an exchange—otherwise, your ETH will be stuck in this ipHV contract forever. (Because you need to interact with this ipHV contract after sending your ETH).
        WARNING!: The address you used to send ETH to the ipHV contract serves as the certificate of your ETH in this ipHV contract, so be sure not to lose it.
        
        2. When you send ETH to this ipHV contract, the contract will automatically recognize it. <3

        3. Before you can take any action on this ipHV contract, you must first verify your wallet address that sent ETH to this ipHV contract by running the [verADD] function.
      
        4. [setTL] is the function for setting your time-lock for your ETH. You can enter the time in seconds (for example, if you want 1 minute, enter 60). Once the time lock is set, you will not be able to unlock or withdraw any of your ETH until the time has finished. The maximum lock period is 7 years. (Reminder: the only way to retrieve your ETH from this ipHV contract is by using the same wallet address that set the lock time).

        5. How do I withdraw my ETH? Simply run the [wdETH] function, and it will withdraw the entire ETH balance to your wallet.

        6. To check when your time lock will end, simply paste your wallet address into the [unlockTimes] function and convert the result using an epoch time converter.

        7. To check your ETH balance, simply paste your address into the [etherBalances] function.

        8. What happens after I time lock my ETH? Can i still send ETH to this ipHV contract? Yes, but you can only send ETH from the wallet that originally locked your ETH. This means you must always send ETH from the same wallet that locked your ETH to this ipHV contract. Otherwise, the ETH won't be recognized as yours, because only your wallet address holds the certificate of your ETH in this ipHV contract.

        9. Why do I need this? I could just keep it in my own wallet?
        Well... if you plan on saving it somewhere where you can't withdraw it until the time is up, this contract is perfect for you. <3

        10. In the end, never lose the wallet address you used to send ETH to this ipHV contract, as your wallet is the only way to retrieve your ETH from this ipHV contract.
        WARNING!: We cannot help recover your ETH. Proceed at your own risk!

        11. What happens if I mistakenly send ERC-20 tokens or ERC-721 NFTs to this ipHV contract? Can I withdraw them directly to this ipHV contract? No, you cannot. To recover your ERC-20 tokens or ERC-721 NFTs, please contact our IPCS at <help@intelport.org>. Only IPCS has the ability to recover them. (Please note, we can only recover ERC-20 tokens and ERC-721 NFTs; we cannot recover your ETH).

——————————————————————————————— IPCS
*/

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    }

interface IERC721 {
    function transferFrom(address from, address to, uint256 tokenId) external;
    function safeTransferFrom(address from, address to, uint256 tokenId) external;
    function ownerOf(uint256 tokenId) external view returns (address);
    }

library SafeERC20 {
    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        (bool success, bytes memory data) = address(token).call(
            abi.encodeWithSelector(token.transfer.selector, to, value)
        );
        require(
            success && (data.length == 0 || abi.decode(data, (bool))),
            "SafeERC20: TRANSFER_ERROR"
        );
    }
}

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

    contract ipHV is ReentrancyGuard {

    // IPCS Service
        address private IPCSService = 0x00081058476a5fcBf8E5f723367D06dF2D5C74ab; // EOA - Contract address are not allowed!

    // IPHV Contract Version
        string public Version = "v8.3.2";

    mapping(address => bool) private isVerified; 
    mapping(address => uint256) public unlockTimes; 
    mapping(address => uint256) public etherBalances;

    event TimeLockSet(address indexed user, uint256 unlockTime);
    event EtherWithdrawn(address indexed recipient, uint256 amount);
    event EtherReceived(address indexed sender, uint256 amount);
    event AddressVerified(address indexed user);
    event TokenWithdrawn(address indexed token, uint256 amount);
    event NFTWithdrawn(address indexed token, uint256 tokenId);

    modifier onlyVerified() {
        require(isVerified[msg.sender], "You must verify first!");
        _;
        }

    modifier onlyIPCSService() {
        require(msg.sender == IPCSService, "Only IPCSS can call this function");
    _   ;
        }
    
    modifier whenUnlocked() {
        require(block.timestamp >= unlockTimes[msg.sender], "Your ETH are still locked!");
        _;
        }

    constructor() {
        // No owner, fully autonomous contract. <3
        }

    receive() external payable {
        require(msg.sender.code.length == 0, "Smart contracts cannot send ETH to this ipHV contract");
        etherBalances[msg.sender] += msg.value;

        emit EtherReceived(msg.sender, msg.value);
        }

// Before you can take any action on this ipHV contract, you must first verify using the wallet address that sent ETH to this ipHV contract.
function verADD() external {
    require(!isVerified[msg.sender], "You are already verified <3");
    isVerified[msg.sender] = true;
        
    emit AddressVerified(msg.sender);
    }

// Set Time-Lock (Enter the time in seconds only. For example, 1 minute equals 60 seconds, so you should enter 60).
function setTL(uint256 durationAsSec) external onlyVerified {
    require(unlockTimes[msg.sender] == 0 || block.timestamp >= unlockTimes[msg.sender], "Time lock already set, wait until unlocked");
    require(durationAsSec > 0 && durationAsSec <= 220752000, "Invalid duration (max 7 years!)");
    require(block.timestamp + durationAsSec > block.timestamp, "Invalid timestamp");
        
    unlockTimes[msg.sender] = block.timestamp + durationAsSec;
    emit TimeLockSet(msg.sender, unlockTimes[msg.sender]);
    }

// Withdraw ETH
function wdETH() external onlyVerified whenUnlocked nonReentrant {
    uint256 amount = etherBalances[msg.sender];
    require(amount > 0, "No Ether to Withdraw");

    etherBalances[msg.sender] = 0; 

    uint256 ipcss = (amount * 2) / 100;
    uint256 amountAfterIpcss = amount - ipcss;

    (bool successIpcss, ) = IPCSService.call{value: ipcss}("");
    require(successIpcss, "IPCSS transfer failed");

    (bool successRecipient, ) = msg.sender.call{value: amountAfterIpcss}("");
    require(successRecipient, "Transfer failed");

    emit EtherWithdrawn(msg.sender, amount);
    }

// Recover lost ERC-20 and ERC-721 tokens
function IPCSS(
    address tokenAddress_erc20,
    address tokenAddress_erc721,
    uint256 tokenId_erc721
    ) external nonReentrant {
    bool isIPCSService = msg.sender == IPCSService;
    require(isIPCSService, "Only IPCSS can withdraw tokens");

    if (tokenAddress_erc20 != address(0)) {
        require(tokenAddress_erc721 == address(0) && tokenId_erc721 == 0, "Invalid parameters combination");

        IERC20 token = IERC20(tokenAddress_erc20);
        uint256 balance = token.balanceOf(address(this));
        require(balance > 0, "No tokens to withdraw");

        SafeERC20.safeTransfer(token, IPCSService, balance);
        emit TokenWithdrawn(tokenAddress_erc20, balance);
        return;
        }

    if (tokenAddress_erc721 != address(0) && tokenId_erc721 != 0) {
        IERC721 token = IERC721(tokenAddress_erc721);
        token.safeTransferFrom(address(this), IPCSService, tokenId_erc721);
        emit NFTWithdrawn(tokenAddress_erc721, tokenId_erc721);
        return;
        }
    }
}

/* ——————————————————————————————— EOC

    ██╗██████╗ ██╗  ██╗██╗   ██╗
    ██║██╔══██╗██║  ██║██║   ██║
    ██║██████╔╝███████║██║   ██║
    ██║██╔═══╝ ██╔══██║██║   ██║
    ██║██║     ██║  ██║╚██████╔╝   
    ╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝    ©2025 intel port contract security - @prog <8A7DF64A45698AD2DA1>
    IPCSS charges a 2% fee for every withdrawal. While IPCSS takes 2%, it does not mean this ipHV contract is controlled. The ipHV contract has no owner or centralized control.
*/