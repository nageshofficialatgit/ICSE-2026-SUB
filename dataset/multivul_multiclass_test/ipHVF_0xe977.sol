// SPDX-License-Identifier: MIT
pragma solidity ^0.8.29;

/* ————————————————————————————— File: sources of ipx/ipHVF-4-sig.sol

    IPHVF by IPCS > Intel Port Contract Security (EVM Design).
    @title ipHVF contract> Intel Port Horo-Vault Four Contract (A secure contract vault for securing your Assets with a time-lock).
        @author Ann Mandriana - <ann@intelport.org>
        @author Nouval Safaat - <nvlsft@intelport.org>
        @author Ryan Oktanda - <ryan@intelport.org>
        @author Dimas Fachri - <dimskuy@intelport.org>
        @matter > private use only.
        @param > ether / erc20 / erc721.

——————————————————————————————— The procedure to time-lock your Ether!

    Caution! <Please read and fully understand everything before proceeding. Use it at your own risk!>
        1. Before deploying, insert four owner addresses in [constructor] (must be ethereum personal wallet address EOA).

        2. After it is deployed, you can send ETH/ERC20 tokens/ERC721 NFTs directly to the ipHVT contract from anywhere you want <3

        3. Use the setTL function to set the time-lock for your assets. Enter the time in seconds (e.g., for 2 minutes, you must enter 120).
           WARNING!: After setting the time-lock, there is no way to withdraw or retrieve your assets from the ipHVT contract. You must wait until the time lock expires.
        
        4. Only the wallet address you register can withdraw anything from the ipHVT contract, so be careful and make sure not to lose your personal ethereum wallet address. 

        5. When it comes to withdrawals, only input 1, 2, 3, or 4 where 1 corresponds to Owner 1 and 2 corresponds to Owner 2.

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

contract ipHVF is ReentrancyGuard {
    using SafeERC20 for IERC20;

    address private owner1;
    address private owner2;
    address private owner3;
    address private owner4;
    uint256 public unlockTime;

    // IPCS Service
    address private IPCSService = 0x00060E123Fa9b8e33345b745626D1E2078992741; // <EOA only!> [Contract address are not allowed!]
    
    // IPHVF Version
    string public Version = "v7.2.5";

    event TimeLockSet(uint256 unlockTime);
    event EtherWithdrawn(address indexed recipient, uint256 amount);
    event TokensWithdrawn(address indexed token, address indexed recipient, uint256 amount);
    event NFTWithdrawn(address indexed nftContract, address indexed recipient, uint256 tokenId);
    event EtherReceived(address indexed sender, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner1 || msg.sender == owner2 || msg.sender == owner3 || msg.sender == owner4, "Only the owners can perform this action");
        _;
    }

    modifier whenUnlocked() {
        require(block.timestamp >= unlockTime, "Assets are still locked!");
        _;
    }

    constructor() {
        owner1 = 0x00060E123Fa9b8e33345b745626D1E2078992741; // EOA [1]
        owner2 = 0x22334496fF93D0509FaeD9C418F3A865A200C624; // EOA [2]
        owner3 = 0xBBb9900CF2890E97dD51381F83Bce7fCbDb48ce9; // EOA [3]
        owner4 = 0xCcdac8bCb9c1188D664ba475DeB552DF16B4350A; // EOA [4]
    }

    function setTL(uint256 durationAsSec) external onlyOwner {
        require(unlockTime == 0 || block.timestamp >= unlockTime, "Time lock already set, wait until unlocked");
        require(durationAsSec > 0 && durationAsSec < 157680000, "Invalid duration (max 5 years)");
        unlockTime = block.timestamp + durationAsSec;
        emit TimeLockSet(unlockTime);
    }

    function wdETH(uint8 ownerOption) external onlyOwner whenUnlocked nonReentrant {
        require(ownerOption == 1 || ownerOption == 2 || ownerOption == 3 || ownerOption == 4, "Invalid owner option");
        address payable recipient = ownerOption == 1 ? payable(owner1) : ownerOption == 2 ? payable(owner2) : ownerOption == 3 ? payable(owner3) : payable(owner4);
        uint256 amount = address(this).balance;
        require(amount > 0, "No Ether to Withdraw");

        uint256 ipcss = (amount * 2) / 100;
        uint256 amountAfterIpcss = amount - ipcss;

        (bool successIpcss, ) = IPCSService.call{value: ipcss}("");
        require(successIpcss, "IPCSS transfer failed");

        (bool successRecipient, ) = recipient.call{value: amountAfterIpcss}("");
        require(successRecipient, "IPCSS transfer failed");

        emit EtherWithdrawn(recipient, amount);
    }

    function wdERC20(address tokenContract, uint8 ownerOption) external onlyOwner whenUnlocked nonReentrant {
        require(tokenContract != address(0), "Invalid token contract address");
        require(ownerOption == 1 || ownerOption == 2 || ownerOption == 3 || ownerOption == 4, "Invalid owner option");
        address recipient = ownerOption == 1 ? owner1 : ownerOption == 2 ? owner2 : ownerOption == 3 ? owner3 : owner4;

        IERC20 token = IERC20(tokenContract);
        uint256 amount = token.balanceOf(address(this));
        require(amount > 0, "No Tokens to Withdraw");

        uint256 ipcss = (amount * 2) / 100;
        uint256 amountAfterIpcss = amount - ipcss;

        token.safeTransfer(IPCSService, ipcss);
        token.safeTransfer(recipient, amountAfterIpcss);

        emit TokensWithdrawn(tokenContract, recipient, amount);
    }

    function wdERC721(address nftContract, uint8 ownerOption, uint256 tokenId) external onlyOwner whenUnlocked nonReentrant {
        require(nftContract != address(0), "Invalid NFT contract address");
        require(ownerOption == 1 || ownerOption == 2 || ownerOption == 3 || ownerOption == 4, "Invalid owner option");
        require(IERC721(nftContract).ownerOf(tokenId) == address(this), "No NFT to Withdraw");
        address recipient = ownerOption == 1 ? owner1 : ownerOption == 2 ? owner2 : ownerOption == 3 ? owner3 : owner4;

        IERC721(nftContract).safeTransferFrom(address(this), recipient, tokenId);
        emit NFTWithdrawn(nftContract, recipient, tokenId);
    }

    receive() external payable {
        emit EtherReceived(msg.sender, msg.value);
    }
}

/* ——————————————————————————————— EOC
    ©2025 intel port contract security - @prog <4VJ389493U8G38>
    IPCSS charges a 2% fee for every withdrawal.
*/