// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract WarOnCancerCoin {
    string public name = "WarOnCancerCoin";
    string public symbol = "WCC";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;

    // St. Jude Children's Hospital ETH address
    address public constant ST_JUDE_ADDRESS = 0x2517E08BA74bBE750A00439Ea298b589c248A609;
    // Your ETH address
    address public constant OWNER_ADDRESS = 0xB51594AB8F098048DfB96dee7C533c806553E660;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Donation(address indexed from, uint256 value);

    constructor(uint256 initialSupply) {
        totalSupply = initialSupply * 10 ** uint256(decimals);
        balanceOf[OWNER_ADDRESS] = totalSupply;
    }

    function transfer(address to, uint256 value) public payable returns (bool success) {
        require(to != address(0), "Invalid recipient address");
        require(balanceOf[msg.sender] >= value, "Insufficient WCC balance");
        
        // Calculate 60% of the ETH sent with the transaction (if any)
        uint256 donationAmount = (msg.value * 60) / 100;
        uint256 senderAmount = msg.value - donationAmount;

        // If ETH is sent with the transaction, handle the split
        if (msg.value > 0) {
            // Send 60% to St. Jude
            (bool sentToStJude, ) = ST_JUDE_ADDRESS.call{value: donationAmount}("");
            require(sentToStJude, "Failed to send ETH to St. Jude");

            // Send remaining 40% back to you (owner)
            (bool sentToOwner, ) = OWNER_ADDRESS.call{value: senderAmount}("");
            require(sentToOwner, "Failed to send ETH to owner");

            emit Donation(msg.sender, donationAmount);
        }

        // Perform the WCC token transfer
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        
        return true;
    }

    // Function to check contract's ETH balance (for testing)
    function getContractBalance() public view returns (uint256) {
        return address(this).balance;
    }

    // Allow owner to withdraw any stuck tokens (safety feature)
    function withdrawTokens(uint256 amount) public {
        require(msg.sender == OWNER_ADDRESS, "Only owner can withdraw");
        require(balanceOf[address(this)] >= amount, "Insufficient contract balance");
        balanceOf[address(this)] -= amount;
        balanceOf[OWNER_ADDRESS] += amount;
        emit Transfer(address(this), OWNER_ADDRESS, amount);
    }
}