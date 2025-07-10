// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PaymentSplitter {
    address[] public recipients;
    uint256[] public shares;
    
    /**
     * @dev Initialize the contract with recipients and their respective shares
     * @param _recipients Array of recipient addresses
     * @param _shares Array of shares (in basis points, where 10000 = 100%)
     */
    constructor(address[] memory _recipients, uint256[] memory _shares) {
        require(_recipients.length == _shares.length, "Recipients and shares length mismatch");
        require(_recipients.length > 0, "No recipients specified");
        
        uint256 totalShares;
        for (uint256 i = 0; i < _shares.length; i++) {
            require(_shares[i] > 0, "Share cannot be zero");
            totalShares += _shares[i];
            recipients.push(_recipients[i]);
            shares.push(_shares[i]);
        }
        
        require(totalShares == 10000, "Total shares must equal 10000 (100%)");
    }
    
    /**
     * @dev Splits the received ETH among recipients according to their shares
     */
    receive() external payable {
        require(msg.value > 0, "No ETH received");
        
        uint256 totalSent;
        for (uint256 i = 0; i < recipients.length; i++) {
            uint256 amount = (msg.value * shares[i]) / 10000;
            if (i == recipients.length - 1) {
                // Handle rounding errors by sending remaining balance to last recipient
                amount = address(this).balance;
            }
            payable(recipients[i]).transfer(amount);
            totalSent += amount;
        }
    }
    
    /**
     * @dev Returns the number of recipients
     */
    function recipientCount() public view returns (uint256) {
        return recipients.length;
    }
}