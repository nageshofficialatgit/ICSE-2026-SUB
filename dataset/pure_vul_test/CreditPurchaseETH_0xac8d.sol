// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

contract CreditPurchaseETH {
    address public owner;
    uint256 public creditPriceInWei;
    
    mapping(string => address) public referralPartners;
    mapping(string => bool) public activeReferralCodes;
    
    event CreditsPurchased(
        address indexed buyer, 
        uint256 amountPaid, 
        uint256 creditsReceived, 
        string referralCode
    );
    event ReferralPartnerAdded(string code, address partner);
    event ReferralPartnerRemoved(string code);

    constructor(uint256 _creditPriceInWei) {
        owner = msg.sender;
        creditPriceInWei = _creditPriceInWei;
    }

    function setOwner(address _new) external {
        require(_new!= address(0), "Invalid address");
        require(msg.sender == owner, "Not authorised");
        owner = _new;
    }

    function purchaseCredits(string memory referralCode) external payable {
        require(msg.value > 0, "No ETH sent");
        
        uint256 credits = msg.value / creditPriceInWei;
        
        // Add 10% bonus credits if using valid referral code
        if (activeReferralCodes[referralCode]) {
            credits += (credits * 10) / 100;
        }
        
        if (bytes(referralCode).length > 0) {
            require(activeReferralCodes[referralCode], "Invalid referral code");
        }
        
        emit CreditsPurchased(msg.sender, msg.value, credits, referralCode);
        
        // If valid referral code, split payment
        if (activeReferralCodes[referralCode]) {
            uint256 referralShare = (msg.value * 10) / 100; // 10% to referrer
            uint256 ownerShare = msg.value - referralShare;
            
            payable(referralPartners[referralCode]).transfer(referralShare);
            payable(owner).transfer(ownerShare);
        } else {
            payable(owner).transfer(msg.value);
        }
    }

    function addReferralPartner(string memory code, address partner) external {
        require(msg.sender == owner, "ERR: Not Owner");
        require(partner != address(0), "Invalid partner address");
        require(!activeReferralCodes[code], "Code already exists");
        
        referralPartners[code] = partner;
        activeReferralCodes[code] = true;
        
        emit ReferralPartnerAdded(code, partner);
    }

    function removeReferralPartner(string memory code) external {
        require(msg.sender == owner, "ERR: Not Owner");
        require(activeReferralCodes[code], "Code doesn't exist");
        
        delete referralPartners[code];
        activeReferralCodes[code] = false;
        
        emit ReferralPartnerRemoved(code);
    }

    function changeCreditCost(uint256 newCost) external {
        require(msg.sender == owner, "ERR: Not Owner");
        creditPriceInWei = newCost;
    }

    function destroy() external {
        require(msg.sender == owner, "ERR: Not Owner");
        selfdestruct(payable(owner));
    }
}