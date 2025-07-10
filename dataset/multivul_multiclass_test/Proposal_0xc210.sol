// SPDX-License-Identifier: MIT
pragma solidity >=0.5.17;

contract Proposal {

    address internal constant gFAddress = 0x8f251a25255CA200598064fC3470C441d39545b1;
    string internal proposalMsg = "I knew I wanted to get to know you before I even talked to you, just from the way you talked to others. And even back then, when I didn't know what to expect, I certainly couldn't have predicted falling for you so completely. You melted my walls and helped me open up to people in ways I'd previously wished I could, but didn't know how to do. You loved me without worry about reciprocation and without expectations. I've loved you for so long, and now that we're able to fully embrace our connection, I want to be with you in every feasible way - now and always. Will you marry me?";
    string internal responseMsg;
    bool internal responded;
    
    //Only target address can change the response message.
    modifier onlyGF() {
        require(msg.sender == gFAddress);
        _;
    }

    // Function to respond to the proposal
    function respond(string memory response) external onlyGF {
        require(responded==false);
        responseMsg = response;
        responded=true;
    }
    
    function getProposalMsg() external view returns (string memory) {
        return proposalMsg;
    }

    function getResponse() external view returns (string memory) {
        return responseMsg;
    }
}