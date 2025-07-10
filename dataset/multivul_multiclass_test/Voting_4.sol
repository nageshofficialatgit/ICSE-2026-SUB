// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract VotingSystem4 {
    struct Candidate {
        string name;
        uint256 votes;
    }
    Candidate[] public candidates;
    mapping(address => bool) public hasVoted;

    constructor(string[] memory candidateNames) {
        for (uint i = 0; i < candidateNames.length; i++) {
            candidates.push(Candidate(candidateNames[i], 0));
        }
    }

    function vote(uint candidateIndex) external {
        require(!hasVoted[msg.sender], "Already voted");
        candidates[candidateIndex].votes += 1;
        hasVoted[msg.sender] = true;
    }
}
