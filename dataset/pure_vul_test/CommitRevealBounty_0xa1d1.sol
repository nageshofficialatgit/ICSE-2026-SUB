// SPDX-License-Identifier: MIT
// License: OPNL and OPNL-2
// Dominic Dabish and Andrew Magdy Kamal
// Code Research Lab and Riemann Computing
pragma solidity ^0.8.0;

contract CommitRevealBounty {
    struct Bounty {
        bytes32 hash;
        uint256 amount;
        address asker;
        bool solved;
        address solver;
    }
    
    struct Commitment {
        bytes32 committedHash;
        address solver;
    }

    mapping(bytes32 => Bounty) public bounties; // Hash -> Bounty
    mapping(address => Commitment) public commitments; // Solver -> Commitment

    event BountyPosted(bytes32 indexed hash, address indexed asker, uint256 amount);
    event SolutionCommitted(address indexed solver, bytes32 committedHash);
    event SolutionRevealed(bytes32 indexed hash, address indexed solver, string solution, uint256 reward);

    // Askers post a bounty with a hash and an ETH reward
    function postBounty(bytes32 _hash) external payable {
        require(msg.value > 0, "Must send ETH as a bounty");
        require(bounties[_hash].amount == 0, "Bounty already exists");

        bounties[_hash] = Bounty({
            hash: _hash,
            amount: msg.value,
            asker: msg.sender,
            solved: false,
            solver: address(0)
        });

        emit BountyPosted(_hash, msg.sender, msg.value);
    }

    // Solvers commit to a solution without revealing it yet
    function commitSolution(bytes32 _committedHash) external {
        require(commitments[msg.sender].committedHash == 0, "Already committed");
        commitments[msg.sender] = Commitment({
            committedHash: _committedHash,
            solver: msg.sender
        });
        emit SolutionCommitted(msg.sender, _committedHash);
    }

    // Solvers reveal their solution and claim the bounty if correct
    function revealSolution(string calldata _solution) external {
        bytes32 computedHash = keccak256(abi.encodePacked(_solution));
        Commitment storage commitment = commitments[msg.sender];
        Bounty storage bounty = bounties[computedHash];

        require(commitment.solver == msg.sender, "No commitment found");
        require(keccak256(abi.encodePacked(_solution)) == commitment.committedHash, "Commitment mismatch");
        require(bounty.amount > 0, "No bounty for this hash");
        require(!bounty.solved, "Bounty already solved");

        bounty.solved = true;
        bounty.solver = msg.sender;
        uint256 reward = bounty.amount;
        bounty.amount = 0; // Prevent reentrancy
        
        payable(msg.sender).transfer(reward);

        emit SolutionRevealed(computedHash, msg.sender, _solution, reward);
    }
}