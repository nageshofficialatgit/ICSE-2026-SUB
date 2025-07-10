// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract GestionFonds {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    // 📌 Vérifie le solde des tokens ERC20 détenus par le contrat
    function soldeToken(address tokenAddress) public view returns (uint256) {
        return IERC20(tokenAddress).balanceOf(address(this));
    }

    // 📌 Retirer des tokens ERC20 vers Metamask (wallet owner)
    function retirerToken(address tokenAddress, uint montant) public {
        require(msg.sender == owner, "Seul le proprietaire peut retirer");
        require(montant > 0, "Le montant doit etre superieur a 0");

        IERC20 token = IERC20(tokenAddress);
        require(token.balanceOf(address(this)) >= montant, "Solde insuffisant en tokens"); // ✅ Ajout de cette vérification
        require(token.transfer(owner, montant), "Echec du transfert");
    }
}