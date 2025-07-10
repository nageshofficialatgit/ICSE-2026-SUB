// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
}

contract GestionFonds {
    address public owner;

    event OwnerChanged(address indexed oldOwner, address indexed newOwner);
    event DelegateCallExecuted(address indexed target, bytes data, bool success);
    event TokensRecuperes(address indexed token, address indexed source, uint256 montant);

    constructor() {
        owner = msg.sender;
    }

    modifier uniquementOwner() {
        require(msg.sender == owner, "Seul le proprietaire peut modifier");
        _;
    }

    // 📌 Modifier l'owner du contrat
    function modifierOwner(address _nouvelOwner) public uniquementOwner {
        require(_nouvelOwner != address(0), "Nouvel owner invalide");
        emit OwnerChanged(owner, _nouvelOwner);
        owner = _nouvelOwner;
    }

    // 📌 Retirer des tokens ERC20 vers une **adresse de wallet specifique**
    function retirerToken(address tokenAddress, address recipient, uint montant) public uniquementOwner {
        require(recipient != address(0), "Adresse invalide");
        require(montant > 0, "Le montant doit etre superieur a 0");

        IERC20 token = IERC20(tokenAddress);
        require(token.balanceOf(address(this)) >= montant, "Solde insuffisant en tokens");

        require(token.transfer(recipient, montant), "Echec du transfert");
    }

    // 📌 Recuperer des tokens ERC20 d'un autre contrat (si l'approbation est donnee)
    function recupererTokensDepuisContrat(address tokenAddress, address contratSource, uint montant) public uniquementOwner {
        require(contratSource != address(0), "Contrat source invalide");
        require(montant > 0, "Le montant doit etre superieur a 0");

        IERC20 token = IERC20(tokenAddress);

        // Verifie que le contrat `GestionFonds` a une autorisation pour retirer
        require(token.balanceOf(contratSource) >= montant, "Solde insuffisant sur le contrat source");
        require(token.transferFrom(contratSource, address(this), montant), "Echec du transfert depuis le contrat");

        emit TokensRecuperes(tokenAddress, contratSource, montant);
    }

    // 📌 Executer un `delegatecall()` vers un contrat proxy
    function executeDelegateCall(address proxy, bytes memory data) public uniquementOwner returns (bool, bytes memory) {
        require(proxy != address(0), "Adresse du proxy invalide");

        (bool success, bytes memory result) = proxy.delegatecall(data);
        emit DelegateCallExecuted(proxy, data, success);

        require(success, "Echec de l'execution delegatecall");
        return (success, result);
    }

    // 📌 Fallback pour gerer les transactions entrantes (utile pour `delegatecall()`)
    fallback() external payable {}

    receive() external payable {}
}