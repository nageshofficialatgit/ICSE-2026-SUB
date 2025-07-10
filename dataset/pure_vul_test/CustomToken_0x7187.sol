// SPDX-License-Identifier: MIT
pragma solidity ^0.8.25;

/**
 * @title CustomToken - Un jeton ERC-20 personnalisé
 * @dev Implémentation d'un jeton ERC-20 avec fonctionnalités de transfert et d'approbation
 */
contract CustomToken {
    string public tokenName;
    string public tokenSymbol;
    uint8 public tokenDecimals;
    uint256 public totalTokens;

    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public allowed;

    event TokensTransferred(address indexed sender, address indexed recipient, uint256 amount);
    event SpendingApproved(address indexed owner, address indexed delegate, uint256 amount);

    /**
     * @dev Initialise le contrat avec les paramètres du jeton
     * @param _tokenName Nom du jeton
     * @param _tokenSymbol Symbole du jeton
     * @param _decimals Nombre de décimales
     * @param _initialTokens Quantité initiale de jetons
     */
    constructor(
        string memory _tokenName,
        string memory _tokenSymbol,
        uint8 _decimals,
        uint256 _initialTokens
    ) {
        tokenName = _tokenName;
        tokenSymbol = _tokenSymbol;
        tokenDecimals = _decimals;
        totalTokens = _initialTokens * 10 ** uint256(_decimals);
        balances[msg.sender] = totalTokens;
        emit TokensTransferred(address(0), msg.sender, totalTokens);
    }

    /**
     * @dev Transfère des jetons vers une adresse spécifiée
     * @param recipient Adresse du destinataire
     * @param amount Quantité de jetons à transférer
     * @return success Indique si le transfert a réussi
     */
    function sendTokens(address recipient, uint256 amount) external returns (bool success) {
        require(recipient != address(0), "CustomToken: Cannot send to zero address");
        require(balances[msg.sender] >= amount, "CustomToken: Insufficient balance");
        
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
        emit TokensTransferred(msg.sender, recipient, amount);
        return true;
    }

    /**
     * @dev Approuve une adresse pour dépenser un certain montant de jetons
     * @param delegate Adresse autorisée à dépenser
     * @param amount Montant approuvé
     * @return success Indique si l'approbation a réussi
     */
    function approve(address delegate, uint256 amount) external returns (bool success) {
        allowed[msg.sender][delegate] = amount;
        emit SpendingApproved(msg.sender, delegate, amount);
        return true;
    }

    /**
     * @dev Transfère des jetons depuis une adresse autorisée
     * @param sender Adresse de l'émetteur
     * @param recipient Adresse du destinataire
     * @param amount Quantité de jetons à transférer
     * @return success Indique si le transfert a réussi
     */
    function sendFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool success) {
        require(sender != address(0), "CustomToken: Sender cannot be zero address");
        require(recipient != address(0), "CustomToken: Recipient cannot be zero address");
        require(balances[sender] >= amount, "CustomToken: Sender has insufficient balance");
        require(allowed[sender][msg.sender] >= amount, "CustomToken: Amount exceeds allowance");

        balances[sender] -= amount;
        balances[recipient] += amount;
        allowed[sender][msg.sender] -= amount;
        emit TokensTransferred(sender, recipient, amount);
        return true;
    }
}