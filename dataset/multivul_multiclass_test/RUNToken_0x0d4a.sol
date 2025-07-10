// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Interface ERC20
 * Interface standard pour les contrats ERC20.
 */
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

/**
 * @title RUNToken
 * Implémentation d'un contrat ERC20 pour le token RUN.
 */
contract RUNToken is IERC20 {
    // Informations publiques du token
    string public constant name = "RUN Token";
    string public constant symbol = "RUN";
    uint8 public constant decimals = 18;

    // Total supply du token
    uint256 private constant _initialSupply = 200_000_000 * (10 ** uint256(decimals));

    // Mapping des soldes des comptes
    mapping(address => uint256) private _balances;

    // Mapping des autorisations (approvals)
    mapping(address => mapping(address => uint256)) private _allowances;

    // Total supply actuelle
    uint256 private _totalSupply;

    /**
     * @dev Constructeur qui initialise le total supply et assigne tous les tokens au déployeur du contrat.
     */
    constructor() {
        _totalSupply = _initialSupply;
        _balances[msg.sender] = _initialSupply;

        // Émettre un événement de transfert depuis l'adresse zéro vers le déployeur
        emit Transfer(address(0), msg.sender, _initialSupply);
    }

    /**
     * @dev Retourne le total supply actuel.
     */
    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    /**
     * @dev Retourne le solde du compte donné.
     */
    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    /**
     * @dev Transfère des tokens au destinataire.
     */
    function transfer(address recipient, uint256 amount) public override returns (bool) {
        require(recipient != address(0), "ERC20: transfer to the zero address");
        require(_balances[msg.sender] >= amount, "ERC20: transfer amount exceeds balance");

        _balances[msg.sender] -= amount;
        _balances[recipient] += amount;

        emit Transfer(msg.sender, recipient, amount);
        return true;
    }

    /**
     * @dev Retourne l'allocation approuvée pour un certain "spender" par le "owner".
     */
    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }

    /**
     * @dev Approuve une allocation de tokens pour un "spender".
     */
    function approve(address spender, uint256 amount) public override returns (bool) {
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    /**
     * @dev Transfère des tokens depuis une adresse source vers une adresse cible.
     */
    function transferFrom(address sender, address recipient, uint256 amount) public override returns (bool) {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");
        require(_balances[sender] >= amount, "ERC20: transfer amount exceeds balance");
        require(_allowances[sender][msg.sender] >= amount, "ERC20: transfer amount exceeds allowance");

        _balances[sender] -= amount;
        _balances[recipient] += amount;
        _allowances[sender][msg.sender] -= amount;

        emit Transfer(sender, recipient, amount);
        return true;
    }
}