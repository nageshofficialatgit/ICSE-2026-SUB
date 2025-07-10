// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;



/*
    Rafael Baena Alvarez  
    Founder  
    FFOLLOWME OÜ (16785919)  
    Harju maakond, Tallinn, Lasnamäe linnaosa, Lõõtsa tn 5 // Sepapaja tn 4, 11415  
    Tallinn (Estonia)  
    LinkedIn: https://www.linkedin.com/in/rafael-baena-828b06181/  
    Email: info@ffollowme.com  
    GitHub: https://github.com/followme-dot  
*/
// File: contracts/ReentrancyGuard.sol


pragma solidity ^0.8.20;

abstract contract ReentrancyGuard {
    uint256 private _status;
    
    constructor() {
        _status = 1; // Estado inicial, sin ejecución
    }

    modifier nonReentrant() {
        require(_status == 1, "ReentrancyGuard: Reentrant call");
        _status = 2; // Marcamos la ejecución en curso
        _;
        _status = 1; // Restauramos el estado tras la ejecución
    }
}

// File: contracts/IBEP20.sol


pragma solidity ^0.8.20;

interface IBEP20 {
    /**
     * @dev Devuelve el nombre del token.
     */
    function name() external view returns (string memory);

    /**
     * @dev Devuelve el símbolo del token.
     */
    function symbol() external view returns (string memory);

    /**
     * @dev Devuelve los decimales del token.
     */
    function decimals() external view returns (uint8);

    /**
     * @dev Devuelve el suministro total de tokens en circulación.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Devuelve el saldo de una dirección.
     * @param account Dirección para consultar el saldo.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Transfiere una cantidad de tokens a una dirección.
     * @param recipient Dirección del destinatario.
     * @param amount Cantidad de tokens a transferir.
     */
    function transfer(address recipient, uint256 amount) external returns (bool);

    /**
     * @dev Permite que un contrato gaste tokens en nombre de un propietario.
     * @param owner Dirección del propietario de los tokens.
     * @param spender Dirección del contratista o usuario autorizado para gastar.
     * @return La cantidad que el contrato ha sido autorizado a gastar.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Aprueba a un tercero para gastar tokens en nombre del propietario.
     * @param spender Dirección del contrato o usuario autorizado.
     * @param amount Cantidad de tokens que el tercero puede gastar.
     */
    function approve(address spender, uint256 amount) external returns (bool);

    /**
     * @dev Transfiere tokens de una cuenta a otra utilizando una autorización previamente dada.
     * @param sender Dirección del propietario de los tokens.
     * @param recipient Dirección del destinatario.
     * @param amount Cantidad de tokens a transferir.
     */
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);

    /**
     * @dev Evento que se emite cuando se realiza una transferencia de tokens.
     * @param from Dirección de la cuenta que envía los tokens.
     * @param to Dirección de la cuenta que recibe los tokens.
     * @param value Cantidad de tokens transferidos.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Evento que se emite cuando se aprueba una cantidad de tokens que un tercero puede gastar.
     * @param owner Dirección del propietario de los tokens.
     * @param spender Dirección del tercero autorizado.
     * @param value Cantidad de tokens aprobados para gastar.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

// File: contracts/OwnableP2P.sol


pragma solidity ^0.8.20;


contract OwnableP2P is ReentrancyGuard {
    address public owner;
    mapping(address => bool) public arbitrators;
    mapping(address => bool) public sellers;
    mapping(address => bool) public buyers;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event ArbitratorAdded(address indexed arbitrator);
    event ArbitratorRemoved(address indexed arbitrator);
    event SellerApproved(address indexed seller);
    event BuyerApproved(address indexed buyer);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    modifier onlyArbitrator() {
        require(arbitrators[msg.sender], "Not an arbitrator");
        _;
    }

    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner is zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function addArbitrator(address _arbitrator) external onlyOwner {
        require(_arbitrator != address(0), "Invalid address");
        arbitrators[_arbitrator] = true;
        emit ArbitratorAdded(_arbitrator);
    }

    function removeArbitrator(address _arbitrator) external onlyOwner {
        require(arbitrators[_arbitrator], "Not an arbitrator");
        arbitrators[_arbitrator] = false;
        emit ArbitratorRemoved(_arbitrator);
    }

    function approveSeller(address _seller) external onlyOwner {
        require(_seller != address(0), "Invalid address");
        sellers[_seller] = true;
        emit SellerApproved(_seller);
    }

    function approveBuyer(address _buyer) external onlyOwner {
        require(_buyer != address(0), "Invalid address");
        buyers[_buyer] = true;
        emit BuyerApproved(_buyer);
    }
}

// File: contracts/IERC20.sol


pragma solidity ^0.8.20;

interface IERC20 {
    /**
     * @dev Devuelve el nombre del token.
     */
    function name() external view returns (string memory);

    /**
     * @dev Devuelve el símbolo del token.
     */
    function symbol() external view returns (string memory);

    /**
     * @dev Devuelve los decimales del token.
     */
    function decimals() external view returns (uint8);

    /**
     * @dev Devuelve el suministro total de tokens en circulación.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Devuelve el saldo de una dirección.
     * @param account Dirección para consultar el saldo.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Transfiere una cantidad de tokens a una dirección.
     * @param recipient Dirección del destinatario.
     * @param amount Cantidad de tokens a transferir.
     */
    function transfer(address recipient, uint256 amount) external returns (bool);

    /**
     * @dev Permite que un contrato gaste tokens en nombre de un propietario.
     * @param owner Dirección del propietario de los tokens.
     * @param spender Dirección del contratista o usuario autorizado para gastar.
     * @return La cantidad que el contrato ha sido autorizado a gastar.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Aprueba a un tercero para gastar tokens en nombre del propietario.
     * @param spender Dirección del contrato o usuario autorizado.
     * @param amount Cantidad de tokens que el tercero puede gastar.
     */
    function approve(address spender, uint256 amount) external returns (bool);

    /**
     * @dev Transfiere tokens de una cuenta a otra utilizando una autorización previamente dada.
     * @param sender Dirección del propietario de los tokens.
     * @param recipient Dirección del destinatario.
     * @param amount Cantidad de tokens a transferir.
     */
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) external returns (bool);

    /**
     * @dev Evento que se emite cuando se realiza una transferencia de tokens.
     * @param from Dirección de la cuenta que envía los tokens.
     * @param to Dirección de la cuenta que recibe los tokens.
     * @param value Cantidad de tokens transferidos.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Evento que se emite cuando se aprueba una cantidad de tokens que un tercero puede gastar.
     * @param owner Dirección del propietario de los tokens.
     * @param spender Dirección del tercero autorizado.
     * @param value Cantidad de tokens aprobados para gastar.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

// File: contracts/EscrowP2P.sol


pragma solidity ^0.8.20;



contract EscrowP2P is OwnableP2P {
    enum TradeStatus { Pending, Completed, Cancelled }

    struct Trade {
        address seller;
        address buyer; 
        address token;
        uint256 amount;
        uint256 price;
        TradeStatus status;
    }

    mapping(uint256 => Trade) public trades;
    uint256 public tradeCounter;

    event TradeCreated(uint256 tradeId, address indexed seller, address indexed buyer, address token, uint256 amount, uint256 price);
    event TradeCompleted(uint256 tradeId);
    event TradeCancelled(uint256 tradeId);

    modifier onlyTradeParticipant(uint256 tradeId) {
        require(msg.sender == trades[tradeId].seller || msg.sender == trades[tradeId].buyer, "Not a trade participant");
        _;
    }

    function createTrade(address _seller, address _buyer, address _token, uint256 _amount, uint256 _price) external nonReentrant {
        require(_buyer != address(0), "Invalid buyer address");
        require(_amount > 0, "Amount must be greater than zero");
        require(_price > 0, "Price must be greater than zero");

        IERC20(_token).transferFrom(_seller, address(this), _amount);
        trades[tradeCounter] = Trade(_seller, _buyer, _token, _amount, _price, TradeStatus.Pending);
        emit TradeCreated(tradeCounter, _seller, _buyer, _token, _amount, _price);
        tradeCounter++;
    }

    function completeTrade(uint256 tradeId) external payable onlyTradeParticipant(tradeId) nonReentrant {
        Trade storage trade = trades[tradeId];
        require(trade.status == TradeStatus.Pending, "Trade not pending");
        require(msg.value == trade.price, "Incorrect payment amount");

        payable(trade.seller).transfer(msg.value);
        IERC20(trade.token).transfer(trade.buyer, trade.amount);
        trade.status = TradeStatus.Completed;
        emit TradeCompleted(tradeId);
    }

    function cancelTrade(uint256 tradeId) external onlyTradeParticipant(tradeId) nonReentrant {
        Trade storage trade = trades[tradeId];
        require(trade.status == TradeStatus.Pending, "Trade not pending");

        IERC20(trade.token).transfer(trade.seller, trade.amount);
        trade.status = TradeStatus.Cancelled;
        emit TradeCancelled(tradeId);
    }
}

// File: contracts/P2PExchange.sol


pragma solidity ^0.8.20;




contract P2PExchange is ReentrancyGuard {
    
    enum TradeStatus { Pending, Completed, Cancelled, Disputed }
    enum Role { Buyer, Seller, Arbitrator }

    struct Trade {
        address seller;
        address buyer;
        address token;
        uint256 amount;
        uint256 price;
        TradeStatus status;
        address arbitrator;
        uint256 timestamp;
    }
    
    mapping(uint256 => Trade) public trades;
    uint256 public tradeCounter;

    // Escrow contract instance for secure holding of tokens
    EscrowP2P private escrow;

    // Events
    event TradeCreated(uint256 tradeId, address indexed seller, address indexed buyer, address indexed token, uint256 amount, uint256 price);
    event TradeCompleted(uint256 tradeId, address indexed buyer);
    event TradeCancelled(uint256 tradeId);
    event TradeDisputed(uint256 tradeId, address indexed arbitrator);
    event TokensReleased(uint256 tradeId);

    modifier onlyBuyer(uint256 tradeId) {
        require(trades[tradeId].buyer == msg.sender, "Not the buyer");
        _;
    }

    modifier onlySeller(uint256 tradeId) {
        require(trades[tradeId].seller == msg.sender, "Not the seller");
        _;
    }

    modifier onlyArbitrator(uint256 tradeId) {
        require(trades[tradeId].arbitrator == msg.sender, "Not the arbitrator");
        _;
    }

    modifier tradeExists(uint256 tradeId) {
        require(trades[tradeId].seller != address(0), "Trade does not exist");
        _;
    }

    constructor(EscrowP2P _escrow) {
        escrow = _escrow;
    }

    // Create a new trade offer
    function createTrade(address token, uint256 amount, uint256 price, address arbitrator) external nonReentrant {
        require(amount > 0 && price > 0, "Invalid amount or price");

        // Create the trade in Escrow
        escrow.createTrade(msg.sender, address(0), token, amount, price);  // Adjusted to match EscrowP2P's createTrade parameters

        tradeCounter++;
        trades[tradeCounter] = Trade({
            seller: msg.sender,
            buyer: address(0),
            token: token,
            amount: amount,
            price: price,
            status: TradeStatus.Pending,
            arbitrator: arbitrator,
            timestamp: block.timestamp
        });

        emit TradeCreated(tradeCounter, msg.sender, address(0), token, amount, price);
    }

    // Buyer accepts the trade offer
    function acceptTrade(uint256 tradeId) external nonReentrant tradeExists(tradeId) {
        Trade storage trade = trades[tradeId];
        require(trade.status == TradeStatus.Pending, "Trade not available");
        require(trade.buyer == address(0), "Trade already has a buyer");

        trade.buyer = msg.sender;
        emit TradeCompleted(tradeId, msg.sender);
    }

    // Complete the trade (release tokens to the seller)
    function completeTrade(uint256 tradeId) external onlyBuyer(tradeId) nonReentrant tradeExists(tradeId) {
        Trade storage trade = trades[tradeId];
        require(trade.status == TradeStatus.Pending, "Trade not pending");

        // Complete the trade via escrow (release tokens to the seller)
        escrow.completeTrade(tradeId);
        
        trade.status = TradeStatus.Completed;
        emit TokensReleased(tradeId);
    }

    // Seller cancels the trade (refunds tokens)
    function cancelTrade(uint256 tradeId) external onlySeller(tradeId) nonReentrant tradeExists(tradeId) {
        Trade storage trade = trades[tradeId];
        require(trade.status == TradeStatus.Pending, "Trade already completed/cancelled");

        // Cancel the trade via escrow (refund tokens to the seller)
        escrow.cancelTrade(tradeId);
        
        trade.status = TradeStatus.Cancelled;
        emit TradeCancelled(tradeId);
    }

    // Arbitrator disputes the trade
    function disputeTrade(uint256 tradeId) external onlyArbitrator(tradeId) nonReentrant tradeExists(tradeId) {
        Trade storage trade = trades[tradeId];
        require(trade.status == TradeStatus.Pending, "Trade not pending");

        trade.status = TradeStatus.Disputed;
        emit TradeDisputed(tradeId, msg.sender);
    }

    // Fetch trade details
    function getTradeDetails(uint256 tradeId) external view returns (
        address seller,
        address buyer,
        address token,
        uint256 amount,
        uint256 price,
        TradeStatus status,
        address arbitrator,
        uint256 timestamp
    ) {
        Trade storage trade = trades[tradeId];
        return (
            trade.seller,
            trade.buyer,
            trade.token,
            trade.amount,
            trade.price,
            trade.status,
            trade.arbitrator,
            trade.timestamp
        );
    }
}