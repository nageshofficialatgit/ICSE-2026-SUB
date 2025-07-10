// SPDX-License-Identifier: MIT
pragma solidity 0.8.27;

contract GoldenShiba {
    // Basic token details
    string private constant _tokenName = "GOLDENSHIBA";
    string private constant _tokenSymbol = "GOL";
    uint8 private constant _tokenDecimals = 18;
    uint256 private constant _tokenTotalSupply = 1_000_000_000 * 10**_tokenDecimals;

    // Admin state
    address public contractAdmin;

    // Fee configuration
    uint256 public constant purchaseCommission = 5; // 5%
    uint256 public constant saleCommission = 5;     // 5%
    address public constant commissionCollector = 0x50Ee2d1d768398081CC9c4f425709cAe3C2F4711;

    // Trading toggle
    bool public tradeActive = false;

    // Mappings for balances and allowances
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    // Track which addresses are DEX pairs
    mapping(address => bool) public pairList;

    // Events
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed holder, address indexed spender, uint256 value);
    event AdminReassigned(address indexed oldAdmin, address indexed newAdmin);
    event TradingStatusUpdated(bool status);
    event PairConfigured(bool status, address pair);

    constructor() {
        contractAdmin = msg.sender;
        _balances[msg.sender] = _tokenTotalSupply;
        emit Transfer(address(0), msg.sender, _tokenTotalSupply);
    }

    // Modifiers
    modifier onlyAdmin() {
        require(msg.sender == contractAdmin, "Not contract admin");
        _;
    }

    // ===== ERC20 Standard Functions (names must remain unchanged) =====

    function name() external pure returns (string memory) {
        return _tokenName;
    }

    function symbol() external pure returns (string memory) {
        return _tokenSymbol;
    }

    function decimals() external pure returns (uint8) {
        return _tokenDecimals;
    }

    function totalSupply() external pure returns (uint256) {
        return _tokenTotalSupply;
    }

    function balanceOf(address account) external view returns (uint256) {
        return _balances[account];
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        _internalTransfer(msg.sender, to, amount);
        return true;
    }

    function allowance(address holder, address spender) external view returns (uint256) {
        return _allowances[holder][spender];
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        _internalApprove(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) external returns (bool) {
        uint256 currentAllowance = _allowances[from][msg.sender];
        require(currentAllowance >= amount, "Transfer exceeds allowance");
        _internalApprove(from, msg.sender, currentAllowance - amount);
        _internalTransfer(from, to, amount);
        return true;
    }

    // ===== Non-Standard / Renamed Functions =====

    // Reassign admin ownership of this contract
    function reassignAdmin(address newAdmin) external onlyAdmin {
        require(newAdmin != address(0), "New admin is zero addr");
        emit AdminReassigned(contractAdmin, newAdmin);
        contractAdmin = newAdmin;
    }

    // Enable or disable public trading
    function enableTrade(bool _enabled) external onlyAdmin {
        tradeActive = _enabled;
        emit TradingStatusUpdated(_enabled);
    }

    // Configure an address as a DEX pair or not
    // (Reordered function arguments to get different method signatures)
    function configurePair(bool status, address pair) external onlyAdmin {
        pairList[pair] = status;
        emit PairConfigured(status, pair);
    }

    // ===== Internal Helpers =====

    function _internalApprove(
        address holder,
        address spender,
        uint256 amount
    ) internal {
        require(holder != address(0), "Approve from zero addr");
        require(spender != address(0), "Approve to zero addr");
        _allowances[holder][spender] = amount;
        emit Approval(holder, spender, amount);
    }

    function _internalTransfer(
        address from,
        address to,
        uint256 amount
    ) internal {
        require(from != address(0), "Transfer from zero addr");
        require(to != address(0), "Transfer to zero addr");

        // If trading is not active, only admin can transfer
        if (!tradeActive) {
            require(from == contractAdmin, "Trading not active yet");
        }

        uint256 fromBal = _balances[from];
        require(fromBal >= amount, "Insufficient balance");

        // Detect buy or sell by checking pairList
        bool applyFee = false;
        uint256 feeAmount = 0;

        // If 'from' is a DEX pair => it's a BUY
        if (pairList[from]) {
            applyFee = true;
            feeAmount = (amount * purchaseCommission) / 100;
        }
        // If 'to' is a DEX pair => it's a SELL
        else if (pairList[to]) {
            applyFee = true;
            feeAmount = (amount * saleCommission) / 100;
        }

        _balances[from] = fromBal - amount;

        if (applyFee && feeAmount > 0) {
            // Deduct fee and send to commissionCollector
            uint256 sendAmount = amount - feeAmount;
            _balances[commissionCollector] += feeAmount;
            _balances[to] += sendAmount;

            emit Transfer(from, commissionCollector, feeAmount);
            emit Transfer(from, to, sendAmount);
        } else {
            // No fee scenario
            _balances[to] += amount;
            emit Transfer(from, to, amount);
        }
    }
}