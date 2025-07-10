// SPDX-License-Identifier: MIT
pragma solidity 0.5.16;

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract SilkChainTransaction {
    address public owner;
    address public usdtToken; // USDT Contract Address

    bool private locked; // Reentrancy guard

    // Structure to hold transaction data
    struct Transaction {
        string ref;
        uint256 amount;
        string currency;
        string transactionCode;
        address senderWallet;
        uint256 usdAmount;
        uint256 gasFee;
    }

    Transaction public transactionData;

    event TransactionStored(
        string ref,
        uint256 amount,
        string currency,
        string transactionCode,
        address senderWallet,
        uint256 usdAmount,
        uint256 gasFee
    );

    event FundsDistributed(address recipient, uint256 amount);
    event ETHWithdrawn(address owner, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    modifier noReentrancy() {
        require(!locked, "No reentrancy allowed");
        locked = true;
        _;
        locked = false;
    }

    constructor(address _usdtToken) public {
        owner = msg.sender;
        usdtToken = _usdtToken; // Set USDT contract address
    }

    // Function to store transaction data
    function storeTransaction(
        string memory _ref,
        uint256 _amount,
        string memory _currency,
        string memory _transactionCode,
        address _senderWallet,
        uint256 _usdAmount,
        uint256 _gasFee
    ) public onlyOwner {
        transactionData = Transaction(
            _ref,
            _amount,
            _currency,
            _transactionCode,
            _senderWallet,
            _usdAmount,
            _gasFee
        );

        emit TransactionStored(_ref, _amount, _currency, _transactionCode, _senderWallet, _usdAmount, _gasFee);
    }

    // Function to withdraw ETH for gas fees
    function withdrawETH(uint256 _amount) public onlyOwner noReentrancy {
        require(address(this).balance >= _amount, "Insufficient ETH balance for withdrawal");
        msg.sender.transfer(_amount);
        emit ETHWithdrawn(msg.sender, _amount);
    }

    // Function to distribute USDT to recipients
    function distributeUSDT(address _recipient, uint256 _amount) public onlyOwner noReentrancy {
        IERC20 token = IERC20(usdtToken);
        require(token.balanceOf(address(this)) >= _amount, "Insufficient USDT balance");

        bool success = token.transfer(_recipient, _amount);
        require(success, "USDT transfer failed");

        emit FundsDistributed(_recipient, _amount);
    }

    // Function to check the contract's USDT balance
    function getUSDTBalance() public view returns (uint256) {
        IERC20 token = IERC20(usdtToken);
        return token.balanceOf(address(this));
    }

    // Function to accept ETH deposits to the contract (needed to withdraw ETH later)
    function depositETH() public payable onlyOwner {}

    // Fallback function to accept ETH sent directly to the contract
    function() external payable {}
}