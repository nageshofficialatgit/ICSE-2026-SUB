// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface LinkTokenInterface {
  function allowance(address owner, address spender) external view returns (uint256 remaining);

  function approve(address spender, uint256 value) external returns (bool success);

  function balanceOf(address owner) external view returns (uint256 balance);

  function decimals() external view returns (uint8 decimalPlaces);

  function decreaseApproval(address spender, uint256 addedValue) external returns (bool success);

  function increaseApproval(address spender, uint256 subtractedValue) external;

  function name() external view returns (string memory tokenName);

  function symbol() external view returns (string memory tokenSymbol);

  function totalSupply() external view returns (uint256 totalTokensIssued);

  function transfer(address to, uint256 value) external returns (bool success);

  function transferAndCall(
    address to,
    uint256 value,
    bytes calldata data
  ) external returns (bool success);

  function transferFrom(
    address from,
    address to,
    uint256 value
  ) external returns (bool success);
}



contract TransactionValidator {
    address public owner;
    address public oracle; // Chainlink oracle address
    bytes32 public jobId; // Chainlink job ID
    uint256 public fee; // LINK fee for Chainlink job
    LinkTokenInterface public linkToken;

    mapping(bytes32 => bool) public processedTxs; // Track validated transactions
    uint256 public totalDeposited; // Total validated deposits

    // Events
    event Deposit(address indexed from, uint256 amount);
    event Withdrawal(address indexed to, uint256 amount);
    event TransactionValidated(bytes32 txHash, uint256 amount);
    event JobIdUpdated(bytes32 oldJobId, bytes32 newJobId); // Event for job ID update

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can perform this action");
        _;
    }

    constructor(address _oracle, bytes32 _jobId, uint256 _fee, address _linkToken) {
        owner = msg.sender;
        oracle = _oracle;
        jobId = _jobId;
        fee = _fee;
        linkToken = LinkTokenInterface(_linkToken);
    }

    // Function to receive ETH
    receive() external payable {}

    // Function to process validated transactions
    function depositValidated(bytes32 txHash, uint256 amount) external {
        require(msg.sender == oracle, "Only the designated Chainlink oracle can validate deposits");
        require(!processedTxs[txHash], "Transaction has already been processed");

        processedTxs[txHash] = true; // Mark the transaction as processed
        totalDeposited += amount; // Update the total deposited amount

        emit Deposit(tx.origin, amount); // Log the deposit
        emit TransactionValidated(txHash, amount); // Log the transaction validation
    }

    // Function to withdraw funds (only the owner can call this)
    function withdraw(uint256 amount) external onlyOwner {
        require(address(this).balance >= amount, "Insufficient balance in the contract");
        payable(owner).transfer(amount);
        emit Withdrawal(owner, amount);
    }

    // Get the contract's total balance
    function getContractBalance() external view returns (uint256) {
        return address(this).balance;
    }

    // Get the LINK balance
    function getLinkBalance() external view returns (uint256) {
        return linkToken.balanceOf(address(this));
    }

    // **New Function**: Update the job ID
    function updateJobId(bytes32 _newJobId) external onlyOwner {
        require(_newJobId != bytes32(0), "Job ID cannot be empty");
        emit JobIdUpdated(jobId, _newJobId); // Emit an event for the job ID update
        jobId = _newJobId; // Update the job ID
    }
}