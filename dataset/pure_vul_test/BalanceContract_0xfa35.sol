//SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract BalanceContract {
    address public oracleAddress;
    uint256 public contractBalance;
    address public owner;

    // Use a mapping to store balances associated with UUIDs
    mapping(bytes32 => uint256) public uuidBalances;

    event Deposit(address indexed oracle, bytes32 indexed uuid, uint256 amount);
    event Withdrawal(address indexed owner, address indexed recipient, uint256 amount);

    constructor(address _oracleAddress) {
        require(_oracleAddress != address(0), "Oracle address cannot be zero");
        oracleAddress = _oracleAddress;
        owner = msg.sender; // Set the deployer as the owner
    }

    modifier onlyOracle() {
        require(msg.sender == oracleAddress, "Only the oracle can call this function");
        _;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    function depositWithUUID(bytes32 _uuid, uint256 _amount) external onlyOracle {
        // Update the UUID-specific balance
        uuidBalances[_uuid] += _amount;

        // Update the overall contract balance
        contractBalance += _amount;

        emit Deposit(msg.sender, _uuid, _amount);
    }

    function withdraw(address _recipient, uint256 _amount) external onlyOwner {
        require(contractBalance >= _amount, "Insufficient contract balance");
        contractBalance -= _amount;
        payable(_recipient).transfer(_amount);
        emit Withdrawal(msg.sender, _recipient, _amount);
    }

    function getBalance() external view returns (uint256) {
        return contractBalance;
    }

    function getBalanceForUUID(bytes32 _uuid) external view returns (uint256) {
        return uuidBalances[_uuid];
    }
}