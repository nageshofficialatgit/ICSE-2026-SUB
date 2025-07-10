// SPDX-License-Identifier: MIT
pragma solidity =0.8.0;

// IERC20 interface for interacting with ERC20 tokens (e.g., DAI)
interface IERC20 {
    function transfer(address to, uint256 amount) external;
    function approve(address spender, uint256 amount) external;
    function balanceOf(address account) external view returns (uint256);
}

// IDyDxSoloMargin interface for interacting with the dYdX Solo Margin protocol
interface IDyDxSoloMargin {
    struct AccountInfo {
        address owner;
        uint256 number;
    }
    struct AssetAmount {
        bool sign;
        uint8 denomination;
        uint8 ref;
        uint256 value;
    }
    struct ActionArgs {
        uint256 actionType;
        uint256 accountIdx;
        AssetAmount amount;
        uint256 primaryMarketId;
        uint256 secondaryMarketId;
        address otherAddress;
        uint256 otherAccountIdx;
        bytes data;
    }
    function operate(AccountInfo[] calldata accounts, ActionArgs[] calldata actions) external;
}

// ICallee interface for the callback function required by dYdX
interface ICallee {
    function callFunction(address sender, IDyDxSoloMargin.AccountInfo calldata accountInfo, bytes calldata data) external;
}

contract DyDxFlashLoan is ICallee {
    // Constants for the dYdX Solo Margin and DAI token addresses (Ethereum mainnet)
    address constant soloMargin = 0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e;
    address constant dai = 0x6B175474E89094C44Da98b954EedeAC495271d0F;
    
    // Owner of the contract
    address public owner;

    // Constructor sets the deployer as the owner
    constructor() {
        owner = msg.sender;
    }

    // Function to initiate a flash loan
    function initiateFlashLoan(uint256 amount) external {
        require(msg.sender == owner, "Only owner can initiate flash loan");

        // Define the account information (this contract, account number 0)
        IDyDxSoloMargin.AccountInfo[] memory accounts = new IDyDxSoloMargin.AccountInfo[](1);
        accounts[0] = IDyDxSoloMargin.AccountInfo(address(this), 0);

        // Define the three actions: Withdraw (borrow), Call (execute logic), Deposit (repay)
        IDyDxSoloMargin.ActionArgs[] memory actions = new IDyDxSoloMargin.ActionArgs[](3);

        // Action 1: Withdraw (borrow DAI from dYdX)
        actions[0] = IDyDxSoloMargin.ActionArgs({
            actionType: 1, // Withdraw action
            accountIdx: 0,
            amount: IDyDxSoloMargin.AssetAmount({
                sign: false, // Borrowing (negative movement)
                denomination: 0, // Wei denomination
                ref: 0, // Delta reference
                value: amount // Amount to borrow
            }),
            primaryMarketId: 3, // DAI market ID (verify this for current dYdX configuration)
            secondaryMarketId: 0, // Not used
            otherAddress: address(this), // This contract receives the funds
            otherAccountIdx: 0,
            data: "" // No additional data
        });

        // Action 2: Call (execute custom logic via callback)
        actions[1] = IDyDxSoloMargin.ActionArgs({
            actionType: 8, // Call action
            accountIdx: 0,
            amount: IDyDxSoloMargin.AssetAmount({
                sign: false,
                denomination: 0,
                ref: 0,
                value: 0 // No asset movement in this action
            }),
            primaryMarketId: 0, // Not used
            secondaryMarketId: 0, // Not used
            otherAddress: address(this), // Callback to this contract
            otherAccountIdx: 0,
            data: abi.encode(amount) // Pass the loan amount to the callback
        });

        // Action 3: Deposit (repay the loan plus fee)
        actions[2] = IDyDxSoloMargin.ActionArgs({
            actionType: 0, // Deposit action
            accountIdx: 0,
            amount: IDyDxSoloMargin.AssetAmount({
                sign: true, // Repaying (positive movement)
                denomination: 0, // Wei denomination
                ref: 0, // Delta reference
                value: amount + 2 // Repay the borrowed amount plus 2 wei fee
            }),
            primaryMarketId: 3, // DAI market ID (verify this)
            secondaryMarketId: 0, // Not used
            otherAddress: address(this), // Funds come from this contract
            otherAccountIdx: 0,
            data: "" // No additional data
        });

        // Approve the Solo Margin contract to spend DAI for repayment
        IERC20(dai).approve(soloMargin, amount + 2);

        // Execute the flash loan operation on dYdX
        IDyDxSoloMargin(soloMargin).operate(accounts, actions);
    }

    // Callback function executed by dYdX during the flash loan
    function callFunction(
        address sender,
        IDyDxSoloMargin.AccountInfo memory accountInfo,
        bytes memory data
    ) public override {
        // Security checks
        require(msg.sender == soloMargin, "Only SoloMargin can call this function");
        require(accountInfo.owner == address(this), "Wrong account owner");

        // Decode the loan amount from the data
        uint256 amount = abi.decode(data, (uint256));

        // Verify the loan was received
        require(IERC20(dai).balanceOf(address(this)) >= amount, "Flash loan not received");

        // *** Add your profitable logic here ***
        // Examples:
        // - Perform arbitrage between DEXs (e.g., Uniswap, SushiSwap)
        // - Liquidate undercollateralized positions
        // - Any operation that generates profit
        // Ensure the contract retains at least `amount + 2` DAI for repayment

        // Note: Repayment is handled automatically by the Deposit action in initiateFlashLoan
    }

    // Function to withdraw any remaining DAI balance (e.g., profits)
    function withdraw() external {
        require(msg.sender == owner, "Only owner can withdraw");
        uint256 balance = IERC20(dai).balanceOf(address(this));
        IERC20(dai).transfer(owner, balance);
    }

    // Fallback function to accept ETH (if needed for custom logic)
    receive() external payable {}
}