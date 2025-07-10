// Sources flattened with hardhat v2.10.1 https://hardhat.org

// File contracts/ERC20/IWETH.sol

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IWETH {
    function deposit() external payable;
    function transfer(address to, uint value) external returns (bool);
    function transferFrom(address src, address dst, uint wad) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function withdraw(uint) external;
}


// File contracts/UniswapFork/BankXLibrary.sol




library BankXLibrary {

    // given some amount of an asset and pair reserves, returns an equivalent amount of the other asset
    function quote(uint amountA, uint reserveA, uint reserveB) internal pure returns (uint amountB) {
        require(amountA > 0, 'BankXLibrary: INSUFFICIENT_AMOUNT');
        require(reserveA > 0 && reserveB > 0, 'BankXLibrary: INSUFFICIENT_LIQUIDITY');
        amountB = (amountA*reserveB) / reserveA;
    }
   
}


// File @openzeppelin/contracts/utils/Context.sol@v4.9.6


// OpenZeppelin Contracts (last updated v4.9.4) (utils/Context.sol)



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
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}


// File @openzeppelin/contracts/token/ERC20/IERC20.sol@v4.9.6


// OpenZeppelin Contracts (last updated v4.9.0) (token/ERC20/IERC20.sol)



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


// File @openzeppelin/contracts/utils/math/SafeMath.sol@v4.9.6


// OpenZeppelin Contracts (last updated v4.9.0) (utils/math/SafeMath.sol)



// CAUTION
// This version of SafeMath should only be used with Solidity 0.8 or later,
// because it relies on the compiler's built in overflow checks.

/**
 * @dev Wrappers over Solidity's arithmetic operations.
 *
 * NOTE: `SafeMath` is generally not needed starting with Solidity 0.8, since the compiler
 * now has built in overflow checking.
 */
library SafeMath {
    /**
     * @dev Returns the addition of two unsigned integers, with an overflow flag.
     *
     * _Available since v3.4._
     */
    function tryAdd(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            uint256 c = a + b;
            if (c < a) return (false, 0);
            return (true, c);
        }
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, with an overflow flag.
     *
     * _Available since v3.4._
     */
    function trySub(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b > a) return (false, 0);
            return (true, a - b);
        }
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, with an overflow flag.
     *
     * _Available since v3.4._
     */
    function tryMul(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            // Gas optimization: this is cheaper than requiring 'a' not being zero, but the
            // benefit is lost if 'b' is also tested.
            // See: https://github.com/OpenZeppelin/openzeppelin-contracts/pull/522
            if (a == 0) return (true, 0);
            uint256 c = a * b;
            if (c / a != b) return (false, 0);
            return (true, c);
        }
    }

    /**
     * @dev Returns the division of two unsigned integers, with a division by zero flag.
     *
     * _Available since v3.4._
     */
    function tryDiv(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a / b);
        }
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers, with a division by zero flag.
     *
     * _Available since v3.4._
     */
    function tryMod(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a % b);
        }
    }

    /**
     * @dev Returns the addition of two unsigned integers, reverting on
     * overflow.
     *
     * Counterpart to Solidity's `+` operator.
     *
     * Requirements:
     *
     * - Addition cannot overflow.
     */
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, reverting on
     * overflow (when the result is negative).
     *
     * Counterpart to Solidity's `-` operator.
     *
     * Requirements:
     *
     * - Subtraction cannot overflow.
     */
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return a - b;
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, reverting on
     * overflow.
     *
     * Counterpart to Solidity's `*` operator.
     *
     * Requirements:
     *
     * - Multiplication cannot overflow.
     */
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        return a * b;
    }

    /**
     * @dev Returns the integer division of two unsigned integers, reverting on
     * division by zero. The result is rounded towards zero.
     *
     * Counterpart to Solidity's `/` operator.
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return a / b;
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers. (unsigned integer modulo),
     * reverting when dividing by zero.
     *
     * Counterpart to Solidity's `%` operator. This function uses a `revert`
     * opcode (which leaves remaining gas untouched) while Solidity uses an
     * invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return a % b;
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, reverting with custom message on
     * overflow (when the result is negative).
     *
     * CAUTION: This function is deprecated because it requires allocating memory for the error
     * message unnecessarily. For custom revert reasons use {trySub}.
     *
     * Counterpart to Solidity's `-` operator.
     *
     * Requirements:
     *
     * - Subtraction cannot overflow.
     */
    function sub(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        unchecked {
            require(b <= a, errorMessage);
            return a - b;
        }
    }

    /**
     * @dev Returns the integer division of two unsigned integers, reverting with custom message on
     * division by zero. The result is rounded towards zero.
     *
     * Counterpart to Solidity's `/` operator. Note: this function uses a
     * `revert` opcode (which leaves remaining gas untouched) while Solidity
     * uses an invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function div(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a / b;
        }
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers. (unsigned integer modulo),
     * reverting with custom message when dividing by zero.
     *
     * CAUTION: This function is deprecated because it requires allocating memory for the error
     * message unnecessarily. For custom revert reasons use {tryMod}.
     *
     * Counterpart to Solidity's `%` operator. This function uses a `revert`
     * opcode (which leaves remaining gas untouched) while Solidity uses an
     * invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function mod(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a % b;
        }
    }
}


// File contracts/ERC20/ERC20Custom.sol





// Due to compiling issues, _name, _symbol, and _decimals were removed


/**
 * @dev Implementation of the {IERC20} interface.
 *
 * This implementation is agnostic to the way tokens are created. This means
 * that a supply mechanism has to be added in a derived contract using {_mint}.
 * For a generic mechanism see {ERC20Mintable}.
 *
 * TIP: For a detailed writeup see our guide
 * https://forum.zeppelin.solutions/t/how-to-implement-erc20-supply-mechanisms/226[How
 * to implement supply mechanisms].
 *
 * We have followed general OpenZeppelin guidelines: functions revert instead
 * of returning `false` on failure. This behavior is nonetheless conventional
 * and does not conflict with the expectations of ERC20 applications.
 *
 * Additionally, an {Approval} event is emitted on calls to {transferFrom}.
 * This allows applications to reconstruct the allowance for all accounts just
 * by listening to said events. Other implementations of the EIP may not emit
 * these events, as it isn't required by the specification.
 *
 * Finally, the non-standard {decreaseAllowance} and {increaseAllowance}
 * functions have been added to mitigate the well-known issues around setting
 * allowances. See {IERC20-approve}.
 */
contract ERC20Custom is Context, IERC20 {
    using SafeMath for uint256;

    mapping (address => uint256) internal _balances;

    mapping (address => mapping (address => uint256)) internal _allowances;

    uint256 private _totalSupply;
    /**
     * @dev See {IERC20-totalSupply}.
     */
    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }

    /**
     * @dev See {IERC20-balanceOf}.
     */
    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }

    /**
     * @dev See {IERC20-transfer}.
     *
     * Requirements:
     *
     * - `recipient` cannot be the zero address.
     * - the caller must have a balance of at least `amount`.
     */
    function transfer(address to, uint256 amount) public virtual override returns (bool) {
        address owner = _msgSender();
        _transfer(owner, to, amount);
        return true;
    }

    /**
     * @dev See {IERC20-allowance}.
     */
    function allowance(address owner, address spender) public view virtual override returns (uint256) {
        return _allowances[owner][spender];
    }

    /**
     * @dev See {IERC20-approve}.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.approve(address spender, uint256 amount)
     */
    function approve(address spender, uint256 amount) public virtual override returns (bool) {
        address owner = _msgSender();
        _approve(owner, spender, amount);
        return true;
    }

    /**
     * @dev See {IERC20-transferFrom}.
     *
     * Emits an {Approval} event indicating the updated allowance. This is not
     * required by the EIP. See the note at the beginning of {ERC20}.
     *
     * NOTE: Does not update the allowance if the current allowance
     * is the maximum `uint256`.
     *
     * Requirements:
     *
     * - `from` and `to` cannot be the zero address.
     * - `from` must have a balance of at least `amount`.
     * - the caller must have allowance for ``from``'s tokens of at least
     * `amount`.
     */
    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) public virtual override returns (bool) {
        address spender = _msgSender();
        _spendAllowance(from, spender, amount);
        _transfer(from, to, amount);
        return true;
    }

    /**
     * @dev Updates `owner` s allowance for `spender` based on spent `amount`.
     *
     * Does not update the allowance amount in case of infinite allowance.
     * Revert if not enough allowance is available.
     *
     * Might emit an {Approval} event.
     */
    function _spendAllowance(
        address owner,
        address spender,
        uint256 amount
    ) internal virtual {
        uint256 currentAllowance = allowance(owner, spender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "ERC20: insufficient allowance");
            unchecked {
                _approve(owner, spender, currentAllowance - amount);
            }
        }
    }

    /**
     * @dev Atomically increases the allowance granted to `spender` by the caller.
     *
     * This is an alternative to {approve} that can be used as a mitigation for
     * problems described in {IERC20-approve}.
     *
     * Emits an {Approval} event indicating the updated allowance.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     */
    function increaseAllowance(address spender, uint256 addedValue) public virtual returns (bool) {
        address owner = _msgSender();
        _approve(owner, spender, allowance(owner, spender) + addedValue);
        return true;
    }

    /**
     * @dev Atomically decreases the allowance granted to `spender` by the caller.
     *
     * This is an alternative to {approve} that can be used as a mitigation for
     * problems described in {IERC20-approve}.
     *
     * Emits an {Approval} event indicating the updated allowance.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     * - `spender` must have allowance for the caller of at least
     * `subtractedValue`.
     */
    function decreaseAllowance(address spender, uint256 subtractedValue) public virtual returns (bool) {
        address owner = _msgSender();
        uint256 currentAllowance = allowance(owner, spender);
        require(currentAllowance >= subtractedValue, "ERC20: decreased allowance below zero");
        unchecked {
            _approve(owner, spender, currentAllowance - subtractedValue);
        }

        return true;
    }

    /**
     * @dev Moves tokens `amount` from `sender` to `recipient`.
     *
     * This is internal function is equivalent to {transfer}, and can be used to
     * e.g. implement automatic token fees, slashing mechanisms, etc.
     *
     * Emits a {Transfer} event.
     *
     * Requirements:
     *
     * - `sender` cannot be the zero address.
     * - `recipient` cannot be the zero address.
     * - `sender` must have a balance of at least `amount`.
     */
    function _transfer(address from, address to, uint256 amount) internal virtual {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");

        _beforeTokenTransfer(from, to, amount);

        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "ERC20: transfer amount exceeds balance");
        unchecked {
            _balances[from] = fromBalance - amount;
        }
        _balances[to] += amount;

        emit Transfer(from, to, amount);

        _afterTokenTransfer(from, to, amount);
    }

    /** @dev Creates `amount` tokens and assigns them to `account`, increasing
     * the total supply.
     *
     * Emits a {Transfer} event with `from` set to the zero address.
     *
     * Requirements
     *
     * - `to` cannot be the zero address.
     */
    function _mint(address account, uint256 amount) internal virtual {
        require(account != address(0), "ERC20: mint to the zero address");

        _beforeTokenTransfer(address(0), account, amount);

        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);

        _afterTokenTransfer(address(0), account, amount);
    }

    /**
     * @dev Destroys `amount` tokens from the caller.
     *
     * See {ERC20-_burn}.
     */
    function burn(uint256 amount) public virtual {
        _burn(_msgSender(), amount);
    }

    /**
     * @dev Destroys `amount` tokens from `account`, deducting from the caller's
     * allowance.
     *
     * See {ERC20-_burn} and {ERC20-allowance}.
     *
     * Requirements:
     *
     * - the caller must have allowance for `accounts`'s tokens of at least
     * `amount`.
     */
    function burnFrom(address account, uint256 amount) public virtual {
        uint256 decreasedAllowance = allowance(account, _msgSender()).sub(amount, "ERC20: burn amount exceeds allowance");

        _approve(account, _msgSender(), decreasedAllowance);
        _burn(account, amount);
    }


    /**
     * @dev Transfers 'tokens' from 'account' to origin address, reducing the
     * total supply.
     *
     * Emits a {Transfer} event with `to` set to the zero address.
     *
     * Requirements
     *
     * - `account` cannot be the zero address.
     * - `account` must have at least `amount` tokens.
     */
    function _burn(address account, uint256 amount) internal virtual {
       require(account != address(0), "ERC20: burn from the zero address");

        _beforeTokenTransfer(account, address(0), amount);

        uint256 accountBalance = _balances[account];
        require(accountBalance >= amount, "ERC20: burn amount exceeds balance");
        unchecked {
            _balances[account] = accountBalance - amount;
        }
        _totalSupply -= amount;

        emit Transfer(account, address(0), amount);

        _afterTokenTransfer(account, address(0), amount);
        
    }

    /**
     * @dev Sets `amount` as the allowance of `spender` over the `owner`s tokens.
     *
     * This is internal function is equivalent to `approve`, and can be used to
     * e.g. set automatic allowances for certain subsystems, etc.
     *
     * Emits an {Approval} event.
     *
     * Requirements:
     *
     * - `owner` cannot be the zero address.
     * - `spender` cannot be the zero address.
     */
    function _approve(
        address owner,
        address spender,
        uint256 amount
    ) internal virtual {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    /**
     * @dev Destroys `amount` tokens from `account`.`amount` is then deducted
     * from the caller's allowance.
     *
     * See {_burn} and {_approve}.
     */
    function _burnFrom(address account, uint256 amount) internal virtual {
        _burn(account, amount);
        _approve(account, _msgSender(), _allowances[account][_msgSender()].sub(amount, "ERC20: burn amount exceeds allowance"));
    }

    /**
     * @dev Hook that is called before any transfer of tokens. This includes
     * minting and burning.
     *
     * Calling conditions:
     *
     * - when `from` and `to` are both non-zero, `amount` of `from`'s tokens
     * will be to transferred to `to`.
     * - when `from` is zero, `amount` tokens will be minted for `to`.
     * - when `to` is zero, `amount` of `from`'s tokens will be burned.
     * - `from` and `to` are never both zero.
     *
     * To learn more about hooks, head to xref:ROOT:using-hooks.adoc[Using Hooks].
     */
    function _beforeTokenTransfer(address from, address to, uint256 amount) internal virtual { }
    /**
     * @dev Hook that is called after any transfer of tokens. This includes
     * minting and burning.
     *
     * Calling conditions:
     *
     * - when `from` and `to` are both non-zero, `amount` of ``from``'s tokens
     * has been transferred to `to`.
     * - when `from` is zero, `amount` tokens have been minted for `to`.
     * - when `to` is zero, `amount` of ``from``'s tokens have been burned.
     * - `from` and `to` are never both zero.
     *
     * To learn more about hooks, head to xref:ROOT:extending-contracts.adoc#using-hooks[Using Hooks].
     */
    function _afterTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal virtual {}
}


// File contracts/XSD/Pools/Interfaces/IXSDWETHpool.sol



interface IXSDWETHpool {
    function PERMIT_TYPEHASH() external pure returns (bytes32);
    function nonces(address owner) external view returns (uint);
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
    function price0CumulativeLast() external view returns (uint);
    function price1CumulativeLast() external view returns (uint);
    function kLast() external view returns (uint);
    function collatDollarBalance() external returns (uint);
    function swap(uint amount0Out, uint amount1Out, address to) external;
    function skim(address to) external;
    function sync() external;
}


// File contracts/XSD/Pools/Interfaces/IBankXWETHpool.sol



interface IBankXWETHpool {
    function PERMIT_TYPEHASH() external pure returns (bytes32);
    function nonces(address owner) external view returns (uint);
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
    function price0CumulativeLast() external view returns (uint);
    function price1CumulativeLast() external view returns (uint);
    function kLast() external view returns (uint);
    function collatDollarBalance() external returns(uint);
    function swap(uint amount0Out, uint amount1Out, address to) external;
    function skim(address to) external;
    function sync() external;
}


// File contracts/XSD/Pools/Interfaces/ICollateralPool.sol



interface ICollateralPool{
    function userProvideLiquidity(address to, uint amount1) external;
    function collat_XSD() external returns(uint);
    function mintAlgorithmicXSD(uint256 bankx_amount_d18, uint256 XSD_out_min) external;
    function collatDollarBalance() external returns(uint);
}


// File contracts/Oracle/AggregatorV3Interface.sol



interface AggregatorV3Interface {

  function decimals() external view returns (uint8);
  function description() external view returns (string memory);
  function version() external view returns (uint256);

  // getRoundData and latestRoundData should both raise "No data present"
  // if they do not have data to report, instead of returning unset values
  // which could be misinterpreted as actual reported values.
  function getRoundData(uint80 _roundId)
    external
    view
    returns (
      uint80 roundId,
      int256 answer,
      uint256 startedAt,
      uint256 updatedAt,
      uint80 answeredInRound
    );
  function latestRoundData()
    external
    view
    returns (
      uint80 roundId,
      int256 answer,
      uint256 startedAt,
      uint256 updatedAt,
      uint80 answeredInRound
    );

}


// File contracts/Oracle/ChainlinkETHUSDPriceConsumer.sol



contract ChainlinkETHUSDPriceConsumer {

    AggregatorV3Interface internal priceFeed;
    //Arbitrum: 0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612
    //Ethereum: 0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419
    //Polygon: 0xAB594600376Ec9fD91F8e885dADF0CE036862dE0
    //Optimism: 0x13e3Ee699D1909E989722E753853AE30b17e08c5
    //Avalanche: 0x0A77230d17318075983913bC2145DB16C7366156
    //Fantom: 0xf4766552D15AE4d256Ad41B6cf2933482B0680dc
    constructor() {
        priceFeed = AggregatorV3Interface(0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419);
    }

    /**
     * Returns the latest price
     */
    function getLatestPrice() public view returns (int) {
        (
            uint80 roundID
            , 
            int price,
            ,
            ,
            uint80 answeredInRound
        ) = priceFeed.latestRoundData();
        require(answeredInRound >= roundID);
        return price;
    }
    
    function getDecimals() public view returns (uint8) {
        return priceFeed.decimals();
    }
}


// File contracts/Oracle/ChainlinkXAGUSDPriceConsumer.sol



contract ChainlinkXAGUSDPriceConsumer {

    AggregatorV3Interface priceFeed;
    //Arbitrum: 0xC56765f04B248394CF1619D20dB8082Edbfa75b1
    //Ethereum: 0x379589227b15F1a12195D3f2d90bBc9F31f95235
    //Polygon: 0x461c7B8D370a240DdB46B402748381C3210136b3
    //Optimism:0x290dd71254874f0d4356443607cb8234958DEe49
    //Avalanche:0x4305FB66699C3B2702D4d05CF36551390A4c69C6
    constructor() {
        priceFeed = AggregatorV3Interface(0x379589227b15F1a12195D3f2d90bBc9F31f95235);
    }

    /**
     * Returns the latest price
     */
    function getLatestPrice() public view returns (int) {
        (
            uint80 roundID
            , 
            int price,
            ,
            ,
            uint80 answeredInRound
        ) = priceFeed.latestRoundData();
        require(answeredInRound >= roundID);
        return price;
    }
    
    function getDecimals() public view returns (uint8) {
        return priceFeed.decimals();
    }
}


// File @openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol@v4.9.6


// OpenZeppelin Contracts v4.4.1 (token/ERC20/extensions/IERC20Metadata.sol)



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


// File @openzeppelin/contracts/token/ERC20/ERC20.sol@v4.9.6


// OpenZeppelin Contracts (last updated v4.9.0) (token/ERC20/ERC20.sol)





/**
 * @dev Implementation of the {IERC20} interface.
 *
 * This implementation is agnostic to the way tokens are created. This means
 * that a supply mechanism has to be added in a derived contract using {_mint}.
 * For a generic mechanism see {ERC20PresetMinterPauser}.
 *
 * TIP: For a detailed writeup see our guide
 * https://forum.openzeppelin.com/t/how-to-implement-erc20-supply-mechanisms/226[How
 * to implement supply mechanisms].
 *
 * The default value of {decimals} is 18. To change this, you should override
 * this function so it returns a different value.
 *
 * We have followed general OpenZeppelin Contracts guidelines: functions revert
 * instead returning `false` on failure. This behavior is nonetheless
 * conventional and does not conflict with the expectations of ERC20
 * applications.
 *
 * Additionally, an {Approval} event is emitted on calls to {transferFrom}.
 * This allows applications to reconstruct the allowance for all accounts just
 * by listening to said events. Other implementations of the EIP may not emit
 * these events, as it isn't required by the specification.
 *
 * Finally, the non-standard {decreaseAllowance} and {increaseAllowance}
 * functions have been added to mitigate the well-known issues around setting
 * allowances. See {IERC20-approve}.
 */
contract ERC20 is Context, IERC20, IERC20Metadata {
    mapping(address => uint256) private _balances;

    mapping(address => mapping(address => uint256)) private _allowances;

    uint256 private _totalSupply;

    string private _name;
    string private _symbol;

    /**
     * @dev Sets the values for {name} and {symbol}.
     *
     * All two of these values are immutable: they can only be set once during
     * construction.
     */
    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
    }

    /**
     * @dev Returns the name of the token.
     */
    function name() public view virtual override returns (string memory) {
        return _name;
    }

    /**
     * @dev Returns the symbol of the token, usually a shorter version of the
     * name.
     */
    function symbol() public view virtual override returns (string memory) {
        return _symbol;
    }

    /**
     * @dev Returns the number of decimals used to get its user representation.
     * For example, if `decimals` equals `2`, a balance of `505` tokens should
     * be displayed to a user as `5.05` (`505 / 10 ** 2`).
     *
     * Tokens usually opt for a value of 18, imitating the relationship between
     * Ether and Wei. This is the default value returned by this function, unless
     * it's overridden.
     *
     * NOTE: This information is only used for _display_ purposes: it in
     * no way affects any of the arithmetic of the contract, including
     * {IERC20-balanceOf} and {IERC20-transfer}.
     */
    function decimals() public view virtual override returns (uint8) {
        return 18;
    }

    /**
     * @dev See {IERC20-totalSupply}.
     */
    function totalSupply() public view virtual override returns (uint256) {
        return _totalSupply;
    }

    /**
     * @dev See {IERC20-balanceOf}.
     */
    function balanceOf(address account) public view virtual override returns (uint256) {
        return _balances[account];
    }

    /**
     * @dev See {IERC20-transfer}.
     *
     * Requirements:
     *
     * - `to` cannot be the zero address.
     * - the caller must have a balance of at least `amount`.
     */
    function transfer(address to, uint256 amount) public virtual override returns (bool) {
        address owner = _msgSender();
        _transfer(owner, to, amount);
        return true;
    }

    /**
     * @dev See {IERC20-allowance}.
     */
    function allowance(address owner, address spender) public view virtual override returns (uint256) {
        return _allowances[owner][spender];
    }

    /**
     * @dev See {IERC20-approve}.
     *
     * NOTE: If `amount` is the maximum `uint256`, the allowance is not updated on
     * `transferFrom`. This is semantically equivalent to an infinite approval.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     */
    function approve(address spender, uint256 amount) public virtual override returns (bool) {
        address owner = _msgSender();
        _approve(owner, spender, amount);
        return true;
    }

    /**
     * @dev See {IERC20-transferFrom}.
     *
     * Emits an {Approval} event indicating the updated allowance. This is not
     * required by the EIP. See the note at the beginning of {ERC20}.
     *
     * NOTE: Does not update the allowance if the current allowance
     * is the maximum `uint256`.
     *
     * Requirements:
     *
     * - `from` and `to` cannot be the zero address.
     * - `from` must have a balance of at least `amount`.
     * - the caller must have allowance for ``from``'s tokens of at least
     * `amount`.
     */
    function transferFrom(address from, address to, uint256 amount) public virtual override returns (bool) {
        address spender = _msgSender();
        _spendAllowance(from, spender, amount);
        _transfer(from, to, amount);
        return true;
    }

    /**
     * @dev Atomically increases the allowance granted to `spender` by the caller.
     *
     * This is an alternative to {approve} that can be used as a mitigation for
     * problems described in {IERC20-approve}.
     *
     * Emits an {Approval} event indicating the updated allowance.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     */
    function increaseAllowance(address spender, uint256 addedValue) public virtual returns (bool) {
        address owner = _msgSender();
        _approve(owner, spender, allowance(owner, spender) + addedValue);
        return true;
    }

    /**
     * @dev Atomically decreases the allowance granted to `spender` by the caller.
     *
     * This is an alternative to {approve} that can be used as a mitigation for
     * problems described in {IERC20-approve}.
     *
     * Emits an {Approval} event indicating the updated allowance.
     *
     * Requirements:
     *
     * - `spender` cannot be the zero address.
     * - `spender` must have allowance for the caller of at least
     * `subtractedValue`.
     */
    function decreaseAllowance(address spender, uint256 subtractedValue) public virtual returns (bool) {
        address owner = _msgSender();
        uint256 currentAllowance = allowance(owner, spender);
        require(currentAllowance >= subtractedValue, "ERC20: decreased allowance below zero");
        unchecked {
            _approve(owner, spender, currentAllowance - subtractedValue);
        }

        return true;
    }

    /**
     * @dev Moves `amount` of tokens from `from` to `to`.
     *
     * This internal function is equivalent to {transfer}, and can be used to
     * e.g. implement automatic token fees, slashing mechanisms, etc.
     *
     * Emits a {Transfer} event.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `from` must have a balance of at least `amount`.
     */
    function _transfer(address from, address to, uint256 amount) internal virtual {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");

        _beforeTokenTransfer(from, to, amount);

        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "ERC20: transfer amount exceeds balance");
        unchecked {
            _balances[from] = fromBalance - amount;
            // Overflow not possible: the sum of all balances is capped by totalSupply, and the sum is preserved by
            // decrementing then incrementing.
            _balances[to] += amount;
        }

        emit Transfer(from, to, amount);

        _afterTokenTransfer(from, to, amount);
    }

    /** @dev Creates `amount` tokens and assigns them to `account`, increasing
     * the total supply.
     *
     * Emits a {Transfer} event with `from` set to the zero address.
     *
     * Requirements:
     *
     * - `account` cannot be the zero address.
     */
    function _mint(address account, uint256 amount) internal virtual {
        require(account != address(0), "ERC20: mint to the zero address");

        _beforeTokenTransfer(address(0), account, amount);

        _totalSupply += amount;
        unchecked {
            // Overflow not possible: balance + amount is at most totalSupply + amount, which is checked above.
            _balances[account] += amount;
        }
        emit Transfer(address(0), account, amount);

        _afterTokenTransfer(address(0), account, amount);
    }

    /**
     * @dev Destroys `amount` tokens from `account`, reducing the
     * total supply.
     *
     * Emits a {Transfer} event with `to` set to the zero address.
     *
     * Requirements:
     *
     * - `account` cannot be the zero address.
     * - `account` must have at least `amount` tokens.
     */
    function _burn(address account, uint256 amount) internal virtual {
        require(account != address(0), "ERC20: burn from the zero address");

        _beforeTokenTransfer(account, address(0), amount);

        uint256 accountBalance = _balances[account];
        require(accountBalance >= amount, "ERC20: burn amount exceeds balance");
        unchecked {
            _balances[account] = accountBalance - amount;
            // Overflow not possible: amount <= accountBalance <= totalSupply.
            _totalSupply -= amount;
        }

        emit Transfer(account, address(0), amount);

        _afterTokenTransfer(account, address(0), amount);
    }

    /**
     * @dev Sets `amount` as the allowance of `spender` over the `owner` s tokens.
     *
     * This internal function is equivalent to `approve`, and can be used to
     * e.g. set automatic allowances for certain subsystems, etc.
     *
     * Emits an {Approval} event.
     *
     * Requirements:
     *
     * - `owner` cannot be the zero address.
     * - `spender` cannot be the zero address.
     */
    function _approve(address owner, address spender, uint256 amount) internal virtual {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");

        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    /**
     * @dev Updates `owner` s allowance for `spender` based on spent `amount`.
     *
     * Does not update the allowance amount in case of infinite allowance.
     * Revert if not enough allowance is available.
     *
     * Might emit an {Approval} event.
     */
    function _spendAllowance(address owner, address spender, uint256 amount) internal virtual {
        uint256 currentAllowance = allowance(owner, spender);
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= amount, "ERC20: insufficient allowance");
            unchecked {
                _approve(owner, spender, currentAllowance - amount);
            }
        }
    }

    /**
     * @dev Hook that is called before any transfer of tokens. This includes
     * minting and burning.
     *
     * Calling conditions:
     *
     * - when `from` and `to` are both non-zero, `amount` of ``from``'s tokens
     * will be transferred to `to`.
     * - when `from` is zero, `amount` tokens will be minted for `to`.
     * - when `to` is zero, `amount` of ``from``'s tokens will be burned.
     * - `from` and `to` are never both zero.
     *
     * To learn more about hooks, head to xref:ROOT:extending-contracts.adoc#using-hooks[Using Hooks].
     */
    function _beforeTokenTransfer(address from, address to, uint256 amount) internal virtual {}

    /**
     * @dev Hook that is called after any transfer of tokens. This includes
     * minting and burning.
     *
     * Calling conditions:
     *
     * - when `from` and `to` are both non-zero, `amount` of ``from``'s tokens
     * has been transferred to `to`.
     * - when `from` is zero, `amount` tokens have been minted for `to`.
     * - when `to` is zero, `amount` of ``from``'s tokens have been burned.
     * - `from` and `to` are never both zero.
     *
     * To learn more about hooks, head to xref:ROOT:extending-contracts.adoc#using-hooks[Using Hooks].
     */
    function _afterTokenTransfer(address from, address to, uint256 amount) internal virtual {}
}

contract XSDStablecoin is ERC20Custom {

    /* ========== STATE VARIABLES ========== */
    enum PriceChoice { XSD, BankX }
    ChainlinkETHUSDPriceConsumer private eth_usd_pricer;
    ChainlinkXAGUSDPriceConsumer private xag_usd_pricer;
    uint8 private eth_usd_pricer_decimals;
    uint8 private xag_usd_pricer_decimals;
    string public symbol;
    string public name;
    uint8 public constant decimals = 18;
    address public treasury; 
    address public collateral_pool_address;
    address public router;
    address public eth_usd_oracle_address;
    address public xag_usd_oracle_address;
    address public smartcontract_owner;
    IBankXWETHpool private bankxEthPool;
    IXSDWETHpool private xsdEthPool;
    uint256 public cap_rate;
    uint256 public genesis_supply; 

    // The addresses in this array are added by the oracle and these contracts are able to mint xsd
    address[] public xsd_pools_array;

    // Mapping is also used for faster verification
    mapping(address => bool) public xsd_pools; 

    // Constants for various precisions
    uint256 private constant PRICE_PRECISION = 1e6;

    /* ========== MODIFIERS ========== */

    modifier onlyPools() {
       require(xsd_pools[msg.sender] == true, "Only xsd pools can call this function");
        _;//check happens before the function is executed 
    } 

    modifier onlyByOwner(){
        require(msg.sender == smartcontract_owner, "You are not the owner");
        _;
    }

    modifier onlyByOwnerOrPool() {
        require(
            msg.sender == smartcontract_owner  
            || xsd_pools[msg.sender] == true, 
            "You are not the owner or a pool");
        _;
    }

    /* ========== CONSTRUCTOR ========== */

    constructor(
        string memory _name,
        string memory _symbol,
        uint256 _pool_amount,
        uint256 _genesis_supply,
        address _smartcontract_owner,
        address _treasury,
        uint256 _cap_rate
    ) {
        require((_smartcontract_owner != address(0))
                && (_treasury != address(0)), "Zero address detected"); 
        name = _name;
        symbol = _symbol;
        genesis_supply = _genesis_supply + _pool_amount;
        treasury = _treasury;
        _mint(_smartcontract_owner, _pool_amount);
        _mint(treasury, _genesis_supply);
        smartcontract_owner = _smartcontract_owner;
        cap_rate = _cap_rate;// Maximum mint amount
    }
    /* ========== VIEWS ========== */

    function eth_usd_price() public view returns (uint256) {
        return (uint256(eth_usd_pricer.getLatestPrice())*PRICE_PRECISION)/(uint256(10) ** eth_usd_pricer_decimals);
    }
    //silver price
    //hard coded value for testing on goerli
    function xag_usd_price() public view returns (uint256) {
        return (uint256(xag_usd_pricer.getLatestPrice())*PRICE_PRECISION)/(uint256(10) ** xag_usd_pricer_decimals);
    }

    /* ========== PUBLIC FUNCTIONS ========== */

    function creatorMint(uint256 amount) public onlyByOwner{
        require(genesis_supply+amount<cap_rate,"cap limit reached");
        super._mint(treasury,amount);
    }

    /* ========== RESTRICTED FUNCTIONS ========== */

    // Used by pools when user redeems
    function pool_burn_from(address b_address, uint256 b_amount) public onlyPools {
        super._burnFrom(b_address, b_amount);
        emit XSDBurned(b_address, msg.sender, b_amount);
    }

    // This function is what other xsd pools will call to mint new XSD 
    function pool_mint(address m_address, uint256 m_amount) public onlyPools {
        super._mint(m_address, m_amount);
        emit XSDMinted(msg.sender, m_address, m_amount);
    }
    

    // Adds collateral addresses supported, such as tether and busd, must be ERC20 
    function addPool(address pool_address) public onlyByOwner {
        require(pool_address != address(0), "Zero address detected");

        require(xsd_pools[pool_address] == false, "Address already exists");
        xsd_pools[pool_address] = true; 
        xsd_pools_array.push(pool_address);

        emit PoolAdded(pool_address);
    }

    // Remove a pool 
    function removePool(address pool_address) public onlyByOwner {
        require(pool_address != address(0), "Zero address detected");

        require(xsd_pools[pool_address] == true, "Address nonexistant");
        
        // Delete from the mapping
        delete xsd_pools[pool_address];

        // 'Delete' from the array by setting the address to 0x0
        for (uint i = 0; i < xsd_pools_array.length; i++){ 
            if (xsd_pools_array[i] == pool_address) {
                xsd_pools_array[i] = address(0); // This will leave a null in the array and keep the indices the same
                break;
            }
        }

        emit PoolRemoved(pool_address);
    }
// create a seperate function for users and the pool
    function burnpoolXSD(uint _xsdamount) public {
        require(msg.sender == router, "Only the router can access this function");
        require(totalSupply()-ICollateralPool(payable(collateral_pool_address)).collat_XSD()>_xsdamount, "uXSD has to be positive");
        super._burn(address(xsdEthPool),_xsdamount);
        xsdEthPool.sync();
        emit XSDBurned(msg.sender, address(this), _xsdamount);
    }
    // add burn function for users
    function burnUserXSD(uint _xsdamount) public {
        require(totalSupply()-ICollateralPool(payable(collateral_pool_address)).collat_XSD()>_xsdamount, "uXSD has to be positive");
        super._burn(msg.sender, _xsdamount);
        emit XSDBurned(msg.sender, address(this), _xsdamount);
    }

    function setTreasury(address _new_treasury) public onlyByOwner {
        require(_new_treasury != address(0), "Zero address detected");
        treasury = _new_treasury;
    }

    function setETHUSDOracle(address _eth_usd_oracle_address) public onlyByOwner {
        require(_eth_usd_oracle_address != address(0), "Zero address detected");

        eth_usd_oracle_address = _eth_usd_oracle_address;
        eth_usd_pricer = ChainlinkETHUSDPriceConsumer(eth_usd_oracle_address);
        eth_usd_pricer_decimals = eth_usd_pricer.getDecimals();

        emit ETHUSDOracleSet(_eth_usd_oracle_address);
    }
    
    function setXAGUSDOracle(address _xag_usd_oracle_address) public onlyByOwner {
        require(_xag_usd_oracle_address != address(0), "Zero address detected");

        xag_usd_oracle_address = _xag_usd_oracle_address;
        xag_usd_pricer = ChainlinkXAGUSDPriceConsumer(xag_usd_oracle_address);
        xag_usd_pricer_decimals = xag_usd_pricer.getDecimals();

        emit XAGUSDOracleSet(_xag_usd_oracle_address);
    }

    function setRouterAddress(address _router) external onlyByOwner {
        require(_router != address(0), "Zero address detected");
        router = _router;
    }

    // Sets the XSD_ETH Uniswap oracle address 
    function setXSDEthPool(address _xsd_pool_addr) public onlyByOwner {
        require(_xsd_pool_addr != address(0), "Zero address detected");
        xsdEthPool = IXSDWETHpool(_xsd_pool_addr); 

        emit XSDETHPoolSet(_xsd_pool_addr);
    }

    // Sets the BankX_ETH Uniswap oracle address 
    function setBankXEthPool(address _bankx_pool_addr) public onlyByOwner {
        require(_bankx_pool_addr != address(0), "Zero address detected");
        bankxEthPool = IBankXWETHpool(_bankx_pool_addr);

        emit BankXEthPoolSet(_bankx_pool_addr);
    }

    //sets the collateral pool address
    function setCollateralEthPool(address _collateral_pool_address) public onlyByOwner {
        require(_collateral_pool_address != address(0), "Zero address detected");
        collateral_pool_address = payable(_collateral_pool_address);
    }

    function setSmartContractOwner(address _smartcontract_owner) external{
        require(msg.sender == smartcontract_owner, "Only the smart contract owner can access this function");
        require(_smartcontract_owner != address(0), "Zero address detected");
        smartcontract_owner = _smartcontract_owner;
    }

    function renounceOwnership() external{
        require(msg.sender == smartcontract_owner, "Only the smart contract owner can access this function");
        smartcontract_owner = address(0);
    }

    
    /* ========== EVENTS ========== */

    // Track XSD burned
    event XSDBurned(address indexed from, address indexed to, uint256 amount);
    // Track XSD minted
    event XSDMinted(address indexed from, address indexed to, uint256 amount);
    event PoolAdded(address pool_address);
    event PoolRemoved(address pool_address);
    event RedemptionFeeSet(uint256 red_fee);
    event MintingFeeSet(uint256 min_fee);
    event ETHUSDOracleSet(address eth_usd_oracle_address);
    event XAGUSDOracleSet(address xag_usd_oracle_address);
    event XSDETHPoolSet(address xsd_pool_addr);
    event BankXEthPoolSet(address bankx_pool_addr);
}


// File contracts/BankX/BankXToken.sol





/// @title BankX Token Contract
/// @notice This contract manages the BankX token, including minting, burning, and setting key addresses.
contract BankXToken is ERC20Custom {

    /* ========== STATE VARIABLES ========== */

    /// @notice The name of the token.
    string public name;
    
    /// @notice The symbol of the token.
    string public symbol;
    
    /// @notice The number of decimals the token uses.
    uint8 public constant decimals = 18;

    /// @notice The address of the router contract.
    address public router;
    
    /// @notice The address of the treasury.
    address public treasury;
    
    /// @notice The instance of the XSD Stablecoin contract.
    XSDStablecoin private XSD;
    
    /// @notice The address of the pool contract.
    address public pool_address; 
    
    /// @notice The total initial supply of the token.
    uint256 public genesis_supply;
    
    /// @notice The owner of the smart contract.
    address public smartcontract_owner;

    /* ========== MODIFIERS ========== */

    /// @notice Modifier to restrict access to only authorized pool contracts.
    modifier onlyPools() {
        require(XSD.xsd_pools(msg.sender) == true, "BANKX:FORBIDDEN");
        _;
    } 
    
    /// @notice Modifier to restrict access to only the owner of the contract.
    modifier onlyByOwner() {
        require(msg.sender == smartcontract_owner, "BANKX:FORBIDDEN");
        _;
    }

    /* ========== CONSTRUCTOR ========== */
    
    /// @notice Constructor to initialize the BankXToken contract.
    /// @param _treasury The address of the treasury.
    /// @param _name The name of the token.
    /// @param _pool_amount The amount of tokens allocated to the pool.
    /// @param _symbol The symbol of the token.
    /// @param _genesis_supply The initial total supply of the token.
    /// @param _smartcontract_owner The owner of the smart contract.
    constructor(
        address _treasury,
        string memory _name,
        uint256 _pool_amount,
        string memory _symbol, 
        uint256 _genesis_supply,
        address _smartcontract_owner
    ) {
        require((_treasury != address(0)), "BANKX:ZEROCHECK"); 
        name = _name;
        symbol = _symbol;
        treasury = _treasury;
        _mint(treasury, _genesis_supply);
        _mint(_msgSender(), _pool_amount);
        smartcontract_owner = _smartcontract_owner;
        genesis_supply = _genesis_supply + _pool_amount;
    }

    /* ========== RESTRICTED FUNCTIONS ========== */
    
    /// @notice Sets the BankX Pool contract address in the contract.
    /// @param new_pool The new BankX Pool contract address.
    function setPool(address new_pool) external onlyByOwner {
        require(new_pool != address(0), "BANKX:ZEROCHECK");
        pool_address = new_pool;
    }

    /// @notice Sets the treasury address in the contract.
    /// @param new_treasury The new treasury address.
    function setTreasury(address new_treasury) external onlyByOwner {
        require(new_treasury != address(0), "BANKX:ZEROCHECK");
        treasury = new_treasury;
    }

    /// @notice Sets the router address in the contract.
    /// @param _router The new router address.
    function setRouterAddress(address _router) external onlyByOwner {
        require(_router != address(0), "BANKX:ZEROCHECK");
        router = _router;
    }

    /// @notice Sets the XSD StableCoin instance in the contract.
    /// @param xsd_contract_address The XSD contract address.
    function setXSDAddress(address xsd_contract_address) external onlyByOwner {
        require(xsd_contract_address != address(0), "BANKX:ZEROCHECK");
        XSD = XSDStablecoin(xsd_contract_address);
        emit XSDAddressSet(xsd_contract_address);
    }

    /// @notice Mints BankX Tokens. Only accessible to pool addresses.
    /// @param to The address receiving the minted tokens.
    /// @param amount The amount of tokens to mint.
    function mint(address to, uint256 amount) public onlyPools {
        _mint(to, amount);
        emit BankXMinted(address(this), to, amount);
    }
    
    /// @notice Allows pool addresses to mint BankX Tokens.
    /// @param m_address The address receiving the minted tokens.
    /// @param m_amount The amount of tokens to mint.
    function pool_mint(address m_address, uint256 m_amount) external onlyPools  {        
        super._mint(m_address, m_amount);
        emit BankXMinted(address(this), m_address, m_amount);
    }

    /// @notice Allows pool addresses to burn BankX Tokens.
    /// @param b_address The address burning the tokens.
    /// @param b_amount The amount of tokens to burn.
    function pool_burn_from(address b_address, uint256 b_amount) external onlyPools {
        super._burnFrom(b_address, b_amount);
        emit BankXBurned(b_address, address(this), b_amount);
    }
    
    /// @notice Burns BankX tokens from the pool when BankX is inflationary.
    /// @param _bankx_amount The amount of tokens to burn.
    function burnpoolBankX(uint _bankx_amount) public {
        require(msg.sender == router, "BANKX:FORBIDDEN");
        require(totalSupply() > genesis_supply, "BankX must be deflationary");
        super._burn(pool_address, _bankx_amount);
        IBankXWETHpool(pool_address).sync();
        emit BankXBurned(msg.sender, address(this), _bankx_amount);
    }

    /// @notice Allows the owner to reset the smart contract address.
    /// @param _smartcontract_owner The new owner address.
    function setSmartContractOwner(address _smartcontract_owner) external {
        require(msg.sender == smartcontract_owner, "BANKX:FORBIDDEN");
        require(_smartcontract_owner != address(0), "BANKX:ZEROCHECK");
        smartcontract_owner = _smartcontract_owner;
    }

    /// @notice Allows the owner to renounce ownership of the contract.
    function renounceOwnership() external {
        require(msg.sender == smartcontract_owner, "BANKX:FORBIDDEN");
        smartcontract_owner = address(0);
    }

    /* ========== EVENTS ========== */

    /// @notice Emitted when BankX tokens are burned.
    /// @param from The address initiating the burn.
    /// @param to The address where the tokens are burned.
    /// @param amount The amount of tokens burned.
    event BankXBurned(address indexed from, address indexed to, uint256 amount);
    
    /// @notice Emitted when BankX tokens are minted.
    /// @param from The address initiating the mint.
    /// @param to The address receiving the minted tokens.
    /// @param amount The amount of tokens minted.
    event BankXMinted(address indexed from, address indexed to, uint256 amount);
    
    /// @notice Emitted when the XSD address is set.
    /// @param addr The address of the XSD contract.
    event XSDAddressSet(address addr);
}


// File contracts/UniswapFork/Interfaces/IRouter.sol




interface IRouter{
    function creatorAddLiquidityTokens(
        address tokenB,
        uint amountB,
        uint deadline
    ) external;

    function creatorAddLiquidityETH(
        address pool,
        uint deadline
    ) external payable;

    function userAddLiquidityETH(
        address pool,
        uint deadline
    ) external payable;

    function userRedeemLiquidity(
        address pool,
        uint deadline
    ) external;

    function swapETHForXSD(uint amountOut,uint deadline) external payable;

    function swapXSDForETH(uint amountOut, uint amountInMax, uint deadline) external;

    function swapETHForBankX(uint amountOut, uint deadline) external payable;
    
    function swapBankXForETH(uint amountOut, uint amountInMax, uint deadline) external;

    function swapBankXForXSD(uint bankx_amount, address sender, uint256 eth_min_amount, uint256 bankx_min_amount, uint256 deadline) external;

    function swapXSDForBankX(uint XSD_amount, address sender, uint256 eth_min_amount, uint256 xsd_min_amount, uint256 deadline) external;
}


// File contracts/Utils/Initializable.sol


/**
 * @title Initializable
 *
 * @dev Helper contract to support initializer functions. To use it, replace
 * the constructor with a function that has the `initializer` modifier.
 * WARNING: Unlike constructors, initializer functions must be manually
 * invoked. This applies both to deploying an Initializable contract, as well
 * as extending an Initializable contract via inheritance.
 * WARNING: When used with inheritance, manual care must be taken to not invoke
 * a parent initializer twice, or ensure that all initializers are idempotent,
 * because this is not dealt with automatically as with constructors.
 */
contract Initializable {

  /**
   * @dev Indicates that the contract has been initialized.
   */
  bool private initialized;

  /**
   * @dev Indicates that the contract is in the process of being initialized.
   */
  bool private initializing;

  /**
   * @dev Modifier to use in the initializer function of a contract.
   */
  modifier initializer() {
    require(initializing || isConstructor() || !initialized, "Contract instance has already been initialized");

    bool isTopLevelCall = !initializing;
    if (isTopLevelCall) {
      initializing = true;
      initialized = true;
    }

    _;

    if (isTopLevelCall) {
      initializing = false;
    }
  }

  /// @dev Returns true if and only if the function is running in the constructor
  function isConstructor() private view returns (bool) {
    // extcodesize checks the size of the code stored in an address, and
    // address returns the current address. Since the code is still not
    // deployed when running a constructor, any checks on its code size will
    // yield zero, making it an effective way to detect if a contract is
    // under construction or not.
    address self = address(this);
    uint256 cs;
    assembly { cs := extcodesize(self) }
    return cs == 0;
  }

  // Reserved storage space to allow for layout changes in the future.
  uint256[50] private ______gap;
}


// File contracts/Oracle/Interfaces/IPIDController.sol



interface IPIDController{
    function bucket1() external view returns (bool);
    function bucket2() external view returns (bool);
    function bucket3() external view returns (bool);
    function diff1() external view returns (uint);
    function diff2() external view returns (uint);
    function diff3() external view returns (uint);
    function amountpaid1() external view returns (uint);
    function amountpaid2() external view returns (uint);
    function amountpaid3() external view returns (uint);
    function bankx_updated_price() external view returns (uint);
    function xsd_updated_price() external view returns (uint);
    function global_collateral_ratio() external view returns(uint);
    function interest_rate() external view returns(uint);
    function neededWETH() external view returns(uint);
    function neededBankX() external view returns(uint);
    function systemCalculations(bytes[] calldata priceUpdateData) external;
    struct PriceCheck{
        uint256 lastpricecheck;
        bool pricecheck;
    }
    function lastPriceCheck(address user) external view returns (PriceCheck memory info);
    function setPriceCheck(address sender) external;
    function amountPaidBankXWETH(uint ethvalue) external;
    function amountPaidXSDWETH(uint ethvalue) external;
    function amountPaidCollateralPool(uint ethvalue) external;
}


// File contracts/XSD/Pools/Interfaces/IRewardManager.sol



interface IRewardManager {
function creatorProvideBankXLiquidity() external;
function creatorProvideXSDLiquidity() external;
function userProvideBankXLiquidity(address to) external;
function userProvideXSDLiquidity(address to) external;
function userProvideCollatPoolLiquidity(address to, uint amount) external;
function LiquidityRedemption(address pool,address to) external;
}


// File @uniswap/lib/contracts/libraries/TransferHelper.sol@v4.0.1-alpha

// helper methods for interacting with ERC20 tokens and sending ETH that do not consistently return true/false
library TransferHelper {
    function safeApprove(
        address token,
        address to,
        uint256 value
    ) internal {
        // bytes4(keccak256(bytes('approve(address,uint256)')));
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0x095ea7b3, to, value));
        require(
            success && (data.length == 0 || abi.decode(data, (bool))),
            'TransferHelper::safeApprove: approve failed'
        );
    }

    function safeTransfer(
        address token,
        address to,
        uint256 value
    ) internal {
        // bytes4(keccak256(bytes('transfer(address,uint256)')));
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0xa9059cbb, to, value));
        require(
            success && (data.length == 0 || abi.decode(data, (bool))),
            'TransferHelper::safeTransfer: transfer failed'
        );
    }

    function safeTransferFrom(
        address token,
        address from,
        address to,
        uint256 value
    ) internal {
        // bytes4(keccak256(bytes('transferFrom(address,address,uint256)')));
        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0x23b872dd, from, to, value));
        require(
            success && (data.length == 0 || abi.decode(data, (bool))),
            'TransferHelper::transferFrom: transferFrom failed'
        );
    }

    function safeTransferETH(address to, uint256 value) internal {
        (bool success, ) = to.call{value: value}(new bytes(0));
        require(success, 'TransferHelper::safeTransferETH: ETH transfer failed');
    }
}


// File @openzeppelin/contracts/security/ReentrancyGuard.sol@v4.9.6


// OpenZeppelin Contracts (last updated v4.9.0) (security/ReentrancyGuard.sol)



/**
 * @dev Contract module that helps prevent reentrant calls to a function.
 *
 * Inheriting from `ReentrancyGuard` will make the {nonReentrant} modifier
 * available, which can be applied to functions to make sure there are no nested
 * (reentrant) calls to them.
 *
 * Note that because there is a single `nonReentrant` guard, functions marked as
 * `nonReentrant` may not call one another. This can be worked around by making
 * those functions `private`, and then adding `external` `nonReentrant` entry
 * points to them.
 *
 * TIP: If you would like to learn more about reentrancy and alternative ways
 * to protect against it, check out our blog post
 * https://blog.openzeppelin.com/reentrancy-after-istanbul/[Reentrancy After Istanbul].
 */
abstract contract ReentrancyGuard {
    // Booleans are more expensive than uint256 or any type that takes up a full
    // word because each write operation emits an extra SLOAD to first read the
    // slot's contents, replace the bits taken up by the boolean, and then write
    // back. This is the compiler's defense against contract upgrades and
    // pointer aliasing, and it cannot be disabled.

    // The values being non-zero value makes deployment a bit more expensive,
    // but in exchange the refund on every call to nonReentrant will be lower in
    // amount. Since refunds are capped to a percentage of the total
    // transaction's gas, it is best to keep them low in cases like this one, to
    // increase the likelihood of the full refund coming into effect.
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
    }

    /**
     * @dev Prevents a contract from calling itself, directly or indirectly.
     * Calling a `nonReentrant` function from another `nonReentrant`
     * function is not supported. It is possible to prevent this from happening
     * by making the `nonReentrant` function external, and making it call a
     * `private` function that does the actual work.
     */
    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {
        // On the first call to nonReentrant, _status will be _NOT_ENTERED
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");

        // Any calls to nonReentrant after this point will fail
        _status = _ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = _NOT_ENTERED;
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == _ENTERED;
    }
}


// File contracts/UniswapFork/Router.sol














//swap first
//then burn 10% using different function maybe
//recalculate price
// do not burn uXSD if there is a deficit
contract Router is IRouter, Initializable, ReentrancyGuard {

    address public WETH;
    address public collateral_pool_address;
    address public XSDWETH_pool_address;
    address public BankXWETH_pool_address;
    address public reward_manager_address;
    address public arbitrage;
    address public bankx_address;
    address public xsd_address;
    address public treasury;
    address public smartcontract_owner;
    uint public block_delay;
    bool public swap_paused;
    bool public liquidity_paused;
    XSDStablecoin private XSD;
    IRewardManager private reward_manager;
    IPIDController private pid_controller;

     // Circuit breaker state variables
    struct Safety {
        bool is_active;
        uint256 last_price;
    }
    
    mapping(uint8 => uint256) price_threshold;
    mapping(uint8 => uint256) volume_threshold;
    // Individual circuit breakers for each swap function
    Safety public eth_to_xsd_breaker;
    Safety public xsd_to_eth_breaker;
    Safety public eth_to_bankx_breaker;
    Safety public bankx_to_eth_breaker;

    // Add this modifier for circuit breaker checks
    modifier safetyCheck(uint8 breaker_type) {
        require(!getSafety(breaker_type).is_active, "Circuit breaker is active");
        _;
    }
/**
 * @dev Modifier to ensure the function is only executed before a certain deadline.
 * @param deadline The deadline timestamp that must be greater than or equal to the current block timestamp.
 */
modifier ensure(uint deadline) {
        require(deadline >= block.timestamp, 'ROUTER:EXPIRED');
        _;
    }

/**
 * @dev Modifier to ensure an address is not the zero address.
 * @param _address The address to check.
 */
modifier nonZeroAddress(address _address) {
    require(_address != address(0), "ROUTER:ZEROCHECK");
    _;
}

/**
 * @dev Modifier to ensure the function can only be called by the contract owner.
 */
modifier onlyByOwner(){
        require(msg.sender == smartcontract_owner, "ROUTER:FORBIDDEN");
        _;
    }

/**
 * @dev Modifier to ensure the function can only be executed when swaps are not paused.
 */
modifier swapPaused(){
        require(!swap_paused, "ROUTER:PAUSED");
        _;
    }

/**
 * @dev Modifier to enforce a block delay between price checks.
 * @notice Updates the price check after the function execution.
 */
modifier blockDelay(){
        require(((pid_controller.lastPriceCheck(msg.sender).lastpricecheck+(block_delay)) <= block.number) && (pid_controller.lastPriceCheck(msg.sender).pricecheck), "ROUTER:BLOCKDELAY");
        _;
        pid_controller.setPriceCheck(msg.sender);
    }
/**
 * @dev Initializes the contract with the specified addresses and parameters.
 * @param _bankx_address Address of the BankX contract.
 * @param _xsd_address Address of the XSD Stablecoin contract.
 * @param _XSDWETH_pool Address of the XSD/WETH pool.
 * @param _BankXWETH_pool Address of the BankX/WETH pool.
 * @param _collateral_pool Address of the collateral pool.
 * @param _reward_manager_address Address of the reward manager contract.
 * @param _pid_address Address of the PID controller contract.
 * @param _treasury Address of the treasury.
 * @param _smartcontract_owner Address of the contract owner.
 * @param _WETH Address of the WETH contract.
 * @param _block_delay Number of blocks required between price checks.
 */
function initialize(address _bankx_address, address _xsd_address,address _XSDWETH_pool, address _BankXWETH_pool,address _collateral_pool,address _reward_manager_address,address _pid_address,address _treasury, address _smartcontract_owner,address _WETH, uint _block_delay) public initializer {
        require((_bankx_address != address(0))
        &&(_xsd_address != address(0))
        &&(_XSDWETH_pool != address(0))
        &&(_BankXWETH_pool != address(0))
        &&(_collateral_pool != address(0))
        &&(_treasury != address(0))
        &&(_pid_address != address(0))
        &&(_smartcontract_owner != address(0))
        &&(_WETH != address(0)), "ROUTER:ZEROCHECK");
        bankx_address = _bankx_address;
        xsd_address = _xsd_address;
        XSDWETH_pool_address = _XSDWETH_pool;
        BankXWETH_pool_address = _BankXWETH_pool;
        collateral_pool_address = _collateral_pool;
        reward_manager_address = _reward_manager_address;
        reward_manager = IRewardManager(_reward_manager_address);
        pid_controller = IPIDController(_pid_address);
        XSD = XSDStablecoin(_xsd_address);
        treasury = _treasury;
        WETH = _WETH;
        smartcontract_owner = _smartcontract_owner;
        block_delay = _block_delay;
        price_threshold[0] = 1000;
        volume_threshold[0] = 1000 ether;
        
        price_threshold[1] = 1000;
        volume_threshold[1] = 1000 ether;
        
        price_threshold[2] = 1000;
        volume_threshold[2] = 1000 ether;
        
        price_threshold[3] = 1000;
        volume_threshold[3] = 1000 ether;

        emit RouterInitialized(
            _bankx_address,
            _xsd_address,
            _XSDWETH_pool,
            _BankXWETH_pool,
            _collateral_pool,
            _block_delay
        );
    }

/**
 * @dev Fallback function to accept ETH only from the WETH contract.
 */
receive() external payable {
        assert(msg.sender == WETH); // only accept ETH via fallback from the WETH contract
    }

    function setSafetyThresholds(
        uint8 breaker_type,
        uint256 _price_threshold,
        uint256 _volume_threshold
    ) external onlyByOwner {
        require(breaker_type <= 3, "Invalid breaker type");
        require(_price_threshold > 0, "Price threshold must be > 0");
        require(_volume_threshold > 0, "Volume threshold must be > 0");
        
        price_threshold[breaker_type] = _price_threshold;
        volume_threshold[breaker_type] = _volume_threshold;
        
        emit SafetyThresholdsUpdated(
            breaker_type,
            _price_threshold,
            _volume_threshold
        );
    }

    function toggleSafety(uint8 breaker_type) external onlyByOwner {
        require(breaker_type <= 3, "Invalid breaker type");
        Safety storage breaker = getSafety(breaker_type);
        breaker.is_active = !breaker.is_active;
        
        string memory breaker_name = getBreakerName(breaker_type);
        emit SafetyToggled(breaker_name, breaker.is_active);
    }

    function getBreakerName(uint8 breaker_type) internal pure returns (string memory) {
        if (breaker_type == 0) return "ETH to XSD";
        if (breaker_type == 1) return "XSD to ETH";
        if (breaker_type == 2) return "ETH to BankX";
        if (breaker_type == 3) return "BankX to ETH";
        revert("Invalid breaker type");
    }

    function getSafety(uint8 breaker_type) internal view returns (Safety storage) {
        if (breaker_type == 0) return eth_to_xsd_breaker;
        if (breaker_type == 1) return xsd_to_eth_breaker;
        if (breaker_type == 2) return eth_to_bankx_breaker;
        if (breaker_type == 3) return bankx_to_eth_breaker;
        revert("Invalid breaker type");
    }

    function checkAndUpdateSafety(
        uint8 breaker_type,
        uint256 current_price,
        uint256 volume
    ) internal {
        require(breaker_type <= 3, "Invalid breaker type");
        Safety storage breaker = getSafety(breaker_type);

        // Volume check
        if (volume > volume_threshold[breaker_type]) {
            breaker.is_active = true;
            emit SafetyTripped(
                getBreakerName(breaker_type),
                "Volume Exceeded",
                volume
            );
        }
        // Price deviation check
        if (breaker.last_price > 0) {
            uint256 price_deviation = abs(current_price, breaker.last_price);
            uint256 deviation_percentage = (price_deviation *10000) / breaker.last_price;

            if (deviation_percentage > price_threshold[breaker_type]) {
                breaker.is_active = true;
                emit SafetyTripped(
                    getBreakerName(breaker_type),
                    "Price Deviation",
                    deviation_percentage
                );
            }
        }
        breaker.last_price = current_price;
    }

    function abs(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a - b : b - a;
    }

    // **** ADD LIQUIDITY ****
    // @notice Allows the creator to provide liquidity to the specified pool (XSD/WETH or BankX/WETH).
// @param pool The address of the pool where liquidity is to be added.
function creatorProvideLiquidity(address pool) internal  {
    if(pool == XSDWETH_pool_address){
        reward_manager.creatorProvideXSDLiquidity();
    }
    else if(pool == BankXWETH_pool_address){
        reward_manager.creatorProvideBankXLiquidity();
    }
}

// @notice Allows a user to provide liquidity to the specified pool (XSD/WETH or BankX/WETH).
// @param pool The address of the pool where liquidity is to be added.
// @param sender The address of the user providing liquidity.
function userProvideLiquidity(address pool, address sender) internal  {
    if(pool == XSDWETH_pool_address){
        reward_manager.userProvideXSDLiquidity(sender);
    }
    else if(pool == BankXWETH_pool_address){
        reward_manager.userProvideBankXLiquidity(sender);
    }
}

// @notice Allows the creator to add liquidity tokens (XSD or BankX) to a pool.
// @param tokenB The address of the token to be added (either XSD or BankX).
// @param amountB The amount of the token to be added as liquidity.
// @param deadline The timestamp after which the transaction will revert if not completed.
function creatorAddLiquidityTokens(
    address tokenB,
    uint amountB,
    uint deadline
) public ensure(deadline) override {
    require(msg.sender == treasury || msg.sender == smartcontract_owner, "ROUTER:FORBIDDEN");
    require(tokenB == xsd_address || tokenB == bankx_address, "token address is invalid");
    require(amountB>0, "Please enter a valid amount");
    if(tokenB == xsd_address){
        TransferHelper.safeTransferFrom(tokenB, msg.sender, XSDWETH_pool_address, amountB);
        reward_manager.creatorProvideXSDLiquidity();
    }
    else if(tokenB == bankx_address){
        TransferHelper.safeTransferFrom(tokenB, msg.sender, BankXWETH_pool_address, amountB);
        reward_manager.creatorProvideBankXLiquidity();
    }
    emit LiquidityTokensAdded(
            tokenB,
            msg.sender,
            amountB,
            block.timestamp
        );
}

// @notice Allows the creator to add ETH as liquidity to a specified pool (XSD/WETH or BankX/WETH).
// @param pool The address of the pool where ETH is to be added.
// @param deadline The timestamp after which the transaction will revert if not completed.
function creatorAddLiquidityETH(
    address pool,
    uint256 deadline
) external ensure(deadline) payable override {
    require(msg.sender == treasury || msg.sender == smartcontract_owner, "ROUTER:FORBIDDEN");
    require(pool == XSDWETH_pool_address || pool == BankXWETH_pool_address, "Pool address is invalid");
    require(msg.value>0,"Please enter a valid amount");
    IWETH(WETH).deposit{value: msg.value}();
    assert(IWETH(WETH).transfer(pool, msg.value));
    creatorProvideLiquidity(pool);
    emit CreatorLiquidityAdded(
            pool,
            msg.sender,
            msg.value,
            block.timestamp
        );
}

// @notice Allows a user to add ETH as liquidity to a specified pool (XSD/WETH, BankX/WETH, or Collateral).
// @param pool The address of the pool where ETH is to be added.
// @param deadline The timestamp after which the transaction will revert if not completed.
function userAddLiquidityETH(
    address pool,
    uint deadline
) external ensure(deadline) payable override{
    require(pool == XSDWETH_pool_address || pool == BankXWETH_pool_address || pool == collateral_pool_address, "Pool address is not valid");
    require(!liquidity_paused, "ROUTER:PAUSED");
    IWETH(WETH).deposit{value: msg.value}();
    assert(IWETH(WETH).transfer(pool, msg.value));
    if(pool==collateral_pool_address){
        reward_manager.userProvideCollatPoolLiquidity(msg.sender, msg.value);
    }
    else{
        userProvideLiquidity(pool, msg.sender);
    }

    emit UserLiquidityAdded(
        pool,
        msg.sender,
        msg.value,
        block.timestamp
    );
}

// @notice Allows a user to redeem liquidity from a specified pool (XSD/WETH, BankX/WETH, or Collateral).
// @param pool The address of the pool from which liquidity is to be redeemed.
// @param deadline The timestamp after which the transaction will revert if not completed.
function userRedeemLiquidity(address pool, uint deadline) external ensure(deadline){
    require(pool == XSDWETH_pool_address || pool == BankXWETH_pool_address || pool == collateral_pool_address, "Invalid pool");
    reward_manager.LiquidityRedemption(pool,msg.sender);
    emit LiquidityRedeemed(
        pool,
        msg.sender,
        block.timestamp
    );
}

    /* **** SWAP **** */

/// @notice Swaps ETH for XSD tokens
/// @dev Uses WETH as an intermediary for the swap. This function checks the reserves of the XSD/WETH pool and requires the output amount to be at least `amountOut`.
/// @param amountOut The minimum amount of XSD tokens expected from the swap
/// @param deadline The time by which the transaction must be confirmed
function swapETHForXSD(uint amountOut, uint deadline)
    external
    swapPaused
    blockDelay
    safetyCheck(0)
    ensure(deadline) 
    nonReentrant
    payable
    override
{
    //price check
    uint current_price = pid_controller.xsd_updated_price();
    (uint reserveA, uint reserveB, ) = IXSDWETHpool(XSDWETH_pool_address).getReserves();
    uint amounts = BankXLibrary.quote(msg.value, reserveB, reserveA);
    require(amounts >= amountOut, 'BankXRouter: INSUFFICIENT_OUTPUT_AMOUNT');
    // Check circuit breaker with current transaction details
    checkAndUpdateSafety(
        0,
        current_price,
        msg.value
    );
    IWETH(WETH).deposit{value: msg.value}();
    assert(IWETH(WETH).transfer(XSDWETH_pool_address, msg.value));
    IXSDWETHpool(XSDWETH_pool_address).swap(amountOut, 0, msg.sender);
    emit ETHSwappedForXSD(
            msg.sender,
            msg.value,
            amountOut,
            block.timestamp
        );
}

/// @notice Swaps XSD tokens for ETH
/// @dev Uses WETH as an intermediary for the swap. This function checks the reserves of the XSD/WETH pool and requires the input amount to be no more than `amountInMax`.
/// It also burns XSD tokens if certain conditions are met.
/// @param amountOut The amount of ETH expected from the swap
/// @param amountInMax The maximum amount of XSD tokens allowed to be spent for the swap
/// @param deadline The time by which the transaction must be confirmed
function swapXSDForETH(uint amountOut, uint amountInMax, uint deadline)
    external
    swapPaused
    blockDelay
    safetyCheck(1)
    ensure(deadline) 
    nonReentrant
    override
{
    (uint reserveA, uint reserveB, ) = IXSDWETHpool(XSDWETH_pool_address).getReserves();
    uint amounts = BankXLibrary.quote(amountOut, reserveB, reserveA);
    require(amounts <= amountInMax, 'BankXRouter: EXCESSIVE_INPUT_AMOUNT');
    // Circuit breaker check
    checkAndUpdateSafety(
        1,
        pid_controller.xsd_updated_price(),
        amountInMax
    );
    TransferHelper.safeTransferFrom(
        xsd_address, msg.sender, XSDWETH_pool_address, amountInMax
    );
    IXSDWETHpool(XSDWETH_pool_address).swap(0, amountOut, address(this));
    IWETH(WETH).withdraw(amountOut);
    TransferHelper.safeTransferETH(msg.sender, amountOut);
    //burn xsd here 
    if(XSD.totalSupply()-ICollateralPool(payable(collateral_pool_address)).collat_XSD()>amountOut/10 && !pid_controller.bucket1()){
        XSD.burnpoolXSD(amountInMax/10);
    }
    emit XSDSwappedForETH(
            msg.sender,
            amountInMax,
            amountOut,
            amountInMax/10,
            block.timestamp
        );
}

/// @notice Swaps ETH for BankX tokens
/// @dev Uses WETH as an intermediary for the swap. This function checks the reserves of the BankX/WETH pool and requires the output amount to be at least `amountOut`.
/// @param amountOut The minimum amount of BankX tokens expected from the swap
/// @param deadline The time by which the transaction must be confirmed
function swapETHForBankX(uint amountOut, uint deadline)
    external
    swapPaused
    blockDelay
    safetyCheck(2)
    ensure(deadline)  
    nonReentrant
    override
    payable
{
    (uint reserveA, uint reserveB, ) = IBankXWETHpool(BankXWETH_pool_address).getReserves();
    uint amounts = BankXLibrary.quote(msg.value, reserveB, reserveA);
    require(amounts >= amountOut, 'BankXRouter: INSUFFICIENT_OUTPUT_AMOUNT');
     // Circuit breaker check
    checkAndUpdateSafety(
        2,
        pid_controller.bankx_updated_price(),
        msg.value
    );
    IWETH(WETH).deposit{value: msg.value}();
    assert(IWETH(WETH).transfer(BankXWETH_pool_address, msg.value));
    IBankXWETHpool(BankXWETH_pool_address).swap(amountOut, 0, msg.sender);
    emit ETHSwappedForBankX(
        msg.sender,
        msg.value,
        amountOut,
        block.timestamp
    );
}

/// @notice Swaps BankX tokens for ETH
/// @dev Uses WETH as an intermediary for the swap. This function checks the reserves of the BankX/WETH pool and requires the input amount to be no more than `amountInMax`.
/// It also burns BankX tokens if certain conditions are met.
/// @param amountOut The amount of ETH expected from the swap
/// @param amountInMax The maximum amount of BankX tokens allowed to be spent for the swap
/// @param deadline The time by which the transaction must be confirmed
function swapBankXForETH(uint amountOut, uint amountInMax, uint deadline)
    external
    swapPaused
    blockDelay
    safetyCheck(3)
    ensure(deadline)  
    nonReentrant
    override
{
    (uint reserveA, uint reserveB, ) = IBankXWETHpool(BankXWETH_pool_address).getReserves();
    uint amounts = BankXLibrary.quote(amountOut, reserveB, reserveA);
    require(amounts <= amountInMax, 'BankXRouter: EXCESSIVE_INPUT_AMOUNT');
    // Circuit breaker check
    checkAndUpdateSafety(
        3,
        pid_controller.bankx_updated_price(),
        amountInMax
    );
    TransferHelper.safeTransferFrom(
        bankx_address, msg.sender, BankXWETH_pool_address, amountInMax
    );
    IBankXWETHpool(BankXWETH_pool_address).swap(0,amountOut, address(this));
    IWETH(WETH).withdraw(amountOut);
    TransferHelper.safeTransferETH(msg.sender, amountOut);
    if((BankXToken(bankx_address).totalSupply() - amountOut/10)>BankXToken(bankx_address).genesis_supply()){
        BankXToken(bankx_address).burnpoolBankX(amountOut/10);
    }
    emit BankXSwappedForETH(
        msg.sender,
        amountInMax,
        amountOut,
        amountOut/10,
        block.timestamp
    );
}
/**
 * @notice Swaps XSD tokens for BankX tokens using the provided amounts and reserves.
 * @dev Requires that the caller is either the sender or the arbitrage contract. 
 *      Ensures the swap is not paused and the deadline is met.
 * @param XSD_amount The amount of XSD tokens to swap.
 * @param sender The address initiating the swap.
 * @param eth_min_amount The minimum amount of ETH expected from the swap.
 * @param bankx_min_amount The minimum amount of BankX tokens expected from the swap.
 * @param deadline The timestamp by which the transaction must be completed.
 */
    function swapXSDForBankX(uint XSD_amount,address sender,uint256 eth_min_amount, uint256 bankx_min_amount, uint deadline)
        external 
        swapPaused
        ensure(deadline)  
        nonReentrant
        override
    {   
        require(msg.sender == sender || msg.sender == arbitrage, "Router:FORBIDDEN");
        (uint reserveA, uint reserveB, ) = IXSDWETHpool(XSDWETH_pool_address).getReserves();
        (uint reserve1, uint reserve2, ) = IBankXWETHpool(BankXWETH_pool_address).getReserves();
        uint ethamount = BankXLibrary.quote(XSD_amount, reserveA, reserveB);
        require(eth_min_amount<= ethamount,'XSDETH: EXCESSIVE_INPUT_AMOUNT');
        uint bankxamount = BankXLibrary.quote(eth_min_amount, reserve2, reserve1);
        require(bankx_min_amount<= bankxamount,'ETHBankX: EXCESSIVE_INPUT_AMOUNT');
        TransferHelper.safeTransferFrom(
            xsd_address, sender, XSDWETH_pool_address, XSD_amount
        );
        IXSDWETHpool(XSDWETH_pool_address).swap(0, ethamount, BankXWETH_pool_address);
        IBankXWETHpool(BankXWETH_pool_address).swap(bankxamount,0,sender);
    }

/**
 * @notice Swaps BankX tokens for XSD tokens using the provided amounts and reserves.
 * @dev Requires that the caller is either the sender or the arbitrage contract. 
 *      Ensures the swap is not paused and the deadline is met.
 * @param bankx_amount The amount of BankX tokens to swap.
 * @param sender The address initiating the swap.
 * @param eth_min_amount The minimum amount of ETH expected from the swap.
 * @param xsd_min_amount The minimum amount of XSD tokens expected from the swap.
 * @param deadline The timestamp by which the transaction must be completed.
 */
    function swapBankXForXSD(uint bankx_amount, address sender, uint256 eth_min_amount, uint256 xsd_min_amount, uint deadline)
        external
        swapPaused
        ensure(deadline)  
        nonReentrant
        override
    {   
        require(msg.sender == sender || msg.sender == arbitrage, "Router:FORBIDDEN");
        (uint reserveA, uint reserveB, ) = IXSDWETHpool(XSDWETH_pool_address).getReserves();
        (uint reserve1, uint reserve2, ) = IBankXWETHpool(BankXWETH_pool_address).getReserves();
        uint ethamount = BankXLibrary.quote(bankx_amount, reserve1, reserve2);
        require(eth_min_amount<=ethamount,'BankXETH: EXCESSIVE_INPUT_AMOUNT');
        uint xsdamount = BankXLibrary.quote(ethamount, reserveB, reserveA);
        require(xsd_min_amount<=xsdamount, "ETHXSD: EXCESSIVE_INPUT_AMOUNT");
        TransferHelper.safeTransferFrom(
            bankx_address, sender, BankXWETH_pool_address, bankx_amount
        );
        IBankXWETHpool(BankXWETH_pool_address).swap(0, ethamount, XSDWETH_pool_address);
        IXSDWETHpool(XSDWETH_pool_address).swap(xsdamount,0,sender);
    }
    
    function setSmartContractOwner(address _smartcontract_owner) external onlyByOwner{
        smartcontract_owner = _smartcontract_owner;
    }

    function renounceOwnership() external onlyByOwner{
        smartcontract_owner = address(0);
    }
    
    // **** LIBRARY FUNCTIONS ****
    function quote(uint amountA, uint reserveA, uint reserveB) internal pure  returns (uint amountB) {
        return BankXLibrary.quote(amountA, reserveA, reserveB);
    }

    function pauseSwaps() external onlyByOwner{
        swap_paused = !swap_paused;
    }

    function pauseLiquidity() external onlyByOwner{
        liquidity_paused = !liquidity_paused;
    }
    
    
    function setBankXAddress(address _bankx_address) external nonZeroAddress(_bankx_address) onlyByOwner{
        bankx_address = _bankx_address;
    }

    function setXSDAddress(address _xsd_address) external nonZeroAddress(_xsd_address) onlyByOwner{
        xsd_address = _xsd_address;
    }

    function setXSDPoolAddress(address _XSDWETH_pool) external nonZeroAddress(_XSDWETH_pool) onlyByOwner{
        XSDWETH_pool_address = _XSDWETH_pool;
    }

    function setBankXPoolAddress(address _BankXWETH_pool) external nonZeroAddress(_BankXWETH_pool) onlyByOwner{
        BankXWETH_pool_address = _BankXWETH_pool;
    }

    function setCollateralPool(address _collateral_pool) external nonZeroAddress(_collateral_pool) onlyByOwner{
        collateral_pool_address = _collateral_pool;
    }

    function setRewardManager(address _reward_manager_address) external nonZeroAddress(_reward_manager_address) onlyByOwner{
        reward_manager_address = _reward_manager_address;
        reward_manager = IRewardManager(_reward_manager_address);
    }

    function setPIDController(address _pid_address) external nonZeroAddress(_pid_address) onlyByOwner{
        pid_controller = IPIDController(_pid_address);
    }

    function setArbitrageAddress(address _arbitrage) external nonZeroAddress(_arbitrage) onlyByOwner{
        arbitrage = _arbitrage;
    }

    function getPriceThreshold(uint8 breaker_type) external view onlyByOwner returns (uint256) {
        require(breaker_type <= 3, "Invalid breaker type");
        return price_threshold[breaker_type];
    }

    function getVolumeThreshold(uint8 breaker_type) external view onlyByOwner returns (uint256) {
        require(breaker_type <= 3, "Invalid breaker type");
        return volume_threshold[breaker_type];
    }

    // Initialization Events
    event RouterInitialized(
        address indexed bankx_address,
        address indexed xsd_address,
        address indexed XSDWETH_pool,
        address BankXWETH_pool,
        address collateral_pool,
        uint256 block_delay
    );

    // Liquidity Events
    event CreatorLiquidityAdded(
        address indexed pool,
        address indexed creator,
        uint256 amount,
        uint256 timestamp
    );

    event UserLiquidityAdded(
        address indexed pool,
        address indexed user,
        uint256 amount,
        uint256 timestamp
    );

    event LiquidityTokensAdded(
        address indexed token,
        address indexed provider,
        uint256 amount,
        uint256 timestamp
    );

    event LiquidityRedeemed(
        address indexed pool,
        address indexed user,
        uint256 timestamp
    );

    // Swap Events
    event ETHSwappedForXSD(
        address indexed user,
        uint256 ethAmount,
        uint256 xsdAmount,
        uint256 timestamp
    );

    event XSDSwappedForETH(
        address indexed user,
        uint256 xsdAmount,
        uint256 ethAmount,
        uint256 burnedAmount,
        uint256 timestamp
    );

    event ETHSwappedForBankX(
        address indexed user,
        uint256 ethAmount,
        uint256 bankxAmount,
        uint256 timestamp
    );

    event BankXSwappedForETH(
        address indexed user,
        uint256 bankxAmount,
        uint256 ethAmount,
        uint256 burnedAmount,
        uint256 timestamp
    );

    // Circuit Breaker Events
    event SafetyTripped(string breaker_name, string reason, uint256 value);
    event SafetyToggled(string breaker_name, bool status);
    event SafetyThresholdsUpdated(
        uint8 breaker_type,
        uint256 price_threshold,
        uint256 volume_threshold
    );
}