// File: @openzeppelin/contracts/utils/cryptography/Hashes.sol


// OpenZeppelin Contracts (last updated v5.1.0) (utils/cryptography/Hashes.sol)

pragma solidity ^0.8.20;




// OpenZeppelin Contracts (last updated v5.1.0) (utils/cryptography/MerkleProof.sol)
// This file was procedurally generated from scripts/generate/templates/MerkleProof.js.


// File: aurora_token_feb_2025/tokensale.sol


pragma solidity ^0.8.0;


// File: @openzeppelin/contracts/utils/math/SafeMath.sol




// OpenZeppelin Contracts (last updated v4.9.0) (utils/math/SafeMath.sol)

pragma solidity ^0.8.0;

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

// File: @openzeppelin/contracts/token/ERC20/IERC20.sol


// OpenZeppelin Contracts (last updated v5.0.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.20;

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
     * @dev Returns the value of tokens in existence.
     */
    function totalSupply() external view returns (uint256);

    /**
     * @dev Returns the value of tokens owned by `account`.
     */
    function balanceOf(address account) external view returns (uint256);

    /**
     * @dev Moves a `value` amount of tokens from the caller's account to `to`.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transfer(address to, uint256 value) external;

    /**
     * @dev Returns the remaining number of tokens that `spender` will be
     * allowed to spend on behalf of `owner` through {transferFrom}. This is
     * zero by default.
     *
     * This value changes when {approve} or {transferFrom} are called.
     */
    function allowance(address owner, address spender) external view returns (uint256);

    /**
     * @dev Sets a `value` amount of tokens as the allowance of `spender` over the
     * caller's tokens.
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
    function approve(address spender, uint256 value) external returns (bool);

    /**
     * @dev Moves a `value` amount of tokens from `from` to `to` using the
     * allowance mechanism. `value` is then deducted from the caller's
     * allowance.
     *
     * Returns a boolean value indicating whether the operation succeeded.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 value) external;

    function decimals() external view returns(uint);
}

contract TokenSale {
    using SafeMath for uint256;

    mapping(address => bool) public admins;
    address public treasuryAddress = 0x92DEdD242B8Aa07d00D236C2afe9136Ed358722d; // Address to receive funds
    IERC20 public tokenForSale = IERC20(0xe374CD55094334eEF5D1841624b001223c6b624D); // ERC20 token contract to be sold
    IERC20 public paymentToken1 = IERC20(0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48); // USDC
    IERC20 public paymentToken2 = IERC20(0xdAC17F958D2ee523a2206206994597C13D831ec7); // USDT
    uint256 public tokenPrice = 66660000000000000; // 10**18 ==> 1 usdt
    uint256 public tokensSold;
    bool public saleActive;

    uint256 public maxInvestmentLimit = 100000 * 10**6; // Default to 5000 USDT/USDC (6 decimals)

    mapping(address => uint256) public totalInvestments; // Tracks total investments per wallet

    event Sell(address _buyer, uint256 _amount);
    event TreasuryAddressChanged(address _newTreasuryAddress);
    event AdminAdded(address _admin);
    event AdminRemoved(address _admin);

    event MaxInvestmentLimitUpdated(uint256 oldLimit, uint256 newLimit);


    modifier onlyAdmin() {
        require(admins[msg.sender], "Only admin can perform this action");
        _;
    }

    constructor() {
        admins[msg.sender] = true; // Set contract deployer as admin by default

        saleActive = false;
    }

    function addAdmin(address _admin) public onlyAdmin {
        admins[_admin] = true;
        emit AdminAdded(_admin);
    }

    function setMaxInvestmentLimit(uint256 _newLimit) public onlyAdmin {
        emit MaxInvestmentLimitUpdated(maxInvestmentLimit, _newLimit);
        maxInvestmentLimit = _newLimit;
    }

    function getTotalPrice(uint256 _paymentTokenId, uint256 _numberOfTokens) public view returns (uint256){

        IERC20 paymentToken;
        uint256 paymentTokenDecimals;

        if (_paymentTokenId == 1) {
            paymentToken = paymentToken1;
            paymentTokenDecimals = paymentToken1.decimals();
        } else if (_paymentTokenId == 2) {
            paymentToken = paymentToken2;
            paymentTokenDecimals = paymentToken2.decimals();
        } else {
            revert("Invalid payment token ID");
        }

        uint256 decimals = tokenForSale.decimals();

        uint256 tokenPriceCorrected = tokenPrice.div(10**(decimals - paymentTokenDecimals));

        return tokenPriceCorrected.mul(_numberOfTokens.div(10**decimals));
    }
    
    function removeAdmin(address _admin) public onlyAdmin {
        require(msg.sender != _admin, "Cannot remove self");
        admins[_admin] = false;
        emit AdminRemoved(_admin);
    }

    function startSale() public onlyAdmin {
        require(!saleActive, "Sale is already active");
        saleActive = true;
    }

    function stopSale() public onlyAdmin {
        require(saleActive, "Sale is not active");
        saleActive = false;
    }

    function setTokenPrice(uint256 _newPrice) public onlyAdmin {
        tokenPrice = _newPrice;
    }

    function setPaymentToken1(address _paymentTokenAddress) public onlyAdmin {
        paymentToken1 = IERC20(_paymentTokenAddress); // Set the first ERC20 token contract address for payment
    }

    function setPaymentToken2(address _paymentTokenAddress) public onlyAdmin {
        paymentToken2 = IERC20(_paymentTokenAddress); // Set the second ERC20 token contract address for payment
    }

    function setTreasuryAddress(address _newTreasuryAddress) public onlyAdmin {
        treasuryAddress = _newTreasuryAddress;
        emit TreasuryAddressChanged(_newTreasuryAddress);
    }

    function buyTokens(uint256 _paymentTokenId, uint256 _numberOfTokens) public {
        require(saleActive, "Sale is not active");

        IERC20 paymentToken;
        uint256 paymentTokenDecimals;

        if (_paymentTokenId == 1) {
            paymentToken = paymentToken1;
            paymentTokenDecimals = paymentToken1.decimals();
        } else if (_paymentTokenId == 2) {
            paymentToken = paymentToken2;
            paymentTokenDecimals = paymentToken2.decimals();
        } else {
            revert("Invalid payment token ID");
        }

        uint256 decimals     = tokenForSale.decimals();

        uint256 tokenPriceCorrected = tokenPrice.div(10**(decimals - paymentTokenDecimals));

        uint256 totalPrice = tokenPriceCorrected.mul(_numberOfTokens).div(10**decimals);

        uint256 newTotalInvestment = totalInvestments[msg.sender].add(totalPrice);
        require(newTotalInvestment <= maxInvestmentLimit, "Investment exceeds max limit");
        totalInvestments[msg.sender] = newTotalInvestment; // Update the investment tracker

        require(paymentToken.allowance(msg.sender, address(this)) >= totalPrice, "Token allowance not enough");

        paymentToken.transferFrom(msg.sender, treasuryAddress, totalPrice);

        require(tokenForSale.balanceOf(address(this)) >= _numberOfTokens, "Insufficient tokens available");

        tokenForSale.transfer(msg.sender, _numberOfTokens);

        tokensSold = tokensSold.add(_numberOfTokens);

        emit Sell(msg.sender, _numberOfTokens);
    }

    function withdrawTokens() public onlyAdmin {
        uint256 contractBalance = tokenForSale.balanceOf(address(this));
        require(contractBalance > 0, "No tokens available to withdraw");
        tokenForSale.transfer(treasuryAddress, contractBalance);
    }

    function withdrawEther() public onlyAdmin {
        payable(treasuryAddress).transfer(address(this).balance);
    }
}