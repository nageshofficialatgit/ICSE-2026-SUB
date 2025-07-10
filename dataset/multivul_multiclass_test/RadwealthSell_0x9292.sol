// SPDX-License-Identifier: MIT
// File: @openzeppelin/contracts/token/ERC20/IERC20.sol


// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC20/IERC20.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC-20 standard as defined in the ERC.
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
    function transfer(address to, uint256 value) external returns (bool);

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
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

// File: @chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol


pragma solidity ^0.8.0;

interface AggregatorV3Interface {
  function decimals() external view returns (uint8);

  function description() external view returns (string memory);

  function version() external view returns (uint256);

  function getRoundData(
    uint80 _roundId
  ) external view returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);

  function latestRoundData()
    external
    view
    returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);
}

// File: contracts/RadwealthSell.sol


pragma solidity ^0.8.0;



contract RadwealthSell {
    address public ownerWallet;
    IERC20 public radwealthToken;
    IERC20 public usdcToken;
    IERC20 public usdtToken;
    AggregatorV3Interface internal priceFeed;
    uint256 public price; // Price in USD with 8 decimals

    modifier onlyOwner() {
        require(msg.sender == ownerWallet, "Only owner can call this function");
        _;
    }

    constructor(
        address _radwealthTokenAddress,
        address _usdcAddress,
        address _usdtAddress,
        address _aggregatorAddress,
        address _ownerWallet,
        uint256 _price
    ) {
        radwealthToken = IERC20(_radwealthTokenAddress);
        usdcToken = IERC20(_usdcAddress);
        usdtToken = IERC20(_usdtAddress);
        priceFeed = AggregatorV3Interface(_aggregatorAddress);
        ownerWallet = _ownerWallet;
        price = _price;
    }

    // Function to sell Radwealth tokens for ETH
    function sellTokensForETH(uint256 tokenAmount) external {
        require(tokenAmount > 0, "Amount should be greater than 0");

        // Calculate the equivalent ETH value
        (, int ethPrice, , , ) = priceFeed.latestRoundData();
        uint256 ethUSDPrice = uint256(ethPrice); // Price of ETH in USD (8 decimals)
        uint256 ethToTransfer = (tokenAmount * price) / ethUSDPrice;

        // Ensure the contract has enough ETH
        require(address(this).balance >= ethToTransfer, "Insufficient ETH liquidity");

        // Transfer tokens from seller to ownerWallet
        require(radwealthToken.transferFrom(msg.sender, ownerWallet, tokenAmount), "Token transfer failed");

        // Transfer ETH to seller
        (bool success, ) = payable(msg.sender).call{value: ethToTransfer}("");
        require(success, "ETH transfer failed");
    }

    // Function to sell Radwealth tokens for USDC
    function sellTokensForUSDC(uint256 tokenAmount) external {
        require(tokenAmount > 0, "Amount should be greater than 0");

        // Calculate the equivalent USDC value
        uint256 usdcToTransfer = (tokenAmount * price) / (10 ** (18 - 6 + 8)); // Adjust for USDC decimals (6)

        // Ensure the contract has enough USDC
        require(usdcToken.balanceOf(address(this)) >= usdcToTransfer, "Insufficient USDC liquidity");

        // Transfer tokens from seller to ownerWallet
        require(radwealthToken.transferFrom(msg.sender, ownerWallet, tokenAmount), "Token transfer failed");

        // Transfer USDC to seller
        require(usdcToken.transfer(msg.sender, usdcToTransfer), "USDC transfer failed");
    }

    // Function to sell Radwealth tokens for USDT
    function sellTokensForUSDT(uint256 tokenAmount) external {
        require(tokenAmount > 0, "Amount should be greater than 0");

        // Calculate the equivalent USDT value
        uint256 usdtToTransfer = (tokenAmount * price) / (10 ** (18 - 6 + 8)); // Adjust for USDT decimals (6)

        // Ensure the contract has enough USDT
        require(usdtToken.balanceOf(address(this)) >= usdtToTransfer, "Insufficient USDT liquidity");

        // Transfer tokens from seller to ownerWallet
        require(radwealthToken.transferFrom(msg.sender, ownerWallet, tokenAmount), "Token transfer failed");

        // Transfer USDT to seller
        require(usdtToken.transfer(msg.sender, usdtToTransfer), "USDT transfer failed");
    }

    // Function to set a new owner wallet
    function setOwnerWallet(address newOwnerWallet) external onlyOwner {
        require(newOwnerWallet != address(0), "Invalid address");
        ownerWallet = newOwnerWallet;
    }

    // Function to set a new Radwealth token contract address
    function setRadwealthToken(address newRadwealthToken) external onlyOwner {
        require(newRadwealthToken != address(0), "Invalid address");
        radwealthToken = IERC20(newRadwealthToken);
    }

    // Function to set a new price (in USD with 8 decimals)
    function setPrice(uint256 newPrice) external onlyOwner {
        require(newPrice > 0, "Price must be greater than 0");
        price = newPrice;
    }

    // Function to deposit ETH into the contract for liquidity
    receive() external payable {}

    // Owner can add USDC/USDT liquidity
    function depositUSDC(uint256 amount) external onlyOwner {
        require(usdcToken.transferFrom(msg.sender, address(this), amount), "USDC transfer failed");
    }

    function depositUSDT(uint256 amount) external onlyOwner {
        require(usdtToken.transferFrom(msg.sender, address(this), amount), "USDT transfer failed");
    }

    // Owner can withdraw liquidity
    function withdrawETH(uint256 amount) external onlyOwner {
        (bool success, ) = payable(ownerWallet).call{value: amount}("");
        require(success, "ETH withdrawal failed");
    }

    function withdrawUSDC(uint256 amount) external onlyOwner {
        require(usdcToken.transfer(ownerWallet, amount), "USDC withdrawal failed");
    }

    function withdrawUSDT(uint256 amount) external onlyOwner {
        require(usdtToken.transfer(ownerWallet, amount), "USDT withdrawal failed");
    }
}