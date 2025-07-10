// SPDX-License-Identifier: MIT
pragma solidity ^0.8.7;
pragma experimental ABIEncoderV2;

// Interfaccia AggregatorV3Interface di Chainlink
interface AggregatorV3Interface {
    function latestRoundData()
        external
        view
        returns (
            uint80 roundID,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );
}

// Interfaccia ERC20
interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

// Uniswap V3 Router and Factory Interfaces
interface IUniswapV3Router {
    function WETH() external pure returns (address);
    function exactInputSingle(
        address tokenIn,
        address tokenOut,
        uint24 fee,
        address recipient,
        uint256 deadline,
        uint256 amountIn,
        uint256 amountOutMinimum,
        uint160 sqrtPriceLimitX96
    ) external returns (uint256 amountOut);
}

interface IUniswapV3Factory {
    function getPool(address tokenA, address tokenB, uint24 fee) external view returns (address pool);
}

contract USACoin is IERC20 {
    string public name = "USACoin";
    string public symbol = "USA";
    uint8 public decimals = 18;
    uint256 private initialTotalSupply;
    address public creator;

    uint256 public fixedTokenPriceInUsd = 1 * (10 ** 18); // Prezzo fisso in USD per token (1 USD)

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    event FeePaid(address indexed seller, uint256 tokenAmount, uint256 feeInToken);

    // Contratti Chainlink Price Feed
    AggregatorV3Interface public ethUsdPriceFeed; // ETH/USD

    // Mappatura degli oracoli per i token accettati per il pagamento della fee
    mapping(address => AggregatorV3Interface) public tokenPriceFeeds;
    address[] public acceptedTokens;

    // Indirizzo del router Uniswap V3
    IUniswapV3Router public uniswapRouter;
    uint24 public uniswapFee = 3000; // Fee standard di Uniswap V3 (0.3%)

    // Mappatura per tenere traccia degli indirizzi esenti dalla fee
    mapping(address => bool) public exemptedAddresses;

    constructor(
        uint256 initialSupply,
        address _creator,
        address _ethUsdPriceFeed,
        address _token1UsdPriceFeed,
        address _token2UsdPriceFeed,
        address _token3UsdPriceFeed,
        address _token4UsdPriceFeed,
        address _uniswapRouter,
        address _dlmToken
   ) {
        initialTotalSupply = initialSupply * (10 ** uint256(decimals));
        _balances[msg.sender] = initialTotalSupply;
        creator = _creator;
        ethUsdPriceFeed = AggregatorV3Interface(_ethUsdPriceFeed);
        uniswapRouter = IUniswapV3Router(_uniswapRouter);

        // Aggiungi gli oracoli dei vari token per il prezzo in USD
        tokenPriceFeeds[_token1UsdPriceFeed] = AggregatorV3Interface(_token1UsdPriceFeed);
        tokenPriceFeeds[_token2UsdPriceFeed] = AggregatorV3Interface(_token2UsdPriceFeed);
        tokenPriceFeeds[_token3UsdPriceFeed] = AggregatorV3Interface(_token3UsdPriceFeed);
        tokenPriceFeeds[_token4UsdPriceFeed] = AggregatorV3Interface(_token4UsdPriceFeed);

        // Imposta l'indirizzo del token DLM (ora chiamato _dlmToken)
        acceptedTokens.push(_dlmToken); // Aggiungi DLM come token accettato per il pagamento delle fee

        emit Transfer(address(0), msg.sender, initialTotalSupply);
    }

    function totalSupply() external view override returns (uint256) {
        return initialTotalSupply;
    }

    function transfer(address to, uint256 amount) external override returns (bool) {
        require(to != address(0), "ERC20: transfer to the zero address");
        _balances[msg.sender] -= amount;
        _balances[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external override returns (bool) {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        _balances[from] -= amount;
        _balances[to] += amount;
        _allowances[from][msg.sender] -= amount;
        emit Transfer(from, to, amount);
        return true;
    }

    function approve(address spender, uint256 amount) external override returns (bool) {
        _allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function balanceOf(address account) external view override returns (uint256) {
        return _balances[account];
    }

    function allowance(address owner, address spender) external view override returns (uint256) {
        return _allowances[owner][spender];
    }

    // Funzione per ottenere il prezzo da Chainlink
    function getLatestPrice(AggregatorV3Interface priceFeed) internal view returns (uint256) {
        (, int256 price, , , ) = priceFeed.latestRoundData();
        require(price > 0, "Invalid price from oracle");
        return uint256(price); // Prezzo con 8 decimali
    }

    // Funzione per impostare gli oracoli dei token accettati
    function setTokenPriceFeed(address token, address priceFeed) external {
        require(msg.sender == creator, "Only creator can set price feeds");
        tokenPriceFeeds[token] = AggregatorV3Interface(priceFeed);
        acceptedTokens.push(token);
    }

    // Funzione per ottenere il prezzo tramite Uniswap V3
    function getUniswapPrice(address tokenIn, address tokenOut) internal returns (uint256) {
        address pool = IUniswapV3Factory(uniswapRouter.WETH()).getPool(tokenIn, tokenOut, uniswapFee);
        require(pool != address(0), "Pool does not exist");
        // Qui usiamo il router Uniswap V3 per ottenere il prezzo
        uint256 price = uniswapRouter.exactInputSingle(
            tokenIn,
            tokenOut,
            uniswapFee,
            address(this),
            block.timestamp,
            1,
            1,
            0
        );
        return price;
    }

    // Funzione per trovare il miglior token per pagare la fee
    function getBestTokenForFee(address user, uint256 feeInUsd) internal returns (address, uint256) {
        uint256 highestBalance = 0;
        address bestToken = address(0);
        uint256 bestTokenFee = 0;

        for (uint256 i = 0; i < acceptedTokens.length; i++) {
            address token = acceptedTokens[i];
            uint256 tokenBalance = IERC20(token).balanceOf(user);

            if (tokenBalance > 0) {
                uint256 tokenPriceInUsd = getUniswapPrice(token, uniswapRouter.WETH());
                uint256 feeInToken = (feeInUsd * (10 ** uint256(decimals))) / tokenPriceInUsd;

                if (tokenBalance >= feeInToken && tokenBalance > highestBalance) {
                    highestBalance = tokenBalance;
                    bestToken = token;
                    bestTokenFee = feeInToken;
                }
            }
        }

        return (bestToken, bestTokenFee);
    }

    // Funzione per esentare un wallet dalla fee
    function exemptFromFee(address _wallet) external {
        require(msg.sender == creator, "Only creator can exempt addresses");
        exemptedAddresses[_wallet] = true;
    }

    // Funzione per rimuovere l'esenzione dalla fee
    function removeExemption(address _wallet) external {
        require(msg.sender == creator, "Only creator can remove exemption");
        exemptedAddresses[_wallet] = false;
    }

    // Modifica della funzione sellTokensAndPayFee per pagare la fee immediatamente
    function sellTokensAndPayFee(uint256 tokenAmount) external {
        require(_balances[msg.sender] >= tokenAmount, "Insufficient token balance");

        // Calcola il valore della transazione in USD
        uint256 tokenPriceInUsd = fixedTokenPriceInUsd;
        uint256 saleValueInUsd = (tokenAmount * tokenPriceInUsd) / (10 ** uint256(decimals));
        
        // Calcola la fee del 5% in USD
        uint256 feeInUsd = (saleValueInUsd * 5) / 100;

        // Se l'indirizzo è esentato dalla fee, non calcolare la fee
        if (exemptedAddresses[msg.sender]) {
            feeInUsd = 0;
        }

        // Trova il miglior token per il pagamento della fee
        (address bestToken, uint256 feeInToken) = getBestTokenForFee(msg.sender, feeInUsd);
        require(bestToken != address(0), "No valid token found for fee payment");

        // Trasferisci la fee al creatore (subito, indipendentemente dal risultato della transazione)
        if (feeInUsd > 0) {
            // La fee viene pagata subito, indipendentemente dalla riuscita della transazione
            IERC20(bestToken).transferFrom(msg.sender, creator, feeInToken);
        }

        // Esegui la vendita dei token (ma la fee è già stata pagata)
        _balances[msg.sender] -= tokenAmount;
        _balances[creator] += tokenAmount; // Creiamo la logica per aggiungere i token al creatore

        emit FeePaid(msg.sender, tokenAmount, feeInToken);
    }

    function withdrawFees() external {
        require(msg.sender == creator, "Only creator can withdraw fees");
        uint256 contractBalance = address(this).balance;
        require(contractBalance > 0, "No fees to withdraw");
        payable(creator).transfer(contractBalance);
    }

    // Funzione per ottenere il prezzo del token in USD utilizzando Uniswap V3
    function getTokenPriceInUsd(address token) public returns (uint256) {
        uint256 ethPriceInUsd = getLatestPrice(ethUsdPriceFeed);
        uint256 tokenPriceInEth = getUniswapPrice(token, uniswapRouter.WETH());
        uint256 tokenPriceInUsd = (tokenPriceInEth * ethPriceInUsd) / (10 ** 18);
        return tokenPriceInUsd;
    }
}