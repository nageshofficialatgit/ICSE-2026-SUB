// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

interface IERC20 {
    function balanceOf(address account) external view returns (uint);
    function transfer(address recipient, uint amount) external returns (bool);
    function approve(address spender, uint amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint);
}

interface IUniswapV2Router {
    function swapExactETHForTokens(
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external payable returns (uint[] memory amounts);

    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);
    function WETH() external pure returns (address);
}

contract DexSniper {
    address public owner;
    address public immutable router;
    bool public tradingEnabled = false;
    uint256 public profitTarget = 0.05 ether;

    event Sniped(address indexed token, uint256 amountOut);
    event Withdrawn(uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    constructor(address _router) {
        owner = msg.sender;
        router = _router;
    }

    function enableTrading(bool _state) external onlyOwner {
        tradingEnabled = _state;
    }

    function setProfitTarget(uint256 _target) external onlyOwner {
        profitTarget = _target;
    }

    function recoverTokens(address token) external onlyOwner {
        IERC20(token).transfer(owner, IERC20(token).balanceOf(address(this)));
    }

    function recoverEth() external onlyOwner {
        uint256 amount = address(this).balance;
        payable(owner).transfer(amount);
        emit Withdrawn(amount);
    }

    function snipe(address tokenOut, uint256 amountOutMin) external onlyOwner {
        require(tradingEnabled, "Trading disabled");
        require(address(this).balance > 0, "No ETH in contract");

        address[] memory path = new address[](2);
        path[0] = IUniswapV2Router(router).WETH();
        path[1] = tokenOut;

        uint256 deadline = block.timestamp + 300;
        uint256 startBalance = IERC20(tokenOut).balanceOf(address(this));

        IUniswapV2Router(router).swapExactETHForTokens{
            value: address(this).balance
        }(
            amountOutMin,
            path,
            address(this),
            deadline
        );

        uint256 endBalance = IERC20(tokenOut).balanceOf(address(this));
        uint256 received = endBalance - startBalance;

        emit Sniped(tokenOut, received);

        if (address(this).balance >= profitTarget) {
            uint256 toSend = address(this).balance;
            payable(owner).transfer(toSend);
            emit Withdrawn(toSend);
        }
    }

    function getEstimatedTokens(address tokenOut, uint256 amountIn) external view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = IUniswapV2Router(router).WETH();
        path[1] = tokenOut;
        uint[] memory amounts = IUniswapV2Router(router).getAmountsOut(amountIn, path);
        return amounts[1];
    }

    receive() external payable {}
}