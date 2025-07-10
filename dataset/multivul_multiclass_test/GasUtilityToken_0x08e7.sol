// SPDX-License-Identifier: MIT
/**
⣿⣿⠛⡋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠭⠤⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣐⢂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠤⠀⠀⢀⠀⠀⢀⠀⢀⣤⣤⣾⣿⣿⣷⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣶⣶⣶⣦⣦⣢⢀⠀⢀⣠⣴⣶⣶⣿⣷⣶⣤⣆⣤⠐⠄⠈⣴⣿⡿⠹⠟⠀⠀⠚⠛⢻⣷⡆⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⡀⠀⠀⠀⠀⠀⢠⣾⣿⣿⠿⠛⠛⠿⢿⣿⣿⣷⣿⠿⠟⠛⠛⠉⠛⠻⢿⣿⣿⣷⣄⠀⠻⣿⣧⣀⡀⠀⠐⠀⠀⠀⣿⣯⠅⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣼⣀⠤⢀⠀⠀⠀⣾⣿⡟⠀⠀⠀⠀⠀⠀⠈⠛⣿⣷⡂⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣤⠀⠈⣽⣯⠀⠀⠀⡀⠀⠛⢿⡏⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿⡿⢻⣿⣀⠀⠂⠀⢠⣿⠿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣿⡀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⡏⠀⠀⠻⣷⣶⣠⣼⢧⣬⣶⣿⠗⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⣾⣿⠋⠀⠀⢈⣙⣷⣆⠀⢸⣿⣵⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⡏⠀⠀⠀⠀⠉⠉⠀⠈⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠿⣿⣱⣴⣷⣼⣿⡿⠏⠀⠀⠘⣿⣆⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⣴⣾⣿⣿⣿⣷⣶⣴⣦⡀⠲⠨⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠈⠁⠀⠀⢀⣴⣿⡿⢧⣀⠒⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⢤⣿⣿⠋⠀⠀⢩⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⣿⣿⣧⡄⠈⠀⢠⣄⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠄⣿⣻⡃⠀⠀⠀⠀⢿⣦⣄⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣺⠀⣠⣴⣶⣄⡀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢸⣿⣼⠇⠀⠀⠀⠀⠈⢻⣿⣷⣦⣤⣤⣤⣾⡄⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⡟⣀⣿⣿⣿⣿⣶⣶⣶⡀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢻⣿⣶⡀⠀⠀⠀⠀⢠⡽⠛⣿⣿⠋⠁⣿⣧⠀⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣿⠇⢴⣿⣿⢿⡿⠛⠻⣿⣿
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣦⣄⠀⠀⣾⡇⠀⠸⣿⣷⣶⣿⣿⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⠟⠁⢈⣿⢿⡁⠀⠀⠀⠙⢿⣿
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠛⡟⠿⣿⣯⠋⢀⡀⠈⠈⠉⠉⠻⣿⣿⣶⣤⣄⡀⠀⠀⠀⠀⣠⣠⣴⠿⣿⣿⣿⣷⡄⠀⢿⣿⣦⣀⣴⣆⣤⣿⣿
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡿⠉⠀⣌⠁⠉⠀⢀⣠⡾⠂⠡⠈⠉⠉⠁⠀⠀⠀⠀⠉⠁⠀⠀⠈⠻⣿⣿⣿⠆⠀⠀⠉⠉⠙⢛⣟⠟⠁
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⡋⣠⣾⠏⣴⣴⣶⡿⠃⠀⠀⠀⠈⠀⠀⢠⣦⠈⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⡀⠀⠀⠀⠉⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣿⣿⣿⣷⣿⣿⣿⣿⣅⠀⠀⠀⠀⠀⠀⢠⣴⣿⣦⣀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⠿⠁⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⡿⠟⠀⠘⠻⣿⣷⣀⡀⠀⠀⣀⣾⡟⠋⠛⢿⣿⣶⣤⣤⣤⣴⣾⣿⣿⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⣿⣿⠟⠁⠀⠀⠀⠀⠂⠈⠙⢻⡿⡿⠟⠛⠉⠀⠀⠀⠀⠈⠙⢛⢿⠿⠿⠻⠻⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⡿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⡟⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⣤⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

 *         This contract implements the $GAS token, a complementary asset
 *         designed to support and enhance the $ASS token ecosystem by
 *         automatically swapping $GAS for $ASS, thereby potentially supporting
 *         its price floor. By using this contract, you agree to the terms and
 *         conditions described herein.
 *
 * @dev 
 *  - This version replaces any ETH-swapping logic with direct swaps for $ASS
 *    (address: 0x1BF56759e95D9E85b6927161f6F8DBC4568642bc).
 *  - Ensure you audit and test this thoroughly before mainnet deployment.
 *
 * DISCLAIMER:
 *  - The information contained herein is provided “as is” and without any
 *    representations or warranties, express or implied. The authors, developers,
 *    and contributors shall not be held liable for any damages or losses arising
 *    from the use of this software, including but not limited to direct, indirect,
 *    incidental, or consequential damages.
 *  - This is not financial advice. Conduct your own due diligence and consult
 *    professional advisors before making any investment decisions.
 *  - Token holders and users are solely responsible for compliance with
 *    applicable laws and regulations related to cryptocurrencies and digital
 *    assets.
 *
 * Web: https://ass.financial
 * Twitter/X: https://x.com/assfinancial
 * Telegram: https://t.me/ass_financial
 */

pragma solidity 0.8.25;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(
        address sender, 
        address recipient, 
        uint256 amount
    ) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return sub(a, b, "SafeMath: subtraction overflow");
    }
    function sub(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b <= a, errorMessage);
        uint256 c = a - b;
        return c;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) { return 0; }
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return div(a, b, "SafeMath: division by zero");
    }
    function div(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b > 0, errorMessage);
        uint256 c = a / b;
        return c;
    }
}

contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor () {
        address msgSender = _msgSender();
        _owner = msgSender;
        emit OwnershipTransferred(address(0), msgSender);
    }
    function owner() public view returns (address) {
        return _owner;
    }
    modifier onlyOwner() {
        require(_owner == _msgSender(), "Ownable: caller is not the owner");
        _;
    }
    function renounceOwnership() public virtual onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }
    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

interface IUniswapV2Factory {
    function createPair(address tokenA, address tokenB) external returns (address pair);
}

interface IUniswapV2Router02 {
    function swapExactTokensForTokensSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;

    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external;

    function factory() external pure returns (address);
    function WETH() external pure returns (address);

    function addLiquidityETH(
        address token,
        uint amountTokenDesired,
        uint amountTokenMin,
        uint amountETHMin,
        address to,
        uint deadline
    ) external payable returns (uint amountToken, uint amountETH, uint liquidity);
}

/**
 * @dev *** IMPORTANT ***
 * This is your $GAS contract with direct swapping for $ASS instead of ETH.
 */
contract GasUtilityToken is Context, IERC20, Ownable {
    using SafeMath for uint256;

    // ========================== NEW: The $ASS token address ==========================
    address public constant ASS_TOKEN = 0x1BF56759e95D9E85b6927161f6F8DBC4568642bc;

    // Balances and allowances
    mapping (address => uint256) private _balances;
    mapping (address => mapping (address => uint256)) private _allowances;

    // Exile logic (likely an exempt or excluded list)
    mapping (address => bool) private isExile;
    // Identifies whether an address is a marketPair (DEX pair)
    mapping (address => bool) public marketPair;

    // Anti-bot logic
    mapping (uint256 => uint256) private perBuyCount;
    uint256 private firstBlock = 0;

    // Taxes
    uint256 private _initialBuyTax=23;
    uint256 private _initialSellTax=23;
    uint256 private _finalBuyTax=2;
    uint256 private _finalSellTax=3;
    uint256 private _reduceBuyTaxAt=23;
    uint256 private _reduceSellTaxAt=23;

    // BUY/SELL counters
    uint256 private _buyCount=0;
    uint256 private sellCount = 0;
    uint256 private lastSellBlock = 0;

    // -------------------- ADD THIS VARIABLE --------------------
    // Set this to the buy count threshold you want before allowing swaps:
    uint256 private _preventSwapBefore = 10;

    // Basic token info
    uint8 private constant _decimals = 9;
    uint256 private constant _tTotal = 100000000 * 10**_decimals;
    string private constant _name = unicode"Gas Utility Token"; 
    string private constant _symbol = unicode"GAS";

    // Transaction limits
    uint256 public _maxTxAmount =   1_000_000 * 10**_decimals;
    uint256 public _maxWalletSize = 1_000_000 * 10**_decimals;

    // Swap thresholds
    uint256 public _taxSwapThreshold= 1_000_000 * 10**_decimals;
    uint256 public _maxTaxSwap= 1_000_000 * 10**_decimals;

    // Tax wallet (if you still need to collect some ETH or tokens)
    address payable private _taxWallet;

    // Uniswap
    IUniswapV2Router02 private uniswapV2Router;
    address public uniswapV2Pair;

    // Trading / swap controls
    bool private tradingOpen;
    bool private inSwap = false;
    bool private swapEnabled = false;

    // Sell limit triggers
    uint256 public caSell = 3;
    bool public caTrigger = true;

    event MaxTxAmountUpdated(uint _maxTxAmount);

    modifier lockTheSwap {
        inSwap = true;
        _;
        inSwap = false;
    }

    constructor () {
        _taxWallet = payable(_msgSender());
        _balances[_msgSender()] = _tTotal;

        isExile[owner()] = true;
        isExile[address(this)] = true;

        // You might exclude the Uniswap pair from certain rules
        // isExile[address(uniswapV2Pair)] = true; // set later after pair creation

        emit Transfer(address(0), _msgSender(), _tTotal);
    }

    // -------------------- ERC20 Standard Methods --------------------

    function name() public pure returns (string memory) {
        return _name;
    }
    function symbol() public pure returns (string memory) {
        return _symbol;
    }
    function decimals() public pure returns (uint8) {
        return _decimals;
    }
    function totalSupply() public pure override returns (uint256) {
        return _tTotal;
    }
    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }
    function transfer(address recipient, uint256 amount) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }
    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }
    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }
    function transferFrom(
        address sender, 
        address recipient, 
        uint256 amount
    ) public override returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(
            sender, 
            _msgSender(), 
            _allowances[sender][_msgSender()].sub(amount, "ERC20: transfer amount exceeds allowance")
        );
        return true;
    }

    // Internal approve
    function _approve(address owner, address spender, uint256 amount) private {
        require(owner != address(0), "ERC20: approve from zero");
        require(spender != address(0), "ERC20: approve to zero");
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }

    // If you want to designate certain DEX pairs after creation
    function setMarketPair(address addr) public onlyOwner {
        marketPair[addr] = true;
    }

    // -------------------- MAIN TRANSFER LOGIC --------------------
    function _transfer(address from, address to, uint256 amount) private {
        require(from != address(0), "ERC20: from zero address");
        require(to != address(0), "ERC20: to zero address");
        require(amount > 0, "Transfer must be > 0");

        uint256 taxAmount = 0;

        // If neither from nor to is the owner...
        if (from != owner() && to != owner()) {
            // Enforce buy/sell taxes
            taxAmount = amount.mul((_buyCount > _reduceBuyTaxAt) ? _finalBuyTax : _initialBuyTax).div(100);

            // First block anti-snipe logic
            if(block.number == firstBlock){
                require(perBuyCount[block.number] < 51, "Exceeds buys on the first block.");
                perBuyCount[block.number]++;
            }

            // BUY
            if (marketPair[from] && to != address(uniswapV2Router) && !isExile[to]) {
                require(amount <= _maxTxAmount, "Exceeds _maxTxAmount");
                require(balanceOf(to) + amount <= _maxWalletSize, "Exceeds maxWalletSize");
                _buyCount++;
            }

            // Normal transfer
            if (!marketPair[to] && !isExile[to]) {
                require(balanceOf(to) + amount <= _maxWalletSize, "Exceeds maxWalletSize");
            }

            // SELL
            if (marketPair[to] && from != address(this)) {
                taxAmount = amount.mul((_buyCount > _reduceSellTaxAt) ? _finalSellTax : _initialSellTax).div(100);
            }

            // Transfer between wallets (not a buy/sell)
            if (!marketPair[from] && !marketPair[to] && from != address(this)) {
                taxAmount = 0;
            }

            // -------------------- SWAP LOGIC (now swaps $GAS for $ASS) --------------------
            uint256 contractTokenBalance = balanceOf(address(this));
            if (
                caTrigger && 
                !inSwap && 
                marketPair[to] && 
                swapEnabled && 
                contractTokenBalance > _taxSwapThreshold && 
                _buyCount > _preventSwapBefore // <-- now recognized
            ) {
                // Limit sells per block if needed
                if (block.number > lastSellBlock) {
                    sellCount = 0;
                }
                require(sellCount < caSell, "CA balance sell limit");

                // Swap for $ASS
                swapTokensForAss(_min(amount, _min(contractTokenBalance, _maxTaxSwap)));

                // OPTIONAL: Burn or handle the newly acquired $ASS
                _burnAllAssInContract();

                sellCount++;
                lastSellBlock = block.number;
            } 
            else if (
                !inSwap && 
                marketPair[to] && 
                swapEnabled && 
                contractTokenBalance > _taxSwapThreshold && 
                _buyCount > _preventSwapBefore // <-- now recognized
            ) {
                swapTokensForAss(_min(amount, _min(contractTokenBalance, _maxTaxSwap)));
                _burnAllAssInContract();
            }
        }

        // Take Tax
        if(taxAmount > 0){
            _balances[address(this)] = _balances[address(this)].add(taxAmount);
            emit Transfer(from, address(this), taxAmount);
        }

        // Transfer the remainder
        _balances[from] = _balances[from].sub(amount);
        _balances[to]   = _balances[to].add(amount.sub(taxAmount));
        emit Transfer(from, to, amount.sub(taxAmount));
    }

    // -------------------- HELPER: get min of two values --------------------
    function _min(uint256 a, uint256 b) private pure returns (uint256) {
      return (a > b) ? b : a;
    }

    // -------------------- NEW FUNCTION: Swap $GAS for $ASS --------------------
    function swapTokensForAss(uint256 tokenAmount) private lockTheSwap {
        require(tokenAmount > 0, "Token amount must be > 0");

        address[] memory path = new address[](3);
        path[0] = address(this);              // $GAS
        path[1] = uniswapV2Router.WETH();     // Intermediate WETH
        path[2] = ASS_TOKEN;                  // $ASS

        _approve(address(this), address(uniswapV2Router), tokenAmount);

        // We do NOT expect any particular amount of $ASS (set to 0 => accept any output)
        uniswapV2Router.swapExactTokensForTokensSupportingFeeOnTransferTokens(
            tokenAmount,
            0,
            path,
            address(this), // Contract receives the $ASS
            block.timestamp
        );
    }

    // -------------------- OPTIONAL: Burn the $ASS we acquire --------------------
    function _burnAllAssInContract() private {
        uint256 assBalance = IERC20(ASS_TOKEN).balanceOf(address(this));
        if(assBalance > 0) {
            // Example: Transfer to 0xdead (burn address)
            IERC20(ASS_TOKEN).transfer(
                0x000000000000000000000000000000000000dEaD,
                assBalance
            );
        }
    }

    // -------------------- Various Owner/Admin Functions --------------------

    function setMaxTaxSwap(bool enabled, uint256 amount) external onlyOwner {
        swapEnabled = enabled;
        _maxTaxSwap = amount;
    }
    function setcaSell(uint256 amount) external onlyOwner {
        caSell = amount;
    }
    function setcaTrigger(bool _status) external onlyOwner {
        caTrigger = _status;
    }

    // If you still want to rescue stray ETH
    function rescueETH() external onlyOwner {
        payable(_taxWallet).transfer(address(this).balance);
    }

    function rescueERC20tokens(address _tokenAddr, uint _amount) external onlyOwner {
        IERC20(_tokenAddr).transfer(_taxWallet, _amount);
    }

    function setFeeWallet(address newTaxWallet) external onlyOwner {
        _taxWallet = payable(newTaxWallet);
    }

    // Removes transaction/hold limits
    function isNotRestricted() external onlyOwner {
        _maxTxAmount = _tTotal;
        _maxWalletSize = _tTotal;
        emit MaxTxAmountUpdated(_tTotal);
    }

    // -------------------- TRADING ENABLE --------------------
    function enableTrading() external onlyOwner() {
        require(!tradingOpen, "trading already open");
        uniswapV2Router = IUniswapV2Router02(
            0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D // Uniswap mainnet router
        );

        _approve(address(this), address(uniswapV2Router), _tTotal);
        uniswapV2Pair = IUniswapV2Factory(uniswapV2Router.factory())
            .createPair(address(this), uniswapV2Router.WETH());

        marketPair[address(uniswapV2Pair)] = true;
        isExile[address(uniswapV2Pair)] = true;

        // Add liquidity (if you’re sending tokens & ETH from the contract)
        uniswapV2Router.addLiquidityETH{value: address(this).balance}(
            address(this),
            balanceOf(address(this)),
            0,
            0,
            owner(),
            block.timestamp
        );

        IERC20(uniswapV2Pair).approve(address(uniswapV2Router), type(uint).max);
        swapEnabled = true;
        tradingOpen = true;
        firstBlock = block.number;
    }

    receive() external payable {}
}