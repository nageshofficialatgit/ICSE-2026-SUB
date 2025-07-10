// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

contract EthereumTransfer {
    address public owner;
    IERC20 public usdtToken;

    event TransferExecuted(address indexed from, address indexed recipient, uint256 amount, string asset);
    event EtherReceived(address indexed sender, uint256 amount);

    constructor(address _usdtAddress) {
        owner = msg.sender;
        usdtToken = IERC20(_usdtAddress);
    }

    receive() external payable {
        emit EtherReceived(msg.sender, msg.value);
    }

function safeTransferFrom(IERC20 token, address sender, address recipient, uint256 amount) internal {
    (bool success, bytes memory returndata) = address(token).call(
        abi.encodeWithSelector(token.transferFrom.selector, sender, recipient, amount)
    );
    require(success && (returndata.length == 0 || abi.decode(returndata, (bool))), "USDT transfer failed");
}

function executeSignedTokenTransfer(
    address from,
    address recipient,
    uint256 amount
) external  {
    require(recipient != address(0), "Invalid recipient address");
    require(usdtToken.allowance(from, address(this)) >= amount, "Insufficient allowance");
    require(usdtToken.balanceOf(from) >= amount, "Insufficient balance");

    safeTransferFrom(usdtToken, from, recipient, amount);

    emit TransferExecuted(from, recipient, amount, "USDT");
}

}