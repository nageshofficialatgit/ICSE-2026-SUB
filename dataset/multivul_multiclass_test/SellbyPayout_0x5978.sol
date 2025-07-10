// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function allowance(
        address owner,
        address spender
    ) external view returns (uint256);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(
        address from,
        address to,
        uint256 value
    ) external returns (bool);
}

library SafeERC20 {
    function safeTransfer(IERC20 _token, address _to, uint256 _value) internal returns (bool) {
        uint256 prevBalance = _token.balanceOf(address(this));

        if (prevBalance < _value) {
            return false;
        }

        address(_token).call(abi.encodeWithSignature("transfer(address,uint256)", _to, _value));

        return prevBalance - _value == _token.balanceOf(address(this));
    }
}

contract SellbyPayout {
    address public owner;

    constructor() payable {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    function distributeEth(
        address[] calldata recipients,
        uint256[] calldata amounts
    ) external onlyOwner {
        require(recipients.length == amounts.length);

        uint256 totalAmount = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }

        require(address(this).balance >= totalAmount);

        for (uint256 i = 0; i < recipients.length; i++) {
            payable(recipients[i]).transfer(amounts[i]);
        }
    }

    function distributeToken(
        IERC20 token,
        address[] calldata recipients,
        uint256[] calldata amounts
    ) external onlyOwner {
        require(recipients.length == amounts.length);

        uint256 totalAmount = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }

        require(totalAmount <= token.balanceOf(address(this)));

        for (uint256 i = 0; i < recipients.length; i++) {
            SafeERC20.safeTransfer(token, recipients[i], amounts[i]);
        }
    }

    function withraw(address addr, uint amount) external onlyOwner {
        payable(addr).transfer(amount);
    }

    function withdrawToken(IERC20 token, address to, uint amount) external onlyOwner {
        SafeERC20.safeTransfer(token, to, amount);
    }

   function balance() public view returns (uint256) {
        return address(this).balance;
    }

    function balanceToken(IERC20 token) public view returns (uint256) {
        return token.balanceOf(address(this));
    }

    receive() external payable {}
}