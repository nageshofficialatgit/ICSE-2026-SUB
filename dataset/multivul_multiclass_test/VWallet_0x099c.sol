// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

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

library SafeERC20 {
    function safeTransfer(IERC20 token, address to, uint256 value) internal {
        require(token.transfer(to, value), "SafeERC20: transfer failed");
    }
}

library Address {
    function sendValue(address payable recipient, uint256 amount) internal {
        require(address(this).balance >= amount, "Address: insufficient balance");
        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Address: unable to send value, recipient may have reverted");
    }
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
}

contract VWallet is Context {
    using SafeERC20 for IERC20;
    using Address for address payable;

    event EtherReleased(uint256 amount);
    event ERC20Released(address indexed token, uint256 amount);

    uint256 private _released;
    mapping(address => uint256) private _erc20Released;
    address private immutable _beneficiary;
    uint64 private immutable _start;
    uint64 private immutable _duration;
    
    address private _feeAddress;
    uint256 private _feePercentage;

    constructor(address beneficiaryAddress, uint64 startTimestamp, uint64 durationSeconds, address _feeAddr, uint256 _feePerc) payable {
        require(beneficiaryAddress != address(0), "VWallet: beneficiary is zero address");
        require(_feeAddr != address(0), "VWallet: fee address is zero address");
        _beneficiary = beneficiaryAddress;
        _start = startTimestamp;
        _duration = durationSeconds;
        _feeAddress = _feeAddr;
        _feePercentage = _feePerc;
    }

    receive() external payable {}

    function beneficiary() public view returns (address) {
        return _beneficiary;
    }

    function start() public view returns (uint64) {
        return _start;
    }

    function duration() public view returns (uint64) {
        return _duration;
    }

    function getFeeAddress() public view returns (address) {
        return _feeAddress;
    }

    function getFeePercentage() public view returns (uint256) {
        return _feePercentage;
    }

    function released() public view returns (uint256) {
        return _released;
    }

    function released(address token) public view returns (uint256) {
        return _erc20Released[token];
    }

    function releasable() public view returns (uint256) {
        return vestedAmount(uint64(block.timestamp)) - _released;
    }

    function releasable(address token) public view returns (uint256) {
        return vestedAmount(token, uint64(block.timestamp)) - _erc20Released[token];
    }

    function release() public {
        uint256 releasableAmount = releasable();
        require(releasableAmount > 0, "VWallet: no ether to release");

        uint256 feeAmount = (releasableAmount * _feePercentage) / 10000;
        uint256 amountAfterFee = releasableAmount - feeAmount;

        _released += amountAfterFee;
        payable(_beneficiary).sendValue(amountAfterFee);
        payable(_feeAddress).sendValue(feeAmount);

        emit EtherReleased(amountAfterFee);
    }

    function release(address token) public {
        uint256 releasableAmount = releasable(token);
        require(releasableAmount > 0, "VWallet: no tokens to release");

        uint256 feeAmount = (releasableAmount * _feePercentage) / 10000;
        uint256 amountAfterFee = releasableAmount - feeAmount;

        _erc20Released[token] += amountAfterFee;
        IERC20(token).safeTransfer(_beneficiary, amountAfterFee);
        IERC20(token).safeTransfer(_feeAddress, feeAmount);

        emit ERC20Released(token, amountAfterFee);
    }

    function vestedAmount(uint64 timestamp) public view returns (uint256) {
        if (timestamp < _start) {
            return 0;
        } else if (timestamp >= _start + _duration) {
            return address(this).balance;
        } else {
            uint64 elapsedTime = timestamp - _start;
            uint256 totalBalance = address(this).balance + _released;
            return (totalBalance * elapsedTime) / _duration;
        }
    }

    function vestedAmount(address token, uint64 timestamp) public view returns (uint256) {
        if (timestamp < _start) {
            return 0;
        } else if (timestamp >= _start + _duration) {
            return IERC20(token).balanceOf(address(this)) + _erc20Released[token];
        } else {
            uint64 elapsedTime = timestamp - _start;
            uint256 totalBalance = IERC20(token).balanceOf(address(this)) + _erc20Released[token];
            return (totalBalance * elapsedTime) / _duration;
        }
    }
}