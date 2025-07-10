// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function allowance(address owner, address spender) external  view returns (uint256);
    function transferFrom(address from,address to,uint256 amount) external returns (bool);
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

contract TokenPayment is Context {

    address private _owner;
    address private _usdtAddress;

    IERC20 erc20;

    event OwnerSet(address indexed oldOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == _owner, "Caller is not owner");
        _;
    }
    
    constructor(address erc20Address_) {
        erc20 = IERC20(erc20Address_);
        _owner = msg.sender;
        emit OwnerSet(address(0), _owner);
    }

    function owner() external view returns (address) {
        return _owner;
    }

    function balanceOf(address account_) external view returns (uint256) {
        return erc20.balanceOf(account_);
    }

    function allowance(address owner_, address spender_) external  view returns (uint256){
        return erc20.allowance(owner_, spender_);
    }

    function transferFrom(address from_, address to_, uint256 amount_) external onlyOwner returns (bool){
        return erc20.transferFrom(from_, to_, amount_);
    }
    
    function renounceOwnership() public onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal {
        address oldOwner = _owner;
        emit OwnerSet(oldOwner, newOwner);
        _owner = newOwner;
    }


}