pragma solidity 0.8.25;

pragma experimental ABIEncoderV2;

// SPDX-License-Identifier: MIT

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        this; // silence state mutability warning without generating bytecode - see https://github.com/ethereum/solidity/issues/2691
        return msg.data;
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

contract AuthorizedChecker is Ownable {

    mapping (address => bool) public deployerAddress;
    mapping (address => bool) public incubatorAddress;
    mapping (address => address) public deployersIncubatorAddress;

    constructor(address _owner){
        incubatorAddress[_owner] = true;
        deployerAddress[_owner] = true;
        transferOwnership(_owner);
    }

    modifier onlyAuthorized {
        require(incubatorAddress[msg.sender], "Not Authorized");
        _;
    }

    function updateIncubator(address _address, bool _isAuthorized) external onlyOwner {
        incubatorAddress[_address]  = _isAuthorized;
    }

    function updateDeployerAddress(address _address, bool _isAuthorized) external onlyAuthorized {
        if(deployersIncubatorAddress[_address] == address(0)){
            deployersIncubatorAddress[_address] = msg.sender;
        } else {
            require(deployersIncubatorAddress[_address] == msg.sender);
        }
        deployerAddress[_address]  = _isAuthorized;
    }
}