// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Love{
    address public admin;
    
    struct Lover {
        string nickname;
        string name;
    }
    
    Lover[] private lovers;
    
    event LoverAdded(string nickname, string name);
    
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this action");
        _;
    }
    
    constructor() {
        admin = 0xbBE332256170F067e8d68211d3C63b936043782D;
    }
    
    function addLover(string memory _nickname, string memory _name) external onlyAdmin {
        lovers.push(Lover(_nickname, _name));
        emit LoverAdded(_nickname, _name);
    }
    
    function love() external view returns (Lover[] memory) {
        return lovers;
    }
}