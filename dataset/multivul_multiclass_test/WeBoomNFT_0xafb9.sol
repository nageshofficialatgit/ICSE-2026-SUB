// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract WeBoomNFT {
    string public name = "WeBoom";
    string public symbol = "WB";
    uint256 public constant MAX_SUPPLY = 70;

    uint256 public totalSupply;
    address public immutable owner;
    
    mapping(uint256 => address) public ownerOf;
    mapping(address => uint256) public balanceOf;

    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);

    constructor() {
        owner = msg.sender;
    }

    function mint() external {
        require(msg.sender == owner, "Only owner can mint");
        require(totalSupply < MAX_SUPPLY, "Max supply reached");

        uint256 tokenId = totalSupply + 1;
        ownerOf[tokenId] = owner;
        balanceOf[owner] += 1;
        totalSupply += 1;

        emit Transfer(address(0), owner, tokenId);
    }

    function transfer(address to, uint256 tokenId) external {
        require(to != address(0), "Cannot transfer to zero address");
        require(ownerOf[tokenId] == msg.sender, "Not the owner");

        ownerOf[tokenId] = to;
        balanceOf[msg.sender] -= 1;
        balanceOf[to] += 1;

        emit Transfer(msg.sender, to, tokenId);
    }
}