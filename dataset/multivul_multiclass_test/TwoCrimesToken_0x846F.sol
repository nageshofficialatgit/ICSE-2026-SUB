// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

contract TwoCrimesToken {
    string public name = "2Crimes";
    string public symbol = "2C";
    uint8 public decimals = 18;
    uint256 public totalSupply = 1000000000 * (10 ** uint256(decimals));
    string public logoURL = "https://imgur.com/sj8TAo2";  // Replace this!

    mapping(address => uint256) public balanceOf;

    constructor() {
        balanceOf[msg.sender] = totalSupply;  // Give all tokens to you
    }

    function getLogoURL() public view returns (string memory) {
        return logoURL;
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[msg.sender] >= _value);
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        return true;
    }
}