// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;


interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
}

contract halakhot {
    address public owner;
    
    address public tokenAddress;
    uint256 public minTokenBalance;
    
    string public publicString1;
    string public publicString2;
    string public publicString3;
    string public publicString4;
    string public publicString5;
    
    string private ownerString1;
    string private ownerString2;
    string private ownerString3;
    string private ownerString4;
    string private ownerString5;
    
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
    event TokenRequirementUpdated(address tokenAddress, uint256 minTokenBalance);

    constructor(address _tokenAddress, uint256 _minTokenBalance) {
        owner = msg.sender;
        tokenAddress = _tokenAddress;
        minTokenBalance = _minTokenBalance;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Seul le proprietaire peut executer cette fonction");
        _;
    }
    
    modifier hasMinimumBalance() {
        require(
            IERC20(tokenAddress).balanceOf(msg.sender) >= minTokenBalance,
            "Solde de tokens insuffisant pour ecrire dans les champs publics"
        );
        _;
    }
    
    function setPublicString1(string memory _value) public hasMinimumBalance {
        publicString1 = _value;
    }
    
    function setPublicString2(string memory _value) public hasMinimumBalance {
        publicString2 = _value;
    }
    
    function setPublicString3(string memory _value) public hasMinimumBalance {
        publicString3 = _value;
    }
    
    function setPublicString4(string memory _value) public hasMinimumBalance {
        publicString4 = _value;
    }
    
    function setPublicString5(string memory _value) public hasMinimumBalance {
        publicString5 = _value;
    }
    
    function setOwnerString1(string memory _value) public onlyOwner {
        ownerString1 = _value;
    }
    
    function setOwnerString2(string memory _value) public onlyOwner {
        ownerString2 = _value;
    }
    
    function setOwnerString3(string memory _value) public onlyOwner {
        ownerString3 = _value;
    }
    
    function setOwnerString4(string memory _value) public onlyOwner {
        ownerString4 = _value;
    }
    
    function setOwnerString5(string memory _value) public onlyOwner {
        ownerString5 = _value;
    }
    
    function getOwnerString1() public view returns (string memory) {
        return ownerString1;
    }
    
    function getOwnerString2() public view returns (string memory) {
        return ownerString2;
    }
    
    function getOwnerString3() public view returns (string memory) {
        return ownerString3;
    }
    
    function getOwnerString4() public view returns (string memory) {
        return ownerString4;
    }
    
    function getOwnerString5() public view returns (string memory) {
        return ownerString5;
    }
    
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "L'adresse ne peut pas etre zero");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
    
    function setTokenRequirement(address _tokenAddress, uint256 _minTokenBalance) public onlyOwner {
        tokenAddress = _tokenAddress;
        minTokenBalance = _minTokenBalance;
        emit TokenRequirementUpdated(_tokenAddress, _minTokenBalance);
    }
}