// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FakeUSDT {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    string public tokenImageURL; // ✅ Stores token image URL
    uint256 public fakePrice = 104; // ✅ Fake price ($1.04 per token)
    address public owner;

    mapping(address => uint256) public balances;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event MetadataUpdated(string newName, string newSymbol);
    event SupplyChanged(uint256 newSupply);
    event ImageUpdated(string newImage);
    event PriceUpdated(uint256 newPrice);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    constructor() {
        owner = msg.sender;
        name = "Tether USD";  // ✅ Initially set as USDT
        symbol = "USDT";
        decimals = 6;
        totalSupply = 1_000_000 * 10**decimals;
        balances[msg.sender] = totalSupply;
        tokenImageURL = "https://ipfs.io/ipfs/QmFakeExample";  // ✅ Default fake image
    }

    // ✅ Change Name & Symbol Anytime
    function changeMetadata(string memory newName, string memory newSymbol) public onlyOwner {
        name = newName;
        symbol = newSymbol;
        emit MetadataUpdated(newName, newSymbol);
    }

    // ✅ Update Token Image
    function updateTokenImage(string memory newImageURL) public onlyOwner {
        tokenImageURL = newImageURL;
        emit ImageUpdated(newImageURL);
    }

    // ✅ Fake Price Manipulation
    function getFakePrice() public view returns (uint256) {
        return fakePrice;  // Always returns $1.04 per token
    }

    function updateFakePrice(uint256 newPrice) public onlyOwner {
        fakePrice = newPrice;
        emit PriceUpdated(newPrice);
    }

    // ✅ Mint More Tokens Anytime
    function increaseSupply(uint256 amount) public onlyOwner {
        totalSupply += amount;
        balances[msg.sender] += amount;
        emit SupplyChanged(totalSupply);
    }

    function transfer(address _to, uint256 _amount) public returns (bool) {
        require(balances[msg.sender] >= _amount, "Not enough Fake USDT");
        balances[msg.sender] -= _amount;
        balances[_to] += _amount;
        emit Transfer(msg.sender, _to, _amount);
        return true;
    }

    function balanceOf(address _owner) public view returns (uint256) {
        return balances[_owner];
    }
}